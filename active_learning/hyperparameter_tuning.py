import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
import torch

PADDING_LEFT = 50
PADDING_RIGHT = 50
TRAIN_TEST_SPLIT = .9
training_filename = './data/annotations.csv'
model_suffix = '_v2'
tokenizer = AutoTokenizer.from_pretrained("cffl/bert-base-styleclassification-subjective-neutral")
model = AutoModelForSequenceClassification.from_pretrained("cffl/bert-base-styleclassification-subjective-neutral")

def make_text_dataset(df):
  texts = [df.loc[i, 'text'][df.loc[i, 'begin']-PADDING_LEFT: df.loc[i, 'end']+PADDING_RIGHT] for i in range(0, len(df))]
  labels = [1 if x=='rejected' else 0 for x in list(df['annotation'])]
  return {
      'label': labels,
      'text': texts
  }

scored_df = pd.read_csv(training_filename)
train_indices = scored_df.sample(frac=TRAIN_TEST_SPLIT).index
test_indices = [x for x in scored_df.index if x not in train_indices]
assert(len(set(train_indices).intersection(test_indices)) == 0)
train_df = scored_df.loc[train_indices].reset_index().drop(columns=['index'])
test_df = scored_df.loc[test_indices].reset_index().drop(columns=['index'])

train_dataset = make_text_dataset(train_df)
test_dataset = make_text_dataset(test_df)

del scored_df, train_df, test_df, train_indices, test_indices

tdf = pd.DataFrame(train_dataset)
edf = pd.DataFrame(test_dataset)
tds = Dataset.from_pandas(tdf)
eds = Dataset.from_pandas(edf)


dataset = DatasetDict()

dataset['train'] = tds
dataset['eval'] = eds

print(dataset)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["eval"]

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

torch.save(model, './models/binary_classifier_model' + model_suffix + '.py')

predictions = trainer.predict(tokenized_datasets["eval"])
print(predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)

import evaluate

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

for metric in [accuracy_metric, precision_metric, recall_metric, f1_metric]:
    print(metric.compute(predictions=preds, references=predictions.label_ids))
