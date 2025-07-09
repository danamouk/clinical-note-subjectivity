import pandas as pd
from tqdm import tqdm

from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import nltk
#nltk.download('punkt')
import random
import numpy as np
import re
PADDING_LEFT = 50
PADDING_RIGHT = 50

main_df = pd.read_csv('./data/NOTEEVENTS.csv', low_memory=False)
MODEL_PATH = './models/binary_classifier_model.py'
CSV_FILEPATH = './data/output_predictions.csv'
ROW_SAMPLER = 1.0
main_df = main_df.sample(frac=ROW_SAMPLER).reset_index().drop(columns=['index'])
words = set([ word[2:-1] for word in pd.read_csv('./data/m_Opinion.csv', header=None)[1]])

tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

def get_sentences(text, frac=1.0):
    sentences = []
    for pat in words:
        for match in re.finditer(pat, text):
            begin = match.start()
            end = match.end()
            sentences.append({
                'sentence': text,
                'begin': begin,
                'end': end,
            })
    if frac < 1.0: return random.sample(sentences, int(np.ceil(len(sentences) * frac)))
    else: return sentences   

def extract_predictions(text, generator, frac=1.0):
    sentences = get_sentences(text, frac=frac)
    gen_output = generator([x['sentence'][x['begin']-PADDING_LEFT: x['end']+PADDING_LEFT] for x in sentences], **tokenizer_kwargs)
    return [dict(gen_output[idx], begin=sentences[idx]['begin'], end=sentences[idx]['end']) for idx in range(0, len(sentences))]
    #sentence=sentences[idx]['sentence']
def write_output_for_each_report(text, id, generator, output_filename, frac=1.0):
    predictions = extract_predictions(text, generator, frac=frac)
    predictions = [dict(item, ROW_ID=id) for item in predictions]

    with open(output_filename, 'a') as f:
        pd.DataFrame(predictions).to_csv(f, header=f.tell()==0, index=None)

infer_model = torch.load(MODEL_PATH, map_location=torch.device('cuda:0'))

tokenizer = AutoTokenizer.from_pretrained("cffl/bert-base-styleclassification-subjective-neutral")
generator = pipeline(task="text-classification", model=infer_model, tokenizer=tokenizer, device=0) #, device=0)

for idx in tqdm(range(0, len(main_df))):
    id = main_df.loc[idx, 'ROW_ID']
    text = main_df.loc[idx, 'TEXT']
    write_output_for_each_report(text, id, generator, CSV_FILEPATH, frac=1)
