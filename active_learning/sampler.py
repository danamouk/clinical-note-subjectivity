import pandas as pd
from tqdm import tqdm
import random

df = pd.read_csv('../data/output_predictions.csv', header=None)
df = df.rename(columns={
    0:'label',
    1: 'score',
    2: 'text',
    3: 'begin',
    4: 'end',
    5: 'id',
})

note_df = pd.read_csv('./data/NOTEEVENTS.csv', low_memory=False)
scored_df = pd.read_csv('../data/annotations.csv')
prev_set = set([int(x.split('-')[0]) for x in scored_df['id'].dropna()])
df = df.loc[df.loc[:, 'id'].isin(prev_set) == False].reset_index().drop(columns=['index'])

sorted_df = df.sort_values(by='score').dropna().reset_index().drop(columns=['index'])

NUM_SETS = 7
NUM_ANNOT_PER_SET = 2
TEXT_PER_SET = 250
RANDOM_SAMPLE_PER_SET = 50
SCORE_THRESHOLD = 1.0 

task_prefix = 'subj_annot_p'
save_dir = '../data/upload_files/round_3/'

sampled_df = sorted_df[sorted_df['score'] < SCORE_THRESHOLD]
sampled_df = sampled_df[['id', 'begin', 'end', 'text', 'score']]
# this keeps track of the number of instances per pattern words. 
# This is to make sure we do not have too many sentences focusing around the same word.
cnt_dict_2 = dict() 

keep_idx = []
for itr in tqdm(range(0, len(sampled_df))):
    row_id = sampled_df.loc[itr, 'id'] 
    begin = sampled_df.loc[itr, 'begin']
    end = sampled_df.loc[itr, 'end']
    full_text = note_df.loc[note_df['ROW_ID'] == row_id, 'TEXT'].values[0]
    pat = full_text[begin:end]
    
    if pat not in cnt_dict_2.keys():
        cnt_dict_2[pat] = 0
    
    cnt_dict_2[pat] += 1
    if cnt_dict_2[pat] > 50:
        continue
    else:
        keep_idx.append(itr)
    
    sampled_df.loc[itr, 'id'] = '-'.join([str(sampled_df.loc[itr, 'id']), str(sampled_df.loc[itr, 'begin']), str(sampled_df.loc[itr, 'end'])])
    sampled_df.loc[itr, 'text'] = full_text
    
    if len(keep_idx) >= TEXT_PER_SET * NUM_SETS:
        break
        
for itr in tqdm(range(0, len(sampled_df))):
    if itr in keep_idx:
        continue
    if len(keep_idx) >= TEXT_PER_SET * NUM_SETS:
        break
    row_id = sampled_df.loc[itr, 'id'] 
    begin = sampled_df.loc[itr, 'begin']
    end = sampled_df.loc[itr, 'end']
    full_text = note_df.loc[note_df['ROW_ID'] == row_id, 'TEXT'].values[0]
    pat = full_text[begin:end]
    
    keep_idx.append(itr)
    
    sampled_df.loc[itr, 'id'] = '-'.join([str(sampled_df.loc[itr, 'id']), str(sampled_df.loc[itr, 'begin']), str(sampled_df.loc[itr, 'end'])])
    sampled_df.loc[itr, 'text'] = full_text
    
    if len(keep_idx) > TEXT_PER_SET * NUM_SETS:
        break
    
random_sampled_df = sorted_df[sorted_df['score'] >= SCORE_THRESHOLD].sample(frac=1)[0:RANDOM_SAMPLE_PER_SET*NUM_SETS].reset_index().drop(columns=['index'])
for itr in tqdm(range(0, len(random_sampled_df))):
    row_id = random_sampled_df.loc[itr, 'id'] 
    begin = random_sampled_df.loc[itr, 'begin']
    end = random_sampled_df.loc[itr, 'end']
    full_text = note_df.loc[note_df['ROW_ID'] == row_id, 'TEXT'].values[0]
    
    random_sampled_df.loc[itr, 'id'] = '-'.join([str(random_sampled_df.loc[itr, 'id']), str(random_sampled_df.loc[itr, 'begin']), str(random_sampled_df.loc[itr, 'end'])])
    random_sampled_df.loc[itr, 'text'] = full_text
    
    if len(keep_idx) > TEXT_PER_SET * NUM_SETS:
        break
        
upload_df = pd.concat([sampled_df.loc[sorted(keep_idx)], random_sampled_df]).sample(frac=1).reset_index().drop(columns=['index', 'score', 'label'])
upload_df.loc[:, 'task'] = 'subjectivity_classification'

total_per_set = TEXT_PER_SET + RANDOM_SAMPLE_PER_SET
for upload_set_itr in range(0, NUM_SETS):
    cur_upload_set = upload_df[upload_set_itr*total_per_set: (upload_set_itr+1)*total_per_set]
    for annot_itr in range(0, NUM_ANNOT_PER_SET):
        task_name = task_prefix + str(upload_set_itr) + '_' + chr(annot_itr+97)
        cur_upload_set.to_csv(save_dir + task_name, index=None, header=None)