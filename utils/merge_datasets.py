import pandas as pd
import json
from functools import reduce

def load_data_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            json_obj = json.loads(line)
            for key, value in json_obj[0].items():
                data.append({'key': key, 'value': value})
    return data

def merge_responses(path):
    answers1 = pd.DataFrame(load_data_jsonl(f'{path}_gemini.jsonl')).rename(columns={"value": "value_gemini"}).drop_duplicates(subset='key', keep='first')
    answers2 = pd.DataFrame(load_data_jsonl(f'{path}_deepseek.jsonl')).rename(columns={"value": "value_deepseek"}).drop_duplicates(subset='key', keep='first')
    answers3 = pd.DataFrame(load_data_jsonl(f'{path}_lechat.jsonl')).rename(columns={"value": "value_lechat"}).drop_duplicates(subset='key', keep='first')
    answers4 = pd.DataFrame(load_data_jsonl(f'{path}_chatgpt.jsonl')).rename(columns={"value": "value_chatgpt"}).drop_duplicates(subset='key', keep='first')
    answers5 = pd.DataFrame(load_data_jsonl(f'{path}_claude.jsonl')).rename(columns={"value": "value_claude"}).drop_duplicates(subset='key', keep='first')

    data_frames = [answers1, answers2, answers3, answers4, answers5]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['key']), data_frames)

    df_merged.to_excel('data/dataset_merged_all.xlsx', index=False)
    return df_merged

path = "data/"
path2 = "data/dataset_completion"
merge_responses(path2)
