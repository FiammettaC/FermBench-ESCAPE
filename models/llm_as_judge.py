from api_config import openai_api_key
from openai import OpenAI
import json
import pandas as pd
from transformers import pipeline

def load_data_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            json_obj = json.loads(line)
            for key, value in json_obj[0].items():
                data.append({'key': key, 'value': value})
    df = pd.DataFrame(data)
    return df

def llm_as_judge_gpt(df, dest_path):    
    client = OpenAI(api_key=openai_api_key())
  
    for question, answer in zip(df['key'], df[f'value']):
        prompt = f"Task: given a specific question, please evaluate the given answer \
        on a 1-5 scale (5 being the highest) for each of these factors: \
        factual accuracy, comprehensiveness, coherence, concision, \
        readability, helpfulness, insightfulness and overall quality. \
        An explanations of the various factors is provided below.\
        Factual accuracy aims to assess to what extent the answer is correct \
        or if the model is hallucinating. \
        Comprehensiveness measures the completeness of the response, \
        whether it answers the question fully or if there is something missing.\
        Coherence focuses on whether the response tackles the question and aligns with user intent. \
        Concision evaluates whether the response is redundant or contains superfluous information. \
        Readability provides an overview of how easy the response is to read and understand \
        or if it is too complex, if the language is fluent, natural and grammatically correct. \
        Helpfulness evaluates whether the answer is useful to the user/student. \
        Insightfulness tackles the depth of the answer and if it offers any novel perspectives \
        or goes beyond a superficial answer. \
        Overall quality provides a general assessment of the response.\
        Please return your answer as a dictionary, where the key is the factor the score is the value. \
        Do not elaborate further. The question asked is: {question}, and its relative answer is: {answer}."
        completion = {}
        response = client.responses.create(
        model="gpt-5.2-2025-12-11",
        input=prompt)
        completion[question] = response.output_text
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')

def llm_as_judge_gpt_pairwise(df, dest_path):    
    client = OpenAI(api_key=openai_api_key())
  
    c = 0    
    for question, answer1, answer2 in zip(df['Question'], df['Response A'], df['Response B']):
        c+=1
        prompt = f"Task: given this question, {question}, \
        please evaluate which is a better response between {answer1} and {answer2}. \
        Please return the response as either 'Response A' or 'Response B'."
        completion = {}
        response = client.responses.create(
        model="gpt-5.2-2025-12-11",
        input=prompt)
        completion[c] = response.output_text
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')
 


#df = load_data("data/dataset_merged_all_reruns.xlsx")
#m = "gemini"


m = "chatgpt"
df = load_data_jsonl(f"data/dataset_completion_{m}.jsonl")
llm_as_judge_gpt(df, f"data/llm_as_judge_gpt_{m}.jsonl")



"""
from json_to_excel import load_data_jsonl
import re

models = ['chatgpt', 'claude', 'gemini', 'deepseek', 'lechat'] #'chatgpt', 
runs = ['', '_rerun1', '_rerun2']
judge = 'gemma'

#dict_chatgpt = {}
#dict_claude = {}
#dict_lechat = {}
#dict_gemini = {}
#dict_deepseek = {}

for model in models:
    dict = {}
    fa = 0
    comp = 0
    coh = 0
    conc = 0
    read = 0
    help = 0
    ins = 0
    oq = 0
    missed = 0
    for run in runs:
        data = load_data_jsonl(f"data/llm_as_judge_{judge}/llm_as_judge_{judge}_{model}{run}.jsonl")
        for idx, d in enumerate(data):
            numbers = re.findall(r'[+-]?\d*\.?\d+', d['value'])
            #print(numbers)
            numbers = [round(float(n)) for n in numbers]
            #print(numbers)
            #print(len(numbers))
            if len(numbers) == 8:
                fa += int(numbers[0])
                comp += int(numbers[1])
                coh += int(numbers[2])
                conc += int(numbers[3])
                read += int(numbers[4])
                help += int(numbers[5])
                ins += int(numbers[6])
                oq += int(numbers[7])
            else:
                #print(model, run, d)
                missed +=1
                continue
        dict['factual_accuracy'] = round(fa/(len(data)*3), 2)
        dict['comprehensiveness'] = round(comp/(len(data)*3), 2)
        dict['coherence'] = round(coh/(len(data)*3), 2)
        dict['concision'] = round(conc/(len(data)*3), 2)
        dict['readability'] = round(read/(len(data)*3), 2)
        dict['helpfulness'] = round(help/(len(data)*3), 2)
        dict['insightfulness'] = round(ins/(len(data)*3), 2)
        dict['overall_quality'] = round(oq/(len(data)*3), 2)  
    print(model)
    print(dict) 
    print(missed) 
                

"""

# Pairwise analysis
#df = pd.read_excel("data/pairwise_dataset.xlsx")
#llm_as_judge_gpt_pairwise(df, "data/ll_as_judge_pairwise_gpt4o.jsonl")
