from openai import OpenAI
import pandas as pd
import json
from api_config import deepseek_key
import time
from tqdm import tqdm

start_time = time.time()

client = OpenAI(api_key=deepseek_key(), base_url="https://api.deepseek.com")

def load_data(path):
    df = pd.read_excel(path)
    return df

def return_responses(df, dest_path):
    for question in tqdm(df["question"]):
        completion = {}
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": question},],
            stream=False)
        completion[question] = response.choices[0].message.content
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')

# Run code
data_path = "data/dataset_curated.xlsx"
output_path = "data/dataset_completion_deepseek.jsonl"
return_responses(load_data(data_path), output_path)
#print("--- %s seconds ---" % (time.time() - start_time))

#--- 4364.234277248383 seconds ---
#--- 4090.1609075069427 seconds ---