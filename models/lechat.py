import os
from mistralai import Mistral
import pandas as pd
import json
from api_config import lechat_key
from tqdm import tqdm

model = "mistral-small-latest"
client = Mistral(api_key=lechat_key())

def load_data(path):
    df = pd.read_excel(path)
    return df

def return_responses(df, dest_path):
    for question in tqdm(df["question"]):
        completion = {}
        chat_response = client.chat.complete(
        model = model,
        messages = [{"role": "user", "content": question}])
        completion[question] = chat_response.choices[0].message.content
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')

# Run code
data_path = "data/dataset_curated.xlsx"
output_path = "data/dataset_completion_lechat.jsonl"
return_responses(load_data(data_path), output_path)