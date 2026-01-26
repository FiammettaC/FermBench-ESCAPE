import anthropic
import os
import pandas as pd
import json
from tqdm import tqdm
from api_config import claude_key

# Set up your API key
client = anthropic.Anthropic(
    api_key=claude_key(),  # Or use os.environ["ANTHROPIC_API_KEY"]
)

def load_data(path):
    df = pd.read_excel(path)
    return df

def return_responses(df, dest_path):

    for question in tqdm(df["question"]):
        completion = {}
        try:
            # Make a simple request
            message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            messages=[{"role": "user", "content": question}])
            
            # Check if response has content
            if message.content and len(message.content) > 0:
                completion[question] = message.content[0].text
            else:
                completion[question] = f"Empty response: {message.stop_reason}"
        except Exception as e:
            completion[question] = f"Error: {str(e)}"
        
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')

# Run code
data_path = "data/dataset_curated.xlsx"
output_path = "data/dataset_completion_claude2.jsonl"
return_responses(load_data(data_path), output_path)