import google.genai as genai
import google.genai.errors as genai_errors
import pandas as pd
import json
import time
import random
import os
from tqdm import tqdm
from api_config import gemini_key_paid

client = genai.Client(api_key=gemini_key_paid())


def load_data(path):
    df = pd.read_excel(path)
    return df


def generate_with_retries(question, model="gemini-3-flash-preview", max_retries=None, base_delay=1.0):
    """Call the Gemini model with retries on ServerError (503 UNAVAILABLE).

    max_retries can be overridden by the environment variable `GEMINI_MAX_RETRIES`.
    """
    if max_retries is None:
        try:
            max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "6"))
        except Exception:
            max_retries = 6

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=question)
            return response
        except genai_errors.ServerError as e:
            # ServerError includes overloaded / 503 conditions — retry with exponential backoff
            if attempt == max_retries:
                raise
            # exponential backoff with jitter
            delay = base_delay * (2 ** (attempt - 1))
            delay += random.uniform(0, min(1.0, delay))
            print(f"ServerError (attempt {attempt}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
        except Exception:
            # For other exceptions, don't mask them — re-raise
            raise


def return_responses(df, dest_path):
    for question in tqdm(df["question"]):
        completion = {}
        try:
            response = generate_with_retries(question)
            text = getattr(response, "text", str(response))
            completion[question] = text
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Failed to get response for question: {e}")
            completion[question] = None

        with open(dest_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps([completion], ensure_ascii=False) + '\n')


# Run code
data_path = "data/dataset_curated.xlsx"
output_path = "data/dataset_completion_gemini_3flash.jsonl"
return_responses(load_data(data_path), output_path)