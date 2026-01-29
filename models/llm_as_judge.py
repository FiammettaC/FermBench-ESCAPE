from xml.parsers.expat import model
from api_config import openai_api_key, claude_key
import anthropic
from openai import OpenAI
import json
import pandas as pd
from transformers import pipeline
import re

def load_data_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            json_obj = json.loads(line)
            for key, value in json_obj[0].items():
                data.append({'key': key, 'value': value})
    df = pd.DataFrame(data)
    return df

def load_data_judge(path):
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            json_obj = json.loads(line)
            for key, value in json_obj[0].items():
                data.append({'key': key, 'value': value})
    return data

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
            
def llm_as_judge_claude(df, dest_path):    
    client = anthropic.Anthropic(
    api_key=claude_key(),  # Or use os.environ["ANTHROPIC_API_KEY"]
    )
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
        try:
            # Make a simple request
            response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}])
            
            # Check if response has content
            if response.content and len(response.content) > 0:
                completion[question] = response.content[0].text
            else:
                completion[question] = f"Empty response: {response.stop_reason}"
        except Exception as e:
            completion[question] = f"Error: {str(e)}"
        
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

def calculate_parameters(models, judge):
    #for model in models:
    #    df = load_data_jsonl(f"data/dataset_completion_{model}.jsonl")
    #    llm_as_judge_claude(df, f"data/llm_as_judge_claude_{model}.jsonl")

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
        data = load_data_judge(f"data/llm_as_judge_{judge}_{model}.jsonl")
        for idx, d in enumerate(data):
            numbers = re.findall(r'[+-]?\d*\.?\d+', d['value'])
            numbers = [round(float(n)) for n in numbers]
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
                missed +=1
                continue
        dict['factual_accuracy'] = round(fa/(len(data)), 2)
        dict['comprehensiveness'] = round(comp/(len(data)), 2)
        dict['coherence'] = round(coh/(len(data)), 2)
        dict['concision'] = round(conc/(len(data)), 2)
        dict['readability'] = round(read/(len(data)), 2)
        dict['helpfulness'] = round(help/(len(data)), 2)
        dict['insightfulness'] = round(ins/(len(data)), 2)
        dict['overall_quality'] = round(oq/(len(data)), 2)  
        print(model)
        print(dict) 
        print(missed) 
           
def calculate_parameters_topics(models, judge):
    for model in models:
        df_params = pd.DataFrame()
        key = []
        fa = []
        comp = []
        coh = []
        conc = []
        read = []
        help = []
        ins = []
        oq = []
        missed =  []
        data = load_data_judge(f"data/llm_as_judge_{judge}_{model}.jsonl")
        for idx, d in enumerate(data):
            numbers = re.findall(r'[+-]?\d*\.?\d+', d['value'])
            numbers = [round(float(n)) for n in numbers]
            if len(numbers) == 8:
                key.append(d['key'])
                fa.append(int(numbers[0]))
                comp.append(int(numbers[1]))
                coh.append(int(numbers[2]))
                conc.append(int(numbers[3]))
                read.append(int(numbers[4]))
                help.append(int(numbers[5]))
                ins.append(int(numbers[6]))
                oq.append(int(numbers[7]))
            else:
                print(model, d)
                missed.append("Missed")
                continue
        df_params['key'] = key
        df_params['factual_accuracy'] = fa
        df_params['comprehensiveness'] = comp
        df_params['coherence'] = coh
        df_params['concision'] = conc
        df_params['readability'] = read
        df_params['helpfulness'] = help
        df_params['insightfulness'] = ins
        df_params['overall_quality'] = oq  
        df_params.to_excel(f"data/result_{judge}_{model}.xlsx", index=False)
    return df_params

def topic_mapping(models, judge):
    theme_mapping = {
        # Modelling & Optimization
        "Modelling": "Modelling & Optimization",
        "Experimental Design": "Modelling & Optimization",
        "Design Of Experiments": "Modelling & Optimization",

        # Bioprocess Engineering
        "Scaling": "Bioprocess Engineering",
        "Bioreactor": "Bioprocess Engineering",
        "Bioreactors Operation": "Bioprocess Engineering",
        "Continuous Fermentation": "Bioprocess Engineering",
        "Oxygen Transfer": "Bioprocess Engineering",
        "Bioprocess Development": "Bioprocess Engineering",
        "Industrial Biotech": "Bioprocess Engineering",

        # Organisms & Cell Biology
        "Bacterial": "Organisms & Cell Biology",
        "Yeast": "Organisms & Cell Biology",
        "Mold": "Organisms & Cell Biology",
        "Pure Strain": "Organisms & Cell Biology",
        "Cell Physiology": "Organisms & Cell Biology",
        "Biology": "Organisms & Cell Biology",

        # Sector Applications
        "Food": "Sector Applications",
        "Food Biotech": "Sector Applications",
        "Pharma": "Sector Applications",
        "Med Biotech": "Sector Applications",
        "Biofuel": "Sector Applications",
        "Alcohol": "Sector Applications",
        "Bioplastic": "Sector Applications",
        "Waste Treatment": "Sector Applications",

        # Specific Biochemistry & Operations
        "Enzymes": "Biochemistry & Operations",
        "Medium Composition": "Biochemistry & Operations",
        "Mixed Media": "Biochemistry & Operations",
        "Yield": "Biochemistry & Operations",
        "Product Toxicity": "Biochemistry & Operations",

        # Fundamentals
        "Basics": "Fundamentals"
    }

    df_topics = pd.read_excel("data/dataset_curated.xlsx")
    for model in models:
        print(model)
        #df_params = pd.read_excel(f"data/result_{judge}_{model}.xlsx")
        df_params = calculate_parameters_topics([model], judge)
        df_params_merged = df_params.join(df_topics.set_index('question'), on='key', how='left')        
        df_params_merged['theme'] = df_params_merged['topic'].map(theme_mapping)
        #df_params_merged.to_excel(f"data/result_{judge}_{model}_with_theme.xlsx", index=False)
    return df_params_merged
  
def calculate_parameters_by_theme(models, judge):
    for model in models:
        print(model)
        #df_params = pd.read_excel(f"data/result_{judge}_{model}_with_theme.xlsx")
        df_params = topic_mapping([model], judge)
        df_new = df_params.groupby(["theme"]).agg({"factual_accuracy": ["mean"], 
                                                 "comprehensiveness": ["mean"], 
                                                 "coherence": ["mean"], 
                                                 "concision": ["mean"], 
                                                 "readability": ["mean"], 
                                                 "helpfulness": ["mean"],
                                                 "insightfulness": ["mean"], 
                                                 "overall_quality": ["mean"]})
        #df_new.to_excel(f"data/result_{judge}_{model}_aggregated.xlsx") 
    return df_new

def aggregate_final_results(models):
    df_aggregated = pd.DataFrame()
    df_aggregated['theme'] = ["Biochemistry & Operations",
                            "Bioprocess Engineering",
                            "Fundamentals",
                            "Modelling & Optimization",
                            "Organisms & Cell Biology",
                            "Sector Applications"]
    for judge in ['claude', 'gpt']:
        print(judge)
        for model in models:
            print(model)
            #df = pd.read_excel(f"data/result_{judge}_{model}_aggregated.xlsx")
            df = calculate_parameters_by_theme([model], judge)
            #print(df["overall_quality"].values.tolist())
            df_aggregated[f'scores_{judge}_{model}'] = df['overall_quality'].values
    df_aggregated.to_excel("data/final_results_aggregated.xlsx", index=False)
    return df_aggregated
        
# RUN CODE 
#models = ['chatgpt', 'claude', 'deepseek', 'lechat', 'gemini_']  # uncomment this
#m = "gemini"
#df = load_data_jsonl(f"data/dataset_completion_{m}.jsonl")
#llm_as_judge_claude(df, f"data/llm_as_judge_claude_{m}.jsonl")
#llm_as_judge_gpt(df, f"data/llm_as_judge_gpt_{m}.jsonl")
#df = pd.read_excel("data/pairwise_dataset.xlsx")
#llm_as_judge_gpt_pairwise(df, "data/ll_as_judge_pairwise_gpt4o.jsonl")
#calculate_parameters(['gemini'], 'gpt')
#df_params = calculate_parameters_topics(models, "claude")
#calculate_parameters(['gemini'], "claude")
#topic_mapping(models, 'claude')
#calculate_parameters_by_theme(models, 'claude') 
#aggregate_final_results(models) # uncomment this