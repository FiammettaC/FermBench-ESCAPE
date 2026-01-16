import pandas as pd

def create_question_dataset(path, dest_path):
    df = pd.read_excel(path)
    questions = []
    topics = []
    for i in range(10):
        questions.extend(df[f"Question {i+1}"].tolist())
        topics.extend(df[f"Topic {i+1}"].tolist())
    # remove empty questions
    #questions = [i for i in questions if str(i) != "nan"]
    #topics = [i for i in topics if str(i) != "nan"]
    # store in dataframe
    df_questions = pd.DataFrame()
    df_questions["id"] = range(len(questions))
    df_questions["question"] = questions
    df_questions["topic"] = topics
    df_questions = df_questions[df_questions["question"].notna()]
    df_questions.to_excel(dest_path, index=False)
    return df_questions

# Run code 
data_path = "data/questions_database.xlsx"
output_path = "data/dataset_curated.xlsx"
create_question_dataset(data_path, output_path)