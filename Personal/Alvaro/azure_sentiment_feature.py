#### Retrieve sentiment analysis from Azure Text Analysis API Tool ####

import http.client, urllib.request, urllib.parse, urllib.error
import json
import pandas as pd
import os
from alvaro_final import *

AZURE_KEY = 'd5948ca3e55a4fa7b39694b3f8d93b72'
ROOT_PATH = "C:/workspace/Repositorios/Politicians/"
PERSONAL_PATH = "Personal/Alvaro"
TRAIN_FPATH = os.path.join(ROOT_PATH, "Data/train_traducido.csv")
TEST_FPATH = os.path.join(ROOT_PATH, "Data/test_traducido.csv")

def get_text_list_working_azure_working_example():
    return {
        "documents": [
            {
                "language": "en",
                "id": "1",
                "text": "Hello world. @MyBest Friend :( This is some input text that I do not love #superkillyou."
            },
            {
                "language": "fr",
                "id": "2",
                "text": "Bonjour tout le monde"
            },
        ]
    }

def get_azure_sentiment(texts_lists, AZURE_KEY):
    headers = {'Content-Type': 'application/json',
               'Ocp-Apim-Subscription-Key': f'{AZURE_KEY}'}
    params = urllib.parse.urlencode({'showStats': '{boolean}'})

    try:
        conn = http.client.HTTPSConnection('westeurope.api.cognitive.microsoft.com')
        conn.request("POST", "/text/analytics/v2.1/sentiment?%s" % params,
                     f"{json.dumps(texts_lists)}",
                     headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()

    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

    return json.loads(data.decode('utf-8'))

def set_up_df_for_sentiment_analysis(df):
    s_df = df[['Id', 'text_clean']].copy()
    s_df.rename(columns={'text_clean': 'text', 'Id': 'id'}, inplace=True)
    s_df['id'] = s_df['id'].map(lambda x: str(x))
    s_df.insert(0, 'language', value='es')

    return s_df

if __name__ == '__main__':

    ## Load data and Preprocess Data ##
    train_df_tr = pd.read_csv(TRAIN_FPATH, delimiter=';')  # Load the traductions file
    train_df_tr = remove_tweets_with_non_identified_language(train_df_tr)
    train_df_tr = add_text_clean_col_to_df(train_df_tr)
    train_df_counts = get_features_of_interest_counts(train_df_tr)
    train_df_languages = get_language_df(train_df_tr)

    test_df_tr = pd.read_csv(TEST_FPATH, delimiter=';')  # Load the traductions file
    test_df_counts = get_features_of_interest_counts(test_df_tr)
    test_df_languages = get_language_df(test_df_tr)
    test_df_tr = add_text_clean_col_to_df(test_df_tr)

    # Do Azure Sentiment Analysis #
    train_s_df = set_up_df_for_sentiment_analysis(train_df_tr)
    test_s_df = set_up_df_for_sentiment_analysis(train_df_tr)

    all_train_texts = train_s_df.to_dict(orient='records')

    first_ = 0
    step_ = 500
    last_ = first_ + step_
    limit_ = len(train_s_df)

    my_results = pd.DataFrame()
    while last_ <= limit_ + step_:
        print("Getting Sentiment from {} to {}".format(first_, last_))
        iteration_texts = all_train_texts[first_:last_]
        formatted_iteration_texts = {"documents": iteration_texts}
        data = get_azure_sentiment(formatted_iteration_texts, AZURE_KEY)
        iteration_results_df = pd.DataFrame(data['documents'])
        my_results = my_results.append(iteration_results_df)
        first_ = last_
        last_ = last_ + step_

    len(my_results)
    my_results.to_csv(os.path.join(ROOT_PATH, "Data/test_sentiment_azure.csv"), index=False, sep=';')

