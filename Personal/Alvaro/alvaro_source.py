###  Import essentials ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from stop_words import get_stop_words
from langdetect import detect
import os
from textblob import TextBlob
import time
import langid
from google.cloud import translate

pd.set_option('max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('precision', 4)

### Functions ###

def stop_words():
    """Retrieve the stop words for vectorization -Feel free to modify this function
    """
    return get_stop_words('es') + get_stop_words('ca') + get_stop_words('en')


def filter_mentions(text):
    """Utility function to remove the mentions of a tweet
    """
    return re.sub("@\S+", "", text)


def filter_hashtags(text):
    """Utility function to remove the hashtags of a tweet
    """
    return re.sub("#\S+", "", text)


def translate_string_google_cloud(string, source, target):
    time.sleep(0.1)
    translation = translate_client.translate(
        string,
        source_language=source,
        target_language=target)
    return translation['translatedText']


def detect_language_google_cloud(string):
    time.sleep(0.1)
    return translate_client.detect_language(string)

def get_language_id(google_translation):
    return google_translation['language']

def text_filtered(string):
    # my_string = filter_hashtags(string)
    # my_string = filter_mentions(my_string)
    my_string = string
    return my_string


def save_submission(prediction):
    import datetime
    t = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    output = pd.DataFrame({'Party': prediction})
    output.index.name = 'Id'
    output.to_csv(f'sample_submission{t}.csv')


if __name__ == '__main__':
    ROOT_PATH = "C:/workspace/my_repos/Capstone DS/Data/"

    train_df = pd.read_excel(os.path.join(ROOT_PATH, "train-20190402-08_55_19.xlsx"))  # Load the `train` file
    train_df = pd.read_excel(os.path.join(ROOT_PATH, 'test-20190402-08_55_19-public.xlsx'))  # Load the `test` file

    # remove 5715
    #train_df = train_df.drop([5715])
    #train_df.reset_index(inplace=True, drop=True)

    # detect lang
    translate_client = translate.Client()

    ini = time.time()
    train_df['no_of_characters'] = list(map(len, train_df['text']))
    train_df['language'] = list(map(detect_language_google_cloud, train_df['text']))
    train_df['language_id'] = list(map(get_language_id, train_df['language']))
    end = time.time()
    print("It took: ", end - ini, " seconds")

    train_df['text_modif'] = list(map(text_filtered, train_df['text']))  # to see text post filtering
    num_lang = train_df['language'].value_counts()

    # Translation
    ini = time.time()
    translations = []
    for idx, tweet in enumerate(train_df['text']):
        print(f"{idx} / {len(train_df['text'])}")
        source_language = train_df.loc[idx, 'language_id']
        text_to_translate = train_df.loc[idx, 'text_modif']
        if source_language == 'es':
            translations.append(text_to_translate) #
        else:
            translation = translate_string_google_cloud(text_to_translate,
                                                        source_language,
                                                        "es")
            translations.append(translation)

    train_df.loc[:, 'traducciones'] = translations
    train_df.to_csv("C:/workspace/my_repos/Capstone DS/Data/test_traducido.csv", sep=';', index=False)

  #
    # train_df['day'] = [x.day for x in train_df.created_at]
    # train_df['month'] = [x.month for x in train_df.created_at]
    # train_df['year'] = [x.year for x in train_df.created_at]
    #
    # train_df.to_csv(r"/Users/oliver/YOGA/Yoga/Personal/Oliver/train_df_tr2.csv", sep=";")
    #
    # tfidf_vectorizer = TfidfVectorizer(
    #     # tfidf_vectorizer = StemmedTfidfVectorizer(
    #     sublinear_tf=True,
    #     strip_accents='unicode',
    #     min_df=3,
    #     norm='l2',
    #     token_pattern='(?u)\w\w+',  # r'[^0-9]\w{1,}',#r'#?[^0-9]\w\w+',
    #     stop_words=stop_words(),
    #     # ngram_range=(1,2),
    #     # max_features=4000
    # )
    #
    # X.columns
    # from sklearn.naive_bayes import BernoulliNB
    # from sklearn.model_selection import train_test_split
    #
    # # Split train and test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #
    # # My first Naive Bayes classifier!
    # clf = BernoulliNB()
    # clf.fit(X_train, y_train)
    # prediction = clf.predict(X_test)
    # print(np.mean([prediction == y_test]))
    #
    # # 1 Train the classifier
    # clf = BernoulliNB()
    # clf.fit(X_train, y_train)
    #
    # # 2 Predict the data (We need to tokenize the data using the same vectorizer object)
    # X_test = vectorizer.transform(test_df['text']).toarray()
    # prediction = clf.predict(X_test)
    #
    # # 3 Create a the results file
    # output = pd.DataFrame({'Party': prediction})
    # output.index.name = 'Id'
    # output.to_csv('sample_submission.csv')
    #
    # # TIP - Copy and paste this function to generate the output file in your code
