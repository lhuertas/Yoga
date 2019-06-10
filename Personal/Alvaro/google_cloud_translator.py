###  Import essentials ###
import numpy as np
import pandas as pd
import re
from stop_words import get_stop_words
import os
import time
from google.cloud import translate
import unidecode
import emoji

pd.set_option('max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('precision', 4)


### Functions ###

def hello_world_translation(translate_client):
    # The text to translate
    text = u'Hola Mundo!'
    # The target language
    target = 'en'

    # Translates some text into Russian
    translation = translate_client.translate(
        text,
        target_language=target)

    print(u'Text: {}'.format(translation['input']))
    print(u'Translation: {}'.format(translation['translatedText']))

    return translation


def clean_text(text):
    # lowercase:
    text_clean = text.lower()
    # remove accents:
    text_clean = unidecode.unidecode(text_clean)
    # remove_hashtags:
    text_clean = re.sub('@', '', text_clean)
    # remove_mentions:
    text_clean = re.sub('#', '', text_clean)
    # transf. emoticons_to_word:
    text_clean = emoji.demojize(text_clean).replace(':', ' ').replace('_', '')
    # remove puntuation, numbers:
    text_clean = re.sub('[^A-Za-z]+', ' ', text_clean)
    return text_clean


def translate_string_google_cloud(string, target):
    time.sleep(0.05)
    translation = translate_client.translate(
        string,
        target_language=target)

    return translation


def detect_language_google_cloud(string):
    time.sleep(0.1)
    return translate_client.detect_language(string)


def get_language_id(google_translation):
    return google_translation['language']


def read_train_test_originals():
    train_df = pd.read_excel(
        os.path.join(ROOT_PATH, "Data", "original", "train-20190402-08_55_19.xlsx"))  # Load the `train` file
    test_df = pd.read_excel(
        os.path.join(ROOT_PATH, "Data", "original", 'test-20190402-08_55_19-public.xlsx'))  # Load the `test` file

    # remove 5715
    train_df = train_df.drop([5715])
    train_df.reset_index(inplace=True, drop=True)

    return train_df, test_df


ROOT_PATH = "C:/workspace/Repositorios/Politicians/"

if __name__ == '__main__':
    translate_client = translate.Client()
    train_df, test_df = read_train_test_originals()
    train_df['text_clean'] = list(map(clean_text, train_df['text']))

    # Translation
    ini = time.time()
    translations_dicts = []
    for idx, tweet in enumerate(train_df['text_clean']):
        print(f"{idx} / {len(train_df['text_clean'])}")
        text_to_translate = train_df.loc[idx, 'text_clean']
        tweet_id = train_df.loc[idx, 'Id']
        translation = translate_string_google_cloud(text_to_translate, "en")
        iter_dict = {
            'id':tweet_id,
            'source' : text_to_translate,
            'language' : translation['detectedSourceLanguage'],
            'text' : translation['translatedText']}
        translations_dicts.append(iter_dict)

    results_df = pd.DataFrame.from_dict(translations_dicts)
    results_df.to_csv(os.path.join(ROOT_PATH, "Data", "train_english_translation.csv"), sep=';', index=False)

