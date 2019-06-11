import sys
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
import files.functions as funcs
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import datetime as dt
from keras.utils.np_utils import to_categorical

def count_features_and_scale(df, df_counts):
    scaler = StandardScaler()
    counts = df[['retweet_count', 'favorite_count']].apply(lambda x: x + 1)
    counts['counts_emoticons'] = df_counts.emoticons.apply(lambda x: x + 1)
    counts['counts_hashtags'] = df_counts.hashtags.apply(lambda x: x + 1)
    counts['counts_mentions'] = df_counts.mentions.apply(lambda x: x + 1)
    counts['counts_words'] = df_counts.words.apply(lambda x: x + 1)
    counts = np.log10(counts)
    features_counts = scaler.fit_transform(counts)
    features_counts = pd.DataFrame(features_counts)

    return features_counts


def remove_tweets_with_non_identified_language(df):
    df.drop(df[(df['language_id'] == 'lb') |
               (df['language_id'] == 'co') |
               (df['language_id'] == 'mr')].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

def get_features_of_interest_counts(df):
    return pd.DataFrame({'words': df['text'].apply(lambda x: funcs.number_words(x)),
                         'emoticons': df['text'].apply(lambda x: funcs.number_emoticons(x)),
                         'hashtags': df['text'].apply(lambda x: funcs.number_hashtags(x)),
                         'mentions': df['text'].apply(lambda x: funcs.number_mentions(x)),
                         })


def get_language_df(df):
    return pd.DataFrame({'ca': np.where(df['language_id'] == 'ca', 1, 0),
                         'es': np.where(df['language_id'] == 'es', 1, 0),
                         'en': np.where(df['language_id'] == 'en', 1, 0),
                         'other': df['language_id'].apply(lambda x: 1 if x not in {'ca', 'es', 'en'} else 0)
                         })


def add_text_clean_col_to_df(df):
    df['text_clean'] = df['traducciones'].apply(lambda x: funcs.clean_text(x))

    return df


if __name__ == '__main__':
    # SET PATHS ##
    ROOT_PATH = "/Users/lina/DataScience/Project/Yoga/"
    PERSONAL_PATH = "Personal/Lina"

    os.chdir(os.path.join(ROOT_PATH, PERSONAL_PATH))
    sys.path.append(os.path.join(ROOT_PATH, PERSONAL_PATH))
    TRAIN_FPATH = os.path.join(ROOT_PATH, "Data/train_traducido.csv")
    TEST_FPATH = os.path.join(ROOT_PATH, "Data/test_traducido.csv")

    ## Load data and Preprocess Data ##
    train_df_tr = pd.read_csv(TRAIN_FPATH, delimiter=';')  # Load the traductions file
    train_df_tr = remove_tweets_with_non_identified_language(train_df_tr)
    train_df_tr = add_text_clean_col_to_df(train_df_tr)
    train_df_counts = get_features_of_interest_counts(train_df_tr)
    train_df_languages = get_language_df(train_df_tr)

    #train_df_sentiment_raw = pd.read_csv(os.path.join(ROOT_PATH, "Data/train_sentiment_azure.csv"),delimiter=';')

    #train_df_sentiment = pd.DataFrame(train_df_tr.merge(train_df_sentiment_raw,
    #                                left_on='Id',
    #                                right_on='id',
    #                                how='left')['score'])

    test_df_tr = pd.read_csv(TEST_FPATH, delimiter=';')  # Load the traductions file
    test_df_counts = get_features_of_interest_counts(test_df_tr)
    test_df_languages = get_language_df(test_df_tr)
    test_df_tr = add_text_clean_col_to_df(test_df_tr)

    y = train_df_tr['party'].values

    # Counts features and scale
    train_features_count = count_features_and_scale(train_df_tr, train_df_counts)
    test_features_count = count_features_and_scale(test_df_tr, test_df_counts)

    # Parameter selection (TFIDF)
    tfidf_vectorizer = funcs.StemmedTfidfVectorizer(
        sublinear_tf=True,  # scaling
        # strip_accents='unicode',
        max_df=0.25,  # 0.5,
        min_df=3,
        norm='l2',
        # token_pattern='#?\w\w+',#r'[^0-9]\w{1,}',#r'#?[^0-9]\w\w+',
        stop_words=funcs.stop_words(),
        ngram_range=(1, 1),
        # max_features=4000
    )

    X_tfidf = tfidf_vectorizer.fit_transform(train_df_tr['text_clean']).toarray()
    X_tfidf = np.append(X_tfidf,train_df_languages,1)
    X_tfidf = np.append(X_tfidf, train_features_count,1)

    X_tfidf_test = tfidf_vectorizer.transform(test_df_tr['text_clean']).toarray()
    X_tfidf_test = np.append(X_tfidf_test,test_df_languages,1)
    X_tfidf_test = np.append(X_tfidf_test, test_features_count,1)

    le = LabelEncoder()
    y_encode = le.fit_transform(train_df_tr['party'])
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_tfidf, y_encode,
                                                                                     train_df_tr.index,
                                                                                     test_size=0.25)

    ## model selection ##
    models, maxVote, Stacking = funcs.model_sel(X_train, y_train, X_test, y_test)

    ## Choose the best model to train all the data and make the predictions ##
    maxVote.fit(X_tfidf, y_encode)
    predictions = maxVote.predict(X_tfidf_test)
    predictions_ = np.argmax(to_categorical(predictions), axis=1)
    predictions_ = le.inverse_transform(predictions_)

    ## Create the results file ##
    funcs.save_submission(predictions_, "sample_submission_mv_")

