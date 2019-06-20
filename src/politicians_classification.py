import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import functions as funcs
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import datetime as dt


def fill_nan_in_matrix(X, value):
    ix_list = np.argwhere(np.isnan(X)).tolist()
    for ix in ix_list:
        X[ix[0], ix[1]] = value

def get_sentiment_features_df(ROOT_PATH, traintest_df, str, ):
    if (str == 'train'):
        fname = 'train_sentiment_features.csv'
    elif (str == 'test'):
        fname = 'test_sentiment_features.csv'
    train_sentiment_raw = pd.read_csv(os.path.join(
        ROOT_PATH, "Data", fname),
        delimiter=';').iloc[:, [0, 1, 2, 3,4]]
    df = train_sentiment_raw.copy()
    df = pd.concat([df, pd.get_dummies(df['amazon_sentiment'])], axis=1).rename(
        columns={'POSITIVE': 'az_positive',
                 "NEGATIVE": 'az_negative',
                 "NEUTRAL": 'az_neutral'}).drop('amazon_sentiment', axis=1)

    df = pd.DataFrame(traintest_df.merge(df,
                                         left_on='Id',
                                         right_on='id',
                                         how='left'))

    cols_to_keep = ['az_positive', 'az_negative', 'az_neutral',
                    'google_sentiment', 'azure_sentiment', 'google_emotion']

    return df[cols_to_keep]


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


def get_amazon_sentiment_dummies_df(df):
    return pd.DataFrame({'az_positive': np.where(df['amazon_sentiment'] == 'POSITIVE', 1, 0),
                         'az_neutral': np.where(df['amazon_sentiment'] == 'NEUTRAL', 1, 0),
                         'az_negative': np.where(df['amazon_sentiment'] == 'NEGATIVE', 1, 0),
                         })

def add_text_clean_col_to_df(df):
    df['text_clean'] = df['traducciones'].apply(lambda x: funcs.clean_text(x))

    return df

def get_day_week(df):
    df.created_at = pd.to_datetime(df.created_at)
    day_week = df['created_at'].dt.day_name()
    ohe = OneHotEncoder(handle_unknown='ignore')
    day_week = np.array(day_week)
    day_week = day_week.reshape(-1, 1)
    day_week = ohe.fit_transform(day_week).toarray()
    return pd.DataFrame(day_week)

def get_part_day(df):
    new_hours = df.created_at[df.created_at.dt.hour == 0]
    new_hours2 = new_hours + dt.timedelta(hours=1)
    df.created_at[df.created_at.dt.hour == 0] = new_hours2
    part_day = pd.cut(df.created_at.dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening'])
    ohe = OneHotEncoder(handle_unknown='ignore')
    part_day = np.array(part_day)
    part_day = part_day.reshape(-1, 1)
    part_day = ohe.fit_transform(part_day).toarray()
    return pd.DataFrame(part_day)
    
def balance_dataset(df):

    #target
    y = df['username'].values
    y1, y2 = np.unique(y, return_inverse=True)
    values, tweet_counts = np.unique(y, return_counts=True)
    politicians = pd.DataFrame({'values': values,
                                'tweet_counts': tweet_counts})

    max_tweet = np.max(politicians['tweet_counts'])

    # todos los políticos con mismo número de tweets
    pol_df_total = df.iloc[0:0]
    for pol in politicians['values']:
        pol_df = df[df.username == pol]
        if pol == 'olallamarga':
            pol_df = pol_df.loc[pol_df.index.repeat(20)]
        pol_df = pd.concat([pol_df, pol_df[0:max_tweet-len(pol_df)]], axis=0)
        pol_df_total = pd.concat([pol_df_total, pol_df], axis=0)
    
    pol_df_total.reset_index(inplace=True, drop=True)
    df = pol_df_total.copy()
    del(pol_df_total)
    
    #target
    y = df['username'].values
    y1, y2 = np.unique(y, return_inverse=True)
    values, tweet_counts = np.unique(y, return_counts=True)
    politicians = pd.DataFrame({'values': values,
                                'tweet_counts': tweet_counts})
    return df, politicians



if __name__ == '__main__':
    # SET PATHS ##
    ROOT_PATH = "C:/workspace/Repositorios/Politicians/"
    PERSONAL_PATH = "Personal/Alvaro"

    os.chdir(os.path.join(ROOT_PATH, PERSONAL_PATH))
    sys.path.append(os.path.join(ROOT_PATH, PERSONAL_PATH))
    TRAIN_FPATH = os.path.join(ROOT_PATH, "Data/train_traducido.csv")
    TEST_FPATH = os.path.join(ROOT_PATH, "Data/test_traducido.csv")

    print("Data Loading...")
    train_df_tr = pd.read_csv(TRAIN_FPATH, delimiter=';')
    train_df_tr.drop('party', axis=1, inplace=True)
    train_df_tr, politicians = balance_dataset(train_df_tr)
    train_df_tr = remove_tweets_with_non_identified_language(train_df_tr)
    train_df_tr = add_text_clean_col_to_df(train_df_tr)
    train_df_counts = get_features_of_interest_counts(train_df_tr)
    train_df_languages = get_language_df(train_df_tr)
    train_df_day_week = get_day_week(train_df_tr)
    train_df_part_day = get_part_day(train_df_tr)
    train_df_sentiment = get_sentiment_features_df(ROOT_PATH, train_df_tr, str='train')
    train_features_count = count_features_and_scale(train_df_tr, train_df_counts)

    test_df_tr = pd.read_csv(TEST_FPATH, delimiter=';')
    test_df_tr = add_text_clean_col_to_df(test_df_tr)
    test_df_counts = get_features_of_interest_counts(test_df_tr)
    test_df_languages = get_language_df(test_df_tr)
    test_df_day_week = get_day_week(test_df_tr)
    test_df_part_day = get_part_day(test_df_tr)
    test_df_sentiment = get_sentiment_features_df(ROOT_PATH, test_df_tr, str='test')
    test_features_count = count_features_and_scale(test_df_tr, test_df_counts)

    y = train_df_tr['username'].values
    y1, y2 = np.unique(y, return_inverse=True)

    ## MODEL ##
    tfidf_vectorizer = funcs.StemmedTfidfVectorizer(
        sublinear_tf=True,
        max_df=0.25,
        min_df=3,
        norm='l2',
        stop_words=funcs.stop_words(),
        ngram_range=(1, 1),
    )



    X_tfidf = tfidf_vectorizer.fit_transform(train_df_tr['text_clean']).toarray()
    X_tfidf = np.append(X_tfidf, train_df_languages, 1)
    X_tfidf = np.append(X_tfidf, train_features_count, 1)
    X_tfidf = np.append(X_tfidf, train_df_day_week, 1)
    X_tfidf = np.append(X_tfidf, train_df_part_day, 1)
    X_tfidf = np.append(X_tfidf, train_df_sentiment, 1)

    len(X_tfidf)
    len(train_df_languages)

    X_tfidf_test = tfidf_vectorizer.transform(test_df_tr['text_clean']).toarray()
    X_tfidf_test = np.append(X_tfidf_test, test_df_languages, 1)
    X_tfidf_test = np.append(X_tfidf_test, test_features_count, 1)
    X_tfidf_test = np.append(X_tfidf_test, test_df_sentiment, 1)
    X_tfidf_test = np.append(X_tfidf_test, test_df_day_week, 1)
    X_tfidf_test = np.append(X_tfidf_test, test_df_part_day, 1)
    fill_nan_in_matrix(X_tfidf_test, value=0.5)

    le = LabelEncoder()
    y_encode = le.fit_transform(y)
    X_train, X_test, y_train, y_test,\
    indices_train, indices_test = train_test_split(X_tfidf,
                                                   y_encode,
                                                   train_df_tr.index,
                                                   test_size=0.25)

    clf = SVC(C=1, kernel='linear')
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print("Prediction: {}".format(np.mean([prediction == y_test])))

    #Create the results file
    clf.fit(X_tfidf, y)
    predictions = clf.predict(X_tfidf_test)
    len(X_tfidf_test)
    save_submission_politicians(predictions, "sample_submission_svc_politicians")