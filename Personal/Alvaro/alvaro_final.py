import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import functions as funcs
from sklearn.svm import SVC


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
    ROOT_PATH = "C:/workspace/Repositorios/Politicians/"
    PERSONAL_PATH = "Personal/Alvaro"

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

    train_df_sentiment_raw = pd.read_csv(os.path.join(ROOT_PATH, "Data/train_sentiment_azure.csv"),delimiter=';')

    train_df_sentiment = pd.DataFrame(train_df_tr.merge(train_df_sentiment_raw,
                                    left_on='Id',
                                    right_on='id',
                                    how='left')['score'])

    test_df_tr = pd.read_csv(TEST_FPATH, delimiter=';')  # Load the traductions file
    test_df_counts = get_features_of_interest_counts(test_df_tr)
    test_df_languages = get_language_df(test_df_tr)
    test_df_tr = add_text_clean_col_to_df(test_df_tr)

    ## LET's DO A NICE MODEL

    y = train_df_tr['party'].values
    y1, y2 = np.unique(y, return_inverse=True)

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
    results_no_sentiment = []
    for ii in range(1,10):
        X_tfidf_train = tfidf_vectorizer.fit_transform(train_df_tr['text_clean']).toarray()
        X_tfidf_train = np.append(X_tfidf_train, train_df_languages, 1)
        X_tfidf_train = np.append(X_tfidf_train, train_features_count, 1)
    #X_tfidf_train = np.append(X_tfidf_train, train_df_sentiment, 1)

    # X_tfidf_test = tfidf_vectorizer.fit_transform(test_df_tr['text_clean']).toarray()
    # X_tfidf_test = np.append(X_tfidf_test, test_df_languages, 1)
    # X_tfidf_test = np.append(X_tfidf_test, test_features_count, 1)

    # ---------------------- Model selection
    # Split train and test
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_tfidf_train, y,
                                                                                     train_df_tr.index,
                                                                                     test_size=0.25)

    # param_grid = [{'estimator__kernel':['rbf'],'estimator__gamma':[1e-4,1e-3,0.5,5,10,50],'estimator__C':[0.001,0.1,1,10,100] },
    #              {'estimator__kernel':['linear'], 'estimator__C':[1, 10, 100, 1000]}]
    # gsv = GridSearchCV(SVC(), param_grid, cv=5, scoring='average_precision', n_jobs=-1)
    # gsv.fit(X_train,y_train)
    # Best HyperParameter:  {'estimator__C': 1, 'estimator__kernel': 'linear'}


        clf = SVC(C=1, kernel='linear')
        assert len(X_train) == len(y_train)
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        print("Prediction: {}".format(np.mean([prediction == y_test])))
        results_no_sentiment.append(prediction)

    # Create the results file
    # assert len(X_tfidf_train) == len(y)
    # clf.fit(X_tfidf_train, y)
    # predictions = clf.predict(X_tfidf_test)
    # len(X_tfidf_test)
    # funcs.save_submission(predictions, "sample_submission_svc")
