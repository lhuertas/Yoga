import sys
import time
from langdetect import language

# sys.modules[__name__].__dict__.clear()

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
from stop_words import get_stop_words


def stop_words():
    return get_stop_words('es')  # + get_stop_words('ca') + get_stop_words('en')


def filter_mentions(text):
    return re.sub("@\S+", "", text)


def filter_hashtags(text):
    return re.sub("#\S+", "", text)


def extract_emojis(str):
    return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)


if __name__ == '__main__':

    root_path = 'C:/Users/Alvaro\OneDrive/3- Vida academica/0_Universidad de Barcelona/Capstone Project/'
    train_df_tr = pd.read_csv(root_path + 'Personal/Oliver/train_df_tr2.csv', delimiter=';')
    test_df_tr = pd.read_csv(root_path + 'Personal/Oliver/test_df_tr2.csv', delimiter=';')

    spanish_stemmer = nltk.stem.SnowballStemmer('spanish')
    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super(TfidfVectorizer, self).build_analyzer()
            return lambda doc: (spanish_stemmer.stem(w) for w in analyzer(doc))

    # Preprocess data
    train_df_tr['emoji_transformation'] = train_df_tr['traducciones'].apply(lambda x: emoji.demojize(x)). \
        apply(lambda x: x.replace(':', ' ')).apply(lambda x: x.replace('_', ''))
    test_df_tr['emoji_transformation'] = test_df_tr['traducciones'].apply(lambda x: emoji.demojize(x)). \
        apply(lambda x: x.replace(':', ' ')).apply(lambda x: x.replace('_', ''))

    tfidf_vectorizer = StemmedTfidfVectorizer(
        sublinear_tf=True,  # scaling
        # strip_accents='unicode',
        min_df=2,
        norm='l2',
        # token_pattern='#?\w\w+',#r'[^0-9]\w{1,}',#r'#?[^0-9]\w\w+',
        stop_words=stop_words(),
        ngram_range=(1, 2),
        # max_features=4000
    )

    X = tfidf_vectorizer.fit_transform(train_df_tr['emoji_transformation']).toarray()
    y = train_df_tr['party'].values
    # language = pd.get_dummies(train_df_tr['language'])
    language_list = ['ca', 'en', 'es']
    language = pd.DataFrame()
    language['ca'] = np.where(train_df_tr['language'] == 'ca', 1, 0)
    language['es'] = np.where(train_df_tr['language'] == 'es', 1, 0)
    language['en'] = np.where(train_df_tr['language'] == 'en', 1, 0)
    language['other'] = train_df_tr['language'].apply(lambda x: 1 if x not in set(language_list) else 0)

    scaler = StandardScaler()

    Features = np.append(X, language, 1)
    counts = train_df_tr[['retweet_count', 'favorite_count']].apply(lambda x: x + 1)
    counts = np.log10(counts)
    features_counts = scaler.fit_transform(counts)
    features_counts = pd.DataFrame(features_counts)
    allFeatures = np.append(Features, features_counts, 1)

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(allFeatures, y, test_size=0.25)

    # My first Naive Bayes classifier!
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print(np.mean([prediction == y_test]))

    # cross validation
    from sklearn.model_selection import KFold
    from sklearn import metrics

    bins = 10
    acc = np.zeros((bins,))
    i = 0
    kf = KFold(n_splits=bins)
    kf.get_n_splits(allFeatures)
    print(kf)
    kf = KFold(n_splits=bins, shuffle=True, random_state=0)

    # We will build the predicted y from the partial predictions on the test of each of the folds
    yhat = y.copy()
    for train_index, test_index in kf.split(allFeatures):
        X_train, X_test = allFeatures[train_index], allFeatures[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # dt = neighbors.KNeighborsClassifier(n_neighbors=1)
        # dt.fit(X_train,y_train)
        clf = BernoulliNB()
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        yhat[test_index] = clf.predict(X_test)
        acc[i] = metrics.accuracy_score(yhat[test_index], y_test)
        # print(acc[i])
        i = i + 1
    print('Mean accuracy: ' + str(np.mean(acc)))
    # Mean accuracy: 0.76
