# Import essentials
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
import nltk.stem
import emoji
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def stop_words():
    return get_stop_words('es') + get_stop_words('ca') + get_stop_words('en')


def filter_mentions(text):
    return re.sub("@\S+", "", text)


def filter_hashtags(text):
    return re.sub("#\S+", "", text)


def save_submission(prediction):
    import datetime
    t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = pd.DataFrame({'Party': prediction})
    output.index.name = 'Id'
    output.to_csv(f'sample_submission{t}.csv')


if __name__ == '__main__':

    root_path = 'C:/Users/Alvaro\OneDrive/3- Vida academica/0_Universidad de Barcelona/Capstone Project/'
    train_df = pd.read_csv(root_path + 'Personal/Oliver/train_df_tr2.csv', delimiter=';')
    test_df = pd.read_csv(root_path + 'Personal/Oliver/test_df_tr2.csv', delimiter=';')

    # Process Emojis
    train_df['emoji_transformation'] = train_df['traducciones'].apply(lambda x: emoji.demojize(x)). \
        apply(lambda x: x.replace(':', ' ')).apply(lambda x: x.replace('_', ''))
    test_df['emoji_transformation'] = test_df['traducciones'].apply(lambda x: emoji.demojize(x)). \
        apply(lambda x: x.replace(':', ' ')).apply(lambda x: x.replace('_', ''))

    # Preprocess data
    vectorizer = CountVectorizer(stop_words=stop_words())
    X = vectorizer.fit_transform(train_df['emoji_transformation']).toarray()
    y = train_df['party'].values

    ## bloque 28 cuaderno 5
    # nuestra X es term_freq_matrix

    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(X)
    tf_idf_matrix = tfidf.transform(X)

    ix = np.where(X[0] != 0)[0].tolist()
    for index in ix:
        print(vectorizer.get_feature_names()[index])

    # Add languages
    language_list = ['ca', 'en', 'es']
    language = pd.DataFrame()
    language['ca'] = np.where(train_df['language'] == 'ca', 1, 0)
    language['es'] = np.where(train_df['language'] == 'es', 1, 0)
    language['en'] = np.where(train_df['language'] == 'en', 1, 0)
    language['other'] = train_df['language']\
        .apply(lambda x: 1 if x not in set(language_list) else 0)

    scaler = StandardScaler()
    tf_idf_matrix_arr = tf_idf_matrix.toarray()
    #Features = np.append(tf_idf_matrix_arr, language, 1)
    counts = train_df[['retweet_count', 'favorite_count']].apply(lambda x: x + 1)
    counts = np.log10(counts)
    features_counts = scaler.fit_transform(counts)
    features_counts = pd.DataFrame(features_counts)
    allFeatures = np.append(tf_idf_matrix_arr, features_counts, 1)

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(allFeatures, y, test_size=0.25)

    # My first Naive Bayes classifier!
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print(np.mean([prediction == y_test]))

    # 1 Train the classifier
    clf = BernoulliNB()
    clf.fit(X_train, y_train)

    # 2 Predict the data (We need to tokenize the data using the same vectorizer object)
    # X_test = vectorizer.transform(test_df['traducciones']).toarray()
    prediction = clf.predict(X_test)

    # 3 Create a the results file
    save_submission(prediction)

    # cross validation
    acc = np.zeros((5,))
    i = 0
    kf = KFold(n_splits=5)
    kf.get_n_splits(allFeatures)
    print(kf)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

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
