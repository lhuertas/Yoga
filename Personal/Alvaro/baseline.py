# Import essentials
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

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

    # Preprocess data
    vectorizer = CountVectorizer(stop_words=stop_words())
    X = vectorizer.fit_transform(train_df['traducciones']).toarray()
    y = train_df['party'].values

    ## bloque 28 cuaderno 5
    # nuestra X es term_freq_matrix

    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(X)
    tf_idf_matrix = tfidf.transform(X)

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y, test_size=0.25)

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
