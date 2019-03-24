# Import essentials
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import nltk.stem
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

    # Preprocess data
    vectorizer = CountVectorizer(stop_words=stop_words())
    X = vectorizer.fit_transform(train_df['traducciones']).toarray()
    y = train_df['party'].values

    #Process emojis
    # train_df['emoji_transformation'] = train_df['traducciones'].apply(lambda x: emoji.demojize(x)). \
    #     apply(lambda x: x.replace(':', ' ')).apply(lambda x: x.replace('_', ''))
    # test_df['emoji_transformation'] = test_df['traducciones'].apply(lambda x: emoji.demojize(x)). \
    #     apply(lambda x: x.replace(':', ' ')).apply(lambda x: x.replace('_', ''))


    ## bloque 28 cuaderno 5
    # nuestra X es term_freq_matrix


    spanish_stemmer = nltk.stem.SnowballStemmer('spanish')
    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super(TfidfVectorizer, self).build_analyzer()
            return lambda doc: (spanish_stemmer.stem(w) for w in analyzer(doc))

    tfidf = StemmedTfidfVectorizer(
        min_df=1,
        stop_words=stop_words(),
        norm='12',
        ngram_range=(1, 1),
    )

    tfidf.fit(X)
    tf_idf_matrix = tfidf.transform(X)

    # tfidf = TfidfTransformer(norm="l2")
    # tfidf.fit(X)
    #tf_idf_matrix = tfidf.transform(X)

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
