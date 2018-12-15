# -*- coding: utf-8 -*-
"""
Text classfication

"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from stop_words import get_stop_words
from sklearn.feature_selection import chi2, SelectKBest
import nltk.stem 

def stop_words():
    """Retrieve the stop words for vectorization -Feel free to modify this function
    """
    return get_stop_words('es') + get_stop_words('ca') + get_stop_words('en')

spanish_stemmer = nltk.stem.SnowballStemmer('english','spanish')


#tfidf = StemmedTfidfVectorizer(min_df=1, stop_words='english', analyzer='word', ngram_range=(1,1)) 

train_df = pd.read_excel('train.xlsx') # Load the `train` file
train_df.sample(frac=0.1)[:10] # Show a sample of the dataset

test_df = pd.read_excel('test.xlsx') # Load the `test` file
test_df.sample(frac=0.1)[:10] # Show a sample of the dataset

tfidf_vectorizer = TfidfVectorizer(
#tfidf_vectorizer = StemmedTfidfVectorizer(
        sublinear_tf = True,
        strip_accents='unicode',
        min_df = 3,
        norm='l2',
        token_pattern='(?u)\w\w+',#r'[^0-9]\w{1,}',#r'#?[^0-9]\w\w+',
        stop_words=stop_words(), 
        #ngram_range=(1,2), 
        #max_features=4000
        )

X = tfidf_vectorizer.fit_transform(train_df['text']).toarray() #features
y = train_df['party'].values

tfidf_vectorizer.get_feature_names()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25)
#
# My first Naive Bayes classifier!
#clf = BernoulliNB()
#clf.fit(X_train, y_train)
#prediction = clf.predict(X_test)
#print(np.mean([prediction == y_test]))    
#
