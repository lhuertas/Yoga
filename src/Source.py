# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:44:15 2018

@author: Òscar
"""

# Import essentials
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

##Data Processing

import re
from stop_words import get_stop_words



train_df = pd.read_excel('./Data/train.xlsx') # Load the `train` file
test_df = pd.read_excel('./Data/test.xlsx') # Load the `test` file
train_df['language'] = list(map(get_idioma, train_df['text']))
train_df[train_df['idiom'] == 'et']


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

X.columns
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# My first Naive Bayes classifier!
clf = BernoulliNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print(np.mean([prediction == y_test]))

# 1 Train the classifier
clf = BernoulliNB()
clf.fit(X_train, y_train)

# 2 Predict the data (We need to tokenize the data using the same vectorizer object)
X_test = vectorizer.transform(test_df['text']).toarray()
prediction = clf.predict(X_test)

# 3 Create a the results file
output = pd.DataFrame({'Party': prediction})
output.index.name = 'Id'
output.to_csv('sample_submission.csv')

# TIP - Copy and paste this function to generate the output file in your code
def save_submission(prediction):
    import datetime
    t = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    output = pd.DataFrame({'Party': prediction})
    output.index.name = 'Id'
    output.to_csv(f'sample_submission{t}.csv')