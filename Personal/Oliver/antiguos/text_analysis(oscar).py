# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 23:51:16 2019
@author: Òscar
"""

# Import essentials

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer


# Import data

train_df = pd.read_csv('C:/Workspace/Yoga/train_df_tr2.csv', sep=';') # Load the `train` file


test_df = pd.read_csv('C:/Workspace/Yoga/train_df_tr2.csv', sep = ";") # Load the `test` file


#Stopwords

import re
from stop_words import get_stop_words

def stop_words():
    """Retrieve the stop words for vectorization -Feel free to modify this function
    """
    return get_stop_words('es') + get_stop_words('ca') + get_stop_words('en')


def filter_mentions(text):
    """Utility function to remove the mentions of a tweet
    """
    return re.sub("@\S+", "", text)


def filter_hashtags(text):
    """Utility function to remove the hashtags of a tweet
    """
    return re.sub("#\S+", "", text)


# Preprocess data. TF-IDF

vectorizer = TfidfVectorizer(min_df=0,stop_words=stop_words())
X = vectorizer.fit_transform(train_df['traducciones']).toarray()

#Feature engineering

y = train_df['party'].values

language_dummy = pd.get_dummies(train_df['language'])

train_df = train_df.values
X = np.column_stack((X,train_df[:,6:8],language_dummy))


# Import model
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)


# Model
clf = BernoulliNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)


# K-Fold Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Result creation
#output = pd.DataFrame({'Party': prediction})
#output.index.name = 'Id'
#output.to_csv('sample_submission.csv')


# TIP - Copy and paste this function to generate the output file in your code
#def save_submission(prediction):
#    import datetime
#    t = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
#    output = pd.DataFrame({'Party': prediction})
#    output.index.name = 'Id'
#    output.to_csv(f'sample_submission{t}.csv')