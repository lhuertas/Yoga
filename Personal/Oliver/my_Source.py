# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:44:15 2018

@author: Òscar
"""

# Import essentials
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

##Data Processing

import re
from stop_words import get_stop_words

# Functions #
from langdetect import detect


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


def get_language(string):
    #my_string = filter_hashtags(string)
    #my_string = filter_mentions(my_string)
    my_string = string
    my_vector = my_string.split()
    my_vector = ",".join(my_vector)

    return detect(my_vector)

def text_filtered(string):
    #my_string = filter_hashtags(string)
    #my_string = filter_mentions(my_string)
    my_string = string
    return my_string

def get_lang(string):
    tweet = TextBlob(string)
    return tweet.detect_language()  


train_df = pd.read_excel('/Users/oliver/YOGA/datos/train-20190402-08_55_19.xlsx') # Load the `train` file
test_df = pd.read_excel('/Users/oliver/YOGA/datos/test-20190402-08_55_19-public.xlsx') # Load the `test` file

#remove 5715
train_df = train_df.drop([5715])
train_df.reset_index(inplace=True, drop=True)

# detect lang 
from textblob import TextBlob
train_df['language'] = list(map(get_lang, train_df['text']))


train_df['lang'] = list(map(get_language, train_df['text']))
train_df['text_modif'] = list(map(text_filtered, train_df['text'])) # to see text post filtering
num_lang = train_df['language'].value_counts()
num_lang
plt.plot(num_lang)

# Translation

import time
ini = time.time()
from gtrans import translate_text

translations = []
for idx, tweet in enumerate(train_df['text']):
    if train_df.loc[idx,'language'] == 'es':
        translations.append(train_df.loc[idx,'text_modif'])
    else:
        translation = translate_text(train_df.loc[idx, 'text_modif'],train_df.loc[idx,'language'],"es")
        translations.append(translation)

train_df.loc[:,'traducciones'] = translations


# por si se ha dejado alguno
a = True
while a == True:
    
    empty = []
    for idx, tweet in enumerate(train_df['traducciones']):
        if tweet == "":
            empty.append(idx)
            print(idx)
    
    if len(empty) == 0:
        a = False
    
    for idx, tweet in enumerate(train_df['traducciones']):
        if tweet == "":
            if train_df.loc[idx,'language'] == 'es':
                train_df.loc[idx, 'traducciones'] = train_df.loc[idx, 'text_modif']
            else:
                translation = translate_text(train_df.loc[idx, 'text_modif'],train_df.loc[idx,'language'],"es")
                train_df.loc[idx, 'traducciones'] = translation


end = time.time()
print("It took: ", end-ini, " seconds")


train_df['day'] = [x.day for x in train_df.created_at]
train_df['month'] = [x.month for x in train_df.created_at]
train_df['year'] = [x.year for x in train_df.created_at]

train_df.to_csv(r"/Users/oliver/YOGA/Yoga/Personal/Oliver/train_df_tr2.csv", sep=";")







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