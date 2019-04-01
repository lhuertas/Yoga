# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 23:51:16 2019

@author: Òscar
"""

# Import essentials

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # Stopwords
from stop_words import get_stop_words # Stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer #getdummies for language

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Import data

train_df = pd.read_csv('C:/Workspace/Yoga/Personal/Oliver/train_df_tr2.csv', sep=';') # Load the `train` file
test_df = pd.read_csv('C:/Workspace/Yoga/Personal/Oliver/test_df_tr2.csv', sep = ";") # Load the `test` file


#Stopwords

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




#train_df['emoji_transformation'] = train_df['traducciones'].apply(lambda x: emoji.demojize(x)). \
#                                      apply(lambda x: x.replace(':',' ')).apply(lambda x: x.replace('_',''))

# Preprocess data. TF-IDF

vectorizer = TfidfVectorizer(min_df=2,
                             stop_words=stop_words(), 
                             sublinear_tf = True,
                             norm='l2',
                             ngram_range=(1,3))
X_t = vectorizer.fit_transform(train_df['traducciones']).toarray()
X_test_s = vectorizer.transform(test_df['traducciones']).toarray()


#Feature engineering. Get dummies for language tweet count and tweet favourite count

y = train_df['party'].values


get_dummies = LabelBinarizer()

X_dummies = get_dummies.fit_transform(train_df['language'])
train_df = train_df.values
X = np.column_stack((X_t,train_df[:,6:8],X_dummies)) #TF+IDF + counts and favourites + dummies language


Test_dummies = get_dummies.transform(test_df['language']) 
test_df = test_df.values
X_test_sub = np.column_stack((X_test_s,test_df[:,3:5],Test_dummies)) #TF+IDF + counts and favourites + dummies language


# Split train and test. For training and validation, not to submit!!!

y_fact = pd.factorize(y) #Random forest doesn't understand string as y labels, we have to convert to float
y_f = y_fact[0]
definitions = y_fact[1]
X_train, X_test, y_train, y_test = train_test_split(X, y_f, test_size=0.01)


# Model

#clf = BernoulliNB()
#clf.fit(X_train, y_train)


rf = RandomForestClassifier(n_estimators = 4000, criterion = 'entropy', random_state = 42)
rf.fit(X_train, y_train)


# K-Fold Cross-Validation
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(rf, X, y_f, cv=4)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#Prediction of test data. Remember to use the test to submit, not the test split to check accuracy !!!

prediction = rf.predict(X_test_sub)
reversefactor = dict(zip(range(6),definitions)) # See line 85, we are reversing the process labels to parties...
y_test = np.vectorize(reversefactor.get)(y_test) # In order to submit it to Kaggle
prediction = np.vectorize(reversefactor.get)(prediction)

#prediction = np.where(prediction == 'xavierdomenechs','comuns',prediction)
#prediction = np.where(prediction == 'adacolau','comuns',prediction)
#prediction = np.where(prediction == 'inesarrimadas','cs',prediction)
#prediction = np.where(prediction == 'albert_rivera','cs',prediction)
#prediction = np.where(prediction == 'martarovira','erc',prediction)
#prediction = np.where(prediction == 'joantarda','erc',prediction)
#prediction = np.where(prediction == 'quimtorraipla','jxcat',prediction)
#prediction = np.where(prediction == 'krls','jxcat',prediction)
#prediction = np.where(prediction == 'albiol_xg','ppc',prediction)
#prediction = np.where(prediction == 'santirodriguez','ppc',prediction)
#prediction = np.where(prediction == 'jaumecollboni','psc',prediction)
#prediction = np.where(prediction == 'miqueliceta','psc',prediction)

# TIP - Copy and paste this function to generate the output file in your code
def save_submission(prediction):
    import datetime
    t = datetime.datetime.now().strftime("%Y%m%d")
    output = pd.DataFrame({'Party': prediction})
    output.index.name = 'Id'
    output.to_csv(f'C:/Workspace/Yoga/Code/sample_submission_{t}.csv')
    
save_submission(prediction)    


#Lina's work

#emojis
#import emoji
#def extract_emojis(str):
#  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

#Dummies language, it categorizes as others languages not cat, es, en
#language = pd.get_dummies(train_df_tr['language'])
#language_list = ['ca','en','es','other']
#languaget = pd.DataFrame()
#languaget['ca'] = np.where(train_df['language']=='ca', 1, 0)
#languaget['es'] = np.where(train_df['language']=='es', 1, 0)
#languaget['en'] = np.where(train_df['language']=='en', 1, 0)
#languaget['other'] = train_df['language'].apply(lambda x: 1 if x not in set(language_list) else 0)

#languages = pd.DataFrame()
#languages['ca'] = np.where(test_df['language']=='ca', 1, 0)
#languages['es'] = np.where(test_df['language']=='es', 1, 0)
#languages['en'] = np.where(test_df['language']=='en', 1, 0)
#languages['other'] = test_df['language'].apply(lambda x: 1 if x not in set(language_list) else 0)