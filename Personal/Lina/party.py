import sys
import time

import os
os.chdir("/Users/lina/DataScience/Project/Yoga/Personal/Lina/")
sys.path.append("/Users/lina/DataScience/Project/Yoga/Personal/Lina/")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, ShuffleSplit
import files.functions as funcs
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


#----------- 1. Load data
train_df_tr = pd.read_csv('/Users/lina/DataScience/Project/Yoga/Data/train_traducido.csv', delimiter=';') # Load the traductions file
test_df_tr = pd.read_csv('/Users/lina/DataScience/Project/Yoga/Data/test_traducido.csv', delimiter=';') # Load the traductions file

#removing indentified languages
train_df_tr.drop(train_df_tr[(train_df_tr['language_id']=='lb') | (train_df_tr['language_id']=='co')].index, inplace=True)
train_df_tr.reset_index(inplace=True, drop=True)

# counts per party
df_counts = pd.DataFrame({ 'party' : train_df_tr['party'].values,
                           'words' : train_df_tr['text'].apply(lambda x: funcs.number_words(x)),
                           'emoticons' : train_df_tr['text'].apply(lambda x: funcs.number_emoticons(x)),
                           'hashtags' : train_df_tr['text'].apply(lambda x: funcs.number_hashtags(x)),
                           'mentions' : train_df_tr['text'].apply(lambda x: funcs.number_mentions(x)),
                         })

#----------- 3. Preprocess data

# target
y = train_df_tr['party'].values
y1, y2 = np.unique(y, return_inverse=True)

train_df_tr['text_clean'] = train_df_tr['traducciones'].apply(lambda x: funcs.clean_text(x))

# language feature
language = pd.DataFrame({ 'ca' : np.where(train_df_tr['language_id']=='ca', 1, 0),
                          'es' : np.where(train_df_tr['language_id']=='es', 1, 0),
                          'en' : np.where(train_df_tr['language_id']=='en', 1, 0),
                          'other' : train_df_tr['language_id'].apply(lambda x: 1 if x not in {'ca','es','en'} else 0)
                        })


# Counts features and scale
scaler = StandardScaler()
counts = train_df_tr[['retweet_count','favorite_count']].apply(lambda x: x+1)
counts = pd.DataFrame()
counts['counts_emoticons'] = df_counts.emoticons.apply(lambda x: x+1)
counts['counts_hashtags'] = df_counts.hashtags.apply(lambda x: x+1)
counts['counts_mentions'] = df_counts.mentions.apply(lambda x: x+1)
counts['counts_words'] = df_counts.words.apply(lambda x: x+1)
counts = np.log10(counts)
counts['engagement_rate'] = df_counts.engagement_rate
features_counts = scaler.fit_transform(counts)
features_counts = pd.DataFrame(features_counts)


# Parameter selection (TFIDF)
# tfidf matrix
tfidf_vectorizer = funcs.StemmedTfidfVectorizer(
        sublinear_tf = True, #scaling
        #strip_accents='unicode',
        max_df = 0.25,#0.5,
        min_df = 3,
        norm='l2',
        #token_pattern='#?\w\w+',#r'[^0-9]\w{1,}',#r'#?[^0-9]\w\w+',
        stop_words=funcs.stop_words(),
        ngram_range=(1,1),
        #max_features=4000
        )

X_tfidf = tfidf_vectorizer.fit_transform(train_df_tr['text_clean']).toarray()
X_tfidf = np.append(X_tfidf,language,1)
X_tfidf = np.append(X_tfidf, features_counts,1)


#---------------------- Model selection
# Split train and test
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_tfidf, y, train_df_tr.index, test_size=0.25)

#param_grid = [{'estimator__kernel':['rbf'],'estimator__gamma':[1e-4,1e-3,0.5,5,10,50],'estimator__C':[0.001,0.1,1,10,100] },
#              {'estimator__kernel':['linear'], 'estimator__C':[1, 10, 100, 1000]}]
#gsv = GridSearchCV(SVC(), param_grid, cv=5, scoring='average_precision', n_jobs=-1)
#gsv.fit(X_train,y_train)
#Best HyperParameter:  {'estimator__C': 1, 'estimator__kernel': 'linear'}


clf= SVC(C= 1, kernel= 'linear')
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print("Prediction: {}".format(np.mean([prediction == y_test])))

# Create the results file
clf = SVC(X_tfidf,y)
predictions = clf.predict(X_tfidf_test)
funcs.save_submission(predictions, "sample_submission_svc")
