import sys
import time
import imp

import os
os.chdir("/Users/oliver/YOGA/Yoga/Personal/Oliver/")
sys.path.append("/Users/oliver/YOGA/Yoga/Personal/Oliver/")
sys.path.insert(0, "/Users/oliver/YOGA/Yoga/Personal/Oliver/")
funcs = imp.load_source('functions', '/Users/oliver/YOGA/Yoga/Personal/Oliver/files/functions.py')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, ShuffleSplit
#import files.functions as funcs
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import datetime as dt

pd.set_option('max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('precision', 4)


#----------- 1. Load data
train_df_tr = pd.read_csv('/Users/oliver/YOGA/Yoga/Data/train_traducido.csv', delimiter=';') # Load the traductions file
test_df_tr = pd.read_csv('/Users/oliver/YOGA/Yoga/Data/test_traducido.csv', delimiter=';') # Load the traductions file

#removing indentified languages
train_df_tr.drop('party', axis=1, inplace=True)

#removing indentified languages
train_df_tr.drop(train_df_tr[(train_df_tr['language_id']=='lb') | (train_df_tr['language_id']=='co')].index, inplace=True)
train_df_tr.reset_index(inplace=True, drop=True)

# counts per politician
df_counts = pd.DataFrame({ 'username' : train_df_tr['username'].values,
                           'words' : train_df_tr['text'].apply(lambda x: funcs.number_words(x)),
                           'emoticons' : train_df_tr['text'].apply(lambda x: funcs.number_emoticons(x)),
                           'hashtags' : train_df_tr['text'].apply(lambda x: funcs.number_hashtags(x)),
                           'mentions' : train_df_tr['text'].apply(lambda x: funcs.number_mentions(x)),
                         })

#----------- 3. Preprocess data

### with imbalanced data ###
y = train_df_tr['username'].values
y1, y2 = np.unique(y, return_inverse=True)


#### without imbalanced data ###
## target
#y = train_df_tr['username'].values
#y1, y2 = np.unique(y, return_inverse=True)
#values, tweet_counts = np.unique(y, return_counts=True)
#politicians = pd.DataFrame({'values': values,
#                            'tweet_counts': tweet_counts})
#politicians['tweet_counts'].plot.hist() # Disbalanced
#
#max_tweet = np.max(politicians['tweet_counts'])
#
## todos los políticos con mismo número de tweets
#pol_df_total = train_df_tr.iloc[0:0]
#for pol in politicians['values']:
#    pol_df = train_df_tr[train_df_tr.username == pol]
#    if pol == 'olallamarga':
#        pol_df = pol_df.loc[pol_df.index.repeat(20)]
#    pol_df = pd.concat([pol_df, pol_df[0:max_tweet-len(pol_df)]], axis=0)
#    pol_df_total = pd.concat([pol_df_total, pol_df], axis=0)
#
#pol_df_total.reset_index(inplace=True, drop=True)
#train_df_tr = pol_df_total.copy()
#del(pol_df_total)
#
## counts per politician
#df_counts = pd.DataFrame({ 'username' : train_df_tr['username'].values,
#                           'words' : train_df_tr['text'].apply(lambda x: funcs.number_words(x)),
#                           'emoticons' : train_df_tr['text'].apply(lambda x: funcs.number_emoticons(x)),
#                           'hashtags' : train_df_tr['text'].apply(lambda x: funcs.number_hashtags(x)),
#                           'mentions' : train_df_tr['text'].apply(lambda x: funcs.number_mentions(x)),
#                         })

# -----------------------------------------------






train_df_tr['text_clean'] = train_df_tr['traducciones'].apply(lambda x: funcs.clean_text(x))

# language feature
language = pd.DataFrame({ 'ca' : np.where(train_df_tr['language_id']=='ca', 1, 0),
                          'es' : np.where(train_df_tr['language_id']=='es', 1, 0),
                          'en' : np.where(train_df_tr['language_id']=='en', 1, 0),
                          'other' : train_df_tr['language_id'].apply(lambda x: 1 if x not in {'ca','es','en'} else 0)
                        })

# day of week feature
train_df_tr.created_at = pd.to_datetime(train_df_tr.created_at)
day_week = train_df_tr['created_at'].dt.day_name()
ohe = OneHotEncoder(handle_unknown='ignore')
day_week = np.array(day_week)
day_week = day_week.reshape(-1, 1)
day_week = ohe.fit_transform(day_week).toarray()

# part of day feature
new_hours = train_df_tr.created_at[train_df_tr.created_at.dt.hour == 0]
new_hours2 = new_hours + dt.timedelta(hours=1)
train_df_tr.created_at[train_df_tr.created_at.dt.hour == 0] = new_hours2
part_day = pd.cut(train_df_tr.created_at.dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening'])
ohe = OneHotEncoder(handle_unknown='ignore')
part_day = np.array(part_day)
part_day = part_day.reshape(-1, 1)
part_day = ohe.fit_transform(part_day).toarray()
del(new_hours, new_hours2)

# Counts features and scale
scaler = StandardScaler()
counts = train_df_tr[['retweet_count','favorite_count']].apply(lambda x: x+1)
counts = pd.DataFrame()
counts['counts_emoticons'] = df_counts.emoticons.apply(lambda x: x+1)
counts['counts_hashtags'] = df_counts.hashtags.apply(lambda x: x+1)
counts['counts_mentions'] = df_counts.mentions.apply(lambda x: x+1)
counts['counts_words'] = df_counts.words.apply(lambda x: x+1)
counts = np.log10(counts)
#counts['engagement_rate'] = df_counts.engagement_rate
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
X_tfidf = np.append(X_tfidf, day_week, 1)
X_tfidf = np.append(X_tfidf, part_day, 1)


#---------------------- Model selection
# Split train and test
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_tfidf, y, train_df_tr.index, test_size=0.25)

#------------------------------------------------------------------------
##### SVM ####
#from sklearn.svm import SVC
#param_grid = [{'estimator__kernel':['rbf'],'estimator__gamma':[1e-3, 1e-4],'estimator__C':[0.001,0.1,1,10,100] },
#              {'estimator__kernel':['linear'], 'estimator__C':[1, 10, 100, 1000]}]
#gsv = GridSearchCV(SVC(), param_grid, cv=5, scoring='average_precision', n_jobs=-1)
#gsv.fit(X_train,y_train)
##Best HyperParameter:  {'estimator__C': 1, 'estimator__kernel': 'linear'}
#
#clf= SVC(C= 1, kernel= 'linear')
#clf.fit(X_train,y_train)
#prediction = clf.predict(X_test)
#print("Prediction: {}".format(np.mean([prediction == y_test])))
#------------------------------------------------------------------------


#------------------------------------------------------------------------
#### KNN ####
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
# define the parameter values that should be searched
k_range = list(range(1, 31))
#print(k_range)

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
#print(param_grid)

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)

# fit the grid with data
grid.fit(X_train,y_train)
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

#    mean_test_score  std_test_score               params
#0   0.1436           0.0134          {'n_neighbors': 1} 
#1   0.1175           0.0101          {'n_neighbors': 2} 
#2   0.1171           0.0130          {'n_neighbors': 3} 
#3   0.1253           0.0109          {'n_neighbors': 4} 
#4   0.1319           0.0093          {'n_neighbors': 5} 
#5   0.1362           0.0057          {'n_neighbors': 6} 
#6   0.1369           0.0077          {'n_neighbors': 7} 
#7   0.1412           0.0086          {'n_neighbors': 8} 
#8   0.1439           0.0084          {'n_neighbors': 9} 
#9   0.1486           0.0088          {'n_neighbors': 10}
#10  0.1476           0.0120          {'n_neighbors': 11}
#11  0.1443           0.0120          {'n_neighbors': 12}
#12  0.1486           0.0155          {'n_neighbors': 13}
#13  0.1504           0.0161          {'n_neighbors': 14}
#14  0.1545           0.0156          {'n_neighbors': 15}
#15  0.1569           0.0169          {'n_neighbors': 16}
#16  0.1591           0.0147          {'n_neighbors': 17}
#17  0.1609           0.0121          {'n_neighbors': 18}
#18  0.1622           0.0126          {'n_neighbors': 19}
#19  0.1648           0.0143          {'n_neighbors': 20}
#20  0.1667           0.0155          {'n_neighbors': 21}
#21  0.1667           0.0162          {'n_neighbors': 22}
#22  0.1670           0.0165          {'n_neighbors': 23}
#23  0.1674           0.0180          {'n_neighbors': 24}
#24  0.1672           0.0151          {'n_neighbors': 25}
#25  0.1669           0.0137          {'n_neighbors': 26}
#26  0.1665           0.0147          {'n_neighbors': 27}
#27  0.1696           0.0137          {'n_neighbors': 28}
#28  0.1676           0.0146          {'n_neighbors': 29}
#29  0.1683           0.0157          {'n_neighbors': 30}

knn = KNeighborsClassifier(n_neighbors=28, weights='uniform')
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
print("Prediction: {}".format(np.mean([prediction == y_test])))
#Prediction: 0.16518847006651885
#------------------------------------------------------------------------


#------------------------------------------------------------------------
#### RandomForest ####
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=42)

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train,y_train)

CV_rfc.best_params_

#{'criterion': 'gini',
# 'max_depth': 8,
# 'max_features': 'auto',
# 'n_estimators': 500}

rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=8, criterion='gini')

rfc1.fit(X_train, y_train)

pred=rfc1.predict(X_test)

print("Prediction: {}".format(np.mean([pred == y_test])))
#Prediction: 0.2860310421286031
#------------------------------------------------------------------------







# Create the results file
clf = SVC(X_tfidf,y)
predictions = clf.predict(X_tfidf_test)
funcs.save_submission(predictions, "sample_submission_svc")
