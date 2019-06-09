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

# counts per party
df_counts = pd.DataFrame({ 'username' : train_df_tr['username'].values,
                           'words' : train_df_tr['text'].apply(lambda x: funcs.number_words(x)),
                           'emoticons' : train_df_tr['text'].apply(lambda x: funcs.number_emoticons(x)),
                           'hashtags' : train_df_tr['text'].apply(lambda x: funcs.number_hashtags(x)),
                           'mentions' : train_df_tr['text'].apply(lambda x: funcs.number_mentions(x)),
                         })

#----------- 3. Preprocess data

# target
y = train_df_tr['username'].values
y1, y2 = np.unique(y, return_inverse=True)
values, counts = np.unique(y, return_counts=True)
politicians = pd.DataFrame({'values': values,
                            'counts': counts})
politicians['counts'].plot.hist() # Disbalanced

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
#### SVM ####
from sklearn.svm import SVC
param_grid = [{'estimator__kernel':['rbf'],'estimator__gamma':[1e-4,1e-3,0.5,5,10,50],'estimator__C':[0.001,0.1,1,10,100] },
              {'estimator__kernel':['linear'], 'estimator__C':[1, 10, 100, 1000]}]
gsv = GridSearchCV(SVC(), param_grid, cv=5, scoring='average_precision', n_jobs=-1)
gsv.fit(X_train,y_train)
#Best HyperParameter:  {'estimator__C': 1, 'estimator__kernel': 'linear'}

clf= SVC(C= 1, kernel= 'linear')
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print("Prediction: {}".format(np.mean([prediction == y_test])))
#------------------------------------------------------------------------


#------------------------------------------------------------------------
#### KNN ####
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
# define the parameter values that should be searched
k_range = list(range(1, 31))
print(k_range)

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
print(param_grid)

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)

# fit the grid with data
grid.fit(X_train,y_train)
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

#    mean_test_score  std_test_score               params
#0          0.411308        0.008945   {'n_neighbors': 1}
#1          0.383962        0.016952   {'n_neighbors': 2}
#2          0.388211        0.013291   {'n_neighbors': 3}
#3          0.396896        0.018203   {'n_neighbors': 4}
#4          0.398374        0.016028   {'n_neighbors': 5}
#5          0.397450        0.015767   {'n_neighbors': 6}
#6          0.398189        0.016545   {'n_neighbors': 7}
#7          0.398189        0.019866   {'n_neighbors': 8}
#8          0.396157        0.020081   {'n_neighbors': 9}
#9          0.394678        0.018251  {'n_neighbors': 10}
#10         0.398189        0.017076  {'n_neighbors': 11}
#11         0.400037        0.018136  {'n_neighbors': 12}
#12         0.393200        0.017951  {'n_neighbors': 13}
#13         0.389505        0.016395  {'n_neighbors': 14}
#14         0.391907        0.018764  {'n_neighbors': 15}
#15         0.391907        0.018375  {'n_neighbors': 16}
#16         0.394124        0.017641  {'n_neighbors': 17}
#17         0.395418        0.018961  {'n_neighbors': 18}
#18         0.393570        0.022339  {'n_neighbors': 19}
#19         0.390613        0.021501  {'n_neighbors': 20}
#20         0.387288        0.020707  {'n_neighbors': 21}
#21         0.383777        0.021821  {'n_neighbors': 22}
#22         0.385994        0.020882  {'n_neighbors': 23}
#23         0.383777        0.023380  {'n_neighbors': 24}
#24         0.382114        0.018740  {'n_neighbors': 25}
#25         0.376571        0.018080  {'n_neighbors': 26}
#26         0.376940        0.018684  {'n_neighbors': 27}
#27         0.373614        0.022256  {'n_neighbors': 28}
#28         0.377679        0.020201  {'n_neighbors': 29}
#29         0.374169        0.020142  {'n_neighbors': 30}



#------------------------------------------------------------------------






# Create the results file
clf = SVC(X_tfidf,y)
predictions = clf.predict(X_tfidf_test)
funcs.save_submission(predictions, "sample_submission_svc")
