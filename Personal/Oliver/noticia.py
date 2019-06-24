import sys
import imp
import os
os.chdir("/Users/oliver/YOGA/Yoga/Personal/Oliver/")
sys.path.append("/Users/oliver/YOGA/Yoga/Personal/Oliver/")
sys.path.insert(0, "/Users/oliver/YOGA/Yoga/Personal/Oliver/")
funcs = imp.load_source('functions', '/Users/oliver/YOGA/Yoga/Personal/Oliver/files/functions.py')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import functions as funcs
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


def fill_nan_in_matrix(X, value):
    ix_list = np.argwhere(np.isnan(X)).tolist()
    for ix in ix_list:
        X[ix[0], ix[1]] = value


def get_sentiment_features_df(ROOT_PATH, str):
    if (str == 'train'):
        fname = 'train_sentiment_features.csv'
    elif (str == 'test'):
        fname = 'test_sentiment_features.csv'
    train_sentiment_raw = pd.read_csv(os.path.join(
        ROOT_PATH, "Data", fname),
        delimiter=';').iloc[:, [0, 1, 2, 3]]
    df = train_sentiment_raw.copy()
    df = pd.concat([df, pd.get_dummies(df['amazon_sentiment'])], axis=1).rename(
        columns={'POSITIVE': 'az_positive',
                 "NEGATIVE": 'az_negative',
                 "NEUTRAL": 'az_neutral'}).drop('amazon_sentiment', axis=1)

    return df


def count_features_and_scale(df, df_counts):
    scaler = StandardScaler()
    counts = df[['retweet_count', 'favorite_count']].apply(lambda x: x + 1)
    counts['counts_emoticons'] = df_counts.emoticons.apply(lambda x: x + 1)
    counts['counts_hashtags'] = df_counts.hashtags.apply(lambda x: x + 1)
    counts['counts_mentions'] = df_counts.mentions.apply(lambda x: x + 1)
    counts['counts_words'] = df_counts.words.apply(lambda x: x + 1)
    counts = np.log10(counts)
    features_counts = scaler.fit_transform(counts)
    features_counts = pd.DataFrame(features_counts)

    return features_counts


def remove_tweets_with_non_identified_language(df):
    df.drop(df[(df['language_id'] == 'lb') |
               (df['language_id'] == 'co') |
               (df['language_id'] == 'mr')].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def get_features_of_interest_counts(df):
    return pd.DataFrame({'words': df['text'].apply(lambda x: funcs.number_words(x)),
                         'emoticons': df['text'].apply(lambda x: funcs.number_emoticons(x)),
                         'hashtags': df['text'].apply(lambda x: funcs.number_hashtags(x)),
                         'mentions': df['text'].apply(lambda x: funcs.number_mentions(x)),
                         })


def get_language_df(df):
    return pd.DataFrame({'ca': np.where(df['language_id'] == 'ca', 1, 0),
                         'es': np.where(df['language_id'] == 'es', 1, 0),
                         'en': np.where(df['language_id'] == 'en', 1, 0),
                         'other': df['language_id'].apply(lambda x: 1 if x not in {'ca', 'es', 'en'} else 0)
                         })


def add_text_clean_col_to_df(df):
    df['text_clean'] = df['traducciones'].apply(lambda x: funcs.clean_text(x))

    return df

def get_day_week(df):
    df.created_at = pd.to_datetime(df.created_at)
    day_week = df['created_at'].dt.day_name()
    ohe = OneHotEncoder(handle_unknown='ignore')
    day_week = np.array(day_week)
    day_week = day_week.reshape(-1, 1)
    day_week = ohe.fit_transform(day_week).toarray()
    return pd.DataFrame(day_week)

def get_part_day(df):
    new_hours = df.created_at[df.created_at.dt.hour == 0]
    new_hours2 = new_hours + dt.timedelta(hours=1)
    df.created_at[df.created_at.dt.hour == 0] = new_hours2
    part_day = pd.cut(df.created_at.dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening'])
    ohe = OneHotEncoder(handle_unknown='ignore')
    part_day = np.array(part_day)
    part_day = part_day.reshape(-1, 1)
    part_day = ohe.fit_transform(part_day).toarray()
    return pd.DataFrame(part_day)
    
#def balance_dataset(df, sentiment):
#    
#    ''' return balanced train df & balanced sentiment analysis '''
#    
#    #target
#    y = df['username'].values
#    y1, y2 = np.unique(y, return_inverse=True)
#    values, tweet_counts = np.unique(y, return_counts=True)
#    politicians = pd.DataFrame({'values': values,
#                                'tweet_counts': tweet_counts})
#    
#    
#    max_tweet = np.max(politicians['tweet_counts'])
#    
#    df = pd.concat([df, sentiment], axis=1)
#
#    # todos los políticos con mismo número de tweets
#    pol_df_total = train_df_tr.iloc[0:0]
#    for pol in politicians['values']:
#        pol_df = train_df_tr[train_df_tr.username == pol]
#        if pol == 'olallamarga':
#            pol_df = pol_df.loc[pol_df.index.repeat(20)]
#        pol_df = pd.concat([pol_df, pol_df[0:max_tweet-len(pol_df)]], axis=0)
#        pol_df_total = pd.concat([pol_df_total, pol_df], axis=0)
#    
#    pol_df_total.reset_index(inplace=True, drop=True)
#    
#    sentiment_cols = ['id', 
#                      'azure_sentiment',
#                      'google_sentiment',
#                      'MIXED',
#                      'az_negative',
#                      'az_neutral',
#                      'az_positive']
#    
#    train_cols = ['Id', 'username', 'text', 'created_at', 'retweet_count',
#                   'favorite_count', 'no_of_characters', 'language', 'text_modif',
#                   'language_id', 'traducciones', 'text_clean']
#    
#    new_train = pol_df_total.copy()
#    new_sentiment = pol_df_total.copy()
#    
#    new_sentiment = new_sentiment.filter(sentiment_cols)
#    new_train = new_train.filter(train_cols)
#    
#    return new_train, new_sentiment








if __name__ == '__main__':
    # SET PATHS ##
    ROOT_PATH = "/Users/oliver/YOGA/Yoga/"
    PERSONAL_PATH = "Personal/Oliver"

    os.chdir(os.path.join(ROOT_PATH, PERSONAL_PATH))
    sys.path.append(os.path.join(ROOT_PATH, PERSONAL_PATH))
    TRAIN_FPATH = os.path.join(ROOT_PATH, "Data/train_traducido.csv")
    TEST_FPATH = os.path.join(ROOT_PATH, "Data/test_traducido.csv")

    ## Load data and Preprocess Data ##
    train_df_tr = pd.read_csv(TRAIN_FPATH, delimiter=';')
    train_df_tr = remove_tweets_with_non_identified_language(train_df_tr)
    train_df_tr = add_text_clean_col_to_df(train_df_tr)
    
    #train_df_tr.drop('party', axis=1, inplace=True) 
    
    ##################################################
          #####     balancing dataset   #######   
    ##################################################

    train_df_sentiment = get_sentiment_features_df(ROOT_PATH, str='train') # load sentiment
    #train_df_tr, train_df_sentiment = balance_dataset(train_df_tr, train_df_sentiment) NO SE PORQ NO VA!!!!!
    
    train_df_tr = pd.concat([train_df_tr, train_df_sentiment], axis=1)

    def giveme_max_min(feature):
        x = np.max(train_df_tr[feature])
        y = np.min(train_df_tr[feature])
        return x,y
    
    def create_plot(feature1, feature2, feature_color):
        x = feature1
        y = feature2
        fig = plt.figure()
        
        categories = np.unique(train_df_tr[feature_color])
        colors = np.linspace(0, 1, len(categories))
        colordict = dict(zip(categories, colors))
        
        train_df_tr["Color"] = train_df_tr[feature_color].apply(lambda x: colordict[x])
        ax = fig.add_subplot(111)
        ax.scatter(train_df_tr[x], train_df_tr[y], c=train_df_tr.Color)
        ax.set_xlim(giveme_max_min(feature1)[1], giveme_max_min(feature1)[0])
        ax.set_ylim(giveme_max_min(feature2)[1], giveme_max_min(feature2)[0])
        ax.set(ylabel=feature2, xlabel=feature1)
        ax.legend(loc='best')
        return plt.show()

    def create_plot2(feature1, feature2, feature_color):
        x = feature1
        y = feature2
        
        lista = list(np.unique(train_df_tr[feature_color]))
        sns.set_style("whitegrid")
        fg = sns.FacetGrid(data=train_df_tr,
                           col=y,
                           row=x,
                           hue=feature_color, hue_order=lista, aspect=1.61)
        fg.map(plt.scatter, 'Weight (kg)', 'Height (cm)').add_legend()
        
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.scatter(train_df_tr[x], train_df_tr[y])
#        ax.set_xlim(giveme_max_min(feature1)[1], giveme_max_min(feature1)[0])
#        ax.set_ylim(giveme_max_min(feature2)[1], giveme_max_min(feature2)[0])
        return plt.show()

    def create_plot3(feature1, feature2, feature_color):
        x = feature1
        y = feature2
        
        list_of_series = [train_df_tr[x], train_df_tr[y], train_df_tr[feature_color]]
        col_names = [x,y,feature_color]
        df = pd.DataFrame(list_of_series, columns=col_names)
        df = pd.concat(list_of_series, axis=1)
        
        
        fg = sns.FacetGrid(data=df, hue=feature_color, hue_order=lista, aspect=1.61)
        fg.map(plt.scatter, x, y, alpha= 0.4).add_legend()
        fig = fg.fig
        fig.set_size_inches(10, 12)
        return plt.show()

    
    def plot_uniqueval_column(feature):
        y1, y2 = np.unique(train_df_tr[feature], return_inverse=True)
        values, feature_ = np.unique(train_df_tr[feature], return_counts=True)
        output = pd.DataFrame({'values': values,
                               'feature': feature_})
        ax = output['feature'].plot()
        #ax.set_xlim(giveme_max_min(feature)[1], giveme_max_min(feature)[0])
        return ax





x = 'azure_sentiment'
y = 'google_sentiment'
feature_color = 'username'

lista = list(np.unique(train_df_tr[feature_color]))

list_of_series = [train_df_tr[x], train_df_tr[y], train_df_tr[feature_color]]
col_names = [x,y,feature_color]
df = pd.DataFrame(list_of_series, columns=col_names)
df = pd.concat(list_of_series, axis=1)

fg = sns.FacetGrid(data=df, hue=feature_color, hue_order=lista, aspect=1.61)
fg.map(plt.scatter, x, y).add_legend()











df = pd.DataFrame(
data=np.random.randn(90, 4),
columns=pd.Series(list("ABCD"), name="walk"),
index=pd.date_range("2015-01-01", "2015-03-31",
                      name="date"))
df = df.cumsum(axis=0).stack().reset_index(name="val")
def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
g = sns.FacetGrid(df, col="walk", col_wrap=2, height=3.5)
g = g.map_dataframe(dateplot, "date", "val")






x = 'created_at'
y = 'google_sentiment'
feature_color = 'party'

lista = list(np.unique(train_df_tr[feature_color]))



list_of_series = [train_df_tr[x], train_df_tr[y], train_df_tr[feature_color]]
col_names = [x,y,feature_color]
df = pd.DataFrame(list_of_series, columns=col_names)
df = pd.concat(list_of_series, axis=1)

df.created_at = pd.to_datetime(df.created_at)
df = df.groupby([pd.Grouper(key='created_at', freq='M'),
                 pd.Grouper(key='party')]).agg(['count', 'mean'])

a = pd.DataFrame(df.to_records(),
                 columns=df.index.names + list(df.columns))

b = df.stack().reset_index()
b = df.stack(level=0).reset_index()
#df.unstack()
#b = df.stack(level=1).reset_index(level=1, drop=True).reset_index()
#------------------------------


#b = pd.melt(b, id_vars =['level_2'], value_vars =['count'])
sns.set()
fg = sns.pairplot(b, hue='party').add_legend()   
fig = fg.fig
fig.set_size_inches(11, 11)


   
a.boxplot(by='created_at')
a.set_xticklabels(rotation=30)    

fg = sns.factorplot(x='created_at', y='google_sentiment', hue='party', 
                        #col='Sex', 
                        data=a, kind='bar')
fg.set_xlabels('')
fg.set_xticklabels(rotation=30)
#df = df.set_index(x)

def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
    
g = sns.FacetGrid(df)
g = g.map_dataframe(dateplot, x, y)

#-------------------------------------------------

np.random.seed(45)
a = pd.DataFrame(index=range(10), 
                 columns=pd.MultiIndex.from_product(
                         iterables=[['2000', '2010'], ['a', 'b']], 
                         names=['Year', 'Text']), 
                 data=np.random.randn(10,4))

b = a.stack(level=0).reset_index(level=0, drop=True).reset_index()
print (b)

































































    
    ##target
    party = train_df_tr['party'].values
    politicians = train_df_tr['username'].values
    
    
    y1, y2 = np.unique(y, return_inverse=True)
    values, tweet_counts = np.unique(y, return_counts=True)
    politicians = pd.DataFrame({'values': values,
                                'tweet_counts': tweet_counts})
    
    
    max_tweet = np.max(politicians['tweet_counts'])
    
    
    
    
#    # todos los políticos con mismo número de tweets
#    pol_df_total = train_df_tr.iloc[0:0]
#    for pol in politicians['values']:
#        pol_df = train_df_tr[train_df_tr.username == pol]
#        if pol == 'olallamarga':
#            pol_df = pol_df.loc[pol_df.index.repeat(20)]
#        pol_df = pd.concat([pol_df, pol_df[0:max_tweet-len(pol_df)]], axis=0)
#        pol_df_total = pd.concat([pol_df_total, pol_df], axis=0)
#    
#    pol_df_total.reset_index(inplace=True, drop=True)
#    
#    sentiment_cols = ['id', 'azure_sentiment','google_sentiment', 'MIXED', 'az_negative', 'az_neutral','az_positive']
#    train_cols = ['Id', 'username', 'text', 'created_at', 'retweet_count',
#                   'favorite_count', 'no_of_characters', 'language', 'text_modif',
#                   'language_id', 'traducciones', 'text_clean']
#    
#    sentiment_df = pol_df_total.filter(sentiment_cols)
#    pol_df_total = pol_df_total.filter(train_cols)
#    
#    train_df_tr = pol_df_total.copy()
#    train_df_sentiment = sentiment_df.copy()
#    del(sentiment_cols, train_cols, pol_df,pol, max_tweet, politicians, 
#        values, tweet_counts, y1, y2, y)
#    ##################################################
    
    
    
    train_df_counts = get_features_of_interest_counts(train_df_tr)
    train_df_languages = get_language_df(train_df_tr)
    train_df_day_week = get_day_week(train_df_tr)
    train_df_part_day = get_part_day(train_df_tr)
    #train_df_sentiment = get_sentiment_features_df(ROOT_PATH, str='train')

    test_df_tr = pd.read_csv(TEST_FPATH, delimiter=';')
    test_df_tr = add_text_clean_col_to_df(test_df_tr)
    test_df_counts = get_features_of_interest_counts(test_df_tr)
    test_df_languages = get_language_df(test_df_tr)
    test_df_day_week = get_day_week(test_df_tr)
    test_df_part_day = get_part_day(test_df_tr)
    test_df_sentiment = get_sentiment_features_df(ROOT_PATH, str='test')


































































    ## LET's DO A NICE MODEL

    y = train_df_tr['username'].values
    y1, y2 = np.unique(y, return_inverse=True)

    # Counts features and scale
    train_features_count = count_features_and_scale(train_df_tr, train_df_counts)
    test_features_count = count_features_and_scale(test_df_tr, test_df_counts)

    # Parameter selection (TFIDF)
    tfidf_vectorizer = funcs.StemmedTfidfVectorizer(
        sublinear_tf=True,  # scaling
        # strip_accents='unicode',
        max_df=0.25,  # 0.5,
        min_df=3,
        norm='l2',
        # token_pattern='#?\w\w+',#r'[^0-9]\w{1,}',#r'#?[^0-9]\w\w+',
        stop_words=funcs.stop_words(),
        ngram_range=(1, 1),
        # max_features=4000
    )

    X_tfidf = tfidf_vectorizer.fit_transform(train_df_tr['text_clean']).toarray()
    X_tfidf = np.append(X_tfidf, train_df_languages, 1)
    X_tfidf = np.append(X_tfidf, train_features_count, 1)
    X_tfidf = np.append(X_tfidf, train_df_sentiment, 1)
    X_tfidf = np.append(X_tfidf, train_df_day_week, 1)
    X_tfidf = np.append(X_tfidf, train_df_part_day, 1)

    X_tfidf_test = tfidf_vectorizer.transform(test_df_tr['text_clean']).toarray()
    X_tfidf_test = np.append(X_tfidf_test, test_df_languages, 1)
    X_tfidf_test = np.append(X_tfidf_test, test_features_count, 1)
    X_tfidf_test = np.append(X_tfidf_test, test_df_sentiment, 1)
    X_tfidf_test = np.append(X_tfidf_test, test_df_day_week, 1)
    X_tfidf_test = np.append(X_tfidf_test, test_df_part_day, 1)
    fill_nan_in_matrix(X_tfidf_test, value=0.5)

    le = LabelEncoder()
    y_encode = le.fit_transform(train_df_tr['username'])
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_tfidf,
                                                                                     y_encode,
                                                                                     train_df_tr.index,
                                                                                     test_size=0.25)

    clf = SVC(C=1, kernel='linear')
    assert len(X_train) == len(y_train)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print("Prediction: {}".format(np.mean([prediction == y_test])))

    #Create the results file
    assert len(X_tfidf) == len(y)
    clf.fit(X_tfidf, y)
    predictions = clf.predict(X_tfidf_test)
    len(X_tfidf_test)
    funcs.save_submission(predictions, "sample_submission_oliver_svc_politicians")
