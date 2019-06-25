import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from parties_classification import remove_tweets_with_non_identified_language
from parties_classification import add_text_clean_col_to_df
from parties_classification import get_sentiment_features_df


def giveme_max_min(feature):
    x = np.max(train_df_tr[feature])
    y = np.min(train_df_tr[feature])
    return x, y


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
    return plt.show()


def create_plot3(feature1, feature2, feature_color):
    x = feature1
    y = feature2

    list_of_series = [train_df_tr[x], train_df_tr[y], train_df_tr[feature_color]]
    col_names = [x, y, feature_color]
    df = pd.DataFrame(list_of_series, columns=col_names)
    df = pd.concat(list_of_series, axis=1)

    fg = sns.FacetGrid(data=df, hue=feature_color, hue_order=lista, aspect=1.61)
    fg.map(plt.scatter, x, y, alpha=0.4).add_legend()
    fig = fg.fig
    fig.set_size_inches(10, 12)
    return plt.show()


def plot_uniqueval_column(feature):
    y1, y2 = np.unique(train_df_tr[feature], return_inverse=True)
    values, feature_ = np.unique(train_df_tr[feature], return_counts=True)
    output = pd.DataFrame({'values': values,
                           'feature': feature_})
    ax = output['feature'].plot()
    return ax


def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)


if __name__ == '__main__':
    ROOT_PATH = "C:/workspace/my_repos/Capstone DS/"
    PERSONAL_PATH = "Personal/Alvaro"

    os.chdir(os.path.join(ROOT_PATH, PERSONAL_PATH))
    sys.path.append(os.path.join(ROOT_PATH, PERSONAL_PATH))
    TRAIN_FPATH = os.path.join(ROOT_PATH, "Data/train_traducido.csv")

    ## Load data and Preprocess Data ##
    train_df_tr = pd.read_csv(TRAIN_FPATH, delimiter=';')
    train_df_tr = remove_tweets_with_non_identified_language(train_df_tr)
    train_df_tr = add_text_clean_col_to_df(train_df_tr)
    train_df_sentiment = get_sentiment_features_df(ROOT_PATH, train_df_tr, str='train')  # load sentiment
    train_df_tr = pd.concat([train_df_tr, train_df_sentiment], axis=1)
    cols_to_keep = ['Id', 'created_at', 'username', 'party',
                    'az_positive', 'az_negative', 'az_neutral',
                    'azure_sentiment', 'google_sentiment', 'google_emotion']
    tweets_df = train_df_tr.loc[:, cols_to_keep]
    tweets_df.created_at = pd.to_datetime(tweets_df.created_at)
    tweets_df = tweets_df.sort_values('created_at')
    tweets_df.head()

    ##feature_color Make beautiful Plots ##
    tweets_df.describe()


    f = {'google_sentiment': 'mean',
         'azure_sentiment': 'mean'
         #'party': 'size'
         }

  #  f_df = tweets_df.loc[tweets_df['party'] == x,]
    f_df = tweets_df[['created_at', 'party', 'azure_sentiment', 'google_sentiment']]
    f_df = f_df.set_index('created_at')
    f_df = f_df.loc['2019-02-21':'2019-02-21']
    f_df = f_df.groupby(['party']) \
        .agg(f) \
        .rename(columns={#'party': 'count',
                         'google_sentiment': 'mean_daily_google_sentiment',
                         'azure_sentiment': 'mean_daily_azure_sentiment'})
    f_df.plot()

    import pandas as pad
    from numpy.random import randn


    # lista = list(np.unique(tweets_df[feature_color]))
    # list_of_series = [tweets_df[x], tweets_df[y], tweets_df[feature_color]]
    # col_names = [x, y, feature_color]
    # df = pd.DataFrame(list_of_series, columns=col_names)
    # df = pd.concat(list_of_series, axis=1)
    # fg = sns.FacetGrid(data=df, hue=feature_color, hue_order=lista, aspect=1.61)
    # fg.map(plt.scatter, x, y).add_legend()
    #
    # df = tweets_df.groupby([pd.Grouper(key='created_at', freq='M'),
    #                         pd.Grouper(key='party')]).agg(['count', 'mean'])
    #
    # a = pd.DataFrame(df.to_records(),
    #                  columns=df.index.names + list(df.columns))
    #
    # b = df.stack().reset_index()
    # b = df.stack(level=0).reset_index()
    #
    # sns.set()
    # fg = sns.pairplot(b, hue='party').add_legend()
    # fig = fg.fig
    # fig.set_size_inches(11, 11)
    #
    # a.boxplot(by='created_at')
    # a.set_xticklabels(rotation=30)
    #
    # fg = sns.factorplot(x='created_at', y='google_sentiment', hue='party',
    #                     col='Sex',
    #                     data=a, kind='bar')
    #
    # fg.set_xlabels('')
    # fg.set_xticklabels(rotation=30)
    #
    # g = sns.FacetGrid(df)
    # g = g.map_dataframe(dateplot, x, y)
    #
    #
c_df = tweets_df.groupby("party").count()\
    .rename(columns={'Id':'counts'})\
    .reset_index(drop=False).iloc[:,[0,1]]
c_dict = dict(zip(c_df.party,c_df.counts))
c_df = tweets_df.groupby("party")\
    .agg('mean')\
    .sort_values('google_sentiment', ascending=False)
c_df = c_df.loc[:,['google_sentiment']]
print(c_df)

c_df.plot.bar()