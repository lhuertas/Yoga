
if __name__ == '__main__':

    ## Celda 1 Notebook ##

    import os
    import sys

    ROOT_PATH = "C:/workspace/my_repos/Capstone DS/"
    PERSONAL_PATH = "Personal/Alvaro"
    sys.path.append(os.path.join(ROOT_PATH, 'src'))
    os.chdir(os.path.join(ROOT_PATH, PERSONAL_PATH))
    sys.path.append(os.path.join(ROOT_PATH, PERSONAL_PATH))
    TRAIN_FPATH = os.path.join(ROOT_PATH, "Data/train_traducido.csv")

    ## Celda 2 Notebook ##

    # All imports goes here
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns;
    from parties_classification import remove_tweets_with_non_identified_language
    from parties_classification import add_text_clean_col_to_df
    from parties_classification import get_sentiment_features_df

    ## Celda 3 Notebook ##

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
    print(tweets_df.head())

    ## Celda 4 Notebook ##

    x = 'created_at'
    y = 'google_sentiment'
    feature_color = 'party'
    lista = list(np.unique(train_df_tr[feature_color]))
    list_of_series = [train_df_tr[x], train_df_tr[y], train_df_tr[feature_color]]
    col_names = [x, y, feature_color]
    df = pd.DataFrame(list_of_series, columns=col_names)
    df = pd.concat(list_of_series, axis=1)
    df.created_at = pd.to_datetime(df.created_at)
    df = df.groupby([pd.Grouper(key='created_at', freq='M'),
                     pd.Grouper(key='party')]).agg(['count', 'mean'])
    b = df.stack(level=0).reset_index()

    ## Celda 5 Notebook ##

    fg = sns.catplot(x="party", y="mean", kind="boxen",
                     data=b)
    fg.set_xticklabels(rotation=30)
    fig = fg.fig
    fig.set_size_inches(14, 10)

    ## Celda 5b Notebook ##

    ## AÑADIR MAPA DE CALOR CORRELACIONES SENTIMIENTOS
    ## ENTRE POLITICOS

    ## Celda 6 Noteboook ##

    ## AÑADIR GRAFICA DE OLIVER CON LOS "COUNTS" ##

    ## Celda 7 Noteboook ##

    f = {'google_sentiment': 'mean',
         'azure_sentiment': 'mean'
         }
    f_df = tweets_df[['created_at', 'party', 'azure_sentiment', 'google_sentiment']]
    f_df = f_df.set_index('created_at')
    f_df = f_df.loc['2018-10-01':'2018-12-30']
    f_df = f_df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day]) \
        .agg(f) \
        .rename(columns={  # 'party': 'count',
        'google_sentiment': 'mean_daily_google_sentiment',
        'azure_sentiment': 'mean_daily_azure_sentiment'})
    f_df.plot()

    ## Celda 8 Notebook ##

    ## Añadir Pequeña grafica con Distribucion de Sentimientos
    ## de Amazon y comentar que su metodologia difiere de las
    ## otras dos nubes ##

  #Grafico de correlacion de sentimiento entre partidos. Semanal de los ultimos 6 meses.
  #Corregir fechas de graficos y titulos, etc.
  #Añadir etiqueta numerica al grafico de Boxplots y si se puede la media.



