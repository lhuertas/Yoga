from utilities import *
import os
import sys
from parties_classification import *

if __name__ == '__main__':

    ROOT_PATH = "C:/workspace/Repositorios/Politicians/"
    PERSONAL_PATH = "Personal/Alvaro"

    os.chdir(os.path.join(ROOT_PATH, PERSONAL_PATH))
    sys.path.append(os.path.join(ROOT_PATH, PERSONAL_PATH))
    TWEETS_FPATH = os.path.join(ROOT_PATH, "Data/train_traducido.csv")

    tweets_df = pd.read_csv(
        TWEETS_FPATH, delimiter=';')
    tweets_df = add_text_clean_col_to_df(tweets_df)
    keep_cols = ['Id', 'party']
    tweets_df = tweets_df.loc[:,keep_cols]
    sent_df = get_sentiment_features_df(
        ROOT_PATH, tweets_df, str='train')
    tweets_df = tweets_df.merge(sent_df,
                                on="Id",
                                how='left')

    c_df = tweets_df.groupby("party").count()\
        .rename(columns={'Id':'counts'})\
        .reset_index(drop=False).iloc[:,[0,1]]
    c_dict = dict(
        zip(c_df.party,
            c_df.counts
            )
    )

    c_df = tweets_df.groupby("party").agg('mean')