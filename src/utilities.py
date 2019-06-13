import re
from stop_words import get_stop_words
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from vecstack import stacking
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras import layers, regularizers
from keras.layers import Dropout, Dense
from keras.wrappers.scikit_learn import KerasClassifier
import emoji
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
import unidecode
import pandas as pd


def stop_words():
    return get_stop_words('es')


def filter_mentions(text):
    return re.sub("@\S+", "", text)


def filter_hashtags(text):
    return re.sub("#\S+", "", text)


def extract_emojis(str):
    return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)


def number_words(tweet):
    word = re.findall('\w+', tweet)
    return len(word)


def number_mentions(tweet):
    mention = re.findall('@\w+', tweet)
    return len(mention)


def number_hashtags(tweet):
    hashtag = re.findall('#\w+', tweet)
    return len(hashtag)


def number_emoticons(tweet):
    tweet_transf = emoji.demojize(tweet)
    emoticon = re.findall(':\w+:', tweet_transf)
    return len(emoticon)


spanish_stemmer = nltk.stem.SnowballStemmer('spanish')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (spanish_stemmer.stem(w) for w in analyzer(doc))


def clean_text(text):
    # lowercase:
    text_clean = text.lower()
    # remove_hashtags:
    text_clean = re.sub('@', '', text_clean)
    # remove_mentions:
    text_clean = re.sub('#', '', text_clean)
    # transf. emoticons_to_word:
    text_clean = emoji.demojize(text_clean).replace(':', ' ').replace('_', '')
    # remove accents:
    text_clean = unidecode.unidecode(text_clean)
    # remove numbers:
    text_clean = re.sub('[^A-za-z]+', ' ', text_clean)
    #
    text_clean = re.sub('quot', ' ', text_clean)
    return text_clean


def Sequential_model(input_dim, n_classes):
    def sm():
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Dropout(0.2))
        model.add(layers.Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    return sm


def model_sel(x_train, y_train, x_test, y_test):
    clf1 = LogisticRegression(random_state=0)
    clf2 = BernoulliNB()
    clf3 = SVC(C=1, kernel='linear')
    clf4 = RandomForestClassifier(n_estimators=800, min_samples_split=5, min_samples_leaf=1, max_features='sqrt',
                                  max_depth=100, bootstrap=False)
    clf5 = xgb.XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.3,
                             n_estimators=100, max_depth=4)
    clf6 = neighbors.KNeighborsClassifier(n_neighbors=9, p=1)
    clf7 = GradientBoostingClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=1, subsample=1,
                                      max_features='sqrt', random_state=10,
                                      learning_rate=0.15, n_estimators=300)
    clf8 = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                              learning_rate=1, n_estimators=200, random_state=1)

    input_dim = x_train.shape[1]
    n_classes = len(np.unique(y_train))
    clf9 = KerasClassifier(Sequential_model(input_dim, n_classes), epochs=8)

    clf = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9]

    for model in clf:
        print(f"Classifier: {model}")
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        print("score: {}".format(score))
        print(" ")

    print("Ensembles.................................")
    print(" ")

    MaxVoting_esemble = VotingClassifier(
        estimators=[('lr', clf1), ('bb', clf2), ('svc', clf3),
                    ('rf', clf4), ('xg', clf5), ('knn', clf6),
                    ('grb', clf7),('ab', clf8), ('nn', clf9)],
        voting='hard')
    MaxVoting_esemble.fit(x_train, y_train)
    score = MaxVoting_esemble.score(x_test, y_test)
    print("MaxVoting: {}".format(score))
    print(" ")

    # staking
    S_train, S_test = stacking(clf,
                               x_train, y_train, x_test,
                               regression=False,
                               mode='oof_pred_bag',
                               needs_proba=False,
                               save_dir=None,
                               metric=accuracy_score,
                               n_folds=4,
                               stratified=True,
                               shuffle=True,
                               random_state=0,
                               verbose=2)

    stacking_ensemble = SVC(C=1, kernel='linear')
    stacking_ensemble = stacking_ensemble.fit(S_train, y_train)
    stacking_score = stacking_ensemble.score(S_test, y_test)
    print("Stacking: {}".format(stacking_score))

    return clf, MaxVoting_esemble, stacking_ensemble


def save_submission(prediction, fileName='sample_submission'):
    import datetime
    t = datetime.datetime.now().strftime("%Y%m%d-%H_%M_")
    output = pd.DataFrame({'Party': prediction})
    output.index.name = 'Id'
    output.to_csv(f'{fileName}_{t}.csv')

