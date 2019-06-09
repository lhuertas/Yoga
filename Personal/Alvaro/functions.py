import re
from stop_words import get_stop_words
import emoji
import unidecode
import pandas as pd
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer

def stop_words():
    """Retrieve the stop words for vectorization -Feel free to modify this function
    """
    return get_stop_words('es')# + get_stop_words('ca') + get_stop_words('en')

def filter_mentions(text):
    """Utility function to remove the mentions of a tweet
    """
    return re.sub("@\S+", "", text)

def filter_hashtags(text):
    """Utility function to remove the hashtags of a tweet
    """
    return re.sub("#\S+", "", text)

def extract_emojis(str):
   return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

def number_words(tweet):
    word = re.findall('\w+',tweet)
    return len(word)

def number_mentions(tweet):
    mention = re.findall('@\w+',tweet)
    return len(mention)

def number_hashtags(tweet):
    hashtag = re.findall('#\w+',tweet)
    return len(hashtag)

def number_emoticons(tweet):
    tweet_transf = emoji.demojize(tweet)
    emoticon = re.findall(':\w+:',tweet_transf)
    return len(emoticon)

spanish_stemmer = nltk.stem.SnowballStemmer('spanish')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
      analyzer = super(TfidfVectorizer, self).build_analyzer()
      return lambda doc: (spanish_stemmer.stem(w) for w in analyzer(doc))

def clean_text(text):
    #lowercase:
    text_clean = text.lower()
    #remove accents:
    text_clean = unidecode.unidecode(text_clean)
    #remove_hashtags:
    text_clean = re.sub('@','',text_clean)
    #remove_mentions:
    text_clean = re.sub('#','',text_clean)
    #transf. emoticons_to_word:
    text_clean = emoji.demojize(text_clean).replace(':',' ').replace('_','')
    #remove puntuation, numbers:
    text_clean = re.sub('[^A-Za-z]+',' ',text_clean)
    return text_clean

def save_submission(prediction, fileName = 'sample_submission'):
    import datetime
    t = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    output = pd.DataFrame({'Party': prediction})
    output.index.name = 'Id'
    #output.to_csv(f'sample_submission{t}.csv')
    output.to_csv(f'{fileName}_{t}.csv')

