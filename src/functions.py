from langdetect import detect


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


def get_language(string):
    my_string = filter_hashtags(string)
    my_string = filter_mentions(string)
    my_vector = string.split()
    my_vector = ",".join(my_vector)

    return detect(my_vector)