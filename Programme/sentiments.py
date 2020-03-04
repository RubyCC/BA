#! c:\Python36\python.exe

"""
    New versions of sentiment analyzing functions for tweets
"""

def get_sentiment(text, **kwargs):
    """
        function to get a sentiment. returns only polarity values mapped to [-1, 1]

        parameter:
            text:               Text or tweet object to be analyzed

        kwargs:
            engine:             Analysis engine to be used
                                    'vader' | 'textblob'
                                    DEFAULT: 'vader'
    """

#   kwargs
    engine = 'vader' if 'engine' not in kwargs else kwargs['engine']

#   calls
    if engine == 'vader':
        return(vader_sentiment(text, ret = 'dict'))
    if engine == 'textblob':
        return(textblob_sentiment(text, ret = 'tuple'))

def vader_sentiment(text, silent = True, **kwargs):
    """
        VADER Sentiment Analyzer

        kwargs:
            ret:                defines the return value
                                    'dict' returns an array [neg, neu, pos, compound]
                                    DEFAULT returns only the compound value

    """

#   kwargs
    ret = 'dict' if 'ret' not in kwargs else kwargs['ret']
    silent = True if 'silent' not in kwargs else kwargs['silent']

#   packages
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#   setup
    analyzer = SentimentIntensityAnalyzer()

#   return
    if ret == 'dict':
        return(analyzer.polarity_scores(str(text)))
    else:
        return(analyzer.polarity_scores(str(text))['compound'])

def textblob_sentiment(text, silent = True, **kwargs):
    """
        TextBlob Sentiment Analyzer

        kwargs:
            ret:                    defines what to return
                                        'tuple' returns a tuple (polarity, subjectivity)
                                        DEFAULT returns just the polarity as float

    """

#   packages
    from textblob import TextBlob

#   return
    if 'ret' in kwargs:
        if kwargs['ret'] == 'tuple':
            blob = TextBlob(text)
            return(tuple([blob.sentiment.polarity, blob.sentiment.subjectivity]))
    else:
        return TextBlob(text).sentiment.polarity
