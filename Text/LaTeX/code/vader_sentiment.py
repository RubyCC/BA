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
