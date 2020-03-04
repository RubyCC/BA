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

