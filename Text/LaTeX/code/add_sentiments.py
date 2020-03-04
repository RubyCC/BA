def add_sentiments(**kwargs):
    """
        fetches datasets without valid sentiment and runs sentiment analyzer on them

        parameters:

        kwargs:
            max                     max amount of datasets picked from database
                                        DEFAULT: no max, picks all
            silent                  whether or not to display status of processing
                                        DEFAULT: false, displaying progress
    """

#   kwargs
    max = None if not 'max' in kwargs else kwargs['max']
    silent = False if not 'silent' in kwargs else kwargs['silent']

#   imports
    from sentiments import get_sentiment                # for getting sentiment values
    from progress.bar import ChargingBar as Bar         # progress bar

#   get unrated tweets
    sql = """
        SELECT id,
               text,
               date,
               flag_sentiment
          FROM tweets
         WHERE flag_sentiment IS NULL OR
               flag_sentiment = 0
    """
    if not max is None:
        sql += ('LIMIT ' + str(max))
    sql += ';'

#   connect db
    con = create_connection(DB_FILE)
    c = create_cursor(con)

#   get data
    c.execute(sql)
    data = c.fetchall()
    if not silent:
        print('Fetched ' + str(len(data)) + ' tweets from ' + DB_FILE + '.')

#   analyze tweets
    bar = Bar('Running sentiment analyzer', max = len(data))
    sent_values = {
        'tb_polarity': '',
        'tb_subjectivity': '',
        'vader_neg': '',
        'vader_neu': '',
        'vader_pos': '',
        'vader_compound': '',
    }
    for tweet in data:
#       TextBlob
        sentiment = get_sentiment(tweet[1], engine = 'textblob')
        sent_values['tb_polarity'] = sentiment[0]
        sent_values['tb_subjectivity'] = sentiment[1]
#       VADER
        sentiment = get_sentiment(tweet[1], engine = "vader")
        sent_values['vader_neg'] = sentiment['neg']
        sent_values['vader_neu'] = sentiment['neu']
        sent_values['vader_pos'] = sentiment['pos']
        sent_values['vader_compound'] = sentiment['compound']
#       update db
        sql = '''
            UPDATE tweets
            SET tb_polarity = ?,
            tb_subjectivity = ?,
            vader_neg = ?,
            vader_neu = ?,
            vader_pos = ?,
            vader_compound = ?,
            flag_sentiment = 1
            WHERE id IS ?
        '''
        c.execute(sql, (sent_values['tb_polarity'],
                        sent_values['tb_subjectivity'],
                        sent_values['vader_neg'],
                        sent_values['vader_neu'],
                        sent_values['vader_pos'],
                        sent_values['vader_compound'],
                        tweet[0]))
#       progress
        bar.next()

#   commit changes
    con.commit()
    con.close()

#   end
    bar.finish()
