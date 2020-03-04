"""
    all functions for writing and reading from sqlite database
"""

DB_FILE = 'tweets.db'

def create_connection(DB_FILE):
    """
        create a connection to a sqlite database specified by db_file

        parameters:
            db_file:                filename of sqlite database

    """

    import sqlite3

    con = None
    try:
        con = sqlite3.connect(DB_FILE)
    except Error as e:
        print(e)

    return(con)

def create_cursor(con):
    """
        create cursor from connection

        parameters:
            con:                    connection as returned from create_connection

    """

    import sqlite3

    c = con.cursor()
    return(c)

def create_table(tablename):
    """
        function to create predefined sqlite tables
    """

    con = create_connection(DB_FILE)
    c = create_cursor(con)

    if tablename == 'tweets':
        sql_command = '''
            CREATE TABLE tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            screen_name TEXT,
            user_name TEXT,
            date DATE,
            text TEXT,
            text_len INT,
            flag_faulty BOOLEAN,
            flag_sentiment BOOLEAN,
            flag_answer BOOLEAN,
            search_id INTEGER,
            tb_polarity REAL,
            tb_subjectivity REAL,
            tb_sentiment REAL,
            vader_neg REAL,
            vader_neu REAL,
            vader_pos REAL,
            vader_compound REAL,
            vader_sentiment REAL,
            total_sentiment REAL
            );
        '''
    elif tablename == 'searches':
        sql_command = '''
            CREATE table searches (
            search_id INTEGER,
            type TEXT,
            query TEXT,
            symbol TEXT
            );
        '''

    elif tablename == 'stocks':
        sql_command = '''
            CREATE table stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date DATE,
            source TEXT,
            high REAL,
            low REAL,
            open REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            UNIQUE(symbol, date) ON CONFLICT ROLLBACK
            );
        '''


    c.execute(sql_command)

def create_database():
    """
        wrapper to create the whole database
    """

    create_table('tweets')
    create_table('searches')

def add_tweet(tweet):
    """
        add a single tweet to database
    """

    con = create_connection(DB_FILE)
    c = create_cursor(con)

    values = (tweet['screen_name'], tweet['user_name'], tweet['date'], tweet['text'])
    c.execute('INSERT INTO tweets (screen_name, user_name, date, text) VALUES (?,?,?,?)', values)
    entry_id = c.lastrowid

    con.commit()
    con.close()

    return(entry_id)

def add_tweets(data, **kwargs):
    """
        add all tweets from data to database
    """

#   handle kwargs
    search_id = 'NULL' if not 'search_id' in kwargs else str(kwargs['search_id'])
    silent = False if not 'silent' in kwargs else kwargs['silent']

#   establish connection
    con = create_connection(DB_FILE)
    c = create_cursor(con)

#   insert new tweets
    for tweet in data['tweets']:
        values = (tweet['screen_name'], tweet['user_name'], tweet['date'], tweet['text'], search_id)
        c.execute('INSERT INTO tweets (screen_name, user_name, date, text, search_id) VALUES (?,?,?,?,?)', values)
        entry_id = c.lastrowid
    if not silent:
        print(str(len(data['tweets'])) + ' tweets were added to ' + DB_FILE)


#   commit and close
    con.commit()
    con.close()

def run_stored_procedure(procedure_name):
    """
        run a stored procedure on the database
    """

    con = create_connection(DB_FILE)
    c = create_cursor(con)

#   add text lengths
    if procedure_name == 'SP_TEXT_LENGTHS':
        sql = """
            UPDATE tweets
            SET text_len = length(text)
            WHERE text_len IS NULL
        """

#   flag faulty
    elif procedure_name == 'SP_FLAG_FAULTY':
        sql = """
            UPDATE tweets
            SET flag_faulty = 1
            WHERE text_len > 280
        """

#   flag answers
    elif procedure_name == 'SP_FLAG_ANSWERS':
        sql = """
            UPDATE tweets
            SET flag_answer = CASE WHEN SUBSTR(text, 1, 10) = 'Antwort an' THEN 1 ELSE 0 END
        """

#   remove duplicates
    elif procedure_name == 'SP_REMOVE_DUPLICATES':
        sql = """
            DELETE FROM tweets
                WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM tweets
                GROUP BY text
            );
            """

#   binary classifier
    elif procedure_name == 'SP_BINARY_CLASSIFIER':
        sql = """
            UPDATE tweets
                SET tb_sentiment = CASE WHEN tb_polarity >= 0.3 THEN 1 WHEN tb_polarity <= -0.3 THEN -1 ELSE 0 END,
                    vader_sentiment = CASE WHEN vader_compound >= 0.3 THEN 1 WHEN vader_compound <= -0.3 THEN -1 ELSE 0 END,
                    total_sentiment = CASE WHEN ((tb_polarity + vader_compound) / 2) >= 0.3 THEN 1 WHEN ((tb_polarity + vader_compound) / 2) <= -0.3 THEN -1 ELSE 0 END;
        """


    c.execute(sql)

    con.commit()
    con.close()

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

def add_stock_data(symbol, start, end, **kwargs):
    """
        function to download stock data and write them to the database

        parameters:
            symbol                  stock symbol
            start                   start date as datetime.date
            end                     end date as datetime.date

        kwargs:
            source                  DEFAULT = 'YAHOO'
                                    source of the stock data
            silent                  DEFAULT = False
                                    print information

        example call:
            add_stock_data('TSLA', datetime.date(2019,1,1), datetime.date(2019,12,31), silent = False, source = 'yahoo')

    """

#   import
    from charts import get_stock_data

#   kwargs
    source = 'yahoo' if not 'source' in kwargs else kwargs['source']
    silent = False if not 'silent' in kwargs else kwargs['silent']

#   download data
    df = get_stock_data(symbol, start, end, value = 'all')

#   establish connection
    con = create_connection(DB_FILE)
    c = create_cursor(con)

#   write to database
    for index, row in df.iterrows():
        values = (symbol, row.date.strftime('%Y-%m-%d'), source, row.High, row.Low, row.Open, row.Close, row['Adj Close'], row.Volume)
#        c.execute('INSERT INTO stocks (symbol, date, source, high, low, open, close, adj_close, volume) VALUES (?,?,?,?,?,?,?,?,?)', values)
        c.execute('''
            INSERT OR REPLACE INTO stocks (symbol, date, source, high, low, open, close, adj_close, volume)
            VALUES (?,?,?,?,?,?,?,?,?)
        ''', values)
        entry_id = c.lastrowid
    if not silent:
        print(str(len(df)) + ' stock data were added to ' + DB_FILE)

#   commit and close
    con.commit()
    con.close()

def get_stock_timeseries(symbol, start, end, **kwargs):
    """
        get stock data time series from start to end to a given symbol

        parameter:
            symbol                          stock symbol
            start                           start date as string
            end                             end date as string

        kwargs:
            value                           DEFAULT='adj_close'
                                            value to be retrieved from database

    """

#   import
    import pandas as pd

#   kwargs
    value = 'adj_close' if 'value' not in kwargs else kwargs['adj_close']

#   establish connection
    con = create_connection(DB_FILE)
    c = create_cursor(con)

#   read data to dataframe
    df = pd.read_sql_query('''
            SELECT date,
                   ''' + value + '''
            FROM stocks AS S
            WHERE S.symbol = ? AND
                  S.date >= ? AND
                  S.date <= ?
    ''', con, params=(symbol, start, end))

#   return
    return(df)

def get_full_timeseries(symbol, start, end, **kwargs):
    """
        get stock data time series from start to end to a given symbol

        parameter:
            symbol                          stock symbol
            start                           start date as string
            end                             end date as string

        kwargs:
            value                           DEFAULT='adj_close'
                                                value to be retrieved from database



    """

#   import
    import pandas as pd

#   kwargs
    value = 'adj_close' if 'value' not in kwargs else kwargs['adj_close']

#   establish connection
    con = create_connection(DB_FILE)
    c = create_cursor(con)

#   read data to dataframe
    df = pd.read_sql_query('''
            SELECT T.date,
                   AVG(T.vader_sentiment) AS AVG,
                   COUNT(T.vader_sentiment) AS COUNT,
                   S.adj_close AS STOCK
              FROM tweets AS T
                   LEFT JOIN
                   searches AS SE ON SE.search_id = T.search_id
                   LEFT JOIN
                   stocks AS S ON SE.symbol = S.symbol AND
                                  S.date = T.date
             WHERE (T.flag_faulty IS NULL OR
                    T.flag_faulty = 0) AND
                   T.date >= ? AND
                   T.date <= ? AND
                   SE.symbol = ?
             GROUP BY T.date,
                      SE.symbol
    ''', con, params = (start, end, symbol))

#   return
    return(df)
