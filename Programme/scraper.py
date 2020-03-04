import pywinauto
import clipboard
import time
import io
import csv
import re
from os import listdir
from os.path import isfile, join
import json
import random
from textblob import TextBlob
import pandas as pd
import math
import statistics
from progress.bar import ChargingBar as Bar



def scroll_and_copy(delay = 1):
    """
    Scroll down in the browser, copy all and return the copied text

    """


#   copy all
    pywinauto.keyboard.send_keys('^a^c')

#   scroll down
    for i in range(5):
        pywinauto.mouse.click(coords = (100, 200))
        pywinauto.mouse.scroll(coords = (200, 400), wheel_dist = -5)
        time.sleep(delay)

#   return
    return(clipboard.paste())

def save_text(text, filename):
    """
    save raw text to a file with filename
    """
    f = io.open(filename, mode = "w", encoding = "utf-8")
    for line in text:
        f.write(line)
    f.close()

def get_filename(filetype, search_id = -1):
    """
        returns the file name as declared in _data_model.txt depending on type
        searchs the subfolder /data for files and increments search by 1.
    """

#   generate search id if not provided
    if search_id == -1:
        files = [f for f in listdir('data/') if isfile(join('data/', f))]
        files.sort()
        search_id = files[0][:files[0].find('_')]

#   generate name
    if filetype == 'meta':
        return('data/' + search_id + '_search.json')
    elif filetype == 'analysis':
        return('data/' + search_id + '_analysis.csv')
    elif filetype == 'raw':
        files = [f for f in listdir('data/') if isfile(join('data/', f))]
        files.sort()
        sr_ids = [file for file in files if file[:len(str(search_id))] == search_id]
        sr_ids.sort()
        print(sr_ids)
        search_run_id = sr_ids[0]

def get_tweets(iterations, filename = "raw_tweets.txt"):
    """
    Function for getting the tweets. Broswer should be opened and search should be already completed.
    Browser has to be 1 alt+tab away before starting.

    parameters:
        iterations:     number of scrolls that shall be done

    """

    # array for copies
    raw = []

    time.sleep(4)

    # change to browser
    #pywinauto.keyboard.send_keys('{VK_MENU down}' '{TAB}' '{TAB}' '{VK_MENU up}')

    # store result length for cancelling
    lengths = []

    # loop
    for i in range (iterations):
        result = scroll_and_copy(delay = 1)
        lengths.append(str(len(result)))
        raw.append(result)
#       wait
        time.sleep(random.randint(1, 3))
#       end if 3 times in a row no different lengths
        if (lengths[-1:] == lengths[-2:-1]) & (lengths[-3:-2] == lengths[-4:-3]):
            break

#    print info
    print('Scraping ended after ' + str(i) + ' loops.')

    # write to file
    save_text(raw, filename)

def find_first_tweet(lines):
    """
        takes lines as list and gets line of the first tweet
        identified by the term "Suchergebnisse"

        parameter:
            lines: list of lines

        return:
            number of line with first tweet

    """

    #return(lines.index('Suchergebnisse\n') + 1)            # not working anymore
    return(lines.index('Timeline durchsuchen\n') + 1)

def find_date_lines(lines):
    """
        returns list with indices of date lines
    """

    dates = []
    for i in range(len(lines)):
        if is_date(lines[i]):
            dates.append(i)
    return(dates)

def is_username(text):
    """
        returns true, if given string is a user name starting with @
    """

    if re.search('^\@.*$', text):
        return True
    else:
        return False

def is_date(text):
    """
        returns true, if given string is twitter-like date string
    """

    if re.search('\d{1,2}\.\s(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|Jan\.|Feb\.|Mrz\.|Apr\.|Mai|Jun\.|Jul\.|Aug\.|Sep\.|Okt\.|Nov\.|Dez\.)\s\d{4}', text):
    #if re.search('\s\d{1,2}\.\s(Jan\.|Feb\.|Mrz\.|Apr\.|Mai|Jun\.|Jul\.|Aug\.|Sep\.|Okt\.|Nov\.|Dez\.)\s\d{4}', text):
        return True
    else:
        return False

def is_reactions(text):
    """
        returns true if line is reactions line starting with 'reactions:'
    """

    if text[0:10] == 'reactions:':
        return True
    else:
        return False

def get_reactions_row(lines, index):
    """
        returns index of the next line with reactions:x/x/x in lines starting from index
    """

    for i in range(len(lines)):
        if lines[i][0:10] == 'reactions:':
            return(i)

def get_date_row(lines, index):
    """
        find row with valid german date format
    """

    for i in range(len(lines)):
        if is_date(lines[i]):
            return(i)

def proc_raw_tweets(filename):
    """
        function to process the raw tweets and convert the text to a csv file
    """

#   set up list of lines
    lines = []

# remove some lines
    remove_words = [' \n', '\u200f\n', '\n', 'Mehr\n', 'Verifizierter Account\n', 'Diesen Thread anzeigen\n', '.\n', '·\n']
    with open(filename, 'r', encoding = 'utf-8') as file:
        for line in file:
            if not line in remove_words:
                lines.append(line)



# remove lines before Tweets
    lines = lines[find_first_tweet(lines):]


#   set up progress bar
    bar = Bar('Processing raw tweet text', max = len(lines))

# remove line break chars at end
    lines = [re.sub('$\n', '', line) for line in lines]


# break counters apart
    for index, line in enumerate(lines):
        bar.next()
        if re.search('\d Antworten\d Retweets\d Gefällt mir', line):
            lines[index] = 'reactions:' + line.replace(' Antworten', '/').replace(' Retweets', '/').replace(' Gefällt mir', '')
        if re.search('Antworten \d? Retweeten \d? Gefällt mir\s?\d?', line):
            lines.pop(index)
            bar.next()
#        if re.search('Seite bereit:\s.+', line):
#            print('deleting rows ' + index-1 + ' to ' + find_first_tweet(lines[index:]))
#            del lines[index-1:find_first_tweet(lines[index:])]

# merge multilines
    """
    # not working

    for index, line in enumerate(lines):
        if is_date(line) and not is_reactions(lines[index+2]):
            print('index:' + str(index))
            print('reactions at:' + str(get_reactions_row(lines, index)))
            lines[index+1] = lines[index+1] + ' ' + lines[index + 2]
            lines[index+2] = ''
            # empty line and delete empty lines afterwars - indexing errors otherwise
            #lines.pop(index+2)
    for i in range(len(lines)-1, 0, -1):
        if lines[i] == '':
            lines.pop(i)
    """



# temp: save
    with open(filename.replace('raw', 'proc'), 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + '\n')


# write to csv
    """
    with open(filename.replace('raw', 'data').replace('txt', 'csv'), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['screen_name', 'user_name', 'date', 'text', 'answers_count', 'retweets_count', 'likes_count'])
        for i in range(0, len(lines)-6, 5):
#            writer.writerow([lines[i], lines[i+1], lines[i+2].strip(), lines[i+3], lines[i+4].split('/')[0], lines[i+4].split('/')[1], lines[i+4].split('/')[2]])
            writer.writerow([lines[i], lines[i+1], lines[i+2].strip(), lines[i+3], lines[i+4], '0', '0'])
    """
# return
    return()

def date_formatter(date, silent = True):
    """
        converts a date string from twitter ("30. Jan. 2018") to an ISO-formatted date ("2018-01-30")
    """
    try:
        day = str(date[:date.index('.')].strip()).zfill(2)
        month = date[date.index('.')+2:date.index(' ', 5)-1].strip()
        month_dict = {
        'Jan': '01',
        'Feb': '02',
        'M\u00e4r': '03',
        'Apr': '04',
        'Ma': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Okt': '10',
        'Nov': '11',
        'Dez': '12'
        }
        month = month_dict[month]
        year = str(date[-5:-1])

        return(year + '-' + month + '-' + day)

    except:
        if not silent:
            print('Error in date_formatter: input: ' + date)
        return(date)

def extract_tweets(filename):
    """
        function to extract tweets out of the proc file.
        saves tweets as json
    """

#   load file
    file = open(filename, 'r', encoding = 'utf-8')
    lines = [line for line in file]
    file.close()

#   set up progress bar
    bar = Bar('Extracting tweets', max = len(lines))

#   extract tweets
    raw_tweets = []
    datelines = find_date_lines(lines)
    for i in range(len(datelines)-1):       # looses one tweet this way
        raw_tweets.append(lines[(datelines[i]-2):(datelines[i+1]-2)])

#   create tweets object
    tweets = {'tweets': []}

#   set up progress bar
    bar = Bar('Creating tweet objects', max = len(raw_tweets))

#   read out information
    for raw_tweet in raw_tweets:
        bar.next(n = 1)
        tweet = {
        'screen_name': raw_tweet[0].replace('\n', ''),
        'user_name': raw_tweet[1].replace('\n', ''),
        'date': date_formatter(raw_tweet[2]),
        'text': ' '.join(raw_tweet[3:len(raw_tweet)]).replace('\n', ''),
        'reactions': ''#raw_tweet[len(raw_tweet)-1].replace('\n', '')
        }
        tweets['tweets'].append(tweet)

#   save tweets json
    with open(filename.replace('proc', 'data').replace('.txt', '.json'), 'w', encoding = 'utf-8') as file:
        json.dump(tweets, file)
