"""
    functions needed to create figures for the final work
"""



def tesla_2018_chart():
    """
        create chart for chapter 1: introduction
        example of tesla's stock vs. elon musks tweets
    """

    import pandas as pd
    import pandas_datareader.data as web
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    import datetime
    import matplotlib.pyplot as plt



    start = datetime.date(2018, 3, 1)
    end = datetime.date(2018, 8, 30)


    df = web.DataReader('TSLA', 'yahoo', start, end)
    #df['date'] = df_stock.index

#   set up figure
#    w, h, d = 6, 2, 300
    #fig = plt.figure(figsize=(w, h), dpi = d)



#   import images
    im1 = plt.imread('images/elonmusk_1.png')
    im2 = plt.imread('images/elonmusk_2.png')

#   connector lines
    x1 = [datetime.date(2018, 4, 1), datetime.date(2018, 4, 1)]
    y1 = [0, df.at['2018-04-02', 'Adj Close']]

    x2 = [datetime.date(2018, 8, 7), datetime.date(2018, 8, 7)]
    y2 = [0, df.at['2018-08-07', 'Adj Close']]

#    fig, (ax1, ax2) = plt.subplots(1, 1, )

    plt.scatter(x1, y1, marker = 'o', color = 'r', linestyle = '-')
    plt.scatter(x2, y2, marker = 'o', color = 'r', linestyle = '-')
    plt.plot(df.index.values, 'Adj Close', data = df, marker = '', linestyle = '-', label = 'TSLA', color = '#1DA1F2')
    plt.ylim(100, max(df['Adj Close']) * 1.02)

    ax1 = plt.axes([0.15, 0.15, 0.3, 0.3], frameon = True)
    ax1.imshow(im1, origin = 'upper')
    ax1.axis('off')

    ax2 = plt.axes([0.55, 0.15, 0.3, 0.3], frameon = True)
    ax2.imshow(im2, origin = 'upper')
    ax2.axis('off')

    plt.show()


#   return
    return(df)

def tesla_2018_volume_chart():
    """
        create chart for chapter 1: introduction
        example of tesla's stock vs. elon musks tweets
    """

    import pandas as pd
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    import datetime
    import matplotlib.pyplot as plt


#   import images
    im1 = plt.imread('images/elonmusk_1.png')
    im2 = plt.imread('images/elonmusk_2.png')

#   get data
    df = pd.read_csv('images/tesla_volumes.csv')

#   connector lines
    x1 = [datetime.date(2018, 4, 1), datetime.date(2018, 4, 1)]
    y1 = df[df.date == '2018-04-02'].volume

    x2 = [datetime.date(2018, 8, 7), datetime.date(2018, 8, 7)]
    y2 = df[df.date == '2018-08-07'].volume

#    fig, (ax1, ax2) = plt.subplots(1, 1, )

#    plt.scatter(x1, y1, marker = 'o', color = 'r', linestyle = '-')
#    plt.scatter(x2, y2, marker = 'o', color = 'r', linestyle = '-')
    plt.plot('date', 'volume', data = df, marker = '', linestyle = '-', label = 'Tweet volume (#tesla, @elonmusk)', color = '#1DA1F2')
    plt.ylim(100, max(df['volume']) * 1.02)

    ax1 = plt.axes([0.15, 0.15, 0.3, 0.3], frameon = True)
    ax1.imshow(im1, origin = 'upper')
    ax1.axis('off')

    ax2 = plt.axes([0.55, 0.15, 0.3, 0.3], frameon = True)
    ax2.imshow(im2, origin = 'upper')
    ax2.axis('off')

    plt.show()


#   return
    return(df)

def get_pattern_synset_entry(word):
    """
        get an entry for a word of the pattern synset

        parameters:
            word:               word to look up
    """
###
###
###
### NOT WORKING
###
###


#   packages
    from pattern.en import sentiment
    import json

#   create dict
    dictstring = str(sentiment.synset)
    dictstring = dictstring[34:-1]
    dictstring = dictstring.replace("'", "\"")
    dictstring = dictstring.replace("None", '"' + 'None' + '"')
    dict = json.loads(dictstring)

    return(dict)

def get_dataset_statistics(name, start, end, **kwargs):
    """
        function to visualize dataset and calculate some statistics

        parameters:
            name                        dataset name
            start                       start date as string (YYYY-MM-DD)
            end                         end date as string (YYYY-MM-DD)


        kwargs:
            statistics                  calculate and print out statistics
                                            'full' | 'None'
                                            DEFAULT: 'None'

            plot                        display a plot of the time series
                                            'full' | 'None'
                                            DEFAULT: 'None'

            fill                        fill NA-values in dataframe, else dropping rows with NA
                                            True | False
                                            DEFAULT: True

    """

#   handle kwargs
    statistics = 'None' if 'statistics' not in kwargs else kwargs['statistics']
    plot = 'None' if 'plot' not in kwargs else kwargs['plot']
    fill = True if 'fill' not in kwargs else kwargs['fill']

#   packages
    from database import get_full_timeseries
    import matplotlib.pyplot as plt

#   load data
    df = get_full_timeseries(name, start = start, end = end)
    if fill:
        df['STOCK'].fillna(method = 'bfill', inplace = True)
    else:
        df.dropna(axis = 0, inplace = True)

#   empty message
    if statistics == 'None' and plot == 'None':
        print('Neither the plot nor the statistics option has been chosen. No output was produced.')
        return()

#   plot
    if plot == 'full':
        fig, axs = plt.subplots(3, sharex = True, sharey = False)
        axs[0].plot('date', 'AVG', data = df, color = '#00709c', label = 'sentiment')
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(6))
        axs[0].legend()

        axs[1].plot('date', 'COUNT', data = df, color = '#00709c', label = 'tweet count')
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(6))
        axs[1].legend()

        axs[2].plot('date', 'STOCK', data = df, color = '#00709c', label = 'stock')
        axs[2].xaxis.set_major_locator(plt.MaxNLocator(6))
        axs[2].legend()
        plt.show()

#   do the stats
    if statistics == 'full':
        from statistics import mean, stdev, median

        print('Dataset: {}'.format(name))
        print('  {} rows'.format(str(len(df))))
        print('  {} - {}\n'.format(start, end))
        print('  value AVG:')
        print('    mean    : {}'.format(str(mean(df['AVG']))))
        print('    std     : {}'.format(str(stdev(df['AVG']))))
        print('    median  : {}\n'.format(str(median(df['AVG']))))

        print('  value COUNT:')
        print('    sum     : {}    (total number of tweets)'.format(str(sum(df['COUNT']))))
        print('    mean    : {}'.format(str(mean(df['COUNT']))))
        print('    std     : {}'.format(str(stdev(df['COUNT']))))
        print('    median  : {}\n'.format(str(median(df['COUNT']))))

def ar_var_comparison(**kwargs):
    """
        analyze and plot ar/var-comparisons
        kwargs:
            stats                   level of detail for stat values
                                        'full' | 'short' | None
                                        DEFAULT: 'full'

            plot                    list of plots to create
                                        'scatter' | 'scatter_avg' | 'scatter_avg_mape'

            savefig                 save a created figure/plot under a preset name
                                        True | False
                                        DEFAULT: False

            showfig                 show figure/plot
                                        True | False
                                        DEFAULT: False

    """
#   kwargs
    stats = 'full' if 'stats' not in kwargs else kwargs['stats']
    plot = ['scatter'] if 'plot' not in kwargs else kwargs['plot']
    savefig = False if 'savefig' not in kwargs else kwargs['savefig']
    showfig = False if 'showfig' not in kwargs else kwargs['showfig']

#   packages
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from statistics import mean

#   load data
    df = pd.read_csv('comparison_ar_var.results',
                     na_values = '#')
    for col in ['tq_var']:
        df[col] = df[col].astype(float)

#   calculate measures
    means_ar = [mean(df[df['len'] == len]['tq_ar']) for len in df['len'].unique()]
    means_var = [mean(df[df['len'] == len]['tq_var']) for len in df['len'].unique()]
    means_ar_mape = [mean(df[df['len'] == len]['mape_ar']) for len in df['len'].unique()]
    means_var_mape = [mean(df[df['len'] == len]['mape_var']) for len in df['len'].unique()]

#   print statistics
    pivot = pd.pivot_table(df, index = ['len'], values = ['tq_var'], columns = ['dataset'], aggfunc = [np.mean])
    if not stats is None:
        if stats == 'full':
            print(pivot)
            print('Comparing means of values')
            for i in range(len(means_ar)):
                print('AR: {0:1.4f}   VAR: {1:1.4f}'.format(means_ar[i], means_var[i]))
#   plotting
#   scatter
    if 'scatter' in plot:
        fig, ax = plt.subplots()
        ax.scatter(x = df['len'] + 0.3, y = 'tq_var', data = df, marker = 'o', color = '#007A9B', alpha = 0.5, label = 'tq_var')
        ax.scatter(x = 'len', y = 'tq_ar', data = df, color = '#6D8300', marker = 'o', alpha = 0.5, label = 'tq_ar')
        ax.set_xlim(2,13)
        ax.set_xticks(df['len'].unique())
        fig.suptitle('Precision')
        ax.legend()
        if savefig:
            fig.savefig('scatter.png')
        if showfig:
            plt.show()

#   scatter with average
    if 'scatter_avg' in plot:
        fig, ax = plt.subplots()
        ax.scatter(x = df['len'] + 0.3, y = 'tq_var', data = df, marker = 'o', color = '#007A9B', alpha = 0.5, label = 'tq_var')
        ax.scatter(x = 'len', y = 'tq_ar', data = df, color = '#6D8300', marker = 'o', alpha = 0.5, label = 'tq_ar')
        ax.plot(df['len'].unique(), means_ar, '-', color = '#6D8300', label = 'average tq_ar')
        ax.plot(df['len'].unique(), means_var, '-', color = '#007A9B', label = 'average tq_var')
        ax.legend()
        ax.set_xlim(2,13)
        ax.set_xticks(df['len'].unique())
        fig.suptitle('Precision')
        if savefig:
            fig.savefig('scatter_avg.png')
        if showfig:
            plt.show()

#   scatter of MAPE with average
    if 'scatter_avg_mape' in plot:
        # drop mape over 40%. these are caused through errors
        faulty = df[(df['mape_ar'] > 0.3) | (df['mape_var'] > 0.3)].index
        df.drop(faulty, inplace = True)
        fig, ax = plt.subplots()
        ax.scatter(x = df['len'] + 0.3, y = 'mape_var', data = df, marker = 'o', color = '#007A9B', alpha = 0.5, label = 'mape_var')
        ax.scatter(x = df['len'], y = 'mape_ar', data = df, marker = 'o', color = '#6D8300', alpha = 0.5, label = 'mape_ar')
        ax.plot(df['len'].unique(), means_ar_mape, '-', color = '#6D8300', label = 'average mape_ar')
        ax.plot(df['len'].unique(), means_var_mape, '-', color = '#007A9B', label = 'average mape_var')
        ax.legend()
        ax.set_xlim(2, 13)
        ax.set_ylim(0,0.18)
        ax.set_xticks(df['len'].unique())
        fig.suptitle('Mean Absolute Percentage Error (MAPE)')
        if savefig:
            fig.savefig('scatter_avg_mape.png')
        if showfig:
            plt.show()
