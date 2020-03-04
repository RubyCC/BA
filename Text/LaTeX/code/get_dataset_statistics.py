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

