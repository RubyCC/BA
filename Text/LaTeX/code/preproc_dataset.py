def preproc_dataset(df, **kwargs):
    """
        function to pre-process the given dataset

        returns:
            list containing the preprocessed dataframe and a dictionary
            containing the original dataframe and information on which order
            processing had to be performed for stationarity


        parameters:
            df                          data frame to be pre-processed

        kwargs:
            silent                  whether or not to display status of processing
                                        True | False
                                        DEFAULT: True

            fill                    fill missing values in data (NA) with backfill
                                        True | False
                                        DEFAULT: True

            add_cols                list containing names for columns to add
                                        [...] | None
                                        DEFAULT: ['PREDICT']

            stat_test               test time series for stationarity
                                        'ADF' | None
                                        DEFAULT: 'ADF'

    """

#   packages
    import pandas as pd
    import numpy as np

#   handle kwargs
    silent = True if 'silent' not in kwargs else kwargs['silent']
    fill = True if 'fill' not in kwargs else kwargs['fill']
    add_cols = ['PREDICT'] if 'add_cols' not in kwargs else kwargs['add_cols']
    stat_test = 'ADF' if 'stat_test' not in kwargs else kwargs['stat_test']

#   pre-process
    df.index = pd.DatetimeIndex(df.date).to_period("D")
    if fill == True:
        if not silent: print('Using ffill to close {} gaps in data.'.format(str(sum(df['STOCK'].isna()))))
        df['STOCK'].fillna(method = 'ffill', inplace = True)
        if sum(df['STOCK'].isna()) > 0:
            df['STOCK'].fillna(method = 'bfill', inplace = True)            # NA in 1st


#   col for prediction results
    if not add_cols == [] and not add_cols is None:
        for col in add_cols:
            df.insert(loc = 1,
                      column = 'PREDICT',
                      value = np.nan)

#   test stationarity and differentiate when needed
    if not stat_test is None:
        adf_output = 'long' if silent is False else None
        df_orig = df.copy()
        diffs = {'df_orig': df_orig, 'AVG': 0, 'COUNT': 0, 'STOCK': 0}
        for col in ['AVG', 'COUNT', 'STOCK']:
            while not adfuller_test(df[col], signif = 0.05, name = col, print_output = adf_output):
                diffs[col] += 1
                if not silent: print('{} is not stationary and needs to be differenced.'.format(str(col)))
                df[col] = df[col].diff()
                df[col].dropna(inplace = True)
        if not silent:
            print('DataFrame was stationarized using differencing:')
            print('   AVG was differenced {} times'.format(str(diffs['AVG'])))
            print('   COUNT was differenced {} times'.format(str(diffs['COUNT'])))
            print('   STOCK was differenced {} times'.format(str(diffs['STOCK'])))
    else:
        diffs = None


#   remove NaN's created through differencing
    df.fillna(0, inplace = True)

#   return
    return([df, diffs])

