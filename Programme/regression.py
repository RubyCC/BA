"""
    all functions used for prediction
"""

def adjust(val, length=6):
    return(str(val).ljust(length))

def adfuller_test(series, signif = 0.05, name = '', print_output = None):
    """
        Function to perform ADF Test for Stationarity of given time series and print a report
        returns True if stationary, False if not

        parameter:
            series                          series to perform ADF test on
            signif                          significance level
            name
            output                          DEFAULT = 'None'
                                                None: no print output, only return value True/False
                                                'short': a short version of the output (not yet implemented)
                                                'long': the long version of the report of the ADF test

        returns
    """

    from statsmodels.tsa.stattools import adfuller

    r = adfuller(series, autolag = 'AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']


#   print report
    if print_output == 'long':
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
        print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
        print(f' Significance Level    = {signif}')
        print(f' Test Statistic        = {output["test_statistic"]}')
        print(f' No. Lags Chosen       = {output["n_lags"]}')

        for key,val in r[4].items():
            print(f' Critical value {adjust(key)} = {round(val, 3)}')

        if p_value <= signif:
            print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            print(f" => Series is Stationary.")
        else:
            print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            print(f" => Series is Non-Stationary.")
        print('\n')

    return(True if p_value <= signif else False)

def invert_transformation(df, diffs, **kwargs):
    """
        roll back transformations made by differencing
    """

#   packages
    import numpy as np

#   retrieve original values
    df_orig = diffs['df_orig']

#   add PREDICT to diffs
    diffs['PREDICT'] = diffs['STOCK']

#   invert for all cols
#   replace first rows
    max_diff = max(diffs['AVG'], diffs['STOCK'], diffs['COUNT'])
    df.iloc[:max_diff, :] = df_orig.iloc[:max_diff, :]
    df.iloc[:max_diff, df.columns.get_loc('PREDICT')] = df_orig.iloc[:max_diff, df.columns.get_loc('STOCK')]

#   calculate the rest
    for col in ['AVG', 'COUNT', 'STOCK', 'PREDICT']:
        while diffs[col] > 0:
            for i in range(max_diff, len(df)):
                df.iat[i, df.columns.get_loc(col)] += df.iat[i-diffs[col], df.columns.get_loc(col)]
            diffs[col] -= 1

#   return
    return([df, diffs])

def forecast_accuracy(forecast, actual, **kwargs):
    """
        function to calculate forecast accuracy measures

        parameters:
            forecast                    series with forecasted values
            actual                      series with actual values

        kwargs:
            silent                  print output
                                        True | False
                                        DEFAULT: True

            dropna                  drop NaN values in series
                                        True | False
                                        DEFAULT: True

    """

#   packages
    import numpy as np

#   handle kwargs
    silent = True if 'silent' not in kwargs else kwargs['silent']
    dropna = True if 'dropna' not in kwargs else kwaergs['dropna']


#   drop NaN values
    if dropna:
        df = pd.DataFrame({'forecast': forecast,
                           'actual': actual})
        df.dropna(axis = 0, inplace = True)


#   calculate measures
    # mean absolute percentage error
    mape = np.mean(np.abs(df['forecast'] - df['actual']) / np.abs(df['actual']))
    # mean error
    me = np.mean(df['forecast'] - df['actual'])
    # mean absolute error
    mae = np.mean(np.abs(df['forecast'] - df['actual']))
    # mean percentage error
    mpe = np.mean((df['forecast'] - df['actual'])/df['actual'])
    # mean squared error
    mse = np.mean((df['forecast'] - df['actual'])**2)
    # root-mean-squared error
    rmse = mse**.5
    # correlation
    corr = np.corrcoef(df['forecast'], df['actual'])[0,1]
    mins = np.amin(np.hstack([df['forecast'][:,None],
                              df['actual'][:,None]]), axis=1)
    maxs = np.amax(np.hstack([df['forecast'][:,None],
                              df['actual'][:,None]]), axis=1)
    # minmax
    minmax = 1 - np.mean(mins / maxs)

    # binary classify of direction
    forecast_diff = df['forecast'].diff() / df['forecast'].diff().abs()
    actual_diff = (df['actual'].diff() / df['actual'].diff().abs())
    df_diff = pd.DataFrame({'forecast_diff': forecast_diff,
                            'actual_diff': actual_diff})
    df_diff.dropna(axis = 0, inplace = True)
    acc_bin = sum(df_diff['forecast_diff'].eq(df_diff['actual_diff'])) / max(len(df_diff), 1)


#   return
    return({'mape':mape, 'me':me, 'mae': mae,
    'mpe': mpe, 'mse': mse, 'rmse':rmse, 'corr':corr, 'minmax':minmax,
    'acc_bin': acc_bin})

def get_train_size(df, **kwargs):
    """
        function to test window size of a given dataframe.

        fits an AR-model based on stock data and measures the different MSEs for
        different windows.

        parameters:
            df                          data frame to be tested

        kwargs:
            silent                  whether or not to display status of processing
                                        DEFAULT: false, displaying progress
            window_sizes            window sizes to be tested
                                        DEFAULT: [1, 40]
            lag_order               lag order for underlying process
                                        DEFAULT: 4
            forecast_len            number of forecast values to be calculated
                                        DEFAULT: 10
            plot                    print a plot of the results
                                        DEFAULT: False
            prog_bar                displays a progress bar
                                        DEFAULT: True
            ret_val                 determines value to return (optimal | all)
                                        DEFAULT: optimal window length
            resample                number of resamples for testing
                                        DEFAULT: 5

    """

#   packages
    import pandas as pd
    import random
    import numpy as np
    from statsmodels.tsa.arima_model import ARIMA


#   handle kwargs
    silent = False if 'silent' not in kwargs else kwargs['silent']
    window_sizes = [i for i in range(1, 61, 5)] if 'window_sizes' not in kwargs else kwargs['window_sizes']
    lag_order = 4 if 'lag_order' not in kwargs else kwargs['lag_order']
    forecast_len = 10 if 'forecast_len' not in kwargs else kwargs['forecast_len']
    plot = False if 'plot' not in kwargs else kwargs['plot']
    prog_bar = True if 'prog_bar' not in kwargs else kwargs['prog_bar']
    ret_val = 'opt' if 'ret_val' not in kwargs else kwargs['ret_val']
    resample = 5 if 'resample' not in kwargs else kwargs['resample']


#   print infos
    if not silent:
        print('Testing for optimal training set size:')#
        print('    VAR({})'.format(str(lag_order)))
        print('    Forecast length: {}'.format(str(forecast_len)))
        print('    Dataset contains {} values.'.format(str(len(df))))
        print('\nTesting window sizes:\n    {}'.format(str(window_sizes)))

#   set up loading bar
    if prog_bar:
        from progress.bar import ChargingBar as Bar

#   set up df for results
    results = pd.DataFrame({'window_size': window_sizes,
                            'MSE_train': [0 for i in range(len(window_sizes))],
                            'MSE_test': [0 for i in range(len(window_sizes))],
                            'MSE_total': [0 for i in range(len(window_sizes))]})
    results.set_index(keys = 'window_size',
                      inplace = True)

#   set up resamples
    resamples = [i for i in range(0, len(df) - max(window_sizes))]

#   test all window sizes
    if prog_bar:
        bar = Bar('Testing window sizes:', max = len(window_sizes) * resample)
    for i in range(resample):
#       get sample seed
        sample_seed = random.choice(resamples)
        resamples.remove(sample_seed)
        if not silent:
            print('Sampling with seed: {}'.format(str(sample_seed)))
        for w in window_sizes:
            if prog_bar:
                bar.next()
            if not silent:
                print('Testing window size: {}'.format(str(w)))

#               split data
            train, test = df.STOCK[:w + sample_seed], df.STOCK[w:w + forecast_len + sample_seed]

#                fit model
            model = ARIMA(train, order = (lag_order, 0, 0))
            model_fitted = model.fit(disp = -1)

#                 forecast and calc error
            fc = model_fitted.forecast(steps = forecast_len,
                            alpha = 0.05,
                            exog = None)

#           get MSEs
            mse_train = sum([(train[i] - model_fitted.fittedvalues[i]) ** 2 for i in train.index])
            mse_test = sum([(fc[0][i] - list(test)[i]) ** 2 for i in range(forecast_len)])

            if not silent:
                print('Mean squared errors:')
                print('    MSE (train)  :  {}'.format(str(mse_train)))
                print('    MSE (test)   :  {}'.format(str(mse_test)))
                print('    MSE (total)  :  {}'.format(str(mse_train + mse_test)))

#           save results
            results.at[w, 'MSE_train'] += mse_train
            results.at[w, 'MSE_test'] += mse_test
            results.at[w, 'MSE_total'] += (mse_train + mse_test)


#   apply resampling
    results.at[w, 'MSE_train'] /= resample
    results.at[w, 'MSE_test'] /= resample
    results.at[w, 'MSE_total'] /= resample


#   finish prog bar
    if prog_bar:
        bar.finish()

#   plot if necessary
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(results.index, 'MSE_test', data = results, color = 'red', label = 'MSE (test)')
        #plt.plot(results.index, 'MSE_train', data = results, color = 'red', label = 'MSE (train)', linestyle = 'dotted')
        #plt.plot(results.index, 'MSE_total', data = results, color = 'grey', label = 'MSE (total)', linestyle = 'dotted')
        plt.legend()
        plt.show()
    print(results)

#   return
    if ret_val == 'opt':
        opt = results['MSE_test'].argmax()
        if not silent:
            print('Optimal window size: {}'.format(str(opt)))
        return(opt)
    elif ret_val == 'all':
        return(results)

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

def get_lag_order(df, **kwargs):
    """
        function to determine the lag order of the model

        parameters:
            df                          data frame to be used

        kwargs:
            silent                  print output
                                        True | False
                                        DEFAULT: True

            crit                    which criterion to use
                                        'AIC' | 'BIC' | 'FPE' | 'HQIC'
                                        DEFAULT: 'AIC'

            orders                  list of orders to be used
                                        DEFAULT: [1,...,6]

            model                   model to be used
                                        'AR' | 'VAR'
                                        DEFAULT: 'VAR' (using AVG, STOCK, COUNT)

    """

#   packages
    import pandas as pd
    import numpy as np
#    from statsmodels.tools.eval_measures import aic


#   handle kwargs
    silent = True if 'silent' not in kwargs else kwargs['silent']
    crit = 'AIC' if 'crit' not in kwargs else kwargs['crit']
    orders = [i for i in range(1, 7)] if 'orders' not in kwargs else kwargs['orders']
    model = 'VAR' if 'model' not in kwargs else kwargs['model']

#   set up df for results
    lag_results = pd.DataFrame({'lag': orders,
                                'AIC': [np.nan for i in range(len(orders))],
                                'BIC': [np.nan for i in range(len(orders))],
                                'FPE': [np.nan for i in range(len(orders))],
                                'HQIC': [np.nan for i in range(len(orders))]})
    lag_results.set_index('lag', inplace = True)

#   use criterion
    if model == 'VAR':
        from statsmodels.tsa.api import VAR
        model = VAR(df[['AVG', 'COUNT', 'STOCK']])
        for i in orders:
            model_fitted = model.fit(i)
            lag_results.at[i, 'AIC'] = model_fitted.aic
            lag_results.at[i, 'BIC'] = model_fitted.bic
            lag_results.at[i, 'FPE'] = model_fitted.fpe
            lag_results.at[i, 'HQIC'] = model_fitted.hqic
    elif model == 'AR':
        lag_results.drop('FPE', axis = 1,inplace = True)
        from statsmodels.tsa.ar_model import AutoReg
        for i in orders:
            model = AutoReg(df[['STOCK']][:], lags = i, trend = 'n')
            model_fitted = model.fit()
            lag_results.at[i, 'AIC'] = model_fitted.aic
            lag_results.at[i, 'BIC'] = model_fitted.bic
            lag_results.at[i, 'HQIC'] = model_fitted.hqic


#   print output
    if not silent:
        print('Using information criteria for lag orders {}:'.format(str(orders)))
        print(lag_results)

#   determine best
    opt_lag = lag_results[lag_results[crit] == lag_results[crit].min()].index[0]
    if not silent: print('Optimal lag by {} is {}'.format(str(crit), str(opt_lag)))

#   return
    return(opt_lag)

def ar_prediction(df, order, **kwargs):
    """
        function to create an AR prediction of simply using the STOCK values
    """


#   import
    import pandas as pd
    import numpy as np
    from progress.bar import ChargingBar as Bar
    from statsmodels.tsa.ar_model import AutoReg

#   handle kwargs
    silent = False if 'silent' not in kwargs else kwargs['silent']
    prog_bar = True if 'prog_bar' not in kwargs else kwargs['prog_bar']
    window_mode = 'rolling' if 'window_mode' not in kwargs else kwargs['window_mode']
    window_size = 20 if 'window_size' not in kwargs else kwargs['window_size']

#   test with window
    if not silent: print('Using AR({}) model'.format(str(order)))
    if prog_bar: bar = Bar('Predicting:', max = len(df)-1-window_size)
    for i in range(window_size, len(df)-1):
        bar.next()
        if window_mode == 'rolling':
            data = df['STOCK'][max(i-window_size, 0):i]
        else:
            data = df['STOCK'][:i]
        if order == 'auto':
            order = get_lag_order(df[:i], crit = 'AIC', model = 'AR')
        model = AutoReg(data, trend = 'n', lags = order)
        model_fitted = model.fit()
        forecast = model_fitted.predict(start = window_size - 1, end = window_size - 1)
        df.iat[i+1, df.columns.get_loc('PREDICT')] = forecast[0]
    bar.finish()

#   return
    return(df)

def var_prediction(df, order, **kwargs):
    """
        function to determine the lag order of the model

        parameters:
            df                          data frame to be used

            order                       p of VAR(p) - order of autoregression
                                        or 'auto' if order shall be picked automatically


        kwargs:
            silent                  print output
                                        True | False
                                        DEFAULT: True

            window_mode             window mode for testing
                                        'rolling' | 'increasing'
                                        DEFAULT: 'rolling'

            window_sizes            window size for fitting
                                        pos INT
                                        DEFAULT: 20

            variables               variables to be used (if col names changed or
                                    analysis needs to be modified)
                                        [list]
                                        DEFAULT: ['STOCK', 'AVG', 'COUNT']

            crit                    information criterion to be used. obsolete
                                    as lag order is fixed and information criterion
                                    is outsourced to get_lag_order
                                        'aic' | 'bic' | 'hqic' | 'fpe'
                                        DEFAULT: 'aic'

    """

#   packages
    import pandas as pd
    import numpy as np
    from progress.bar import ChargingBar as Bar
    from statsmodels.tsa.api import VAR

#   handle kwargs
    silent = False if 'silent' not in kwargs else kwargs['silent']
    window_mode = 'rolling' if 'window_mode' not in kwargs else kwargs['window_mode']
    window_size = 20 if 'window_size' not in kwargs else kwargs['window_size']
    variables = ['STOCK', 'AVG', 'COUNT'] if 'variables' not in kwargs else kwargs['variables']
    crit = 'aic' if 'crit' not in kwargs else kwargs['crit']


#   test with window
    bar = Bar('Predicting:', max = len(df)-1-window_size)
    for i in range(window_size, len(df)-1):
        bar.next()
        if window_mode == 'rolling':
            data = df[variables][max(i-window_size, 0):i]
        else:
            data = df[variables][:i]
        model = VAR(data)
        if order == 'auto':
            order = get_lag_order(df, crit = 'AIC')
        model_fitted = model.fit(verbose = False, maxlags = order)
        forecast = model_fitted.forecast(model_fitted.endog, steps = 1)
        df.iat[i+1, df.columns.get_loc('PREDICT')] = forecast[0][0]
    bar.finish()

#   return
    return(df)
