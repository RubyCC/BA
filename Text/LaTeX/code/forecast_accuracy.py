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

