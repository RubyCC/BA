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

