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

