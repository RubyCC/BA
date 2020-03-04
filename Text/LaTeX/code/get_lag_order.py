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

