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

