############################################################
## Neutralising a signal (the factor you're testing) to remove the correlation between the signal and common industry or style factors (beta) in order to get pure alpha.
## signal - (winsorised) values of the testing factor of a series of selected stocks
## exposure - exposure of industry and/or style factors on a series of selected stocks
#############################################################
def neutralize(signal,exposure):
    ids = signal.index
    y = signal.ix[ids].values
    X = np.column_stack([exposure.loc[signal.index,:]].values)
    se_weight = pd.Series(np.ones_like(ids), index = ids)
    
    mod_wls = sm.OLS(y,X)
    mod_fit = mod_wls.fit()
    neu_signal = pd.Series(mod_fit.resid, index=ids)
    return neu_signal


def standardize(signal):
    return (signal-signal.mean())/signal.std()

#################################################
# More winsorisation methods to be discovered (eg by quantile/p-value)
##############################################
def winsorize(signal):
    md = np.median(signal)
    upper = md + 3 * signal.std()
    lower = md - 3 * signal.std()
        
    win_signal = np.where(signal>upper, upper, signal)
    win_signal = np.where(win_signal<lower, lower, signal)
    return win_signal       
    
