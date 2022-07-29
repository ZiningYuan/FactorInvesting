def neutralize(signal,exposure):
    ids = signal.index
    y = signal.ix[ids].values
    X = np.column_stack([exposure.loc[signal.index,:]].values)
    se_weight = pd.Series(np.ones_like(ids), index = ids)
    
    mod_wls = sm.OLS(y,X)
    mod_fit = mod_wls.fit()
    neu_signal = pd.Series(mod_fit.resid, index=ids)
    return neu_signal
