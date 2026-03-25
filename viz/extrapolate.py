import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def sat_hyper(x, ymax, k):
    return ymax * x / (k + x)

def sat_hill(x, ymax, x50, h):
    return ymax / (1 + (x50 / x)**h)

def logistic(x, ymax, k, x0):
    return ymax / (1 + np.exp(-k*(x-x0)))

def gompertz(x, ymax, k, x0):
    return ymax * np.exp(-np.exp(-k*(x-x0)))

def exp_approach(x, ymax, k):
    return ymax * (1 - np.exp(-k*x))


def extrapolate(df, n_new, fct):
    result = []
    for method in df['method'].unique():
        df_method = df[df['method'] == method]
        popt, _ = curve_fit(fct, df_method['n'], df_method['score'])
        score_fit = fct(n_new, *popt)
        result.append(pd.DataFrame({'method': method, 'n': n_new, 'score': score_fit}))
    return pd.concat(result, ignore_index=True)