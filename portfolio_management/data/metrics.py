import numpy as np
import pandas as pd

def normalize_weights(w):
    w = np.asarray(w, float)
    s = w.sum()
    if s == 0:
        return w
    return w / s

def daily_to_annual_mean(d, periods=252):
    return d * periods

def daily_to_annual_vol(s, periods=252):
    return s * np.sqrt(periods)

def sharpe_ratio(paths, rf=0.0, periods=252):
    r = paths[1:] / paths[:-1] - 1
    m = r.mean()
    s = r.std()
    if s == 0:
        return np.nan
    return ((m * periods) - rf) / (s * np.sqrt(periods))

def sortino_ratio(paths, rf=0.0, periods=252):
    r = paths[1:] / paths[:-1] - 1
    m = r.mean()
    d = r[r < 0]
    if d.size == 0:
        return np.nan
    s = d.std()
    if s == 0:
        return np.nan
    return ((m * periods) - rf) / (s * np.sqrt(periods))

def max_drawdown(paths):
    peak = np.maximum.accumulate(paths, axis=0)
    d = (paths - peak) / peak
    return d.min()

def calmar_ratio(paths, periods=252):
    r = paths[1:] / paths[:-1] - 1
    m = r.mean() * periods
    d = max_drawdown(paths)
    if d == 0:
        return np.nan
    return m / abs(d)

def correlation_from_returns(returns: pd.DataFrame):
    return returns.corr()

def nearest_psd(matrix, eps=1e-8):
    A = (matrix + matrix.T) / 2
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    B = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(B), eps, None))
    C = B / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)

def beta(asset_returns, benchmark_returns):
    a = asset_returns.values.flatten()
    b = benchmark_returns.values.flatten()
    cov = np.cov(a, b)[0, 1]
    var = np.var(b)
    if var == 0:
        return np.nan
    return cov / var

def alpha(asset_returns, benchmark_returns, rf=0.0, periods=252):
    b = beta(asset_returns, benchmark_returns)
    a_ann = asset_returns.mean() * periods
    b_ann = benchmark_returns.mean() * periods
    return a_ann - (rf + b * (b_ann - rf))

def portfolio_variance(weights, cov):
    w = np.asarray(weights, float)
    return float(w @ cov @ w)

def portfolio_return(weights, mu):
    w = np.asarray(weights, float)
    return float(w @ mu)
