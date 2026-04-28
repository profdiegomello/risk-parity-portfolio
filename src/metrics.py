import numpy as np
import pandas as pd


def _clean_numeric_series(values):
    series = pd.Series(values, copy=False)
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series


def _has_invalid_compounding_returns(returns_series):
    series = _clean_numeric_series(returns_series)
    if series.empty:
        return True

    return bool((series <= -1.0).any())


def calcular_vetor_sharpe(ret_medios, cov_matrix, rf_dinamico):
    volatilidades = np.sqrt(np.diag(cov_matrix))
    excessos = ret_medios - rf_dinamico
    sharpes = np.full_like(excessos, -np.inf, dtype=float)
    mascara_validos = volatilidades > 1e-6
    np.divide(excessos, volatilidades, out=sharpes, where=mascara_validos)
    return sharpes

def annualized_return(returns_series, periods_per_year=252):
    returns_series = _clean_numeric_series(returns_series)
    n_years = len(returns_series) / periods_per_year
    if n_years <= 0:
        return np.nan
    if _has_invalid_compounding_returns(returns_series):
        return np.nan
    wealth_index = (1 + returns_series).prod()
    if wealth_index <= 0:
        return np.nan
    return wealth_index ** (1 / n_years) - 1

def annualized_volatility(returns_series, periods_per_year=252):
    returns_series = _clean_numeric_series(returns_series)
    if returns_series.empty:
        return np.nan
    return returns_series.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns_series, rf_series, periods_per_year=252):
    returns_series = _clean_numeric_series(returns_series)
    rf_series = _clean_numeric_series(rf_series)
    if returns_series.empty or rf_series.empty:
        return np.nan

    returns_series, rf_series = returns_series.align(rf_series, join="inner")
    if returns_series.empty:
        return np.nan

    excess_returns = returns_series - rf_series
    std_excess = excess_returns.std()
    
    if std_excess < 1e-6:
        return 0.0
        
    return (excess_returns.mean() / std_excess) * np.sqrt(periods_per_year)

def sortino_ratio(returns_series, rf_series, periods_per_year=252):
    returns_series = _clean_numeric_series(returns_series)
    rf_series = _clean_numeric_series(rf_series)
    if returns_series.empty or rf_series.empty:
        return np.nan

    returns_series, rf_series = returns_series.align(rf_series, join="inner")
    if returns_series.empty:
        return np.nan

    excess_returns = returns_series - rf_series
    downside_diff = excess_returns[excess_returns < 0]
    
    if len(downside_diff) == 0:
        return 0.0
        
    downside_deviation = np.sqrt(np.mean(downside_diff**2))
    
    if downside_deviation < 1e-6:
        return 0.0
        
    return (excess_returns.mean() / downside_deviation) * np.sqrt(periods_per_year)

def maximum_drawdown(returns_series):
    returns_series = _clean_numeric_series(returns_series)
    if returns_series.empty or _has_invalid_compounding_returns(returns_series):
        return np.nan

    cumulative_returns = (1 + returns_series).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_turnover(pesos_antigos, pesos_novos):
    return np.sum(np.abs(pesos_novos - pesos_antigos)) / 2.0
