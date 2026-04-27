import numpy as np
import pandas as pd

def calcular_vetor_sharpe(ret_medios, cov_matrix, rf_dinamico):
    volatilidades = np.sqrt(np.diag(cov_matrix))
    sharpes = np.where(volatilidades > 1e-6, (ret_medios - rf_dinamico) / volatilidades, -np.inf)
    return sharpes

def annualized_return(returns_series, periods_per_year=252):
    n_years = len(returns_series) / periods_per_year
    if n_years <= 0:
        return 0.0
    wealth_index = (1 + returns_series).prod()
    if wealth_index <= 0:
        return -1.0 
    return wealth_index ** (1 / n_years) - 1

def annualized_volatility(returns_series, periods_per_year=252):
    return returns_series.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns_series, risk_free_rate_annual=0.0, periods_per_year=252):
    ann_ret = annualized_return(returns_series, periods_per_year)
    ann_vol = annualized_volatility(returns_series, periods_per_year)
    if ann_vol < 1e-6:
        return 0.0
    return (ann_ret - risk_free_rate_annual) / ann_vol

def sortino_ratio(returns_series, risk_free_rate_annual=0.0, periods_per_year=252):
    ann_ret = annualized_return(returns_series, periods_per_year)
    negative_returns = returns_series[returns_series < 0]
    downside_vol = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0.0
    if downside_vol < 1e-6:
        return 0.0
    return (ann_ret - risk_free_rate_annual) / downside_vol

def maximum_drawdown(returns_series):
    cumulative_returns = (1 + returns_series).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_turnover(pesos_antigos, pesos_novos):
    return np.sum(np.abs(pesos_novos - pesos_antigos)) / 2.0
