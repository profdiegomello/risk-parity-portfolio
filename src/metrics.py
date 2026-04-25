import numpy as np
import pandas as pd

def annualized_return(daily_returns, days_per_year=252):
    cumulative_return = np.prod(1 + daily_returns)
    n_years = len(daily_returns) / days_per_year
    return cumulative_return ** (1 / n_years) - 1 if n_years > 0 else 0.0

def annualized_volatility(daily_returns, days_per_year=252):
    return np.std(daily_returns, ddof=1) * np.sqrt(days_per_year)

def sharpe_ratio(daily_returns, rf_annual, days_per_year=252):
    ann_ret = annualized_return(daily_returns, days_per_year)
    ann_vol = annualized_volatility(daily_returns, days_per_year)
    return (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else 0.0

def maximum_drawdown(daily_returns):
    cumulative = (1 + daily_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return np.min(drawdowns)

def calculate_turnover(old_weights_vector, new_weights_vector):
    # Assume que os vetores estão alinhados pelo universo total de ativos (N)
    return np.sum(np.abs(new_weights_vector - old_weights_vector))
