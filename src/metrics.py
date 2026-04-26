import numpy as np
import pandas as pd

def annualized_return(daily_returns, days_per_year=252):
    cumulative_return = np.prod(1 + daily_returns)
    n_years = len(daily_returns) / days_per_year
    return cumulative_return ** (1 / n_years) - 1 if n_years > 0 else 0.0

def annualized_volatility(daily_returns, days_per_year=252):
    return np.std(daily_returns, ddof=1) * np.sqrt(days_per_year)

def sharpe_ratio(daily_returns, rf_annual, days_per_year=252):
    rf_daily = (1 + rf_annual) ** (1/days_per_year) - 1
    excess_returns = daily_returns - rf_daily
    
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(daily_returns, ddof=1)
    
    if std_returns > 0:
        return (mean_excess / std_returns) * np.sqrt(days_per_year)
    return 0.0

def sortino_ratio(daily_returns, rf_annual, days_per_year=252):
    rf_daily = (1 + rf_annual) ** (1/days_per_year) - 1
    excess_returns = daily_returns - rf_daily
    
    negative_excess = excess_returns[excess_returns < 0]
    
    if len(negative_excess) == 0:
        return 0.0
        
    downside_std = np.sqrt(np.mean(negative_excess**2)) * np.sqrt(days_per_year)
    ann_ret = annualized_return(daily_returns, days_per_year)
    
    if downside_std > 0:
        return (ann_ret - rf_annual) / downside_std
    return 0.0

def maximum_drawdown(daily_returns):
    cumulative = (1 + daily_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return np.min(drawdowns)

def calculate_turnover(old_weights_vector, new_weights_vector):
    return np.sum(np.abs(new_weights_vector - old_weights_vector))
