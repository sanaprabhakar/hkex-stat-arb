import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series):
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.
    H0: Unit root present (Non-stationary)
    H1: Process is stationary
    """
    clean_series = series.dropna()
    adf_result = adfuller(clean_series, autolag='AIC')
    
    return {
        "Test Statistic": adf_result[0],
        "p-value": adf_result[1],
        "Lags Used": adf_result[2],
        "Observations": adf_result[3],
        "Critical Values": adf_result[4],
        "Is Stationary (5%)": adf_result[1] < 0.05
    }

def run_engle_granger(y_series, x_series):
    """
    Two-Step Engle-Granger Cointegration Test:
    1. OLS regression: Y_t = alpha + beta * X_t + e_t
    2. ADF test on estimated residuals e_t
    """
    X_const = sm.add_constant(x_series)
    model = sm.OLS(y_series, X_const).fit()
    
    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    residuals = y_series - (alpha + beta * x_series)
    
    adf_metrics = test_stationarity(residuals)
    
    return {
        "alpha_ols": alpha,
        "beta_ols": beta,
        "residuals": residuals,
        "adf_stat": adf_metrics["Test Statistic"],
        "p_value": adf_metrics["p-value"],
        "is_cointegrated": adf_metrics["Is Stationary (5%)"]
    }

def calculate_half_life(spread_series):
    """
    Calculates the analytical half-life of mean-reversion by modeling
    the dynamic spread as a continuous-time Ornstein-Uhlenbeck (OU) process:
        dS_t = theta * (mu - S_t) dt + sigma * dW_t
    Discretized into an AR(1) linear regression:
        delta S_t = lambda * S_{t-1} + const + eps_t
        where theta = -lambda / dt
    Half-life tau = ln(2) / theta
    """
    s = spread_series.dropna()
    delta_s = s.diff().dropna()
    lag_s = s.shift(1).dropna()
    
    # Align lengths
    delta_s = delta_s.iloc[1:]
    lag_s = lag_s.iloc[1:]
    
    lag_s_const = sm.add_constant(lag_s)
    ar1_model = sm.OLS(delta_s, lag_s_const).fit()
    
    lambda_param = ar1_model.params.iloc[1]
    
    # Guard against explosive or non-mean-reverting series (lambda >= 0)
    if lambda_param >= 0:
        return np.nan
    
    half_life_days = -np.log(2.0) / lambda_param
    return half_life_days
