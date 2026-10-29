import numpy as np
import pandas as pd

def run_kalman_filter(y_series, x_series, delta=1e-4, ve=1e-3):
    """
    Dynamic state-space hedge ratio estimation using a 2D Kalman Filter.
    Measurement equation: Y_t = alpha_t + beta_t * X_t + e_t
    State transition:     theta_t = theta_{t-1} + w_t
    """
    n = len(y_series)
    
    # State vector: [alpha, beta]
    theta = np.zeros((2, n))
    
    # State covariance matrix R
    R = np.zeros((2, 2, n))
    
    # Process noise covariance W (modeled as random walk)
    W = delta / (1 - delta) * np.eye(2)
    
    # Measurement noise variance V
    V = ve
    
    # Initial state assumptions
    theta[:, 0] = [0.0, 0.7]
    R[:, :, 0] = np.eye(2)
    
    e = np.zeros(n)      # Innovation residual (raw spread)
    Q = np.zeros(n)      # Innovation variance
    
    for t in range(1, n):
        # 1. State Prediction
        theta_pred = theta[:, t-1]
        R_pred = R[:, :, t-1] + W
        
        # 2. Measurement Vector H = [1, X_t]
        H = np.array([1.0, x_series.iloc[t]])
        
        # 3. Innovation Residual
        e[t] = y_series.iloc[t] - np.dot(H, theta_pred)
        
        # 4. Innovation Covariance
        Q[t] = np.dot(H, np.dot(R_pred, H.T)) + V
        
        # 5. Kalman Gain
        K = np.dot(R_pred, H.T) / Q[t]
        
        # 6. State Update
        theta[:, t] = theta_pred + K * e[t]
        
        # 7. Covariance Update
        R[:, :, t] = R_pred - np.outer(K, np.dot(H, R_pred))
        
    alpha = theta[0, :]
    beta = theta[1, :]
    
    return alpha, beta, e

def calculate_rolling_zscore(residuals, window=30):
    """
    Standardize innovation residuals using a rolling window 
    to strictly prevent lookahead bias.
    """
    res_series = pd.Series(residuals)
    rolling_mean = res_series.rolling(window=window).mean()
    rolling_std = res_series.rolling(window=window).std()
    
    z_score = (res_series - rolling_mean) / rolling_std
    return z_score.fillna(0).values
```[cite: 6, 7]
