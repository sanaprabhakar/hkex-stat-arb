import numpy as np
import pandas as pd

def run_kalman_filter(y_series, x_series, delta=1e-4, ve=1.0):
    """
    State-Space Hedge Ratio Estimation:
    State transition:     theta_t = theta_{t-1} + w_t,  w_t ~ N(0, W_t)
    Observation model:    Y_t = H_t * theta_t + v_t,    v_t ~ N(0, V_t)
    where theta_t = [alpha_t, beta_t]^T and H_t = [1, X_t]
    """
    n = len(y_series)
    theta = np.zeros((2, n))
    R = np.zeros((2, 2, n))
    
    # Process covariance matrix W (governed by drift delta)
    W = (delta / (1.0 - delta)) * np.eye(2)
    V = float(ve)
    
    # Initial state assumptions matching Colab: initial_state_mean=[0, 0], covariance=ones((2,2))
    theta[:, 0] = [0.0, 0.0]
    R[:, :, 0] = np.ones((2, 2))
    
    # Step 0 Update
    H0 = np.array([1.0, x_series.iloc[0]])
    e0 = y_series.iloc[0] - np.dot(H0, theta[:, 0])
    Q0 = np.dot(H0, np.dot(R[:, :, 0], H0.T)) + V
    K0 = np.dot(R[:, :, 0], H0.T) / Q0
    theta[:, 0] = theta[:, 0] + K0 * e0
    R[:, :, 0] = R[:, :, 0] - np.outer(K0, np.dot(H0, R[:, :, 0]))
    
    # Recursive Kalman cycle
    for t in range(1, n):
        # 1. State Prediction
        theta_pred = theta[:, t-1]
        R_pred = R[:, :, t-1] + W
        
        # 2. Measurement Vector H = [1, X_t]
        H = np.array([1.0, x_series.iloc[t]])
        
        # 3. Innovation Residual
        e = y_series.iloc[t] - np.dot(H, theta_pred)
        
        # 4. Innovation Covariance
        Q = np.dot(H, np.dot(R_pred, H.T)) + V
        
        # 5. Kalman Gain
        K = np.dot(R_pred, H.T) / Q
        
        # 6. State Update
        theta[:, t] = theta_pred + K * e
        
        # 7. Covariance Update
        R[:, :, t] = R_pred - np.outer(K, np.dot(H, R_pred))
        
    alpha = theta[0, :]
    beta = theta[1, :]
    
    # Filtered Dynamic Spread: Y_t - (alpha_t + beta_t * X_t)
    dynamic_spread = y_series.values - (alpha + beta * x_series.values)
    
    return alpha, beta, dynamic_spread

def calculate_rolling_zscore(spread_series, window=30):
    """
    Dimensionless Z-score computed on the dynamic spread over a rolling window.
    """
    s = pd.Series(spread_series)
    rolling_mean = s.rolling(window=window).mean()
    rolling_std = s.rolling(window=window).std()
    
    z_score = (s - rolling_mean) / rolling_std
    return z_score.fillna(0).values
