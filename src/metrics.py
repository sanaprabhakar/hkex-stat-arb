import numpy as np
import pandas as pd

def calculate_performance_metrics(returns_series, cum_series, n_years, risk_free_rate=0.03):
    """
    Computes institutional risk-adjusted performance metrics.
    """
    total_ret = cum_series.iloc[-1] - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    ann_vol = returns_series.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    
    # Sortino Ratio (Penalizes only downside volatility)
    downside_returns = returns_series[returns_series < 0]
    downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    sortino = (cagr - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0
    
    # Maximum Drawdown
    running_max = cum_series.cummax()
    drawdown = (cum_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": total_ret,
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown
    }

def compute_breakeven_frontier(df, n_years, risk_free_rate=0.03, max_bps=25.0, steps=100):
    """
    Evaluates strategy Net Sharpe across varying transaction cost levels
    to identify the exact Break-Even Friction Frontier.
    """
    fee_levels = np.linspace(0, max_bps, steps)
    sharpe_curve = []
    
    turnover = df['turnover']
    borrow_cost = df['borrow_cost']
    gross_strategy_ret = df['strategy_ret']
    
    for fee_bps in fee_levels:
        cost_rate = fee_bps / 10000.0
        trade_costs = turnover * cost_rate
        net_ret = gross_strategy_ret - trade_costs - borrow_cost
        
        tot_ret = (1.0 + net_ret).cumprod().iloc[-1] - 1.0
        ann_ret = (1.0 + tot_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
        ann_vol = net_ret.std() * np.sqrt(252)
        
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
        sharpe_curve.append(sharpe)
        
    sharpe_curve = np.array(sharpe_curve)
    idx_zero = np.where(sharpe_curve <= 0)[0]
    breakeven_bps = fee_levels[idx_zero[0]] if len(idx_zero) > 0 else fee_levels[-1]
    
    return fee_levels, sharpe_curve, breakeven_bps
