import numpy as np
import pandas as pd

def calculate_performance_metrics(returns_series, cum_series, positions_series=None, n_years=5.0, risk_free_rate=0.03):
    """
    Computes an institutional quantitative performance tearsheet.
    Defensively handles argument order and data types.
    """
    # Defensive guard: auto-correct if positions_series and n_years were swapped
    if not isinstance(n_years, (int, float, np.number)):
        positions_series, n_years = n_years, positions_series
        if not isinstance(n_years, (int, float, np.number)):
            n_years = 5.0
            
    n_years = float(n_years) if n_years > 0 else 5.0
    
    total_ret = cum_series.iloc[-1] - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / n_years) - 1.0
    ann_vol = returns_series.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    
    # Sortino Ratio (Downside deviation only)
    downside_returns = returns_series[returns_series < 0]
    downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-6
    sortino = (cagr - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0
    
    # Maximum Drawdown & Calmar Ratio
    running_max = cum_series.cummax()
    drawdown = (cum_series - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0
    
    # Institutional Trade Analytics
    win_rate = 0.0
    profit_factor = 0.0
    num_trades = 0
    avg_duration = 0.0
    
    if positions_series is not None and isinstance(positions_series, (pd.Series, np.ndarray)):
        pos_series = pd.Series(positions_series)
        pos_diff = pos_series.diff().abs().fillna(0)
        num_trades = int((pos_diff > 0).sum())
        
        active_days = pos_series != 0
        active_returns = returns_series[active_days]
        
        if len(active_returns) > 0:
            winning_days = active_returns[active_returns > 0]
            losing_days = active_returns[active_returns < 0]
            
            win_rate = len(winning_days) / len(active_returns) if len(active_returns) > 0 else 0.0
            gross_profit = winning_days.sum()
            gross_loss = abs(losing_days.sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
            
            # Average Holding Period in trading days
            blocks = (pos_series != pos_series.shift(1)).cumsum()
            active_blocks = blocks[pos_series != 0]
            if len(active_blocks) > 0:
                avg_duration = active_blocks.value_counts().mean()
    
    return {
        "Total Return": total_ret,
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Trade Adjustments": num_trades,
        "Avg Holding Days": avg_duration
    }

def compute_breakeven_frontier(df, n_years=5.0, risk_free_rate=0.03, max_bps=25.0, steps=100):
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
