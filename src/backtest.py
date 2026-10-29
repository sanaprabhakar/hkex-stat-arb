import numpy as np
import pandas as pd

def run_pairs_backtest(
    df,
    y_col='HSBC',
    x_col='StanChart',
    z_col='z_score',
    beta_col='beta_kalman',
    z_entry=2.0,
    z_exit=0.0,
    z_stop=3.5,
    stamp_duty=0.0010,
    exchange_fee=0.0003,
    slippage=0.0002,
    annual_borrow_rate=0.02
):
    """
    Simulates statistical arbitrage trading with lookahead protection
    and realistic HKEX microstructure friction deductions.
    """
    data = df.copy()
    n = len(data)
    
    # 1. State Machine for Position Generation
    positions = np.zeros(n)
    current_pos = 0
    z_scores = data[z_col].values
    
    for i in range(1, n):
        z = z_scores[i]
        
        # Priority 1: Emergency Stop-Loss
        if abs(z) >= z_stop:
            current_pos = 0
        # Priority 2: Long Spread (Buy Y, Short X)
        elif z <= -z_entry:
            current_pos = 1
        # Priority 3: Short Spread (Short Y, Buy X)
        elif z >= z_entry:
            current_pos = -1
        # Priority 4: Reversion Exit to Cash
        elif (current_pos == 1 and z >= -z_exit) or (current_pos == -1 and z <= z_exit):
            current_pos = 0
            
        positions[i] = current_pos
        
    # 2. Shift positions by 1 day (Signal at close t -> Executed at t+1)
    data['strategy_position'] = pd.Series(positions, index=data.index).shift(1).fillna(0)
    
    # 3. Calculate Returns
    data['ret_y'] = data[y_col].pct_change().fillna(0)
    data['ret_x'] = data[x_col].pct_change().fillna(0)
    beta_lag = data[beta_col].shift(1).bfill().fillna(0.7)
    
    # Gross Strategy Daily Return = Position * (Ret_Y - Beta * Ret_X)
    data['strategy_ret'] = data['strategy_position'] * (data['ret_y'] - beta_lag * data['ret_x'])
    
    # 4. HKEX Microstructure Friction Deduction
    cost_per_leg = stamp_duty + exchange_fee + slippage
    pos_diff = data['strategy_position'].diff().abs().fillna(0)
    
    # Basket size is 1 share of Y + beta shares of X
    basket_size = 1.0 + beta_lag
    data['turnover'] = pos_diff * basket_size
    data['transaction_costs'] = data['turnover'] * cost_per_leg
    
    # Accrue daily short borrow financing fee
    daily_borrow = annual_borrow_rate / 252.0
    data['borrow_cost'] = 0.0
    data.loc[data['strategy_position'] == 1, 'borrow_cost'] = daily_borrow * beta_lag
    data.loc[data['strategy_position'] == -1, 'borrow_cost'] = daily_borrow * 1.0
    
    # Net Daily Strategy Return
    data['net_strategy_ret'] = data['strategy_ret'] - data['transaction_costs'] - data['borrow_cost']
    
    # Cumulative Curves
    data['cum_gross_ret'] = (1 + data['strategy_ret']).cumprod()
    data['cum_net_ret'] = (1 + data['net_strategy_ret']).cumprod()
    
    # 50/50 Buy & Hold Benchmark
    data['benchmark_ret'] = 0.5 * data['ret_y'] + 0.5 * data['ret_x']
    data['cum_benchmark_ret'] = (1 + data['benchmark_ret']).cumprod()
    
    return data
