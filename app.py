import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

from src.kalman import run_kalman_filter, calculate_rolling_zscore
from src.backtest import run_pairs_backtest
from src.metrics import calculate_performance_metrics, compute_breakeven_frontier

# 1. Page Configuration & Institutional Theme
st.set_page_config(
    page_title="HKEX Statistical Arbitrage Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS Styling
st.markdown("""
<style>
    .reportview-container, .main {
        background-color: #0E1117;
        color: #E6EDF3;
    }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        font-family: monospace;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #8B949E;
        margin-top: 4px;
    }
    hr {
        border-color: #30363D;
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar Controls
st.sidebar.markdown("## Execution Controls")

pair_choice = st.sidebar.selectbox(
    "Target HKEX Equity Pair",
    [
        "HSBC (0005.HK) vs StanChart (2888.HK)",
        "Tencent (0700.HK) vs Alibaba (9988.HK)",
        "China Mobile (0941.HK) vs China Telecom (0728.HK)"
    ],
    index=0
)

ticker_map = {
    "HSBC (0005.HK) vs StanChart (2888.HK)": ("0005.HK", "2888.HK", "HSBC", "StanChart"),
    "Tencent (0700.HK) vs Alibaba (9988.HK)": ("0700.HK", "9988.HK", "Tencent", "Alibaba"),
    "China Mobile (0941.HK) vs China Telecom (0728.HK)": ("0941.HK", "0728.HK", "ChinaMobile", "ChinaTelecom")
}

ticker_y, ticker_x, label_y, label_x = ticker_map[pair_choice]

st.sidebar.markdown("---")
st.sidebar.markdown("### Signal Hyperparameters")
z_entry = st.sidebar.slider("Entry Threshold (z_entry)", 1.2, 3.0, 2.0, 0.1)
z_exit = st.sidebar.slider("Exit Threshold (z_exit)", 0.0, 1.0, 0.0, 0.1)
z_stop = st.sidebar.slider("Emergency Stop-Loss (z_stop)", 2.5, 5.0, 3.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Microstructure Friction Engine")
apply_duty = st.sidebar.checkbox("HKEX Statutory Stamp Duty (0.10%)", value=True)
stamp_rate = 0.0010 if apply_duty else 0.0

slippage_bps = st.sidebar.slider("Bid-Ask Spread Slippage (bps)", 0, 10, 2, 1)
slippage_rate = slippage_bps / 10000.0

borrow_rate = st.sidebar.slider("Annual Short Borrow Fee (%)", 0.0, 5.0, 2.0, 0.5) / 100.0
exchange_fee = 0.0003  # Fixed SFC, AFRC, HKEX levies & broker fees

# 3. Data Ingestion Engine with Caching
@st.cache_data(ttl=86400)
def fetch_hkex_pair(symbol_y, symbol_x, label_y, label_x):
    start_date = "2021-01-01"
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    df_y = yf.download(symbol_y, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
    df_x = yf.download(symbol_x, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
    
    combined = pd.DataFrame({label_y: df_y.squeeze(), label_x: df_x.squeeze()}).dropna()
    return combined

data_raw = fetch_hkex_pair(ticker_y, ticker_x, label_y, label_x)


# 4. Run Quantitative Pipeline
alpha_t, beta_t, innovations = run_kalman_filter(data_raw[label_y], data_raw[label_x])

data = data_raw.copy()
data['alpha_kalman'] = alpha_t
data['beta_kalman'] = beta_t
data['innovation'] = innovations
data['z_score'] = calculate_rolling_zscore(innovations, window=30)

bt_results = run_pairs_backtest(
    data,
    y_col=label_y,
    x_col=label_x,
    z_col='z_score',
    beta_col='beta_kalman',
    z_entry=z_entry,
    z_exit=z_exit,
    z_stop=z_stop,
    stamp_duty=stamp_rate,
    exchange_fee=exchange_fee,
    slippage=slippage_rate,
    annual_borrow_rate=borrow_rate
)

# Time Horizon
n_years = len(bt_results) / 252.0

# Performance Analytics
gross_stats = calculate_performance_metrics(bt_results['strategy_ret'], bt_results['cum_gross_ret'], n_years)
net_stats = calculate_performance_metrics(bt_results['net_strategy_ret'], bt_results['cum_net_ret'], n_years)
bm_stats = calculate_performance_metrics(bt_results['benchmark_ret'], bt_results['cum_benchmark_ret'], n_years)

total_friction_drag = (gross_stats['Total Return'] - net_stats['Total Return']) * 100.0

fee_axis, sharpe_frontier, breakeven_fee = compute_breakeven_frontier(bt_results, n_years)

# 5. Front-End Dashboard Presentation
st.title("Empirical Statistical Arbitrage on HKEX Equities")
st.caption(f"State-Space Hedge Ratios (Kalman Filter) vs. Real-World Microstructure Frictions | Target: {label_y} / {label_x}")

# Top Executive KPI Grid
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Gross Strategy Return</div>
        <div class="metric-value" style="color: #00C076;">{gross_stats['Total Return']*100:.2f}%</div>
        <div class="metric-sub">Sharpe: {gross_stats['Sharpe Ratio']:.2f} | Vol: {gross_stats['Annualized Volatility']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    net_color = "#00C076" if net_stats['Total Return'] >= 0 else "#FF4D4D"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Net Strategy Return</div>
        <div class="metric-value" style="color: {net_color};">{net_stats['Total Return']*100:.2f}%</div>
        <div class="metric-sub">Sharpe: {net_stats['Sharpe Ratio']:.2f} | Max DD: {net_stats['Max Drawdown']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Friction Drag Penalty</div>
        <div class="metric-value" style="color: #FF7B72;">-{total_friction_drag:.2f}%</div>
        <div class="metric-sub">Turnover Adjustments: {int(bt_results['turnover'].gt(0).sum())} trades</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Break-Even Frontier Limit</div>
        <div class="metric-value" style="color: #A371F7;">{breakeven_fee:.2f} bps</div>
        <div class="metric-sub">Actual Friction: {(stamp_rate+exchange_fee+slippage_rate)*10000:.1f} bps/leg</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Interactive Visualizations (Plotly)

# Panel 1: Dynamic State-Space Tracking
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Scatter(x=bt_results.index, y=bt_results[label_y], name=f"{label_y} Close", line=dict(color="#58A6FF", width=1.5)), secondary_y=False)
fig1.add_trace(go.Scatter(x=bt_results.index, y=bt_results[label_x], name=f"{label_x} Close", line=dict(color="#8B949E", width=1.2, dash='dot')), secondary_y=False)
fig1.add_trace(go.Scatter(x=bt_results.index, y=bt_results['beta_kalman'], name="Kalman Beta (β_t)", line=dict(color="#E3B341", width=1.8)), secondary_y=True)

fig1.update_layout(
    title="Panel 1: Asset Price Dynamics & Dynamic Hedge Ratio (β_t)",
    template="plotly_dark",
    paper_bgcolor="#161B22",
    plot_bgcolor="#161B22",
    height=380,
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig1.update_yaxes(title_text="Stock Price (HKD)", secondary_y=False, gridcolor="#21262D")
fig1.update_yaxes(title_text="Kalman Hedge Ratio", secondary_y=True, gridcolor="#21262D")
st.plotly_chart(fig1, use_container_width=True)

# Panel 2: Normalized Innovation Z-Score
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=bt_results.index, y=bt_results['z_score'], name="Spread Z-Score", line=dict(color="#58A6FF", width=1.3)))
fig2.add_hline(y=z_entry, line_dash="dash", line_color="#FF4D4D", annotation_text=f"Short Spread (+{z_entry}σ)")
fig2.add_hline(y=-z_entry, line_dash="dash", line_color="#00C076", annotation_text=f"Long Spread (-{z_entry}σ)")
fig2.add_hline(y=z_stop, line_dash="dot", line_color="#8B949E", annotation_text=f"Stop-Loss (+{z_stop}σ)")
fig2.add_hline(y=-z_stop, line_dash="dot", line_color="#8B949E", annotation_text=f"Stop-Loss (-{z_stop}σ)")
fig2.add_hline(y=0.0, line_color="#30363D", line_width=1)

fig2.update_layout(
    title="Panel 2: Rolling 30-Day Innovation Z-Score & Execution Bounds",
    template="plotly_dark",
    paper_bgcolor="#161B22",
    plot_bgcolor="#161B22",
    height=330,
    margin=dict(l=40, r=40, t=50, b=40),
    yaxis=dict(gridcolor="#21262D")
)
st.plotly_chart(fig2, use_container_width=True)

# Panel 3: Gross vs. Net vs. Benchmark Performance Triad
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=bt_results.index, y=bt_results['cum_gross_ret'], name="Gross Strategy (0 Fee)", line=dict(color="#00C076", width=1.8)))
fig3.add_trace(go.Scatter(x=bt_results.index, y=bt_results['cum_net_ret'], name="Net Strategy (HKEX Frictions)", line=dict(color="#FF4D4D", width=2.0)))
fig3.add_trace(go.Scatter(x=bt_results.index, y=bt_results['cum_benchmark_ret'], name="50/50 Buy & Hold Benchmark", line=dict(color="#8B949E", width=1.2, dash='dot')))

fig3.update_layout(
    title="Panel 3: Cumulative Performance Triad (Friction Stress Test)",
    template="plotly_dark",
    paper_bgcolor="#161B22",
    plot_bgcolor="#161B22",
    height=400,
    margin=dict(l=40, r=40, t=50, b=40),
    yaxis=dict(title="Growth of HK$1.00", gridcolor="#21262D"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig3, use_container_width=True)

# Panel 4: Break-Even Friction Frontier
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=fee_axis, y=sharpe_frontier, mode='lines', name='Net Sharpe Ratio', line=dict(color="#A371F7", width=2.2)))
fig4.add_hline(y=0.0, line_dash="dash", line_color="#E6EDF3", annotation_text="Sharpe = 0.0 Threshold")
fig4.add_vline(x=breakeven_fee, line_dash="dot", line_color="#00C076", annotation_text=f"Break-Even: {breakeven_fee:.2f} bps")
fig4.add_vline(x=(stamp_rate+exchange_fee+slippage_rate)*10000, line_dash="dash", line_color="#FF4D4D", annotation_text="Active Friction")

fig4.update_layout(
    title="Panel 4: The Break-Even Friction Frontier (Viability Threshold)",
    xaxis_title="Transaction Cost per Single Leg (Basis Points)",
    yaxis_title="Annualized Strategy Net Sharpe Ratio",
    template="plotly_dark",
    paper_bgcolor="#161B22",
    plot_bgcolor="#161B22",
    height=340,
    margin=dict(l=40, r=40, t=50, b=40),
    yaxis=dict(gridcolor="#21262D"),
    xaxis=dict(gridcolor="#21262D")
)
st.plotly_chart(fig4, use_container_width=True)
