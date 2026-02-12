import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================

# FULL ASSET LIST
NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 'SBIN.NS', 
    'INFY.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 
    'HCLTECH.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ADANIENT.NS', 
    'TATASTEEL.NS', 'KOTAKBANK.NS', 'NTPC.NS', 'AXISBANK.NS', 'POWERGRID.NS', 
    'M&M.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'COALINDIA.NS', 'WIPRO.NS', 'BAJAJFINSV.NS', 
    'NESTLEIND.NS', 'ADANIPORTS.NS', 'JSWSTEEL.NS', 'GRASIM.NS', 'CIPLA.NS', 
    'HINDALCO.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'TECHM.NS', 'SBILIFE.NS', 
    'BRITANNIA.NS', 'BPCL.NS', 'HEROMOTOCO.NS', 'DIVISLAB.NS', 'TATACONSUM.NS', 
    'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS', 'LTIM.NS', 'INDUSINDBK.NS'
]

METALS = ['GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'AA', 'FCX', 'SCCO']

SP_TOP_50 = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO', 
    'JPM', 'V', 'UNH', 'XOM', 'MA', 'JNJ', 'HD', 'PG', 'COST', 'ABBV', 'MRK', 'CRM', 
    'AMD', 'CVX', 'NFLX', 'WMT', 'ACN', 'BAC', 'PEP', 'KO', 'DIS', 'CSCO', 'VZ', 
    'CMCSA', 'ADBE', 'INTC', 'T', 'PFE', 'WFC', 'INTU', 'QCOM', 'TXN', 'HON', 'AMGN'
]

ALL_TICKERS = NIFTY_50 + METALS + SP_TOP_50

# SCAN SETTINGS (Added 5m back)
SCAN_CONFIGS = [
    {"label": "5m",  "interval": "5m",  "period": "5d",   "resample": None},
    {"label": "15m", "interval": "15m", "period": "15d",  "resample": None},
    {"label": "1h",  "interval": "1h",  "period": "60d",  "resample": None},
    {"label": "4h",  "interval": "1h",  "period": "300d", "resample": "4h"}, 
]

# ==========================================
# 2. CORE LOGIC (STRICT)
# ==========================================

def get_pivots(series, order=8):
    values = series.values
    if len(values) == 0: return [], []
    high_idx = argrelextrema(values, np.greater, order=order)[0]
    low_idx = argrelextrema(values, np.less, order=order)[0]
    return high_idx, low_idx

def check_line_integrity(series, idx_start, idx_end, slope, intercept, mode="upper"):
    if idx_end <= idx_start: return False
    x_range = np.arange(idx_start, idx_end + 1)
    line_values = slope * x_range + intercept
    actual_values = series.iloc[idx_start : idx_end + 1].values
    tolerance = 0.002 
    
    if mode == "upper":
        violations = actual_values > (line_values * (1 + tolerance))
    else:
        violations = actual_values < (line_values * (1 - tolerance))
    
    if np.any(violations): return False
    return True

def analyze_ticker(df):
    if len(df) < 50: return None

    # Pivots
    high_idxs, low_idxs = get_pivots(df['High'], order=8)
    if len(high_idxs) < 2 or len(low_idxs) < 2: return None

    Ax, Cx = high_idxs[-2], high_idxs[-1]
    Ay, Cy = df['High'].iloc[Ax], df['High'].iloc[Cx]
    Bx, Dx = low_idxs[-2], low_idxs[-1]
    By, Dy = df['Low'].iloc[Bx], df['Low'].iloc[Dx]

    # Geometry
    if not (Ay > Cy and By < Dy): return None

    # Math
    slope_upper = (Cy - Ay) / (Cx - Ax)
    intercept_upper = Ay - (slope_upper * Ax)
    slope_lower = (Dy - By) / (Dx - Bx)
    intercept_lower = By - (slope_lower * Bx)

    # Integrity
    if not check_line_integrity(df['High'], Ax, Cx, slope_upper, intercept_upper, "upper"): return None
    if not check_line_integrity(df['Low'], Bx, Dx, slope_lower, intercept_lower, "lower"): return None

    # Projection
    current_idx = len(df) - 1
    proj_upper = (slope_upper * current_idx) + intercept_upper
    proj_lower = (slope_lower * current_idx) + intercept_lower
    
    current_price = df['Close'].iloc[-1]
    
    if not (proj_lower < current_price < proj_upper): return None
    
    width_pct = (proj_upper - proj_lower) / current_price
    
    if width_pct < 0.035: # Slightly looser for visual confirmation
        return {
            "pivots": {"Ax": Ax, "Ay": Ay, "Cx": Cx, "Cy": Cy, "Bx": Bx, "By": By, "Dx": Dx, "Dy": Dy},
            "slopes": {"upper": slope_upper, "lower": slope_lower},
            "intercepts": {"upper": intercept_upper, "lower": intercept_lower},
            "coil_width": width_pct
        }
    return None

def resample_data(df, interval):
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    return df.resample(interval).agg(logic).dropna()

def plot_triangle(df, ticker, data_dict):
    """ Generates an interactive Plotly chart with the triangle lines """
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name=ticker)])

    # Trendline Coordinates calculation
    # We extend the line slightly into the future for visual clarity
    x_indices = np.arange(len(df))
    
    # Upper Line (Resistance)
    slope_u = data_dict['slopes']['upper']
    int_u = data_dict['intercepts']['upper']
    # Start drawing from first pivot (Ax) to current candle
    start_idx = data_dict['pivots']['Ax']
    y_vals_upper = slope_u * x_indices[start_idx:] + int_u
    
    # Lower Line (Support)
    slope_l = data_dict['slopes']['lower']
    int_l = data_dict['intercepts']['lower']
    start_idx_l = data_dict['pivots']['Bx']
    y_vals_lower = slope_l * x_indices[start_idx_l:] + int_l

    # Add Lines
    fig.add_trace(go.Scatter(x=df.index[start_idx:], y=y_vals_upper, mode='lines', name='Resistance', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=df.index[start_idx_l:], y=y_vals_lower, mode='lines', name='Support', line=dict(color='green', width=2)))

    fig.update_layout(title=f"Triangular Coil: {ticker}", xaxis_rangeslider_visible=False, height=400)
    return fig

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Triangle Hunter", layout="wide")
st.title("🔻 Geometric Triangle Scanner")
st.markdown(f"Scanning **{len(ALL_TICKERS)} Assets** across **5m, 15m, 1h, 4h** timeframes.")

if st.button("🚀 Run Market Scan"):
    
    results_container = st.container()
    
    with st.spinner("Fetching Market Data... (This takes 10-20 seconds)"):
        
        for config in SCAN_CONFIGS:
            label = config['label']
            st.subheader(f"⏱️ Timeframe: {label}")
            
            # Batch Download
            try:
                data = yf.download(ALL_TICKERS, period=config['period'], interval=config['interval'], group_by='ticker', progress=False, threads=True)
            except Exception as e:
                st.error(f"Data Error: {e}")
                continue

            # Process
            cols = st.columns(3) # Grid layout for charts
            col_idx = 0
            found_any = False
            
            for ticker in ALL_TICKERS:
                try:
                    if len(ALL_TICKERS) > 1: df = data[ticker].dropna()
                    else: df = data.dropna()
                    if df.empty: continue
                    if config['resample']: df = resample_data(df, config['resample'])

                    match = analyze_ticker(df)
                    
                    if match:
                        found_any = True
                        with cols[col_idx % 3]:
                            st.success(f"**{ticker}** (Coil: {match['coil_width']*100:.2f}%)")
                            fig = plot_triangle(df, ticker, match)
                            st.plotly_chart(fig, use_container_width=True)
                            col_idx += 1
                except: continue
            
            if not found_any:
                st.caption("No strict patterns found in this timeframe.")
            
            st.divider()