import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import time
import threading
import requests
import schedule

# ==========================================
# 1. CONFIGURATION & SECRETS
# ==========================================

# 🔐 SECURITY SETTINGS
APP_PASSWORD = "JaiBabaKi"  # <--- CHANGE THIS PASSWORD

# 📱 TELEGRAM SETTINGS
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
ENABLE_TELEGRAM = False 

# ASSET LISTS
NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 'SBIN.NS', 
    'INFY.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 
    'HCLTECH.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ADANIENT.NS', 'TATASTEEL.NS', 
    'KOTAKBANK.NS', 'NTPC.NS', 'AXISBANK.NS', 'POWERGRID.NS', 'M&M.NS', 
    'ULTRACEMCO.NS', 'ONGC.NS', 'COALINDIA.NS', 'WIPRO.NS', 'BAJAJFINSV.NS', 
    'NESTLEIND.NS', 'ADANIPORTS.NS', 'JSWSTEEL.NS', 'GRASIM.NS', 'CIPLA.NS', 
    'HINDALCO.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'TECHM.NS', 'SBILIFE.NS', 
    'BRITANNIA.NS', 'BPCL.NS', 'HEROMOTOCO.NS', 'DIVISLAB.NS', 'TATACONSUM.NS', 
    'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS', 'LTIM.NS', 'INDUSINDBK.NS'
]
METALS = ['GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'AA', 'FCX', 'SCCO']
SP_TOP_50 = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO', 
    'JPM', 'V', 'UNH', 'XOM', 'MA', 'JNJ', 'HD', 'PG', 'COST', 'ABBV', 'MRK', 
    'CRM', 'AMD', 'CVX', 'NFLX', 'WMT', 'ACN', 'BAC', 'PEP', 'KO', 'DIS', 'CSCO'
]
ALL_TICKERS = NIFTY_50 + METALS + SP_TOP_50

SCAN_CONFIGS = [
    {"label": "5m",  "interval": "5m",  "period": "5d",   "resample": None},
    {"label": "15m", "interval": "15m", "period": "15d",  "resample": None},
    {"label": "1h",  "interval": "1h",  "period": "60d",  "resample": None},
    {"label": "4h",  "interval": "1h",  "period": "300d", "resample": "4h"}, 
]

# ==========================================
# 2. CORE LOGIC
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
    tolerance = 0.003 
    
    if mode == "upper":
        violations = actual_values > (line_values * (1 + tolerance))
    else:
        violations = actual_values < (line_values * (1 - tolerance))
    
    return not np.any(violations)

def analyze_ticker(df):
    if len(df) < 50: return None
    high_idxs, low_idxs = get_pivots(df['High'], order=8)
    if len(high_idxs) < 2 or len(low_idxs) < 2: return None

    Ax, Cx = high_idxs[-2], high_idxs[-1]
    Ay, Cy = df['High'].iloc[Ax], df['High'].iloc[Cx]
    Bx, Dx = low_idxs[-2], low_idxs[-1]
    By, Dy = df['Low'].iloc[Bx], df['Low'].iloc[Dx]

    if not (Ay > Cy and By < Dy): return None

    slope_upper = (Cy - Ay) / (Cx - Ax)
    intercept_upper = Ay - (slope_upper * Ax)
    slope_lower = (Dy - By) / (Dx - Bx)
    intercept_lower = By - (slope_lower * Bx)

    if not check_line_integrity(df['High'], Ax, Cx, slope_upper, intercept_upper, "upper"): return None
    if not check_line_integrity(df['Low'], Bx, Dx, slope_lower, intercept_lower, "lower"): return None

    current_idx = len(df) - 1
    proj_upper = (slope_upper * current_idx) + intercept_upper
    proj_lower = (slope_lower * current_idx) + intercept_lower
    current_price = df['Close'].iloc[-1]
    
    if not (proj_lower < current_price < proj_upper): return None
    
    width_pct = (proj_upper - proj_lower) / current_price
    
    if width_pct < 0.035:
        return {
            "pivots": {"Ax": Ax, "Ay": Ay, "Cx": Cx, "Cy": Cy, "Bx": Bx, "By": By, "Dx": Dx, "Dy": Dy},
            "slopes": {"upper": slope_upper, "lower": slope_lower},
            "intercepts": {"upper": intercept_upper, "lower": intercept_lower},
            "coil_width": width_pct,
            "price": current_price
        }
    return None

def resample_data(df, interval):
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    return df.resample(interval).agg(logic).dropna()

# ==========================================
# 3. ADVANCED CHARTING
# ==========================================

def plot_triangle_clean(df, ticker, data_dict, interval_label):
    # 1. Zoom Logic: Show pattern start minus 20 bars context
    start_view_idx = max(0, min(data_dict['pivots']['Ax'], data_dict['pivots']['Bx']) - 25)
    df_slice = df.iloc[start_view_idx:].copy()
    
    # 2. X-Axis Formatting
    # For 5m/15m charts, we want "Day Hour:Minute" (e.g., "12 14:30")
    # For 1h/4h charts, we want "Month Day" (e.g., "Feb 12")
    if interval_label in ["5m", "15m"]:
        date_format = "%d %H:%M" # Day + Time
    else:
        date_format = "%b %d"    # Month + Day

    # Convert index to string for "Category" mode (removes gaps)
    df_slice['date_str'] = df_slice.index.strftime(date_format)

    fig = go.Figure(data=[go.Candlestick(
        x=df_slice['date_str'], 
        open=df_slice['Open'], high=df_slice['High'],
        low=df_slice['Low'], close=df_slice['Close'], 
        name=ticker
    )])

    # 3. Line Logic
    x_indices = np.arange(len(df))
    
    # Upper Line
    slope_u = data_dict['slopes']['upper']
    int_u = data_dict['intercepts']['upper']
    line_start_u = data_dict['pivots']['Ax']
    y_vals_upper = slope_u * x_indices[line_start_u:] + int_u
    
    # Lower Line
    slope_l = data_dict['slopes']['lower']
    int_l = data_dict['intercepts']['lower']
    line_start_l = data_dict['pivots']['Bx']
    y_vals_lower = slope_l * x_indices[line_start_l:] + int_l

    # Map indices to the sliced string dates
    # We must offset the indices by 'start_view_idx' to align with the new sliced X-axis
    def get_slice_dates(start_idx):
        # The line starts at absolute index 'start_idx'
        # If start_idx is BEFORE our zoom view, we clip it
        eff_start = max(start_idx, start_view_idx)
        # Get the corresponding dates from the slice
        return df_slice['date_str'][eff_start - start_view_idx:].tolist()
    
    # Get Y-values corresponding to the clipped range
    y_u_clipped = y_vals_upper[max(0, start_view_idx - line_start_u):]
    y_l_clipped = y_vals_lower[max(0, start_view_idx - line_start_l):]

    fig.add_trace(go.Scatter(
        x=get_slice_dates(line_start_u), y=y_u_clipped, 
        mode='lines', name='Res', line=dict(color='red', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=get_slice_dates(line_start_l), y=y_l_clipped, 
        mode='lines', name='Sup', line=dict(color='green', width=2)
    ))

    fig.update_layout(
        title=f"{ticker} (Coil: {data_dict['coil_width']*100:.2f}%)",
        xaxis_rangeslider_visible=False,
        xaxis_type='category', 
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(tickangle=-45, nticks=10) # Angled labels for readability
    )
    return fig

def send_telegram_alert(message):
    if not ENABLE_TELEGRAM: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try: requests.post(url, json=payload)
    except: pass

# ==========================================
# 4. BACKGROUND WORKER
# ==========================================

@st.cache_resource
class BackgroundScanner:
    def __init__(self):
        self.running = False
        self.thread = None
        
    def scan_job(self):
        print("⏰ Auto-Scan Triggered...")
        for config in SCAN_CONFIGS:
            try:
                data = yf.download(ALL_TICKERS, period=config['period'], interval=config['interval'], group_by='ticker', progress=False, threads=True)
                for ticker in ALL_TICKERS:
                    try:
                        if len(ALL_TICKERS) > 1: df = data[ticker].dropna()
                        else: df = data.dropna()
                        if df.empty: continue
                        if config['resample']: df = resample_data(df, config['resample'])
                        
                        match = analyze_ticker(df)
                        if match:
                            msg = f"🚀 {ticker} ({config['label']}) Alert!\nPrice: {match['price']:.2f}\nCoil: {match['coil_width']*100:.1f}%"
                            print(msg)
                            send_telegram_alert(msg)
                    except: continue
            except: continue

    def start(self):
        if not self.running:
            self.running = True
            def loop():
                schedule.every(15).minutes.do(self.scan_job)
                while True:
                    schedule.run_pending()
                    time.sleep(1)
            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()

scanner = BackgroundScanner()
scanner.start()

# ==========================================
# 5. UI & AUTHENTICATION (FIXED)
# ==========================================

st.set_page_config(page_title="Triangle Pro", layout="wide")

# Initialize Session State
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- LOGIN SCREEN (FIXED) ---
if not st.session_state.authenticated:
    st.title("🔒 Restricted Access")
    st.markdown("Please enter the secure access code below.")
    
    # Using a FORM prevents the "disappearing input" glitch
    with st.form("login_form"):
        password = st.text_input("Access Code", type="password")
        submit = st.form_submit_button("Unlock Dashboard")
        
        if submit:
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun() # Force reload to show the main app
            else:
                st.error("❌ Incorrect Password")

else:
    # --- MAIN DASHBOARD ---
    st.title("🔻 Triangle Hunter Pro")
    
    # Control Bar
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Status: System Active | Monitoring {len(ALL_TICKERS)} Assets")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()

    # Timeframe Tabs
    tabs = st.tabs(["5 Min", "15 Min", "1 Hour", "4 Hour"])

    for i, config in enumerate(SCAN_CONFIGS):
        with tabs[i]:
            if st.button(f"🔎 Scan {config['label']} Market", key=f"btn_{i}"):
                
                results_container = st.container()
                
                with st.spinner("Analyzing Market Geometry..."):
                    try:
                        data = yf.download(ALL_TICKERS, period=config['period'], interval=config['interval'], group_by='ticker', progress=False, threads=True)
                        
                        cols = st.columns(3)
                        c_idx = 0
                        found = False
                        
                        for ticker in ALL_TICKERS:
                            try:
                                if len(ALL_TICKERS) > 1: df = data[ticker].dropna()
                                else: df = data.dropna()
                                if df.empty: continue
                                if config['resample']: df = resample_data(df, config['resample'])

                                match = analyze_ticker(df)
                                if match:
                                    found = True
                                    with cols[c_idx % 3]:
                                        # Pass the label (e.g. "5m") to clean up the chart
                                        fig = plot_triangle_clean(df, ticker, match, config['label'])
                                        st.plotly_chart(fig, use_container_width=True)
                                        c_idx += 1
                            except: continue
                        
                        if not found: st.info("No tight patterns found right now.")
                            
                    except Exception as e:
                        st.error(f"Data Error: {e}")

