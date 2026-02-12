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
APP_PASSWORD = "trading-god-mode"  # <--- CHANGE THIS PASSWORD

# 📱 TELEGRAM NOTIFICATION SETTINGS (Optional)
# 1. Search for "@BotFather" on Telegram -> /newbot -> Get Token
# 2. Search for "@userinfobot" -> Get your numeric ID
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
ENABLE_TELEGRAM = False  # Set to True after filling above

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
# 2. CORE PATTERN LOGIC
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
    tolerance = 0.003 # Increased slightly to forgive tiny wicks
    
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
# 3. CHARTING & NOTIFICATIONS
# ==========================================

def plot_triangle_clean(df, ticker, data_dict):
    """ 
    FIXED: Uses 'category' axis type to remove overnight gaps 
    """
    # Slice the dataframe to show only the relevant history (start of pattern - 20 bars)
    start_view_idx = max(0, min(data_dict['pivots']['Ax'], data_dict['pivots']['Bx']) - 20)
    df_slice = df.iloc[start_view_idx:]
    
    # We must re-calculate indices for the slice (since index 0 is now start_view_idx)
    def adj(idx): return idx - start_view_idx
    
    fig = go.Figure(data=[go.Candlestick(
        x=df_slice.index.strftime("%Y-%m-%d %H:%M"), # Convert to string for Category axis
        open=df_slice['Open'], high=df_slice['High'],
        low=df_slice['Low'], close=df_slice['Close'], 
        name=ticker
    )])

    # Generate Line Points
    x_indices = np.arange(len(df)) # Original indices
    
    # Upper Line
    slope_u = data_dict['slopes']['upper']
    int_u = data_dict['intercepts']['upper']
    line_start_u = data_dict['pivots']['Ax']
    # Calculate Y values for the whole range, then slice
    y_vals_upper = slope_u * x_indices[line_start_u:] + int_u
    
    # Lower Line
    slope_l = data_dict['slopes']['lower']
    int_l = data_dict['intercepts']['lower']
    line_start_l = data_dict['pivots']['Bx']
    y_vals_lower = slope_l * x_indices[line_start_l:] + int_l

    # Add Traces (Using formatted date strings for X)
    # We need to map the integer indices back to the string dates
    date_map = df.index.strftime("%Y-%m-%d %H:%M").tolist()
    
    # Safely get dates for lines
    x_dates_upper = [date_map[i] for i in range(line_start_u, len(df))]
    x_dates_lower = [date_map[i] for i in range(line_start_l, len(df))]

    fig.add_trace(go.Scatter(x=x_dates_upper, y=y_vals_upper, mode='lines', name='Resistance', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=x_dates_lower, y=y_vals_lower, mode='lines', name='Support', line=dict(color='green', width=2)))

    fig.update_layout(
        title=f"{ticker} (Coil: {data_dict['coil_width']*100:.2f}%)",
        xaxis_rangeslider_visible=False,
        xaxis_type='category', # <--- THIS FIXES THE JUNK CHART ISSUE
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def send_telegram_alert(message):
    if not ENABLE_TELEGRAM: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try: requests.post(url, json=payload)
    except: pass

# ==========================================
# 4. BACKGROUND WORKER (AUTO-RUNNER)
# ==========================================

# This class caches the background thread so it survives page reloads
@st.cache_resource
class BackgroundScanner:
    def __init__(self):
        self.running = False
        self.thread = None
        
    def scan_job(self):
        print("⏰ Auto-Scan Started...")
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
                            msg = f"🚀 ALERT [{config['label']}]\n{ticker} is Coiling!\nPrice: {match['price']:.2f}\nTightness: {match['coil_width']*100:.1f}%"
                            print(msg)
                            send_telegram_alert(msg)
                    except: continue
            except: continue
        print("✅ Auto-Scan Finished.")

    def start(self):
        if not self.running:
            self.running = True
            
            def loop():
                # Run immediately once
                self.scan_job()
                # Then schedule every 15 mins
                schedule.every(15).minutes.do(self.scan_job)
                while True:
                    schedule.run_pending()
                    time.sleep(1)
            
            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()

# Start the background scanner
scanner = BackgroundScanner()
scanner.start()

# ==========================================
# 5. STREAMLIT FRONTEND
# ==========================================

st.set_page_config(page_title="Triangle Pro", layout="wide")

# --- AUTHENTICATION ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    if st.session_state.password_input == APP_PASSWORD:
        st.session_state.authenticated = True
        del st.session_state.password_input # Don't store password
    else:
        st.error("❌ Access Denied")

if not st.session_state.authenticated:
    st.title("🔒 Restricted Access")
    st.text_input("Enter Access Code:", type="password", key="password_input", on_change=check_password)
else:
    # --- MAIN APP INTERFACE ---
    st.title("🔻 Triangle Hunter Pro")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Status:** System Running | Scanned **{len(ALL_TICKERS)}** Assets")
    with col2:
        if st.button("🔄 Manual Rescan"):
            st.rerun()

    tabs = st.tabs(["5 Min", "15 Min", "1 Hour", "4 Hour"])

    # Loop through tabs and configs
    for i, config in enumerate(SCAN_CONFIGS):
        with tabs[i]:
            if st.button(f"Scan {config['label']} Market", key=f"btn_{i}"):
                with st.spinner("Analyzing Charts..."):
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
                                        fig = plot_triangle_clean(df, ticker, match)
                                        st.plotly_chart(fig, use_container_width=True)
                                        c_idx += 1
                            except: continue
                        
                        if not found: st.info("No patterns found right now.")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
