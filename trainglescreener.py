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
import concurrent.futures
from datetime import datetime, timedelta, timezone

# ==========================================
# 1. CONFIGURATION
# ==========================================
APP_PASSWORD = "JaiBabaKi"
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
ENABLE_TELEGRAM = False

# --- ASSET GROUPS ---
CRYPTO = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD']
COMMODITIES = ['GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'CL=F', 'NG=F', 'BZ=F']
LIQUID_FNO = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS',
    'SBILIFE.NS', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'LT.NS',
    'TATAMOTORS.NS', 'MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS',
    'SUNPHARMA.NS', 'CIPLA.NS', 'DRREDDY.NS', 'ITC.NS', 'HINDUNILVR.NS',
    'TITAN.NS', 'ASIANPAINT.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
    'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS', 'CANBK.NS', 'AUBANK.NS',
    'IDFCFIRSTB.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'CHOLAFIN.NS',
    'SHRIRAMFIN.NS', 'RECLTD.NS', 'PFC.NS', 'SBICARD.NS', 'MUTHOOTFIN.NS',
    'WIPRO.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'MPHASIS.NS',
    'EICHERMOT.NS', 'HEROMOTOCO.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BHARATFORG.NS',
    'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'COALINDIA.NS', 'BPCL.NS', 'IOC.NS',
    'TATAPOWER.NS', 'ADANIGREEN.NS', 'GAIL.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS',
    'HINDALCO.NS', 'VEDL.NS', 'NMDC.NS', 'SAIL.NS', 'JINDALSTEL.NS', 'NATIONALUM.NS',
    'BRITANNIA.NS', 'GODREJCP.NS', 'TATACONSUM.NS', 'DABUR.NS', 'DIVISLAB.NS',
    'APOLLOHOSP.NS', 'LUPIN.NS', 'DLF.NS', 'GODREJPROP.NS', 'HAL.NS', 'BEL.NS',
    'MAZDOCK.NS', 'BHEL.NS', 'ZOMATO.NS', 'TRENT.NS', 'IRCTC.NS', 'INDIGO.NS',
    'JIOFIN.NS', 'ABBOTINDIA.NS', 'SIEMENS.NS', 'ABB.NS', 'POLYCAB.NS', 'HAVELLS.NS', 'VOLTAS.NS'
]
SP_LIQUID_FNO = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'NFLX',
    'JPM', 'BAC', 'WFC', 'V', 'MA', 'DIS', 'CSCO', 'XOM', 'CVX', 'PEP', 'KO',
    'AVGO', 'QCOM', 'INTC', 'MU', 'TXN', 'AMAT', 'LRCX', 'ADI', 'SMCI', 'ARM', 'TSM',
    'GS', 'MS', 'AXP', 'BLK', 'PYPL', 'COIN', 'HOOD', 'CRM', 'ADBE', 'ORCL', 'IBM',
    'NOW', 'PANW', 'PLTR', 'SNOW', 'CRWD', 'SQ', 'SHOP', 'UBER', 'ABNB', 'CMCSA',
    'TMUS', 'VZ', 'T', 'WMT', 'COST', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD',
    'LULU', 'CMG', 'BKNG', 'MAR', 'LLY', 'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',
    'AMGN', 'GILD', 'ISRG', 'CVS', 'COP', 'SLB', 'OXY', 'GE', 'CAT', 'BA', 'LMT',
    'RTX', 'HON', 'UPS', 'UNP', 'DE', 'PG', 'PM', 'MO', 'CL'
]

# Helper to classify for coloring
NON_STOCK_ASSETS = set(CRYPTO + COMMODITIES)

SCAN_CONFIGS = [
    {"label": "5m",  "interval": "5m",  "period": "3d",   "resample": None, "ttl": 300},
    {"label": "15m", "interval": "15m", "period": "10d",  "resample": None, "ttl": 900},
    {"label": "1h",  "interval": "1h",  "period": "40d",  "resample": None, "ttl": 3600},
    {"label": "4h",  "interval": "1h",  "period": "200d", "resample": "4h", "ttl": 14400},
]

# ==========================================
# 2. CACHED DATA FUNCTIONS
# ==========================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(tickers, period, interval):
    try:
        return yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=False, threads=True)
    except Exception as e:
        return None

# ==========================================
# 3. CORE LOGIC
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

def check_market_status(df):
    if df.empty: return False, "N/A"
    last_candle_time = df.index[-1]
    if last_candle_time.tzinfo is None:
        now = datetime.now()
    else:
        now = datetime.now(timezone.utc)
        last_candle_time = last_candle_time.astimezone(timezone.utc)
    diff = now - last_candle_time
    is_online = diff < timedelta(minutes=60)
    time_str = last_candle_time.strftime("%H:%M")
    return is_online, time_str

def analyze_ticker(df):
    if len(df) < 60: return None
    
    high_idxs, low_idxs = get_pivots(df['High'], order=8)
    if len(high_idxs) < 2 or len(low_idxs) < 2: return None

    Ax, Cx = high_idxs[-2], high_idxs[-1]
    Ay, Cy = df['High'].iloc[Ax], df['High'].iloc[Cx]
    Bx, Dx = low_idxs[-2], low_idxs[-1]
    By, Dy = df['Low'].iloc[Bx], df['Low'].iloc[Dx]

    slope_upper = (Cy - Ay) / (Cx - Ax)
    intercept_upper = Ay - (slope_upper * Ax)
    slope_lower = (Dy - By) / (Dx - Bx)
    intercept_lower = By - (slope_lower * Bx)

    # === SIDEWAYS ENFORCEMENT ===
    tolerance = 1e-4
    if slope_upper > tolerance and slope_lower > tolerance: return None # Rising Wedge
    if slope_upper < -tolerance and slope_lower < -tolerance: return None # Falling Wedge

    # PROPORTIONALITY
    width_upper = Cx - Ax
    width_lower = Dx - Bx
    if width_upper == 0 or width_lower == 0: return None
    ratio = min(width_upper, width_lower) / max(width_upper, width_lower)
    if ratio < 0.25: return None
    # === END ===

    if abs(slope_upper - slope_lower) < 1e-5: return None
    x_apex = (intercept_lower - intercept_upper) / (slope_upper - slope_lower)
    current_idx = len(df) - 1
    
    if x_apex < current_idx: return None
    pattern_len = max(Cx, Dx) - min(Ax, Bx)
    if x_apex > current_idx + (pattern_len * 3): return None

    if not check_line_integrity(df['High'], Ax, Cx, slope_upper, intercept_upper, "upper"): return None
    if not check_line_integrity(df['Low'], Bx, Dx, slope_lower, intercept_lower, "lower"): return None

    proj_upper = (slope_upper * current_idx) + intercept_upper
    proj_lower = (slope_lower * current_idx) + intercept_lower
    current_price = df['Close'].iloc[-1]
    
    if not (proj_lower * 0.99 < current_price < proj_upper * 1.01): return None

    width_pct = (proj_upper - proj_lower) / current_price
    
    if width_pct < 0.06:
        is_online, last_time = check_market_status(df)
        return {
            "pivots": {"Ax": Ax, "Ay": Ay, "Cx": Cx, "Cy": Cy, "Bx": Bx, "By": By, "Dx": Dx, "Dy": Dy},
            "slopes": {"upper": slope_upper, "lower": slope_lower},
            "intercepts": {"upper": intercept_upper, "lower": intercept_lower},
            "coil_width": width_pct,
            "price": current_price,
            "is_online": is_online,
            "last_time": last_time
        }
    return None

def resample_data(df, interval):
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    return df.resample(interval).agg(logic).dropna()

def plot_triangle_clean(df, ticker, data_dict, interval_label):
    pattern_start_idx = min(data_dict['pivots']['Ax'], data_dict['pivots']['Bx'])
    pattern_len = len(df) - pattern_start_idx
    history_buffer = int(pattern_len * 4) 
    start_view_idx = max(0, pattern_start_idx - history_buffer)
    df_slice = df.iloc[start_view_idx:].copy()
    
    date_format = "%d %H:%M" if interval_label in ["5m", "15m"] else "%b %d"
    df_slice['date_str'] = df_slice.index.strftime(date_format)

    fig = go.Figure(data=[go.Candlestick(
        x=df_slice['date_str'], 
        open=df_slice['Open'], high=df_slice['High'],
        low=df_slice['Low'], close=df_slice['Close'], 
        name=ticker
    )])

    x_indices = np.arange(len(df))
    y_vals_upper = data_dict['slopes']['upper'] * x_indices + data_dict['intercepts']['upper']
    y_vals_lower = data_dict['slopes']['lower'] * x_indices + data_dict['intercepts']['lower']

    def get_slice_vals(idx_start, y_vals):
        valid_indices = [i for i in range(idx_start, len(df)) if i >= start_view_idx]
        x_plot = [df_slice['date_str'].iloc[i - start_view_idx] for i in valid_indices]
        y_plot = y_vals[valid_indices]
        return x_plot, y_plot

    xu, yu = get_slice_vals(data_dict['pivots']['Ax'], y_vals_upper)
    xl, yl = get_slice_vals(data_dict['pivots']['Bx'], y_vals_lower)

    fig.add_trace(go.Scatter(x=xu, y=yu, mode='lines', name='Resistance', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=xl, y=yl, mode='lines', name='Support', line=dict(color='green', width=2)))

    fig.update_layout(
        title=f"{ticker} | Coil: {data_dict['coil_width']*100:.2f}%",
        xaxis_rangeslider_visible=False,
        xaxis_type='category', height=350, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(tickangle=-45, nticks=10)
    )
    return fig

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Triangle Pro 2.4", layout="wide")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔻 Triangle Pro 2.4")
        with st.form("login_form"):
            password = st.text_input("Enter Access Code", type="password")
            if st.form_submit_button("Unlock Dashboard", type="primary"):
                if password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("❌ Incorrect Access Code")
else:
    # --- SIDEBAR SELECTION ---
    with st.sidebar:
        st.title("⚙️ Scanner Settings")
        asset_choice = st.radio(
            "Select Market:",
            ("All Assets", "NSE F&O Stocks", "S&P 500 Stocks", "Crypto", "Commodities")
        )
        
        # LOGIC TO SET ACTIVE LIST
        if asset_choice == "All Assets":
            ACTIVE_TICKERS = CRYPTO + COMMODITIES + LIQUID_FNO + SP_LIQUID_FNO
        elif asset_choice == "NSE F&O Stocks":
            ACTIVE_TICKERS = LIQUID_FNO
        elif asset_choice == "S&P 500 Stocks":
            ACTIVE_TICKERS = SP_LIQUID_FNO
        elif asset_choice == "Crypto":
            ACTIVE_TICKERS = CRYPTO
        elif asset_choice == "Commodities":
            ACTIVE_TICKERS = COMMODITIES
            
        st.info(f"Scanning {len(ACTIVE_TICKERS)} Assets")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()

    st.title(f"🔻 Triangle Pro 2.4 ({asset_choice})")
    
    tabs = st.tabs(["⚡ 5 Min", "⏱️ 15 Min", "hourly 1 Hour", "📅 4 Hour"])

    def process_ticker(ticker, data_source, config):
        try:
            if len(ACTIVE_TICKERS) > 1: df = data_source[ticker].dropna()
            else: df = data_source.dropna()
            
            if df.empty: return None
            if config['resample']: df = resample_data(df, config['resample'])
            match = analyze_ticker(df)
            if match:
                fig = plot_triangle_clean(df, ticker, match, config['label'])
                item = {"ticker": ticker, "data": match, "fig": fig}
                is_stock = ticker not in NON_STOCK_ASSETS
                return (match['is_online'], is_stock, item)
            return None
        except: return None

    for i, config in enumerate(SCAN_CONFIGS):
        with tabs[i]:
            scan_key = f"scan_{config['label']}_{asset_choice}" # Unique key per asset class
            
            if st.button(f"Start {config['label']} Scan ({asset_choice})", key=f"btn_{i}", type="primary"):
                with st.spinner(f"Scanning {len(ACTIVE_TICKERS)} assets..."):
                    try:
                        data = fetch_market_data(ACTIVE_TICKERS, config['period'], config['interval'])
                        if data is None or data.empty:
                            st.error("No data received.")
                        else:
                            results = {"on_s": [], "on_r": [], "off_s": [], "off_r": []}
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                futures = {executor.submit(process_ticker, t, data, config): t for t in ACTIVE_TICKERS}
                                for future in concurrent.futures.as_completed(futures):
                                    res = future.result()
                                    if res:
                                        is_on, is_stk, item = res
                                        if is_on: results["on_s" if is_stk else "on_r"].append(item)
                                        else: results["off_s" if is_stk else "off_r"].append(item)
                            st.session_state.scan_results[scan_key] = results
                    except Exception as e: st.error(f"Error: {e}")

            if scan_key in st.session_state.scan_results:
                res = st.session_state.scan_results[scan_key]
                total_found = len(res["on_s"]) + len(res["on_r"]) + len(res["off_s"]) + len(res["off_r"])
                
                if total_found == 0:
                    st.info("Scan complete. No patterns found.")
                else:
                    st.markdown(f"### 🟢 Live Markets ({len(res['on_s']) + len(res['on_r'])})")
                    if res["on_s"] or res["on_r"]:
                        for k, v in [("#### 🏢 Stocks", res["on_s"]), ("#### 🪙 Crypto & Commodities", res["on_r"])]:
                            if v:
                                st.markdown(k)
                                cols = st.columns(3)
                                for idx, item in enumerate(v):
                                    with cols[idx % 3]:
                                        st.success(f"**{item['ticker']}** | {item['data']['last_time']}")
                                        st.plotly_chart(item['fig'], use_container_width=True)
                    st.divider()
                    with st.expander(f"🔴 Offline Markets ({len(res['off_s']) + len(res['off_r'])})"):
                        for k, v in [("#### 🏢 Stocks", res["off_s"]), ("#### 🪙 Crypto & Commodities", res["off_r"])]:
                            if v:
                                st.markdown(k)
                                cols = st.columns(3)
                                for idx, item in enumerate(v):
                                    with cols[idx % 3]:
                                        st.warning(f"**{item['ticker']}** | {item['data']['last_time']}")
                                        st.plotly_chart(item['fig'], use_container_width=True)
