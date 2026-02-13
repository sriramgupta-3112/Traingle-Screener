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

APP_PASSWORD = "JaiBabaKi" 
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
ENABLE_TELEGRAM = False 

CRYPTO = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']

COMMODITIES = [
    'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'CL=F', 'NG=F', 'BZ=F'
]

LIQUID_FNO = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 
    'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS', 'CANBK.NS', 'AUBANK.NS', 
    'IDFCFIRSTB.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'CHOLAFIN.NS', 
    'SHRIRAMFIN.NS', 'RECLTD.NS', 'PFC.NS', 'SBICARD.NS', 'MUTHOOTFIN.NS',
    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS', 'LTIM.NS', 
    'COFORGE.NS', 'PERSISTENT.NS', 'MPHASIS.NS',
    'TATAMOTORS.NS', 'MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 
    'HEROMOTOCO.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BHARATFORG.NS', 
    'RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'COALINDIA.NS', 
    'BPCL.NS', 'IOC.NS', 'TATAPOWER.NS', 'ADANIGREEN.NS', 'ADANIENT.NS', 
    'ADANIPORTS.NS', 'GAIL.NS',
    'TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'NMDC.NS', 
    'SAIL.NS', 'JINDALSTEL.NS', 'NATIONALUM.NS',
    'ITC.NS', 'HINDUNILVR.NS', 'TITAN.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS', 
    'BRITANNIA.NS', 'GODREJCP.NS', 'TATACONSUM.NS', 'DABUR.NS', 'SUNPHARMA.NS', 
    'CIPLA.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS', 'LUPIN.NS', 
    'DLF.NS', 'GODREJPROP.NS',
    'HAL.NS', 'BEL.NS', 'MAZDOCK.NS', 'BHEL.NS', 'ZOMATO.NS', 'TRENT.NS', 
    'IRCTC.NS', 'INDIGO.NS', 'JIOFIN.NS', 'ABBOTINDIA.NS', 'SIEMENS.NS', 
    'ABB.NS', 'POLYCAB.NS', 'HAVELLS.NS', 'VOLTAS.NS'
]

SP_LIQUID_FNO = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'AVGO', 
    'QCOM', 'INTC', 'MU', 'TXN', 'AMAT', 'LRCX', 'ADI', 'SMCI', 'ARM', 'TSM',
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 'PYPL', 'COIN', 'HOOD',
    'CRM', 'ADBE', 'ORCL', 'IBM', 'NOW', 'PANW', 'PLTR', 'SNOW', 'CRWD', 'SQ', 'SHOP', 'UBER', 'ABNB',
    'NFLX', 'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'WMT', 'COST', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 
    'LULU', 'CMG', 'BKNG', 'MAR', 'LLY', 'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'AMGN', 'GILD', 'ISRG', 'CVS',
    'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'GE', 'CAT', 'BA', 'LMT', 'RTX', 'HON', 'UPS', 'UNP', 'DE', 
    'KO', 'PEP', 'PG', 'PM', 'MO', 'CL'
]

ALL_TICKERS = CRYPTO + COMMODITIES + LIQUID_FNO + SP_LIQUID_FNO
NON_STOCK_ASSETS = set(CRYPTO + COMMODITIES)

SCAN_CONFIGS = [
    {"label": "5m",  "interval": "5m",  "period": "5d",   "resample": None},
    {"label": "15m", "interval": "15m", "period": "15d",  "resample": None},
    {"label": "1h",  "interval": "1h",  "period": "60d",  "resample": None},
    {"label": "4h",  "interval": "1h",  "period": "300d", "resample": "4h"}, 
]

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
    tolerance = 0.005 
    
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

    # === ACCURACY FILTER 1: APEX CONVERGENCE ===
    if abs(slope_upper - slope_lower) < 1e-5: return None 
    x_apex = (intercept_lower - intercept_upper) / (slope_upper - slope_lower)
    current_idx = len(df) - 1
    
    if x_apex < current_idx: return None 
    pattern_len = max(Cx, Dx) - min(Ax, Bx)
    if x_apex > current_idx + (pattern_len * 2): return None 

    if not check_line_integrity(df['High'], Ax, Cx, slope_upper, intercept_upper, "upper"): return None
    if not check_line_integrity(df['Low'], Bx, Dx, slope_lower, intercept_lower, "lower"): return None

    # === ACCURACY FILTER 2: VOLUME EXHAUSTION ===
    try:
        start_idx = min(Ax, Bx)
        vol_pattern = df['Volume'].iloc[start_idx:].mean()
        vol_pre = df['Volume'].iloc[max(0, start_idx-pattern_len):start_idx].mean()
        if vol_pattern > (vol_pre * 1.5): return None 
    except: pass

    proj_upper = (slope_upper * current_idx) + intercept_upper
    proj_lower = (slope_lower * current_idx) + intercept_lower
    current_price = df['Close'].iloc[-1]
    
    if not (proj_lower < current_price < proj_upper): return None
    
    width_pct = (proj_upper - proj_lower) / current_price
    
    if width_pct < 0.035:
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
    
    if interval_label in ["5m", "15m"]:
        date_format = "%d %H:%M" 
    else:
        date_format = "%b %d"    

    df_slice['date_str'] = df_slice.index.strftime(date_format)

    fig = go.Figure(data=[go.Candlestick(
        x=df_slice['date_str'], 
        open=df_slice['Open'], high=df_slice['High'],
        low=df_slice['Low'], close=df_slice['Close'], 
        name=ticker
    )])

    x_indices = np.arange(len(df))
    slope_u = data_dict['slopes']['upper']
    int_u = data_dict['intercepts']['upper']
    line_start_u = data_dict['pivots']['Ax']
    y_vals_upper = slope_u * x_indices + int_u
    
    slope_l = data_dict['slopes']['lower']
    int_l = data_dict['intercepts']['lower']
    line_start_l = data_dict['pivots']['Bx']
    y_vals_lower = slope_l * x_indices + int_l

    def get_slice_vals(idx_start, y_vals):
        valid_indices = [i for i in range(idx_start, len(df)) if i >= start_view_idx]
        x_plot = [df_slice['date_str'].iloc[i - start_view_idx] for i in valid_indices]
        y_plot = y_vals[valid_indices]
        return x_plot, y_plot

    xu, yu = get_slice_vals(line_start_u, y_vals_upper)
    xl, yl = get_slice_vals(line_start_l, y_vals_lower)

    fig.add_trace(go.Scatter(x=xu, y=yu, mode='lines', name='Res', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=xl, y=yl, mode='lines', name='Sup', line=dict(color='green', width=2)))

    fig.update_layout(
        title=f"{ticker} (Coil: {data_dict['coil_width']*100:.2f}%)",
        xaxis_rangeslider_visible=False,
        xaxis_type='category', 
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(tickangle=-45, nticks=15) 
    )
    return fig

def send_telegram_alert(message):
    if not ENABLE_TELEGRAM: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try: requests.post(url, json=payload)
    except: pass

@st.cache_resource
class BackgroundScanner:
    def __init__(self):
        self.running = False
        self.thread = None
        
    def scan_job(self):
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
                            if match['is_online'] or config['label'] == '4h':
                                status_icon = "🟢" if match['is_online'] else "🔴"
                                msg = f"{status_icon} {ticker} ({config['label']}) Alert!\nPrice: {match['price']:.2f}\nCoil: {match['coil_width']*100:.1f}%"
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

st.set_page_config(page_title="Triangle Finder Pro 2.0", layout="wide")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔒 Triangle Finder Pro 2.0")
        with st.form("login_form"):
            password = st.text_input("Enter Access Code", type="password")
            submit = st.form_submit_button("Unlock Dashboard", type="primary")
            if submit:
                if password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect Access Code")

else:
    st.title("🔻 Triangle Finder Pro 2.0")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(f"✅ System Active | Monitoring {len(ALL_TICKERS)} Assets | Apex & Volume Filters Enabled")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()

    tabs = st.tabs(["⚡ 5 Min", "⏱️ 15 Min", "hourly 1 Hour", "📅 4 Hour"])

    def process_single_ticker(ticker, data_source, config):
        try:
            if len(ALL_TICKERS) > 1: df = data_source[ticker].dropna()
            else: df = data_source.dropna()
            
            if df.empty: return None
            if config['resample']: df = resample_data(df, config['resample'])

            match = analyze_ticker(df)
            if match:
                fig = plot_triangle_clean(df, ticker, match, config['label'])
                item = {"ticker": ticker, "data": match, "fig": fig}
                
                is_stock = ticker not in NON_STOCK_ASSETS
                is_online = match['is_online']
                return (is_online, is_stock, item)
            return None
        except:
            return None

    for i, config in enumerate(SCAN_CONFIGS):
        with tabs[i]:
            if st.button(f"Start {config['label']} Scan", key=f"btn_{i}", type="primary"):
                
                with st.spinner(f"Acquiring Data & Calculating Geometry ({config['label']})..."):
                    try:
                        data = yf.download(ALL_TICKERS, period=config['period'], interval=config['interval'], group_by='ticker', progress=False, threads=True)
                        
                        online_stocks = []
                        online_rest = []
                        offline_stocks = []
                        offline_rest = []

                        # PARALLEL PROCESSING IMPLEMENTATION
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            futures = {executor.submit(process_single_ticker, ticker, data, config): ticker for ticker in ALL_TICKERS}
                            
                            for future in concurrent.futures.as_completed(futures):
                                result = future.result()
                                if result:
                                    is_online, is_stock, item = result
                                    if is_online:
                                        if is_stock: online_stocks.append(item)
                                        else: online_rest.append(item)
                                    else:
                                        if is_stock: offline_stocks.append(item)
                                        else: offline_rest.append(item)

                        # RENDER LIVE MARKETS
                        st.markdown("### 🟢 Live Markets (Trading Now)")
                        if online_stocks or online_rest:
                            if online_stocks:
                                st.markdown("#### 🏢 Stocks")
                                cols = st.columns(3)
                                for idx, item in enumerate(online_stocks):
                                    with cols[idx % 3]:
                                        st.success(f"**{item['ticker']}** | Live @ {item['data']['last_time']}")
                                        st.plotly_chart(item['fig'], use_container_width=True)
                            
                            if online_rest:
                                st.markdown("#### 🪙 Crypto & Commodities")
                                cols = st.columns(3)
                                for idx, item in enumerate(online_rest):
                                    with cols[idx % 3]:
                                        st.success(f"**{item['ticker']}** | Live @ {item['data']['last_time']}")
                                        st.plotly_chart(item['fig'], use_container_width=True)
                        else:
                            st.info("No patterns found in currently open markets.")
                        
                        st.divider()

                        # RENDER OFFLINE MARKETS
                        with st.expander(f"🔴 Offline Markets (Closed) - Found {len(offline_stocks) + len(offline_rest)}"):
                            if offline_stocks:
                                st.markdown("#### 🏢 Stocks")
                                cols = st.columns(3)
                                for idx, item in enumerate(offline_stocks):
                                    with cols[idx % 3]:
                                        st.warning(f"**{item['ticker']}** | Closed @ {item['data']['last_time']}")
                                        st.plotly_chart(item['fig'], use_container_width=True)
                            
                            if offline_rest:
                                st.markdown("#### 🪙 Crypto & Commodities")
                                cols = st.columns(3)
                                for idx, item in enumerate(offline_rest):
                                    with cols[idx % 3]:
                                        st.warning(f"**{item['ticker']}** | Closed @ {item['data']['last_time']}")
                                        st.plotly_chart(item['fig'], use_container_width=True)
                            
                            if not offline_stocks and not offline_rest:
                                st.caption("No patterns found in closed markets.")
                    
                    except Exception as e:
                        st.error(f"Data Error: {e}")
