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
from datetime import datetime, timedelta, timezone

APP_PASSWORD = "JaiBabaKi" 
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
ENABLE_TELEGRAM = False 

CRYPTO = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']
COMMODITIES = ['GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'CL=F', 'NG=F', 'BZ=F']
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
    if len(df) < 50: return None
    is_online, last_time = check_market_status(df)
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
    y_vals_upper = data_dict['slopes']['upper'] * x_indices[data_dict['pivots']['Ax']:] + data_dict['intercepts']['upper']
    y_vals_lower = data_dict['slopes']['lower'] * x_indices[data_dict['pivots']['Bx']:] + data_dict['intercepts']['lower']

    def get_slice_dates(start_idx):
        eff_start = max(start_idx, start_view_idx)
        return df_slice['date_str'][eff_start - start_view_idx:].tolist()
    
    fig.add_trace(go.Scatter(x=get_slice_dates(data_dict['pivots']['Ax']), y=y_vals_upper[max(0, start_view_idx - data_dict['pivots']['Ax']):], mode='lines', name='Res', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=get_slice_dates(data_dict['pivots']['Bx']), y=y_vals_lower[max(0, start_view_idx - data_dict['pivots']['Bx']):], mode='lines', name='Sup', line=dict(color='green', width=2)))

    fig.update_layout(title=f"{ticker} ({data_dict['coil_width']*100:.2f}%)", xaxis_rangeslider_visible=False, xaxis_type='category', height=450, margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(tickangle=-45, nticks=15))
    return fig

def send_telegram_alert(message):
    if not ENABLE_TELEGRAM: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})
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
                        df = data[ticker].dropna() if len(ALL_TICKERS) > 1 else data.dropna()
                        if df.empty: continue
                        if config['resample']: df = resample_data(df, config['resample'])
                        match = analyze_ticker(df)
                        if match and (match['is_online'] or config['label'] == '4h'):
                            status_icon = "🟢" if match['is_online'] else "🔴"
                            send_telegram_alert(f"{status_icon} {ticker} ({config['label']}) Alert!\nPrice: {match['price']:.2f}\nCoil: {match['coil_width']*100:.1f}%")
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

st.set_page_config(page_title="Triangle Pro", layout="wide")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔒 Triangle Hunter Pro 2.0")
        with st.form("login_form"):
            password = st.text_input("Enter Access Code", type="password")
            if st.form_submit_button("Unlock Dashboard", type="primary"):
                if password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect Access Code")
else:
    st.title("🔻 Triangle Hunter Pro")
    c1, c2 = st.columns([4, 1])
    with c1: st.caption(f"✅ Active | Monitoring {len(ALL_TICKERS)} Assets")
    with c2: 
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()

    tabs = st.tabs(["⚡ 5 Min", "⏱️ 15 Min", "hourly 1 Hour", "📅 4 Hour"])
    for i, config in enumerate(SCAN_CONFIGS):
        with tabs[i]:
            if st.button(f"Start {config['label']} Scan", key=f"btn_{i}", type="primary"):
                with st.spinner("Analyzing Market Geometry..."):
                    try:
                        data = yf.download(ALL_TICKERS, period=config['period'], interval=config['interval'], group_by='ticker', progress=False, threads=True)
                        res = {"on_s": [], "on_r": [], "off_s": [], "off_r": []}
                        for ticker in ALL_TICKERS:
                            try:
                                df = data[ticker].dropna() if len(ALL_TICKERS) > 1 else data.dropna()
                                if df.empty: continue
                                if config['resample']: df = resample_data(df, config['resample'])
                                match = analyze_ticker(df)
                                if match:
                                    item = {"ticker": ticker, "data": match, "fig": plot_triangle_clean(df, ticker, match, config['label'])}
                                    is_stock = ticker not in NON_STOCK_ASSETS
                                    if match['is_online']:
                                        res["on_s"].append(item) if is_stock else res["on_r"].append(item)
                                    else:
                                        res["off_s"].append(item) if is_stock else res["off_r"].append(item)
                            except: continue

                        st.markdown("### 🟢 Live Markets")
                        for k, v in [("#### 🏢 Stocks", res["on_s"]), ("#### 🪙 Crypto & Commodities", res["on_r"])]:
                            if v:
                                st.markdown(k)
                                cols = st.columns(3)
                                for idx, item in enumerate(v):
                                    with cols[idx % 3]:
                                        st.success(f"**{item['ticker']}** | {item['data']['last_time']}")
                                        st.plotly_chart(item['fig'], use_container_width=True)
                        
                        st.divider()
                        with st.expander(f"🔴 Offline Markets - Found {len(res['off_s']) + len(res['off_r'])}"):
                            for k, v in [("#### 🏢 Stocks", res["off_s"]), ("#### 🪙 Crypto & Commodities", res["off_r"])]:
                                if v:
                                    st.markdown(k)
                                    cols = st.columns(3)
                                    for idx, item in enumerate(v):
                                        with cols[idx % 3]:
                                            st.warning(f"**{item['ticker']}** | {item['data']['last_time']}")
                                            st.plotly_chart(item['fig'], use_container_width=True)
                    except Exception as e: st.error(f"Data Error: {e}")

