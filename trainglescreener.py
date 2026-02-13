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

CRYPTO = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD']
COMMODITIES = ['GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'CL=F', 'NG=F', 'BZ=F']
LIQUID_FNO = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS',
    'SBILIFE.NS',
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'LT.NS',
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

ALL_TICKERS = CRYPTO + COMMODITIES + LIQUID_FNO + SP_LIQUID_FNO
NON_STOCK_ASSETS = set(CRYPTO + COMMODITIES)

SCAN_CONFIGS = [
    {"label": "5m",  "interval": "5m",  "period": "5d",   "resample": None},
    {"label": "15m", "interval": "15m", "period": "15d",  "resample": None},
    {"label": "1h",  "interval": "1h",  "period": "60d",  "resample": None},
    {"label": "4h",  "interval": "1h",  "period": "300d", "resample": "4h"},
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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

    if Cy > Ay * 1.015: return None
    if Dy < By * 0.985: return None

    slope_upper = (Cy - Ay) / (Cx - Ax)
    intercept_upper = Ay - (slope_upper * Ax)
    slope_lower = (Dy - By) / (Dx - Bx)
    intercept_lower = By - (slope_lower * Bx)

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

    pattern_start_idx = min(Ax, Bx)
    
    if pattern_start_idx < 20: 
        wave_label = "Insufficient Data"
    else:
        lookback = int(pattern_len * 1.5) 
        trend_start = max(0, pattern_start_idx - lookback)
        
        price_start_trend = df['Close'].iloc[trend_start]
        price_at_pattern = df['Close'].iloc[pattern_start_idx]
        
        trend_move_pct = (price_at_pattern - price_start_trend) / price_start_trend
        triangle_height_pct = (Ay - By) / By
        
        df['RSI'] = calculate_rsi(df['Close'])
        rsi_current = df['RSI'].iloc[-1]
        
        is_strong_trend = abs(trend_move_pct) > (triangle_height_pct * 1.5)
        is_rsi_neutral = 35 < rsi_current < 65
        
        if is_strong_trend and is_rsi_neutral:
            direction = "Bullish" if trend_move_pct > 0 else "Bearish"
            wave_label = f"🌊 Potential Wave 4 ({direction})"
        elif not is_strong_trend:
             wave_label = "⚠️ Potential Wave B / Neutral"
        else:
            wave_label = "Unknown Structure"

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
            "last_time": last_time,
            "wave_label": wave_label
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

    title_text = f"{ticker} | {data_dict['wave_label']} | Coil: {data_dict['coil_width']*100:.2f}%"
    fig.update_layout(
        title=title_text,
        xaxis_rangeslider_visible=False,
        xaxis_type='category', height=450, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(tickangle=-45, nticks=15)
    )
    return fig

def send_telegram_alert(message):
    if not ENABLE_TELEGRAM: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

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
                        if match:
                            if match['is_online'] or config['label'] == '4h':
                                status_icon = "🟢" if match['is_online'] else "🔴"
                                if "Wave 4" in match['wave_label']:
                                    msg = f"{status_icon} {ticker} ({config['label']}) ELLIOTT ALERT!\n{match['wave_label']}\nPrice: {match['price']:.2f}"
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

st.set_page_config(page_title="Triangle Finder Pro 3.0 (Elliott Wave)", layout="wide")

if 'authenticated' not in st.session_state: st.session_state.authenticated = False

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🌊 Triangle Pro 3.0")
        with st.form("login_form"):
            password = st.text_input("Enter Access Code", type="password")
            if st.form_submit_button("Unlock Dashboard", type="primary"):
                if password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("❌ Incorrect Access Code")
else:
    st.title("🌊 Triangle Finder Pro 3.0 (Elliott Wave Edition)")
    
    col1, col2 = st.columns([4, 1])
    with col1: st.caption(f"✅ Active | Monitoring {len(ALL_TICKERS)} Assets | Wave 4 Detection Enabled")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()

    tabs = st.tabs(["⚡ 5 Min", "⏱️ 15 Min", "hourly 1 Hour", "📅 4 Hour"])

    def process_ticker(ticker, data_source, config):
        try:
            df = data_source[ticker].dropna() if len(ALL_TICKERS) > 1 else data_source.dropna()
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
            if st.button(f"Start {config['label']} Scan", key=f"btn_{i}", type="primary"):
                with st.spinner(f"Analyzing Waves & Geometry ({config['label']})..."):
                    try:
                        data = yf.download(ALL_TICKERS, period=config['period'], interval=config['interval'], group_by='ticker', progress=False, threads=True)
                        results = {"on_s": [], "on_r": [], "off_s": [], "off_r": []}

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            futures = {executor.submit(process_ticker, t, data, config): t for t in ALL_TICKERS}
                            for future in concurrent.futures.as_completed(futures):
                                res = future.result()
                                if res:
                                    is_on, is_stk, item = res
                                    if is_on: results["on_s" if is_stk else "on_r"].append(item)
                                    else: results["off_s" if is_stk else "off_r"].append(item)

                        st.markdown("### 🟢 Live Markets")
                        if results["on_s"] or results["on_r"]:
                            for k, v in [("#### 🏢 Stocks", results["on_s"]), ("#### 🪙 Crypto & Commodities", results["on_r"])]:
                                if v:
                                    st.markdown(k)
                                    cols = st.columns(3)
                                    for idx, item in enumerate(v):
                                        with cols[idx % 3]:
                                            label = item['data']['wave_label']
                                            if "Wave 4" in label:
                                                st.success(f"**{item['ticker']}** | {label}")
                                            else:
                                                st.warning(f"**{item['ticker']}** | {label}")
                                            st.plotly_chart(item['fig'], use_container_width=True)
                        else: st.info("No patterns found in live markets.")
                        
                        st.divider()
                        
                        with st.expander(f"🔴 Offline Markets - Found {len(results['off_s']) + len(results['off_r'])}"):
                            for k, v in [("#### 🏢 Stocks", results["off_s"]), ("#### 🪙 Crypto & Commodities", results["off_r"])]:
                                if v:
                                    st.markdown(k)
                                    cols = st.columns(3)
                                    for idx, item in enumerate(v):
                                        with cols[idx % 3]:
                                            label = item['data']['wave_label']
                                            if "Wave 4" in label:
                                                st.success(f"**{item['ticker']}** | {label}")
                                            else:
                                                st.warning(f"**{item['ticker']}** | {label}")
                                            st.plotly_chart(item['fig'], use_container_width=True)

                    except Exception as e: st.error(f"Error: {e}")
