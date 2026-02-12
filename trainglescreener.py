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

# 📱 TELEGRAM SETTINGS (Optional)
# 1. Search "@BotFather" -> /newbot -> Get Token
# 2. Search "@userinfobot" -> Get ID
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
ENABLE_TELEGRAM = False  # Set to True to enable alerts

# --- 🌍 EXTENDED LIQUID ASSET LISTS ---

# 1. NIFTY F&O (High Option Volume)
LIQUID_FNO = [
    # BANKING & FINANCE
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 
    'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS', 'CANBK.NS', 'AUBANK.NS', 
    'IDFCFIRSTB.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'CHOLAFIN.NS', 
    'SHRIRAMFIN.NS', 'RECLTD.NS', 'PFC.NS', 'SBICARD.NS', 'MUTHOOTFIN.NS',

    # IT & TECH
    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS', 'LTIM.NS', 
    'COFORGE.NS', 'PERSISTENT.NS', 'MPHASIS.NS',

    # AUTO
    'TATAMOTORS.NS', 'MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 
    'HEROMOTOCO.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BHARATFORG.NS', 

    # ENERGY & POWER
    'RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'COALINDIA.NS', 
    'BPCL.NS', 'IOC.NS', 'TATAPOWER.NS', 'ADANIGREEN.NS', 'ADANIENT.NS', 
    'ADANIPORTS.NS', 'GAIL.NS',

    # METALS
    'TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'NMDC.NS', 
    'SAIL.NS', 'JINDALSTEL.NS', 'NATIONALUM.NS',

    # CONSUMER / PHARMA / REALTY
    'ITC.NS', 'HINDUNILVR.NS', 'TITAN.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS', 
    'BRITANNIA.NS', 'GODREJCP.NS', 'TATACONSUM.NS', 'DABUR.NS', 'SUNPHARMA.NS', 
    'CIPLA.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS', 'LUPIN.NS', 
    'DLF.NS', 'GODREJPROP.NS',

    # HIGH BETA / MOMENTUM
    'HAL.NS', 'BEL.NS', 'MAZDOCK.NS', 'BHEL.NS', 'ZOMATO.NS', 'TRENT.NS', 
    'IRCTC.NS', 'INDIGO.NS', 'JIOFIN.NS', 'ABBOTINDIA.NS', 'SIEMENS.NS', 
    'ABB.NS', 'POLYCAB.NS', 'HAVELLS.NS', 'VOLTAS.NS'
]

# 2. GLOBAL METALS & COMMODITIES
METALS = ['GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'AA', 'FCX', 'SCCO']

# 3. S&P 500 (Liquid Options Only)
SP_LIQUID_FNO = [
    # MAGNIFICENT SEVEN
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA',

    # SEMIS & AI
    'AMD', 'AVGO', 'QCOM', 'INTC', 'MU', 'TXN', 'AMAT', 'LRCX', 'ADI', 
    'SMCI', 'ARM', 'TSM',

    # FINANCE
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 
    'PYPL', 'COIN', 'HOOD',

    # SOFTWARE / CLOUD
    'CRM', 'ADBE', 'ORCL', 'IBM', 'NOW', 'PANW', 'PLTR', 'SNOW', 'CRWD', 
    'SQ', 'SHOP', 'UBER', 'ABNB',

    # MEDIA / RETAIL / PHARMA
    'NFLX', 'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'WMT', 'COST', 'TGT', 'HD', 
    'LOW', 'NKE', 'SBUX', 'MCD', 'LULU', 'CMG', 'BKNG', 'MAR', 'LLY', 'UNH', 
    'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'AMGN', 'GILD', 'ISRG', 'CVS',

    # INDUSTRIAL / ENERGY
    'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'GE', 'CAT', 'BA', 'LMT', 'RTX', 
    'HON', 'UPS', 'UNP', 'DE', 'KO', 'PEP', 'PG', 'PM', 'MO', 'CL'
]

ALL_TICKERS = LIQUID_FNO + METALS + SP_LIQUID_FNO

# SCAN CONFIGURATION
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
    """ Finds significant Highs and Lows """
    values = series.values
    if len(values) == 0: return [], []
    high_idx = argrelextrema(values, np.greater, order=order)[0]
    low_idx = argrelextrema(values, np.less, order=order)[0]
    return high_idx, low_idx

def check_line_integrity(series, idx_start, idx_end, slope, intercept, mode="upper"):
    """ Ensures price does not cut through the trendline """
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
    """ Main Pattern Recognition Function """
    if len(df) < 50: return None
    
    # 1. Find Pivots
    high_idxs, low_idxs = get_pivots(df['High'], order=8)
    if len(high_idxs) < 2 or len(low_idxs) < 2: return None

    # 2. Get Last 2 Major Swings
    Ax, Cx = high_idxs[-2], high_idxs[-1]
    Ay, Cy = df['High'].iloc[Ax], df['High'].iloc[Cx]
    Bx, Dx = low_idxs[-2], low_idxs[-1]
    By, Dy = df['Low'].iloc[Bx], df['Low'].iloc[Dx]

    # 3. Geometry Check (Triangle Shape)
    if not (Ay > Cy and By < Dy): return None

    # 4. Math (Slopes)
    slope_upper = (Cy - Ay) / (Cx - Ax)
    intercept_upper = Ay - (slope_upper * Ax)
    slope_lower = (Dy - By) / (Dx - Bx)
    intercept_lower = By - (slope_lower * Bx)

    # 5. Integrity Check (No Cuts)
    if not check_line_integrity(df['High'], Ax, Cx, slope_upper, intercept_upper, "upper"): return None
    if not check_line_integrity(df['Low'], Bx, Dx, slope_lower, intercept_lower, "lower"): return None

    # 6. Projection (The "E" Leg)
    current_idx = len(df) - 1
    proj_upper = (slope_upper * current_idx) + intercept_upper
    proj_lower = (slope_lower * current_idx) + intercept_lower
    current_price = df['Close'].iloc[-1]
    
    # Price must be INSIDE the triangle
    if not (proj_lower < current_price < proj_upper): return None
    
    # Triangle must be TIGHT (Coiling)
    width_pct = (proj_upper - proj_lower) / current_price
    
    if width_pct < 0.035: # < 3.5% Range
        return {
            "pivots": {"Ax": Ax, "Ay": Ay, "Cx": Cx, "Cy": Cy, "Bx": Bx, "By": By, "Dx": Dx, "Dy": Dy},
            "slopes": {"upper": slope_upper, "lower": slope_lower},
            "intercepts": {"upper": intercept_upper, "lower": intercept_lower},
            "coil_width": width_pct,
            "price": current_price
        }
    return None

def resample_data(df, interval):
    """ Resamples 1h data to 4h """
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    return df.resample(interval).agg(logic).dropna()

# ==========================================
# 3. PRO CHARTING (2X ZOOM)
# ==========================================

def plot_triangle_clean(df, ticker, data_dict, interval_label):
    # 1. ZOOM LOGIC (Dynamic 2X Context)
    pattern_start_idx = min(data_dict['pivots']['Ax'], data_dict['pivots']['Bx'])
    pattern_len = len(df) - pattern_start_idx
    # Show pattern + 1.5x history before it
    history_buffer = int(pattern_len * 1.5) 
    start_view_idx = max(0, pattern_start_idx - history_buffer)
    
    df_slice = df.iloc[start_view_idx:].copy()
    
    # 2. FORMAT X-AXIS (Readable Dates)
    if interval_label in ["5m", "15m"]:
        date_format = "%d %H:%M" 
    else:
        date_format = "%b %d"    

    df_slice['date_str'] = df_slice.index.strftime(date_format)

    # 3. CREATE CHART
    fig = go.Figure(data=[go.Candlestick(
        x=df_slice['date_str'], 
        open=df_slice['Open'], high=df_slice['High'],
        low=df_slice['Low'], close=df_slice['Close'], 
        name=ticker
    )])

    # 4. DRAW LINES
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

    # Clip lines to view
    def get_slice_dates(start_idx):
        eff_start = max(start_idx, start_view_idx)
        return df_slice['date_str'][eff_start - start_view_idx:].tolist()
    
    u_offset = max(0, start_view_idx - line_start_u)
    l_offset = max(0, start_view_idx - line_start_l)

    fig.add_trace(go.Scatter(
        x=get_slice_dates(line_start_u), y=y_vals_upper[u_offset:], 
        mode='lines', name='Res', line=dict(color='red', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=get_slice_dates(line_start_l), y=y_vals_lower[l_offset:], 
        mode='lines', name='Sup', line=dict(color='green', width=2)
    ))

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

# ==========================================
# 4. BACKGROUND AUTOMATION
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
# 5. STREAMLIT UI (SECURE)
# ==========================================

st.set_page_config(page_title="Triangle Pro", layout="wide")

# Initialize Session
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- LOGIN FORM ---
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔒 Triangle Hunter Pro")
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
    # --- MAIN DASHBOARD ---
    st.title("🔻 Triangle Hunter Pro")
    
    # Status Bar
    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(f"✅ System Active | Monitoring {len(ALL_TICKERS)} Liquid Assets")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()

    # Timeframe Tabs
    tabs = st.tabs(["⚡ 5 Min", "⏱️ 15 Min", "hourly 1 Hour", "📅 4 Hour"])

    for i, config in enumerate(SCAN_CONFIGS):
        with tabs[i]:
            if st.button(f"Start {config['label']} Scan", key=f"btn_{i}", type="primary"):
                
                with st.spinner("Analyzing Market Geometry..."):
                    try:
                        # Batch Download
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
                                        # Visual Card
                                        st.success(f"**{ticker}** | Coil: {match['coil_width']*100:.2f}%")
                                        # Pass the label to fix X-Axis
                                        fig = plot_triangle_clean(df, ticker, match, config['label'])
                                        st.plotly_chart(fig, use_container_width=True)
                                        c_idx += 1
                            except: continue
                        
                        if not found: st.info("No tight patterns found in this timeframe.")
                            
                    except Exception as e:
                        st.error(f"Data Error: {e}")
