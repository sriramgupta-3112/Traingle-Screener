import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import concurrent.futures
from datetime import datetime, timedelta, timezone

# ==========================================
# 1. CONFIGURATION
# ==========================================
APP_PASSWORD = "JaiBabaKi"
ENABLE_TELEGRAM = False

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

NON_STOCK_ASSETS = set(CRYPTO + COMMODITIES)

SCAN_CONFIGS = {
    "5m": {"interval": "5m", "period": "10d", "ttl": 300},
    "15m": {"interval": "15m", "period": "40d", "ttl": 900},
    "1h": {"interval": "1h", "period": "200d", "ttl": 3600},
    "1d": {"interval": "1d", "period": "1y", "ttl": 14400},
}

# ==========================================
# 2. DATA ENGINE
# ==========================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(tickers, period, interval):
    try:
        return yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=False, threads=True)
    except Exception:
        return None

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

def count_touches(series, slope, intercept, tolerance=0.005):
    indices = np.arange(len(series))
    line_values = slope * indices + intercept
    actual_values = series.values
    diff = np.abs(actual_values - line_values)
    touches = np.sum(diff < (line_values * tolerance))
    return touches

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================

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

    convergence = abs(slope_upper - slope_lower)
    if convergence < 0.0002: return None 

    tolerance = 1e-4
    if slope_upper > tolerance and slope_lower > tolerance: return None
    if slope_upper < -tolerance and slope_lower < -tolerance: return None

    width_upper = Cx - Ax
    width_lower = Dx - Bx
    if width_upper == 0 or width_lower == 0: return None
    ratio_time = min(width_upper, width_lower) / max(width_upper, width_lower)
    if ratio_time < 0.25: return None

    height_A = Ay - (slope_lower * Ax + intercept_lower)
    height_C = Cy - (slope_lower * Cx + intercept_lower)
    if height_A == 0: return None
    if (height_C / height_A) < 0.20: return None 

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
    if not (proj_lower <= current_price <= proj_upper): return None

    pattern_start_idx = min(Ax, Bx)
    wave_label = "Unclear"
    
    if pattern_start_idx > 20:
        lookback = int(pattern_len * 1.5)
        trend_start = max(0, pattern_start_idx - lookback)
        price_start = df['Close'].iloc[trend_start]
        price_end = df['Close'].iloc[pattern_start_idx]
        move_pct = abs((price_end - price_start) / price_start)
        pattern_height_pct = (Ay - By) / By
        df['RSI'] = calculate_rsi(df['Close'])
        current_rsi = df['RSI'].iloc[-1]
        
        if move_pct > (pattern_height_pct * 1.5) and (35 < current_rsi < 65):
            wave_label = "Wave 4"
        elif move_pct < pattern_height_pct:
            wave_label = "Wave B"

    start_search = min(Ax, Bx)
    touches_u = count_touches(df['High'].iloc[start_search:], slope_upper, intercept_upper)
    touches_l = count_touches(df['Low'].iloc[start_search:], slope_lower, intercept_lower)
    
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
            "touches": touches_u + touches_l,
            "wave_label": wave_label
        }
    return None

# ==========================================
# 4. CHARTING
# ==========================================

def plot_triangle_clean(df, ticker, data_dict, interval_label):
    view_len = 200
    start_view_idx = max(0, len(df) - view_len)
    df_slice = df.iloc[start_view_idx:].copy()
    
    # Simple, minimal date format
    if interval_label == "1d":
        df_slice['date_str'] = df_slice.index.strftime("%d %b") # 12 Feb
    else:
        df_slice['date_str'] = df_slice.index.strftime("%H:%M") # 14:30

    fig = go.Figure(data=[go.Ohlc(
        x=df_slice['date_str'], 
        open=df_slice['Open'], high=df_slice['High'],
        low=df_slice['Low'], close=df_slice['Close'], 
        name=ticker,
        increasing_line_color='#26a69a', # Teal for up (Modern)
        decreasing_line_color='#ef5350', # Soft Red for down (Modern)
        line_width=1
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

    fig.add_trace(go.Scatter(x=xu, y=yu, mode='lines', name='Res', line=dict(color='RoyalBlue', width=2)))
    fig.add_trace(go.Scatter(x=xl, y=yl, mode='lines', name='Sup', line=dict(color='DarkOrange', width=2)))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis_type='category', 
        height=320, 
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=-45, nticks=8, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        plot_bgcolor='white', 
        paper_bgcolor='white',
        showlegend=False
    )
    return fig

# ==========================================
# 5. STREAMLIT APP
# ==========================================

st.set_page_config(page_title="Screener Pro 1.3", layout="wide", page_icon="📊")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.title("🔒 Login")
        with st.form("login_form"):
            password = st.text_input("Access Code", type="password")
            if st.form_submit_button("Unlock", type="primary", use_container_width=True):
                if password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("Invalid Code")
else:
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        asset_choice = st.selectbox(
            "Market",
            ("All Assets", "NSE F&O Stocks", "S&P 500 Stocks", "Crypto", "Commodities")
        )
        
        timeframe_choice = st.select_slider(
            "Timeframe",
            options=["5m", "15m", "1h", "1d"],
            value="15m"
        )
        
        if asset_choice == "All Assets": ACTIVE_TICKERS = CRYPTO + COMMODITIES + LIQUID_FNO + SP_LIQUID_FNO
        elif asset_choice == "NSE F&O Stocks": ACTIVE_TICKERS = LIQUID_FNO
        elif asset_choice == "S&P 500 Stocks": ACTIVE_TICKERS = SP_LIQUID_FNO
        elif asset_choice == "Crypto": ACTIVE_TICKERS = CRYPTO
        elif asset_choice == "Commodities": ACTIVE_TICKERS = COMMODITIES
            
        st.caption(f"Universe: {len(ACTIVE_TICKERS)} assets")
        
        start_scan = st.button("🚀 Run Scan", type="primary", use_container_width=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    # Main Area
    st.title("📊 Screener Pro 1.3")
    
    def process_ticker(ticker, data_source, config):
        try:
            if len(ACTIVE_TICKERS) > 1: df = data_source[ticker].dropna()
            else: df = data_source.dropna()
            if df.empty: return None
            match = analyze_ticker(df)
            if match:
                fig = plot_triangle_clean(df, ticker, match, config['interval'])
                item = {"ticker": ticker, "data": match, "fig": fig, "touches": match['touches']}
                is_stock = ticker not in NON_STOCK_ASSETS
                return (match['is_online'], is_stock, item)
            return None
        except Exception: return None

    config = SCAN_CONFIGS[timeframe_choice]
    scan_key = f"scan_{timeframe_choice}_{asset_choice}"

    if start_scan:
        status = st.status(f"Scanning {asset_choice} ({timeframe_choice})...", expanded=True)
        try:
            status.write("📥 Fetching market data...")
            data = fetch_market_data(ACTIVE_TICKERS, config['period'], config['interval'])
            
            if data is None or data.empty:
                status.error("Data fetch failed.")
            else:
                status.write("🧠 Analyzing patterns...")
                results = {"on_s": [], "on_r": [], "off_s": [], "off_r": []}
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(process_ticker, t, data, config): t for t in ACTIVE_TICKERS}
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res:
                            is_on, is_stk, item = res
                            if is_on: results["on_s" if is_stk else "on_r"].append(item)
                            else: results["off_s" if is_stk else "off_r"].append(item)
                
                for key in results:
                    results[key].sort(key=lambda x: x['touches'], reverse=True)
                    
                st.session_state.scan_results[scan_key] = results
                status.update(label="Scan Complete!", state="complete", expanded=False)
        except Exception as e: status.error(f"Error: {e}")

    # Results Display
    if scan_key in st.session_state.scan_results:
        res = st.session_state.scan_results[scan_key]
        
        # 1. Live Section
        live_items = res["on_s"] + res["on_r"]
        if live_items:
            st.subheader(f"🟢 Live Opportunities ({len(live_items)})")
            # Display in grid
            cols = st.columns(3)
            for i, item in enumerate(live_items):
                with cols[i % 3]:
                    # Card-like container
                    with st.container(border=True):
                        # Header Metrics
                        c1, c2, c3 = st.columns([2, 1, 1])
                        c1.markdown(f"**{item['ticker']}**")
                        c1.caption(f"🕒 {item['data']['last_time']}")
                        
                        lbl = item['data']['wave_label']
                        color = "green" if "Wave 4" in lbl else "orange"
                        c2.markdown(f":{color}[{lbl}]")
                        
                        width = item['data']['coil_width']*100
                        c3.markdown(f"**{width:.1f}%**")
                        
                        st.plotly_chart(item['fig'], use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No patterns in live markets.")

        # 2. Offline Section
        off_items = res["off_s"] + res["off_r"]
        if off_items:
            with st.expander(f"🔴 Closed Markets ({len(off_items)})"):
                cols = st.columns(3)
                for i, item in enumerate(off_items):
                    with cols[i % 3]:
                        with st.container(border=True):
                            st.markdown(f"**{item['ticker']}** | {item['data']['wave_label']}")
                            st.plotly_chart(item['fig'], use_container_width=True, config={'displayModeBar': False})
