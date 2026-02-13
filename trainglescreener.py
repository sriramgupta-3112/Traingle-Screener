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
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
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

# INCREASED PERIODS TO ENSURE 200 CANDLES ALWAYS
SCAN_CONFIGS = {
    "5m": {"interval": "5m", "period": "10d", "resample": None, "ttl": 300},
    "15m": {"interval": "15m", "period": "40d", "resample": None, "ttl": 900},
    "1h": {"interval": "1h", "period": "200d", "resample": None, "ttl": 3600},
    "4h": {"interval": "1h", "period": "700d", "resample": "4h", "ttl": 14400},
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(tickers, period, interval):
    try:
        return yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=False, threads=True)
    except Exception as e:
        return None

# ==========================================
# 2. ALGORITHMS & INDICATORS
# ==========================================

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

def count_touches(series, slope, intercept, mode, tolerance=0.005):
    indices = np.arange(len(series))
    line_values = slope * indices + intercept
    actual_values = series.values
    diff = np.abs(actual_values - line_values)
    touches = np.sum(diff < (line_values * tolerance))
    return touches

def analyze_ticker(df):
    if len(df) < 60: return None
    
    high_idxs, low_idxs = get_pivots(df['High'], order=8)
    if len(high_idxs) < 2 or len(low_idxs) < 2: return None

    Ax, Cx = high_idxs[-
