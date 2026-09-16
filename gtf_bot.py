import os
import sys
import logging
import subprocess

# ==============================================================================
# AUTO-INSTALL DEPENDENCIES
# ==============================================================================
def install_and_import(package, import_name=None):
    if import_name is None: import_name = package
    try:
        __import__(import_name)
    except ImportError:
        logging.info(f"Package '{package}' not found. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Removed pandas-ta to prevent Python version dependency errors on GitHub Actions
required_packages = [
    ('requests', 'requests'), ('numpy', 'numpy'), ('pandas', 'pandas'), 
    ('yfinance', 'yfinance'), ('mplfinance', 'mplfinance'), 
    ('scipy', 'scipy'), ('matplotlib', 'matplotlib')
]

for pip_name, imp_name in required_packages:
    install_and_import(pip_name, imp_name)

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# ==============================================================================
# CONFIGURATION & GTF SOP v4.2 PARAMETERS
# ==============================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_CAPITAL = 100000.0  
RISK_PERCENT = 1.0       
RISK_PER_TRADE = BASE_CAPITAL * (RISK_PERCENT / 100.0)
SWING_WINDOW = 15        

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ACTIVE_TRIPLET = os.getenv("ACTIVE_TRIPLET", "MIT")

TRIPLETS = {
    "HIT": {"htf": "60m", "itf": "15m", "ltf": "5m",  "p_htf": "1mo", "p_itf": "5d",  "p_ltf": "2d"},
    "DIT": {"htf": "1d",  "itf": "60m", "ltf": "15m", "p_htf": "2y",  "p_itf": "1mo", "p_ltf": "5d"},
    "WIT": {"htf": "1wk", "itf": "1d",  "ltf": "60m", "p_htf": "5y",  "p_itf": "1y",  "p_ltf": "1mo"},
    "MIT": {"htf": "1mo", "itf": "1wk", "ltf": "1d",  "p_htf": "10y", "p_itf": "3y",  "p_ltf": "1y"}
}

WATCHLIST = [
    "ABB.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", "ATGL.NS", 
    "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "DMART.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", 
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "BANKBARODA.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS", 
    "BPCL.NS", "BHARTIARTL.NS", "BOSCHLTD.NS", "BRITANNIA.NS", "CANBK.NS", "CHOLAFIN.NS", 
    "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "CROMPTON.NS", 
    "CUMMINSIND.NS", "DIVISLAB.NS", "DIXON.NS", "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", 
    "GAIL.NS", "GICRE.NS", "GODREJCP.NS", "GODREJPROP.NS", "GRASIM.NS", "HAVELLS.NS", "HCLTECH.NS", 
    "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HAL.NS", 
    "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "ITC.NS", 
    "IOC.NS", "IRCTC.NS", "IREDA.NS", "IRFC.NS", "INDUSINDBK.NS", "NAUKRI.NS", "INFY.NS", "INDIGO.NS", 
    "JSWINFRA.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "JIOFIN.NS", "KOTAKBANK.NS", "LT.NS", "LTIM.NS", 
    "LUPIN.NS", "M&M.NS", "MARICO.NS", "MARUTI.NS", "MUTHOOTFIN.NS", "NMDC.NS", "NTPC.NS", 
    "NESTLEIND.NS", "ONGC.NS", "PAGEIND.NS", "PIIND.NS", "PIDILITIND.NS", "PFC.NS", 
    "POWERGRID.NS", "PNB.NS", "RECLTD.NS", "RELIANCE.NS", "SBICARD.NS", "SBILIFE.NS", 
    "SBIN.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "SUNPHARMA.NS", "TVSMOTOR.NS", 
    "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATAELXSI.NS", "TATAPOWER.NS", "TATASTEEL.NS", 
    "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "ULTRACEMCO.NS", "VBL.NS", 
    "VEDL.NS", "WIPRO.NS", "ZOMATO.NS", "ZYDUSLIFE.NS"
]

def send_telegram_alert(message: str, chart_path: str = None):
    """Dispatches HTML-formatted institutional alert and optional chart to Telegram."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.info("Telegram not configured. Printing locally:\n\n" + message)
        return
        
    if chart_path and os.path.exists(chart_path):
        url_photo = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        try:
            with open(chart_path, 'rb') as photo:
                requests.post(url_photo, data={'chat_id': CHAT_ID}, files={'photo': photo}, timeout=15)
        except Exception as e:
            logging.error(f"Telegram Photo Send Failure: {e}")

    url_msg = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url_msg, json=payload, timeout=10)
        res.raise_for_status()
    except Exception as e:
        logging.error(f"Telegram Text Notification Failure: {e}")

# ==============================================================================
# PHASE 1: DATA INGESTION & ADVANCED CLASSIFICATION
# ==============================================================================
def get_historical_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def classify_candles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # GTF Classifications
    df['Range'] = df['High'] - df['Low']
    df['Range'] = np.where(df['Range'] == 0, 1e-4, df['Range'])
    df['Body'] = np.abs(df['Close'] - df['Open'])
    df['Ratio'] = df['Body'] / df['Range']

    df['Is_Exciting'] = df['Ratio'] > 0.50
    df['Is_Base'] = df['Ratio'] <= 0.50
    df['Is_Green'] = df['Close'] > df['Open']
    df['Is_Red'] = df['Close'] < df['Open']

    df['Gap_Up'] = df['Open'] > df['Close'].shift(1)
    df['Gap_Down'] = df['Open'] < df['Close'].shift(1)

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['SMA7'] = df['Close'].rolling(window=7).mean()
    
    # --- Native Institutional Indicators (Replaces pandas-ta) ---
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # True Range & ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ADX
    up = df['High'] - df['High'].shift(1)
    down = df['Low'].shift(1) - df['Low']
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_sm = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (tr_sm + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (tr_sm + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + (2 * std20)
    df['BB_Lower'] = sma20 - (2 * std20)
    
    # Fill safe defaults for start of array
    df.fillna({'RSI': 50, 'ADX': 0, 'ATR': 0, 'BB_Upper': df['Close'], 'BB_Lower': df['Close']}, inplace=True)
        
    return df

# ==============================================================================
# PHASE 2: ZONE DETECTION & CURVE EVALUATION
# ==============================================================================
def find_all_zones(df: pd.DataFrame, zone_type: str = "Demand", max_base: int = 5) -> list:
    zones = []
    n = len(df)
    if n < 8: return zones
        
    cmp = float(df['Close'].iloc[-1])

    for i in range(n - 2, 3, -1):
        leg_out = df.iloc[i + 1]
        is_leg_out_valid = leg_out['Is_Exciting'] or \
            (zone_type == "Demand" and leg_out['Gap_Up']) or \
            (zone_type == "Supply" and leg_out['Gap_Down'])
        
        if not is_leg_out_valid: continue
        if zone_type == "Demand" and not leg_out['Is_Green']: continue
        if zone_type == "Supply" and not leg_out['Is_Red']: continue

        base_candles, base_indices = [], []
        for j in range(i, max(-1, i - max_base - 1), -1):
            if df['Is_Base'].iloc[j]:
                base_candles.append(df.iloc[j])
                base_indices.append(j)
            else: break

        base_count = len(base_candles)
        if base_count < 1 or base_count > max_base: continue

        leg_in_idx = base_indices[-1] - 1
        if leg_in_idx < 0: continue
        leg_in = df.iloc[leg_in_idx]
        if not leg_in['Is_Exciting']: continue

        base_df = pd.DataFrame(base_candles)
        
        if zone_type == "Demand":
            pattern = "DBR" if leg_in['Is_Red'] else "RBR"
            leg_in_high = max(leg_in['Open'], leg_in['Close'])
            closing_concept = leg_out['Close'] > leg_in_high
            pl = float(base_df[['Open', 'Close']].max().max())
            base_min_wick = float(base_df['Low'].min())
            
            dl = float(min(leg_in['Low'], base_min_wick, leg_out['Low'])) if pattern == "DBR" else float(min(base_min_wick, leg_out['Low']))
            if pl >= cmp: continue

        else:
            pattern = "RBD" if leg_in['Is_Green'] else "DBD"
            leg_in_low = min(leg_in['Open'], leg_in['Close'])
            closing_concept = leg_out['Close'] < leg_in_low
            pl = float(base_df[['Open', 'Close']].min().min())
            base_max_wick = float(base_df['High'].max())
            
            dl = float(max(leg_in['High'], base_max_wick, leg_out['High'])) if pattern == "RBD" else float(max(base_max_wick, leg_out['High']))
            if pl <= cmp: continue

        post_zone_df = df.iloc[i + 2:]
        is_fresh, is_breached, tested_count = True, False, 0

        if not post_zone_df.empty:
            for _, candle in post_zone_df.iterrows():
                if zone_type == "Demand":
                    if candle['Low'] < dl: is_breached = True; break
                    elif candle['Low'] <= pl: is_fresh = False; tested_count += 1
                else:
                    if candle['High'] > dl: is_breached = True; break
                    elif candle['High'] >= pl: is_fresh = False; tested_count += 1

        if is_breached: continue

        fresh_pts = 3.0 if is_fresh else (1.5 if tested_count == 1 else 0.0)
        has_double_exciting = (len(df) > i + 2 and df['Is_Exciting'].iloc[i + 2])
        strength_pts = 2.0 if (has_double_exciting or leg_out['Gap_Up'] or leg_out['Gap_Down']) else 1.0
        time_pts = 2.0 if base_count <= 3 else (1.0 if base_count <= 5 else 0.0)
        
        zones.append({
            "Type": zone_type, "Pattern": pattern,
            "PL": round(pl, 2), "DL": round(dl, 2),
            "Date": str(df.index[base_indices[-1]])[:10],
            "Base_Count": base_count, "Is_Fresh": is_fresh,
            "Tested_Count": tested_count, "Closing_Concept": closing_concept,
            "Base_Score": fresh_pts + strength_pts + time_pts, "Index": i
        })

    return zones

def evaluate_htf_curve(df_htf: pd.DataFrame, cmp: float):
    htf_demands = find_all_zones(df_htf, "Demand")
    htf_supplies = find_all_zones(df_htf, "Supply")
    recent_htf_dem = htf_demands[0] if len(htf_demands) > 0 else None
    recent_htf_sup = htf_supplies[0] if len(htf_supplies) > 0 else None

    if not recent_htf_dem and not recent_htf_sup:
        return "Equilibrium (No Zones)", "Neutral", 0.0, 0.0, 0.0, None, None

    dem_pl = recent_htf_dem['PL'] if recent_htf_dem else cmp * 0.85
    sup_pl = recent_htf_sup['PL'] if recent_htf_sup else cmp * 1.15
    spread = sup_pl - dem_pl

    if spread <= 0: return "Equilibrium (Compressed)", "Follow Trend", dem_pl, sup_pl, 0.0, recent_htf_dem, recent_htf_sup

    lower_third = dem_pl + (spread / 3.0)
    upper_third = sup_pl - (spread / 3.0)

    if cmp <= dem_pl: location, bias = "Very Low on Curve (Inside Demand)", "Definitely Buy"
    elif dem_pl < cmp <= lower_third: location, bias = "Low on Curve", "Buy"
    elif lower_third < cmp < upper_third: location, bias = "Equilibrium (EQB)", "Follow Trend"
    elif upper_third <= cmp < sup_pl: location, bias = "High on Curve", "Sell"
    else: location, bias = "Very High on Curve (Inside Supply)", "Definitely Sell"

    return location, bias, dem_pl, sup_pl, spread, recent_htf_dem, recent_htf_sup

def evaluate_itf_trend(df_itf: pd.DataFrame) -> dict:
    cmp = float(df_itf['Close'].iloc[-1])
    ema20, ema50 = float(df_itf['EMA20'].iloc[-1]), float(df_itf['EMA50'].iloc[-1])
    sma7 = float(df_itf['SMA7'].iloc[-1])
    sma7_prev = float(df_itf['SMA7'].iloc[-3])

    golden_cross = ema20 > ema50
    above_ema20 = cmp > ema20
    sma7_rising = sma7 > sma7_prev

    recent_highs = df_itf['High'].iloc[-10:].values
    recent_lows = df_itf['Low'].iloc[-10:].values
    is_hh_hl = recent_highs[-1] > np.median(recent_highs) and recent_lows[-1] > np.median(recent_lows)

    if above_ema20 and golden_cross: direction = "Bullish"
    elif not above_ema20 and not golden_cross: direction = "Bearish"
    else: direction = "Sideways / Transition"

    return {
        "Direction": direction, "EMA20": round(ema20, 2), "EMA50": round(ema50, 2),
        "Golden_Cross": golden_cross, "Above_EMA20": above_ema20,
        "SMA7_Direction": "Rising" if sma7_rising else "Falling",
        "Structure": "HH/HL" if is_hh_hl else "LH/LL"
    }

# ==============================================================================
# PHASE 3: CHARTING & FIBONACCI
# ==============================================================================
def get_swings_and_fibs(df: pd.DataFrame, window=SWING_WINDOW):
    highs, lows = df['High'].values, df['Low'].values
    local_max_idx = argrelextrema(highs, np.greater, order=window)[0]
    local_min_idx = argrelextrema(lows, np.less, order=window)[0]

    if len(local_max_idx) == 0 or len(local_min_idx) == 0:
        return None, None, None, {}

    last_high_idx, last_low_idx = local_max_idx[-1], local_min_idx[-1]
    swing_high = {'date': df.index[last_high_idx], 'price': highs[last_high_idx]}
    swing_low = {'date': df.index[last_low_idx], 'price': lows[last_low_idx]}

    trend = "Upward" if last_low_idx < last_high_idx else "Downward"
    H, L = swing_high['price'], swing_low['price']
    diff = H - L
    
    ratios = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
    fib_levels = {}
    for ratio in ratios:
        fib_levels[f"{ratio:.3f}"] = H - (ratio * diff) if trend == "Upward" else L + (ratio * diff)

    return swing_high, swing_low, trend, fib_levels

def generate_chart(ticker: str, df: pd.DataFrame, swing_high: dict, swing_low: dict, fib_levels: dict, score: float):
    plot_df = df.tail(120).copy()
    if plot_df.empty: return None
    
    chart_file = f"chart_{ticker}.png"
    add_plots = [
        mpf.make_addplot(plot_df['EMA20'], color='blue', width=1.5, label='EMA 20'),
        mpf.make_addplot(plot_df['EMA50'], color='orange', width=1.5, label='EMA 50')
    ]

    hlines_data = []
    if "0.500" in fib_levels and "0.618" in fib_levels:
        hlines_data = [fib_levels["0.500"], fib_levels["0.618"]]

    hlines = dict(hlines=hlines_data, colors=['g', 'r'], linestyle='--', linewidths=1.5, alpha=0.7) if hlines_data else None

    fig, axlist = mpf.plot(
        plot_df, type='candle', style='yahoo', addplot=add_plots, hlines=hlines,
        title=f"\n{ticker} - Score: {score}/9.0", ylabel='Price', volume=False, 
        figsize=(10, 6), returnfig=True
    )

    ax = axlist[0]
    start_date = plot_df.index[0]
    
    if swing_high and swing_high['date'] >= start_date:
        ax.annotate('Swing High', xy=(swing_high['date'], swing_high['price']),
                    xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->", color='red'))
                    
    if swing_low and swing_low['date'] >= start_date:
        ax.annotate('Swing Low', xy=(swing_low['date'], swing_low['price']),
                    xytext=(10, -20), textcoords='offset points', arrowprops=dict(arrowstyle="->", color='green'))

    fig.savefig(chart_file, bbox_inches='tight')
    plt.close(fig)
    return chart_file

# ==============================================================================
# PHASE 4: EXECUTION & SCORING (GTF + MURPHY + LEBEAU + PRING)
# ==============================================================================
def run_sop_analysis(ticker: str):
    cfg = TRIPLETS[ACTIVE_TRIPLET]
    
    df_htf = get_historical_data(ticker, cfg['htf'], cfg['p_htf'])
    df_itf = get_historical_data(ticker, cfg['itf'], cfg['p_itf'])
    df_ltf = get_historical_data(ticker, cfg['ltf'], cfg['p_ltf'])

    if df_htf.empty or df_itf.empty or df_ltf.empty: return

    df_htf, df_itf, df_ltf = classify_candles(df_htf), classify_candles(df_itf), classify_candles(df_ltf)
    
    curr = df_ltf.iloc[-1]
    cmp = float(curr['Close'])

    curve_loc, curve_bias, dem_pl, sup_pl, spread, htf_dem, htf_sup = evaluate_htf_curve(df_htf, cmp)
    itf_trend = evaluate_itf_trend(df_itf)

    ltf_demands = find_all_zones(df_ltf, "Demand")
    ltf_supplies = find_all_zones(df_ltf, "Supply")

    trade_side, target_zones, opposing_zones = None, [], []
    if "Buy" in curve_bias or (curve_loc.startswith("Equilibrium") and itf_trend["Direction"] == "Bullish"):
        trade_side, target_zones, opposing_zones = "BUY", ltf_demands, ltf_supplies
    elif "Sell" in curve_bias or (curve_loc.startswith("Equilibrium") and itf_trend["Direction"] == "Bearish"):
        trade_side, target_zones, opposing_zones = "SELL", ltf_supplies, ltf_demands
    else: return

    if not target_zones: return

    # --- SIGNAL DETECTION LOGIC (GTF) ---
    selected_zone = None
    action_status = ""
    
    for z in target_zones:
        pl, dl = z['PL'], z['DL']
        cushion = round(abs(pl - dl) * 0.05, 2)
        
        entry_px = round(pl + cushion, 2) if trade_side == "BUY" else round(pl - cushion, 2)
        stop_px = round(dl - cushion, 2) if trade_side == "BUY" else round(dl + cushion, 2)

        if trade_side == "BUY":
            if stop_px <= cmp <= entry_px:
                action_status = "🟢 ACTIVE ENTRY: Price inside Demand Zone"
                selected_zone = z; break
            elif entry_px < cmp <= entry_px * 1.035:
                action_status = "🟡 SETUP: Approaching Demand Zone"
                selected_zone = z; break
        else: # SELL
            if entry_px <= cmp <= stop_px:
                action_status = "🔴 ACTIVE ENTRY: Price inside Supply Zone"
                selected_zone = z; break
            elif entry_px * 0.965 <= cmp < entry_px:
                action_status = "🟡 SETUP: Approaching Supply Zone"
                selected_zone = z; break

    if not selected_zone: return

    # Base Confluence Score
    confluence_score = selected_zone['Base_Score']
    ltf_ema20, ltf_ema50 = float(df_ltf['EMA20'].iloc[-1]), float(df_ltf['EMA50'].iloc[-1])

    ema_aligned = abs(selected_zone['PL'] - ltf_ema20) / selected_zone['PL'] <= 0.02
    if ema_aligned: confluence_score += 1.0

    cross_aligned = (trade_side == "BUY" and ltf_ema20 > ltf_ema50) or (trade_side == "SELL" and ltf_ema20 < ltf_ema50)
    if cross_aligned: confluence_score += 1.0

    if confluence_score < 5.5: return

    swing_high, swing_low, trend, fib_levels = get_swings_and_fibs(df_itf, SWING_WINDOW)

    entry_type = "Entry Type 1 (Set & Forget)" if selected_zone['Base_Score'] >= 7.0 and selected_zone['Is_Fresh'] else \
                 "Entry Type 2 (Wait Inside Confirm)" if selected_zone['Base_Score'] >= 6.0 else "Entry Type 3 (Wait Exit Confirm)"

    # --- ADVANCED INSTITUTIONAL FILTERS ---
    
    # 1. Murphy Trend Filter (EMA 9/21 + ADX > 25)
    adx_val = round(curr.get('ADX', 0), 2)
    has_adx = adx_val > 25
    if trade_side == "BUY":
        murphy_aligned = (curr.get('EMA9', 0) > curr.get('EMA21', 0)) and has_adx
    else:
        murphy_aligned = (curr.get('EMA9', 0) < curr.get('EMA21', 0)) and has_adx
    murphy_status = "✅ Trend Confirmed" if murphy_aligned else f"❌ Unconfirmed (ADX: {adx_val})"

    # 2. LeBeau Chandelier Exit (Dynamic Volatility Stop)
    atr = curr.get('ATR', 0)
    if trade_side == "BUY":
        high_22 = df_ltf['High'].tail(22).max()
        chandelier_stop = round(high_22 - (3 * atr), 2)
    else:
        low_22 = df_ltf['Low'].tail(22).min()
        chandelier_stop = round(low_22 + (3 * atr), 2)

    # 3. Pring Exhaustion & Divergence
    rsi = round(curr.get('RSI', 50), 2)
    bb_upper = curr.get('BB_Upper', cmp * 1.5)
    bb_lower = curr.get('BB_Lower', cmp * 0.5)
    
    exhaustion_warn = "✅ Momentum Clear"
    if trade_side == "BUY" and (rsi > 70 or cmp >= bb_upper):
        exhaustion_warn = "⚠️ HIGH RISK (Overbought / BB Pierced)"
    elif trade_side == "SELL" and (rsi < 30 or cmp <= bb_lower):
        exhaustion_warn = "⚠️ HIGH RISK (Oversold / BB Pierced)"
        
    recent_price_high = df_ltf['High'].iloc[-15:-1].max()
    recent_rsi_high = df_ltf['RSI'].iloc[-15:-1].max()
    if trade_side == "BUY" and cmp > recent_price_high and rsi < recent_rsi_high:
        exhaustion_warn += " | 🚨 Bearish Divergence"

    # --- EXECUTIONS & SIZING ---
    pl, dl = selected_zone['PL'], selected_zone['DL']
    cushion = round(abs(pl - dl) * 0.05, 2)
    nearest_opp_pl = opposing_zones[0]['PL'] if opposing_zones else None
    
    if trade_side == "BUY":
        entry_px, stop_px = round(pl + cushion, 2), round(dl - cushion, 2)
        risk_per_share = round(entry_px - stop_px, 2)
        target_2r = round(entry_px + (2 * risk_per_share), 2)
        dynamic_exit_target = nearest_opp_pl if nearest_opp_pl and nearest_opp_pl > entry_px else "N/A"
    else:
        entry_px, stop_px = round(pl - cushion, 2), round(dl + cushion, 2)
        risk_per_share = round(stop_px - entry_px, 2)
        target_2r = round(entry_px - (2 * risk_per_share), 2)
        dynamic_exit_target = nearest_opp_pl if nearest_opp_pl and nearest_opp_pl < entry_px else "N/A"

    if risk_per_share <= 0: return
    qty = int(RISK_PER_TRADE / risk_per_share)
    trade_capital = round(qty * entry_px, 2)

    fib_500 = fib_levels.get('0.500') if fib_levels else None
    fib_618 = fib_levels.get('0.618') if fib_levels else None
    is_golden = (min(fib_500, fib_618) <= pl <= max(fib_500, fib_618)) if fib_500 and fib_618 else False
    golden_status = "✅ PL inside Golden Zone" if is_golden else "❌ PL outside Golden Zone" if fib_500 else "N/A"

    # Message Generation with combined filters
    alert_msg = (
        f"<b>{action_status}</b>\n"
        f"{'🟢 LONG SETUP' if trade_side == 'BUY' else '🔴 SHORT SETUP'}: <b>{ticker}</b>\n\n"
        
        f"<b>SECTION I: INSTITUTIONAL FILTERS</b>\n"
        f"• Murphy Trend (EMA 9/21 + ADX): {murphy_status}\n"
        f"• Pring Exhaustion: {exhaustion_warn} (RSI: {rsi})\n"
        f"• LeBeau Chandelier Stop: Rs {chandelier_stop}\n\n"

        f"<b>SECTION II: ENTRY & EXITS (GTF SOP v4.2)</b>\n"
        f"• Current Price (CMP): Rs {round(cmp, 2)}\n"
        f"• Trigger Entry Order: Rs {entry_px}\n"
        f"• Structural Stop Loss: Rs {stop_px}\n"
        f"• Exit Signal 1 (Opposing Zone): Rs {dynamic_exit_target}\n"
        f"• Exit Signal 2 (2:1 RR Target): Rs {target_2r}\n\n"

        f"<b>SECTION III: RISK METRICS (Capital Rs {BASE_CAPITAL})</b>\n"
        f"• Recommended Position: <b>{qty} Shares</b>\n"
        f"• Total Risk Allocation: Rs {round(RISK_PER_TRADE, 2)}\n"
        f"• Total Outlay: Rs {trade_capital}\n\n"
        
        f"<b>SECTION IV: ZONE ARCHITECTURE (LTF: {cfg['ltf']})</b>\n"
        f"• Proximal Line: Rs {pl} | Distal Line: Rs {dl}\n"
        f"• Pattern: {selected_zone['Pattern']} ({selected_zone['Type']})\n"
        f"• Base Candles: {selected_zone['Base_Count']} | Freshness: {'Fresh' if selected_zone['Is_Fresh'] else 'Tested'}\n"
        f"• GTF Final Score: <b>{confluence_score}/9.0</b> ({entry_type})\n\n"

        f"<b>SECTION V: CURVE & TREND</b>\n"
        f"• HTF Curve: {curve_loc} ({curve_bias})\n"
        f"• ITF Trend: {itf_trend['Direction']} ({itf_trend['Structure']})\n"
        f"• Golden Fib Status: {golden_status}\n"
    )

    chart_path = generate_chart(ticker, df_itf, swing_high, swing_low, fib_levels, confluence_score)
    send_telegram_alert(alert_msg, chart_path)

    if chart_path and os.path.exists(chart_path):
        os.remove(chart_path)

if __name__ == "__main__":
    for ticker_symbol in WATCHLIST:
        try:
            logging.info(f"Scanning {ticker_symbol}...")
            run_sop_analysis(ticker_symbol)
        except Exception as ex:
            logging.error(f"Processing exception on {ticker_symbol}: {ex}")
