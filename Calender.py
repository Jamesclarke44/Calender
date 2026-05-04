"""
📈 Entry Signal Analyzer - Clean & Fixed Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Tuple

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Entry Analyzer", page_icon="📈", layout="wide")

# ============================================================================
# SESSION STATE INIT (FIXED)
# ============================================================================

defaults = {
    "analysis_result": None,
    "shared_ticker": "AAPL",
    "shared_entry": 180.0,
    "shared_direction": "LONG",
    "shared_score": 0,
    "shared_support": 0,
    "shared_resistance": 0,
    "shared_atr": 0,
    "shared_stop": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def robust_scalar(value):
    if value is None:
        return 0.0
    if isinstance(value, pd.Series):
        return float(value.iloc[-1]) if len(value) > 0 else 0.0
    try:
        return float(value)
    except:
        return 0.0

def get_column(df, name):
    for col in df.columns:
        if col.lower() == name.lower():
            return df[col]
    return None

# ============================================================================
# INDICATORS (IMPROVED)
# ============================================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    line = ema12 - ema26
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist

def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr_val = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_val)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

# ============================================================================
# ANALYSIS ENGINE
# ============================================================================

def analyze_entry(ticker, entry_price, direction):

    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)

        if df.empty or len(df) < 50:
            return {"error": "Not enough data"}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        current_price = close.iloc[-1]
        prev_close = close.iloc[-2]
        change = ((current_price - prev_close) / prev_close) * 100

        # Moving averages
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]

        ema9 = ema(close, 9).iloc[-1]
        ema21 = ema(close, 21).iloc[-1]

        # Indicators
        rsi_val = rsi(close).iloc[-1]
        macd_line, signal_line, hist = macd(close)
        hist_val = hist.iloc[-1]
        hist_prev = hist.iloc[-2]

        atr_val = atr(high, low, close).iloc[-1]
        atr_pct = (atr_val / current_price) * 100

        avg_vol = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1

        adx_val = adx(high, low, close).iloc[-1]

        # Support / resistance
        support = low.tail(50).min()
        resistance = high.tail(50).max()

        # ====================================================================
        # SCORING
        # ====================================================================

        score = 50
        reasons = []
        warnings = []

        # Trend
        if current_price > sma20 > sma50:
            score += 15
            reasons.append("✅ Strong uptrend")

        if ema9 > ema21:
            score += 10
            reasons.append("✅ Bullish momentum")

        # MACD
        if hist_val > 0 and hist_val > hist_prev:
            score += 15
            reasons.append("✅ MACD strengthening")

        # RSI
        if 40 <= rsi_val <= 60:
            score += 10
        elif rsi_val < 40:
            score += 5
        elif rsi_val > 70:
            score -= 10
            warnings.append("⚠️ Overbought")

        # Volume
        if vol_ratio > 2:
            score += 12
            reasons.append("🚀 Volume spike")
        elif vol_ratio > 1.2:
            score += 5

        # Trend strength
        trend_strength = abs(sma20 - sma50) / current_price * 100
        if trend_strength > 2:
            score += 5
        elif trend_strength < 0.5:
            score -= 5

        # Entry quality
        if direction == "LONG":
            if entry_price <= current_price:
                score += 5
            else:
                score -= 5
                warnings.append("⚠️ Chasing price")

        # ADX
        if adx_val > 25:
            score += 10
            reasons.append("✅ Strong trend (ADX)")

        # Clamp score
        score = max(0, min(100, score))

        # Signal
        if score >= 75:
            signal = "STRONG BUY"
            color = "#22c55e"
        elif score >= 60:
            signal = "BUY"
            color = "#10b981"
        elif score >= 40:
            signal = "NEUTRAL"
            color = "#f59e0b"
        else:
            signal = "WAIT"
            color = "#ef4444"

        # Stop loss
        if direction == "LONG":
            stop = max(support * 0.995, current_price - atr_val * 1.2)
        else:
            stop = min(resistance * 1.005, current_price + atr_val * 1.2)

        return {
            "ticker": ticker,
            "price": current_price,
            "change": change,
            "score": score,
            "signal": signal,
            "color": color,
            "rsi": rsi_val,
            "adx": adx_val,
            "volume": vol_ratio,
            "support": support,
            "resistance": resistance,
            "stop": stop,
            "reasons": reasons,
            "warnings": warnings
        }

    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# SIDEBAR (FIXED)
# ============================================================================

with st.sidebar:

    st.header("📈 Configuration")

    ticker = st.text_input("Ticker", st.session_state.shared_ticker).upper()

    direction = st.radio(
        "Direction",
        ["LONG", "SHORT"],
        index=0 if st.session_state.shared_direction == "LONG" else 1
    )

    entry_price = st.number_input(
        "Entry Price",
        value=st.session_state.shared_entry,
        step=0.01
    )

    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

# ============================================================================
# MAIN UI
# ============================================================================

st.title("📈 Entry Signal Analyzer")

if analyze_btn:
    st.session_state.analysis_result = analyze_entry(ticker, entry_price, direction)

result = st.session_state.analysis_result

if result:

    if "error" in result:
        st.error(result["error"])

    else:
        st.markdown(f"""
        ## {result['signal']}
        Score: **{result['score']}**
        """)

        col1, col2, col3 = st.columns(3)

        col1.metric("Price", f"${result['price']:.2f}", f"{result['change']:.2f}%")
        col2.metric("RSI", f"{result['rsi']:.1f}")
        col3.metric("ADX", f"{result['adx']:.1f}")

        st.divider()

        st.subheader("Levels")
        st.write(f"Support: {result['support']:.2f}")
        st.write(f"Resistance: {result['resistance']:.2f}")
        st.write(f"Stop: {result['stop']:.2f}")

        st.divider()

        if result["reasons"]:
            st.subheader("✅ Reasons")
            for r in result["reasons"]:
                st.write(r)

        if result["warnings"]:
            st.subheader("⚠️ Warnings")
            for w in result["warnings"]:
                st.write(w)