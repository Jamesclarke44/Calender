import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ----------------- CONFIG -----------------

st.set_page_config(page_title="Calendar Go / No-Go", layout="centered")

# Thresholds (easy to tweak)
RSI_NO_GO = 35
ADX_NO_GO = 30
IVR_NO_GO = 45

RSI_GO_MIN = 40
RSI_GO_MAX = 60
ADX_GO_MAX = 25
IVR_GO_MIN = 20
IVR_GO_MAX = 35

# ----------------- VWAP (Daily Approximation) -----------------

def calc_vwap_daily(data):
    """Stable VWAP using daily OHLCV (works 100% of the time)."""
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    vwap = (typical_price * data["Volume"]).sum() / data["Volume"].sum()
    return float(vwap)

# ----------------- CORE LOGIC -----------------

def classify_environment(price, bbl_low, bbl_high, rsi, ivr, adx, vwap):
    """
    Returns (status, reason)
    status ∈ {"GO", "CAUTION", "NO GO"}
    """

    # VWAP check
    if price < vwap:
        return "NO GO", "Price below VWAP."

    # Hard NO-GO conditions
    if price <= bbl_low:
        return "NO GO", "Price at/under lower Bollinger band."
    if rsi < RSI_NO_GO:
        return "NO GO", f"RSI oversold (< {RSI_NO_GO})."
    if adx > ADX_NO_GO:
        return "NO GO", f"Strong trend (ADX > {ADX_NO_GO})."
    if ivr > IVR_NO_GO:
        return "NO GO", f"IVR too high (> {IVR_NO_GO})."

    # Ideal GO conditions
    bbl_mid = (bbl_low + bbl_high) / 2
    if (
        RSI_GO_MIN <= rsi <= RSI_GO_MAX and
        adx < ADX_GO_MAX and
        IVR_GO_MIN <= ivr <= IVR_GO_MAX and
        bbl_mid <= price <= bbl_high and
        price >= vwap
    ):
        return "GO", "Neutral, stable environment for calendars."

    # Everything else
    return "CAUTION", "Mixed signals. Monitor, don’t force entries."

# ----------------- SIMPLE TECH CALCS -----------------

def calc_bollinger(close, length=20, mult=2):
    ma = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper = ma + mult * std
    lower = ma - mult * std
    return lower, upper

def calc_rsi(close, length=14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_rol = pd.Series(gain).rolling(length).mean()
    loss_rol = pd.Series(loss).rolling(length).mean()
    rs = gain_rol / (loss_rol + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_adx(high, low, close, length=14):
    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(length).mean()

    plus_di = 100 * (pd.Series(plus_dm).rolling(length).sum() / (atr + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(length).sum() / (atr + 1e-9))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.rolling(length).mean()
    return adx

# Dummy IVR placeholder
def dummy_ivr():
    return 30.0

# ----------------- UI -----------------

st.title("📊 Calendar Spread Go / No-Go")

# -------- Manual Input Section --------

st.header("Manual Check")

col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Price", value=454.59)
    bbl_low = st.number_input("BBL Low", value=452.41)
    bbl_high = st.number_input("BBL High", value=460.80)
    vwap_manual = st.number_input("VWAP", value=455.00)
with col2:
    rsi = st.number_input("RSI", value=28.87)
    ivr = st.number_input("IVR", value=41.0)
    adx = st.number_input("ADX", value=47.28)

if st.button("Evaluate Manual Inputs"):
    status, reason = classify_environment(price, bbl_low, bbl_high, rsi, ivr, adx, vwap_manual)
    if status == "GO":
        st.success(f"GO ✅ — {reason}")
    elif status == "CAUTION":
        st.warning(f"CAUTION ⚠️ — {reason}")
    else:
        st.error(f"NO GO ⛔ — {reason}")

st.markdown("---")

# -------- Ticker-Based Auto Metrics --------

st.header("Ticker Auto-Check (basic)")

ticker = st.text_input("Ticker", value="DIA").upper()
period = st.selectbox("History period", ["1mo", "3mo", "6mo"], index=0)

if st.button("Fetch & Evaluate Ticker"):
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        st.error("No data returned for this ticker/period.")
    else:
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        last_price = float(close.iloc[-1])

        bbl_low_series, bbl_high_series = calc_bollinger(close)
        last_bbl_low = float(bbl_low_series.iloc[-1])
        last_bbl_high = float(bbl_high_series.iloc[-1])

        rsi_series = calc_rsi(close)
        last_rsi = float(rsi_series.iloc[-1])

        adx_series = calc_adx(high, low, close)
        last_adx = float(adx_series.iloc[-1])

        last_ivr = dummy_ivr()

        last_vwap = calc_vwap_daily(data)

        st.write(f"**{ticker} latest metrics:**")
        st.write(f"- Price: {last_price:.2f}")
        st.write(f"- VWAP: {last_vwap:.2f}")
        st.write(f"- BBL Low: {last_bbl_low:.2f}")
        st.write(f"- BBL High: {last_bbl_high:.2f}")
        st.write(f"- RSI: {last_rsi:.2f}")
        st.write(f"- ADX: {last_adx:.2f}")
        st.write(f"- IVR (dummy): {last_ivr:.2f}")

        status, reason = classify_environment(
            last_price,
            last_bbl_low,
            last_bbl_high,
            last_rsi,
            last_ivr,
            last_adx,
            last_vwap
        )

        if status == "GO":
            st.success(f"GO ✅ — {reason}")
        elif status == "CAUTION":
            st.warning(f"CAUTION ⚠️ — {reason}")
        else:
            st.error(f"NO GO ⛔ — {reason}")

st.markdown("---")

# -------- Simple Scanner --------

st.header("Scanner (basic, using dummy IVR)")

tickers_input = st.text_input(
    "Tickers (comma-separated)",
    value="SPY, QQQ, DIA, IWM"
)

if st.button("Run Scan"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    rows = []

    for t in tickers:
        data = yf.Ticker(t).history(period="3mo")
        if data.empty:
            continue

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        last_price = float(close.iloc[-1])
        bbl_low_series, bbl_high_series = calc_bollinger(close)
        last_bbl_low = float(bbl_low_series.iloc[-1])
        last_bbl_high = float(bbl_high_series.iloc[-1])
        rsi_series = calc_rsi(close)
        last_rsi = float(rsi_series.iloc[-1])
        adx_series = calc_adx(high, low, close)
        last_adx = float(adx_series.iloc[-1])
        last_ivr = dummy_ivr()
        last_vwap = calc_vwap_daily(data)

        status, reason = classify_environment(
            last_price,
            last_bbl_low,
            last_bbl_high,
            last_rsi,
            last_ivr,
            last_adx,
            last_vwap
        )

        rows.append({
            "Ticker": t,
            "Price": round(last_price, 2),
            "VWAP": round(last_vwap, 2),
            "RSI": round(last_rsi, 2),
            "ADX": round(last_adx, 2),
            "IVR(dummy)": round(last_ivr, 2),
            "Status": status,
            "Reason": reason,
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)
    else:
        st.info("No data returned for any tickers.")