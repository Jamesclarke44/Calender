import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Neutral Calendar Pro", layout="centered")

# ----------------- STYLE -----------------

st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    height: 60px;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ----------------- CORE FUNCTIONS -----------------

def expected_move(price, iv, dte):
    return price * (iv / 100) * math.sqrt(dte / 365)

def classify_trade(price, vwap, rsi, adx, atr_pct, ivr, bbl_low, bbl_high, move):

    score = 0
    reasons = []

    if adx > 30:
        return "NO GO", 0, ["Trending market (ADX too high)"]

    if atr_pct > 3:
        return "NO GO", 0, ["Volatility too large"]

    if rsi < 35 or rsi > 65:
        return "NO GO", 0, ["Directional momentum"]

    if ivr > 50:
        return "NO GO", 0, ["IV too high"]

    if abs(price - vwap) / price < 0.01:
        score += 1
        reasons.append("Near VWAP")

    if 40 <= rsi <= 60:
        score += 1
        reasons.append("Neutral RSI")

    if adx < 20:
        score += 1
        reasons.append("Weak trend")

    if atr_pct < 2:
        score += 1
        reasons.append("Low volatility")

    if 20 <= ivr <= 40:
        score += 1
        reasons.append("Ideal IVR")

    if bbl_low <= price <= bbl_high:
        score += 1
        reasons.append("Inside Bollinger range")

    if move / price < 0.05:
        score += 1
        reasons.append("Tight expected move")

    if score >= 6:
        return "GO", score, reasons
    elif score >= 4:
        return "CAUTION", score, reasons
    else:
        return "NO GO", score, reasons

# ----------------- STRATEGY RECOMMENDER -----------------

def recommend_strategy(adx, rsi, vwap_drift, ivr):
    """
    Returns a strategy recommendation based on your laminated neutral rules.
    """

    # A+ neutral: single calendar
    if adx < 20 and 45 <= rsi <= 55 and vwap_drift < 0.005 and 20 <= ivr <= 40:
        return "Single Calendar — A+ Neutral Setup"

    # Extra safe neutral: double calendar
    if adx < 20 and 40 <= rsi <= 60 and vwap_drift < 0.0075 and 20 <= ivr <= 45:
        return "Double Calendar — Extra Safe, Wide Neutral Tent"

    # Neutral but drifting: wide iron condor
    if 20 <= adx <= 25 and 50 <= rsi <= 60 and 0.005 <= vwap_drift <= 0.01 and ivr >= 30:
        return "Wide Iron Condor — Neutral but Drifting"

    # High IV, low trend: jade lizard
    if adx < 20 and ivr > 40 and rsi < 60:
        return "Jade Lizard — High IV, Low Trend, No Upside Risk"

    # Defined-risk theta: wide credit spread
    if adx < 25 and ivr >= 30 and 40 <= rsi <= 60:
        return "Wide Credit Spread — Defined-Risk Theta Play"

    # No clear edge
    return "No Clear Edge — Stay Flat or Use Small Calendar Size"

# ----------------- REAL P&L ENGINE -----------------

def real_calendar_pnl(price, strike, move, front_iv, back_iv, debit):

    prices = np.linspace(price - move*2, price + move*2, 100)

    pnl = []

    for p in prices:

        distance = abs(p - strike)

        front_decay = front_iv * (1 - (distance / move))
        back_value = back_iv * (1 - (distance / (move * 1.5)))

        front_decay = max(front_decay, 0)
        back_value = max(back_value, 0)

        value = back_value - front_decay

        pnl_value = (value * 10) - debit

        pnl.append(pnl_value)

    pnl = np.array(pnl)

    fig, ax = plt.subplots()

    ax.plot(prices, pnl)
    ax.axvline(strike, linestyle="--")
    ax.axhline(0)

    ax.set_title("Real Calendar P&L")
    ax.set_xlabel("Price")
    ax.set_ylabel("Profit / Loss ($)")

    return fig

# ----------------- OPTIMIZER -----------------

def optimize_calendar(price, strikes, front_expiries, back_expiries, front_iv, back_iv):

    best_score = -999
    best = None

    for strike in strikes:
        for f in front_expiries:
            for b in back_expiries:

                score = 0

                # Strike proximity
                score += max(0, 1 - abs(price - strike) / price)

                # IV structure
                if back_iv > front_iv:
                    score += 2
                else:
                    score -= 2

                # IV spread magnitude
                score += min((back_iv - front_iv) / 10, 2)

                # Preferred DTE ranges
                if 7 <= f <= 14:
                    score += 1

                if 30 <= b <= 60:
                    score += 1

                if score > best_score:
                    best_score = score
                    best = {
                        "strike": strike,
                        "front_dte": f,
                        "back_dte": b,
                        "score": score
                    }

    return best

# ----------------- UI -----------------

st.title("📊 Neutral Calendar Pro V13")

ticker = st.text_input("Ticker", value="SPY").upper()

st.header("Market Inputs")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Price", value=450.0)
    vwap = st.number_input("VWAP", value=450.0)
    bbl_low = st.number_input("BB Low", value=445.0)
    bbl_high = st.number_input("BB High", value=455.0)

with col2:
    rsi = st.number_input("RSI", value=50.0)
    adx = st.number_input("ADX", value=18.0)
    atr = st.number_input("ATR", value=5.0)
    ivr = st.number_input("IV Rank", value=30.0)
    iv = st.number_input("IV (%)", value=25.0)
    dte = st.number_input("DTE", value=30)

st.header("Options Inputs")

front_iv = st.number_input("Front IV (%)", value=20.0)
back_iv = st.number_input("Back IV (%)", value=25.0)
debit = st.number_input("Debit Paid ($)", value=2.50)

front_dte = st.number_input("Front Expiry (DTE)", value=10)
back_dte = st.number_input("Back Expiry (DTE)", value=45)

# ----------------- RUN -----------------

if st.button("🚀 Evaluate Trade"):

    atr_pct = (atr / price) * 100
    move = expected_move(price, iv, dte)
    vwap_drift = abs(price - vwap) / price

    decision, score, reasons = classify_trade(
        price, vwap, rsi, adx, atr_pct, ivr, bbl_low, bbl_high, move
    )

    strike = round(price)

    st.markdown("---")
    st.subheader(f"📈 {ticker} Analysis")

    if decision == "GO":
        st.success(f"GO ✅ (Score {score}/7)")
    elif decision == "CAUTION":
        st.warning(f"CAUTION ⚠️ (Score {score}/7)")
    else:
        st.error(f"NO GO ⛔ (Score {score}/7)")

    st.subheader("🧠 Reasoning")
    for r in reasons:
        st.write(f"- {r}")

    st.subheader("📊 Key Metrics")
    st.write(f"- VWAP Drift: {vwap_drift*100:.2f}%")
    st.write(f"- ATR %: {atr_pct:.2f}%")

    # ----------------- STRATEGY RECOMMENDATION -----------------

    st.subheader("🎯 Strategy Recommendation")
    strategy = recommend_strategy(adx, rsi, vwap_drift, ivr)
    st.write(strategy)

    st.subheader("📈 Expected Move")
    st.write(f"± {move:.2f}")

    # Profit zones
    max_profit_low = strike - (move * 0.25)
    max_profit_high = strike + (move * 0.25)

    st.subheader("💰 Profit Zone")
    st.write(f"🟢 Max Profit: {max_profit_low:.2f} → {max_profit_high:.2f}")
    st.write(f"🟡 Expected Range: {price - move:.2f} → {price + move:.2f}")

    # ----------------- OPTIMIZER -----------------

    st.subheader("🎯 Auto Strike & Expiry Optimizer")

    strikes = [round(price - 5), round(price), round(price + 5)]
    front_expiries = list(range(front_dte - 2, front_dte + 3))
    back_expiries = list(range(back_dte - 5, back_dte + 6))

    best = optimize_calendar(price, strikes, front_expiries, back_expiries, front_iv, back_iv)

    if best:
        st.write(f"Strike: {best['strike']}")
        st.write(f"Front DTE: {best['front_dte']}")
        st.write(f"Back DTE: {best['back_dte']}")
        st.write(f"Score: {best['score']}")

    # ----------------- REAL P&L -----------------

    st.subheader("📈 Real P&L Visualization")

    fig = real_calendar_pnl(price, strike, move, front_iv, back_iv, debit)
    st.pyplot(fig)

    # ----------------- JOURNAL -----------------

    if "journal" not in st.session_state:
        st.session_state.journal = []

    st.session_state.journal.append({
        "time": datetime.now().strftime("%H:%M"),
        "ticker": ticker,
        "decision": decision,
        "score": score,
        "strike": strike,
        "strategy": strategy
    })

# ----------------- JOURNAL -----------------

st.markdown("---")
st.subheader("📓 Trade Journal")

if "journal" in st.session_state:
    for j in reversed(st.session_state.journal):
        st.write(j)
else:
    st.write("No trades yet.")
