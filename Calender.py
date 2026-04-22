"""
2_📈_Entry_Analyzer.py - Entry Signal Analyzer
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Tuple

# ... (keep all your existing Calender.py code) ...
Calender

"""
Calender.py - Enhanced Entry Signal Analyzer
Full technical analysis with scoring, regime detection, and recommendations
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
# HELPER FUNCTIONS
# ============================================================================

def robust_scalar(value):
    """Safely convert pandas objects to float"""
    if value is None:
        return 0.0
    if isinstance(value, pd.Series):
        return float(value.iloc[0]) if len(value) > 0 and not pd.isna(value.iloc[0]) else 0.0
    if isinstance(value, pd.DataFrame):
        return float(value.iloc[0, 0]) if not value.empty and not pd.isna(value.iloc[0, 0]) else 0.0
    try:
        return float(value)
    except:
        return 0.0

def get_column(df, col_name):
    """Safely extract column from DataFrame"""
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        for i, col_tuple in enumerate(df.columns):
            if any(str(c).lower() == col_name.lower() for c in col_tuple):
                return df.iloc[:, i]
        return None
    for col in df.columns:
        if col.lower() == col_name.lower():
            return df[col]
    return None

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD Line, Signal Line, Histogram"""
    ema_12 = calculate_ema(series, 12)
    ema_26 = calculate_ema(series, 26)
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: Upper, Middle, Lower"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx

def detect_market_regime(df: pd.DataFrame) -> Dict:
    """Detect if market is trending, choppy, or reversing"""
    if len(df) < 20:
        return {"regime": "Unknown", "adx": 0, "atr_ratio": 1}
    
    high = get_column(df, 'High')
    low = get_column(df, 'Low')
    close = get_column(df, 'Close')
    
    if high is None or low is None or close is None:
        return {"regime": "Unknown", "adx": 0, "atr_ratio": 1}
    
    adx = calculate_adx(high, low, close, 14)
    current_adx = robust_scalar(adx.iloc[-1])
    
    atr = calculate_atr(high, low, close, 14)
    current_atr = robust_scalar(atr.iloc[-1])
    avg_atr = robust_scalar(atr.rolling(window=50).mean().iloc[-1]) if len(atr) >= 50 else current_atr
    atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1
    
    if current_adx > 25 and atr_ratio > 1.1:
        regime = "Strong Trending"
    elif current_adx > 20:
        regime = "Weak Trending"
    elif current_adx < 20 and atr_ratio < 0.9:
        regime = "Potential Reversal"
    else:
        regime = "Choppy/Ranging"
    
    return {"regime": regime, "adx": current_adx, "atr_ratio": atr_ratio}

def find_key_levels(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """Find support and resistance levels"""
    if len(df) < lookback:
        return {"support": 0, "resistance": 0, "pivot": 0}
    
    high = get_column(df, 'High')
    low = get_column(df, 'Low')
    close = get_column(df, 'Close')
    
    if high is None or low is None or close is None:
        return {"support": 0, "resistance": 0, "pivot": 0}
    
    recent_high = robust_scalar(high.tail(lookback).max())
    recent_low = robust_scalar(low.tail(lookback).min())
    pivot = (recent_high + recent_low + robust_scalar(close.iloc[-1])) / 3
    
    return {"support": recent_low, "resistance": recent_high, "pivot": pivot}

# ============================================================================
# ENTRY ANALYZER
# ============================================================================

def analyze_entry(ticker: str, entry_price: float, direction: str) -> Dict:
    """Main analysis function"""
    
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        
        if df.empty or len(df) < 50:
            return {"error": "Insufficient data"}
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        close = get_column(df, 'Close')
        high = get_column(df, 'High')
        low = get_column(df, 'Low')
        volume = get_column(df, 'Volume')
        
        if close is None:
            return {"error": "No price data"}
        
        # Current price and change
        current_price = robust_scalar(close.iloc[-1])
        prev_close = robust_scalar(close.iloc[-2])
        daily_change = ((current_price - prev_close) / prev_close) * 100
        
        # Moving Averages
        sma_20 = robust_scalar(close.rolling(20).mean().iloc[-1])
        sma_50 = robust_scalar(close.rolling(50).mean().iloc[-1])
        sma_200 = robust_scalar(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else 0
        ema_9 = robust_scalar(calculate_ema(close, 9).iloc[-1])
        ema_21 = robust_scalar(calculate_ema(close, 21).iloc[-1])
        
        # MACD
        macd, signal, hist = calculate_macd(close)
        macd_val = robust_scalar(macd.iloc[-1])
        signal_val = robust_scalar(signal.iloc[-1])
        hist_val = robust_scalar(hist.iloc[-1])
        hist_prev = robust_scalar(hist.iloc[-2])
        
        # RSI
        rsi = robust_scalar(calculate_rsi(close, 14).iloc[-1])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        bb_upper_val = robust_scalar(bb_upper.iloc[-1])
        bb_lower_val = robust_scalar(bb_lower.iloc[-1])
        bb_position = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val) if bb_upper_val > bb_lower_val else 0.5
        
        # Volume Analysis
        avg_volume = robust_scalar(volume.rolling(20).mean().iloc[-1]) if volume is not None else 0
        current_volume = robust_scalar(volume.iloc[-1]) if volume is not None else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # ATR
        atr = robust_scalar(calculate_atr(high, low, close, 14).iloc[-1])
        atr_percent = (atr / current_price) * 100
        
        # Market Regime
        regime_info = detect_market_regime(df)
        
        # Key Levels
        levels = find_key_levels(df)
        support = levels["support"]
        resistance = levels["resistance"]
        
        dist_to_support = ((current_price - support) / current_price) * 100 if support > 0 else 100
        dist_to_resistance = ((resistance - current_price) / current_price) * 100 if resistance > 0 else 100
        
        # ======================================================================
        # SCORING SYSTEM
        # ======================================================================
        
        score = 50
        reasons = []
        warnings = []
        
        if direction == "LONG":
            # Trend alignment
            if current_price > sma_20 > sma_50:
                score += 15
                reasons.append("✅ Price above 20 & 50 SMA (bullish trend)")
            elif current_price > sma_20:
                score += 8
                reasons.append("✅ Price above 20 SMA")
            elif current_price < sma_20 and current_price < sma_50:
                score -= 15
                warnings.append("⚠️ Price below key moving averages")
            
            # EMA cross
            if ema_9 > ema_21:
                score += 10
                reasons.append("✅ 9 EMA above 21 EMA (bullish momentum)")
            else:
                score -= 5
                warnings.append("⚠️ 9 EMA below 21 EMA")
            
            # MACD
            if hist_val > 0 and hist_val > hist_prev:
                score += 15
                reasons.append("✅ MACD histogram positive and increasing")
            elif hist_val > 0:
                score += 8
                reasons.append("✅ MACD histogram positive")
            elif macd_val > signal_val:
                score += 5
                reasons.append("✅ MACD above signal line")
            else:
                score -= 5
                warnings.append("⚠️ MACD bearish")
            
            # RSI
            if 40 <= rsi <= 60:
                score += 10
                reasons.append(f"✅ RSI neutral at {rsi:.1f}")
            elif 30 <= rsi < 40:
                score += 15
                reasons.append(f"✅ RSI oversold at {rsi:.1f} (bounce potential)")
            elif rsi > 70:
                score -= 10
                warnings.append(f"⚠️ RSI overbought at {rsi:.1f}")
            elif rsi < 30:
                score += 5
                reasons.append(f"✅ RSI deeply oversold at {rsi:.1f}")
            
            # Bollinger Bands
            if bb_position < 0.2:
                score += 10
                reasons.append("✅ Near lower Bollinger Band (oversold)")
            elif bb_position > 0.8:
                score -= 5
                warnings.append("⚠️ Near upper Bollinger Band (overbought)")
            
            # Distance to support
            if dist_to_support < 3:
                score += 15
                reasons.append(f"✅ Near support ({dist_to_support:.1f}% away)")
            elif dist_to_support < 5:
                score += 10
                reasons.append(f"✅ Close to support ({dist_to_support:.1f}% away)")
            elif dist_to_support > 10:
                score -= 5
                warnings.append(f"⚠️ Far from support ({dist_to_support:.1f}% away)")
        else:
            # SHORT logic
            if current_price < sma_20 < sma_50:
                score += 15
                reasons.append("✅ Price below 20 & 50 SMA (bearish trend)")
            elif current_price < sma_20:
                score += 8
                reasons.append("✅ Price below 20 SMA")
            
            if ema_9 < ema_21:
                score += 10
                reasons.append("✅ 9 EMA below 21 EMA (bearish momentum)")
            else:
                score -= 5
                warnings.append("⚠️ 9 EMA above 21 EMA")
            
            if hist_val < 0 and hist_val < hist_prev:
                score += 15
                reasons.append("✅ MACD histogram negative and decreasing")
            elif hist_val < 0:
                score += 8
                reasons.append("✅ MACD histogram negative")
            
            if 60 <= rsi <= 70:
                score += 15
                reasons.append(f"✅ RSI overbought at {rsi:.1f}")
            elif rsi > 70:
                score += 10
                reasons.append(f"✅ RSI extremely overbought at {rsi:.1f}")
            elif rsi < 30:
                score -= 10
                warnings.append(f"⚠️ RSI oversold at {rsi:.1f}")
            
            if bb_position > 0.8:
                score += 10
                reasons.append("✅ Near upper Bollinger Band (overbought)")
            
            if dist_to_resistance < 3:
                score += 15
                reasons.append(f"✅ Near resistance ({dist_to_resistance:.1f}% away)")
            elif dist_to_resistance < 5:
                score += 10
                reasons.append(f"✅ Close to resistance ({dist_to_resistance:.1f}% away)")
        
        # Volume confirmation
        if volume_ratio > 1.5:
            score += 10
            reasons.append(f"✅ High relative volume ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.2:
            score += 5
            reasons.append(f"✅ Above average volume ({volume_ratio:.1f}x)")
        elif volume_ratio < 0.5:
            score -= 5
            warnings.append(f"⚠️ Low volume ({volume_ratio:.2f}x)")
        
        # Market Regime
        if regime_info["regime"] == "Strong Trending":
            score += 15
            reasons.append("✅ Strong trending market")
        elif regime_info["regime"] == "Weak Trending":
            score += 8
            reasons.append("✅ Weak trending market")
        elif regime_info["regime"] == "Choppy/Ranging":
            score -= 10
            warnings.append("⚠️ Choppy/ranging market - reduce expectations")
        elif regime_info["regime"] == "Potential Reversal":
            score += 5
            reasons.append("⚠️ Potential reversal setup - use caution")
        
        # ATR check
        if 1.5 <= atr_percent <= 5.0:
            score += 5
            reasons.append(f"✅ ATR {atr_percent:.1f}% - good for trading")
        elif atr_percent > 5.0:
            warnings.append(f"⚠️ High volatility - ATR {atr_percent:.1f}%")
        
        # Cap score
        score = max(0, min(100, score))
        
        # Determine signal
        if score >= 75:
            signal = "STRONG BUY"
            signal_color = "#22c55e"
        elif score >= 60:
            signal = "BUY"
            signal_color = "#10b981"
        elif score >= 40:
            signal = "NEUTRAL"
            signal_color = "#f59e0b"
        elif score >= 25:
            signal = "WAIT"
            signal_color = "#ef4444"
        else:
            signal = "AVOID"
            signal_color = "#6b7280"
        
        # Suggested stop loss
        if direction == "LONG":
            suggested_stop = min(support * 0.99, current_price - (atr * 1.5))
        else:
            suggested_stop = max(resistance * 1.01, current_price + (atr * 1.5))
        
        return {
            "ticker": ticker,
            "direction": direction,
            "entry_price": entry_price,
            "current_price": current_price,
            "daily_change": daily_change,
            "signal": signal,
            "signal_color": signal_color,
            "score": score,
            "reasons": reasons,
            "warnings": warnings,
            "regime": regime_info["regime"],
            "adx": regime_info["adx"],
            "rsi": rsi,
            "macd_histogram": hist_val,
            "volume_ratio": volume_ratio,
            "atr": atr,
            "atr_percent": atr_percent,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "support": support,
            "resistance": resistance,
            "dist_to_support": dist_to_support,
            "dist_to_resistance": dist_to_resistance,
            "bb_position": bb_position,
            "suggested_stop": suggested_stop
        }
        
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# UI STYLING
# ============================================================================

st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .signal-box {
        text-align: center;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
    }
    .metric-card {
        background: #1e2a3a;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #334155;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
    }
    .reason-good { color: #22c55e; }
    .reason-warning { color: #f59e0b; }
    .reason-bad { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📈 Configuration")
    
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True)
    entry_price = st.number_input("Entry Price ($)", value=180.00, step=0.01, format="%.2f")
    
    st.divider()
    st.subheader("🎯 Scoring Guide")
    st.markdown("""
    - **75-100:** ⭐ Strong Buy
    - **60-74:** 📈 Buy
    - **40-59:** ⚠️ Neutral
    - **25-39:** ⏳ Wait
    - **0-24:** ❌ Avoid
    """)
    
    st.divider()
    analyze_btn = st.button("🔍 Analyze Entry", type="primary", use_container_width=True)
    
    st.divider()
    st.caption("After analysis, use **Exit Planner** to calculate stops and position size.")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="header">📈 Entry Signal Analyzer</h1>', unsafe_allow_html=True)
st.caption("Comprehensive technical analysis for trade entry confirmation")

if analyze_btn:
    with st.spinner(f"Analyzing {ticker}... This may take a few seconds."):
        result = analyze_entry(ticker, entry_price, direction)
        st.session_state.analysis_result = result

if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
        st.info("Try a different ticker or check your internet connection.")
    else:
        # ======================================================================
        # SIGNAL DISPLAY
        # ======================================================================
        
        st.markdown(f"""
        <div class="signal-box" style="background: {result['signal_color']}15; border: 2px solid {result['signal_color']};">
            <div style="font-size: 1.2rem; color: #94a3b8; margin-bottom: 10px;">ENTRY SIGNAL</div>
            <div style="font-size: 3.5rem; font-weight: 800; color: {result['signal_color']};">{result['signal']}</div>
            <div style="font-size: 1.5rem; color: white; margin-top: 10px;">Score: {result['score']}/100</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ======================================================================
        # QUICK METRICS ROW
        # ======================================================================
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"${result['current_price']:.2f}", 
                     f"{result['daily_change']:+.2f}%")
        with col2:
            st.metric("Market Regime", result['regime'])
        with col3:
            st.metric("RSI (14)", f"{result['rsi']:.1f}")
        with col4:
            st.metric("Volume Ratio", f"{result['volume_ratio']:.2f}x")
        with col5:
            st.metric("ATR %", f"{result['atr_percent']:.2f}%")
        
        st.divider()
        
        # ======================================================================
        # MAIN ANALYSIS COLUMNS
        # ======================================================================
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("### 📊 Key Levels")
            
            level_col1, level_col2 = st.columns(2)
            with level_col1:
                st.metric("Support", f"${result['support']:.2f}", 
                         f"{result['dist_to_support']:.1f}% away")
                st.metric("20 SMA", f"${result['sma_20']:.2f}")
                st.metric("50 SMA", f"${result['sma_50']:.2f}")
            
            with level_col2:
                st.metric("Resistance", f"${result['resistance']:.2f}", 
                         f"{result['dist_to_resistance']:.1f}% away")
                st.metric("9 EMA", f"${result['ema_9']:.2f}")
                st.metric("21 EMA", f"${result['ema_21']:.2f}")
            
            if result['sma_200'] > 0:
                st.metric("200 SMA", f"${result['sma_200']:.2f}")
            
            st.markdown("### 🎯 Suggested Stop Loss")
            stop_color = "#22c55e" if result['direction'] == "LONG" else "#ef4444"
            st.markdown(f"""
            <div style="background: {stop_color}20; border: 1px solid {stop_color}; 
                        border-radius: 12px; padding: 15px; text-align: center;">
                <span style="font-size: 1.8rem; font-weight: 700; color: {stop_color};">
                    ${result['suggested_stop']:.2f}
                </span>
                <br>
                <span style="color: #94a3b8;">Based on ATR and support/resistance</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Confirming factors
            if result['reasons']:
                st.markdown("### ✅ Confirming Factors")
                for reason in result['reasons']:
                    st.markdown(f"- {reason}")
            
            # Warning signs
            if result['warnings']:
                st.markdown("### ⚠️ Warning Signs")
                for warning in result['warnings']:
                    st.markdown(f"- {warning}")
            
            # Additional indicators
            st.markdown("### 📈 Additional Indicators")
            
            ind_col1, ind_col2 = st.columns(2)
            with ind_col1:
                st.metric("ADX", f"{result['adx']:.1f}")
                st.metric("BB Position", f"{result['bb_position']*100:.0f}%")
            with ind_col2:
                st.metric("MACD Hist", f"{result['macd_histogram']:.4f}")
        
        # ======================================================================
        # RECOMMENDATION
        # ======================================================================
        
        st.divider()
        st.markdown("### 🎯 Trade Management Recommendation")
        
        if result['signal'] == "STRONG BUY":
            st.success("""
            **Full Position Size** - This setup has strong confluence across multiple indicators.
            - Use standard position size (100%)
            - Place stop below support or 1.5x ATR
            - Consider scaling out at resistance levels
            """)
        elif result['signal'] == "BUY":
            st.info("""
            **Standard Position** - Good setup with positive factors.
            - Use 75-100% of normal position size
            - Standard stop loss rules apply
            - Monitor for additional confirmation
            """)
        elif result['signal'] == "NEUTRAL":
            st.warning("""
            **Reduced Position** - Mixed signals, proceed with caution.
            - Reduce position size to 25-50%
            - Tighten stop loss
            - Take profits quicker than usual
            """)
        elif result['signal'] == "WAIT":
            st.error("""
            **Paper Trade Only** - Setup needs more confirmation.
            - Track but do not commit capital
            - Wait for specific trigger (break above resistance, volume spike, etc.)
            - Re-evaluate in 1-2 days
            """)
        else:
            st.error("""
            **Avoid This Setup** - Risk/reward not favorable.
            - Pass on this trade
            - Look for better setups in your scanner
            - Capital preservation is priority
            """)
        
        # ======================================================================
        # NEXT STEPS
        # ======================================================================
        
        st.divider()
        st.markdown("### 🚀 Next Steps")
        
        next_col1, next_col2 = st.columns(2)
        
        with next_col1:
            st.markdown(f"""
            **1. Copy these values for Exit Planner:**
            - Ticker: **{result['ticker']}**
            - Direction: **{result['direction']}**
            - Entry Price: **${result['entry_price']:.2f}**
            - Suggested Stop: **${result['suggested_stop']:.2f}**
            - Support: **${result['support']:.2f}**
            - Resistance: **${result['resistance']:.2f}**
            - ATR: **${result['atr']:.2f}** ({result['atr_percent']:.2f}%)
            """)
        
        with next_col2:
            st.markdown("""
            **2. Open Exit Strategy Command Center**
            - Input the values above
            - Calculate exact position size
            - Set take profit levels
            - Execute with defined exit plan
            """)

else:
    st.info("👈 Enter a ticker and click 'Analyze Entry' to begin")
    
    st.markdown("""
    ### 📋 What This Analyzer Checks
    
    | Category | Indicators |
    |---|---|
    | **Trend** | SMA 20/50/200, EMA 9/21 alignment |
    | **Momentum** | MACD histogram, RSI (14) |
    | **Volatility** | ATR, Bollinger Bands |
    | **Volume** | Volume ratio vs 20-day average |
    | **Market Regime** | ADX (trending vs choppy) |
    | **Key Levels** | Support/resistance from 50-day range |
    
    ### ⭐ Scoring System
    
    - Starts at **50 points** (neutral)
    - **+15** Strong trend alignment
    - **+15** MACD confirmation
    - **+15** Near key support/resistance
    - **+10** Favorable RSI
    - **+10** High relative volume
    - **+15** Strong trending market
    
    **Score 60+ is a valid entry.** Use the Exit Planner for stops and position size.
    """)

# Footer
st.divider()
st.caption("📈 Entry Signal Analyzer — Comprehensive technical analysis for confident entries")
# In the sidebar, use shared state:
with st.sidebar:
    st.header("📈 Configuration")
    
    # Use shared ticker from session state
    ticker = st.text_input("Ticker Symbol", value=st.session_state.shared_ticker).upper()
    direction = st.radio("Direction", ["LONG", "SHORT"], 
                        index=0 if st.session_state.shared_direction == "LONG" else 1,
                        horizontal=True)
    entry_price = st.number_input("Entry Price ($)", 
                                  value=st.session_state.shared_entry, 
                                  step=0.01, format="%.2f")
    
    # Quick load button
    if st.button("📥 Load from Scanner", use_container_width=True):
        st.success(f"Loaded {st.session_state.shared_ticker}")
    
    # ... rest of sidebar ...

# After analysis, update shared state:
if st.session_state.analysis_result and "error" not in st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    # Update shared state
    st.session_state.shared_ticker = ticker
    st.session_state.shared_entry = entry_price
    st.session_state.shared_direction = direction
    st.session_state.shared_score = result['score']
    st.session_state.shared_support = result['support']
    st.session_state.shared_resistance = result['resistance']
    st.session_state.shared_atr = result['atr']
    st.session_state.shared_stop = result['suggested_stop']
    
    # Button to go to exit planner
    if st.button("🎯 Go to Exit Planner", use_container_width=True, type="primary"):
        st.switch_page("pages/3_🎯_Exit_Planner.py")