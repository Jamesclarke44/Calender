"""
Strategy.py - Entry Signals & Trade Management
DEBUG VERSION - Shows what's happening with trades
Run with: streamlit run Strategy.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Entry Strategy Manager",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# ENUMS
# ============================================================================

class EntrySignal(Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    NEUTRAL = "Neutral"
    WAIT = "Wait"
    AVOID = "Avoid"

class TradeStatus(Enum):
    PLANNED = "Planned"
    ACTIVE = "Active"
    PARTIAL_EXIT = "Partial Exit"
    CLOSED_WIN = "Closed (Win)"
    CLOSED_LOSS = "Closed (Loss)"
    CLOSED_BREAKEVEN = "Closed (Breakeven)"

class TrendRegime(Enum):
    STRONG_TRENDING = "Strong Trending"
    WEAK_TRENDING = "Weak Trending"
    CHOPPY = "Choppy/Ranging"
    REVERSAL = "Potential Reversal"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def robust_scalar(value):
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
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = calculate_ema(series, 12)
    ema_26 = calculate_ema(series, 26)
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def detect_trend_regime(df: pd.DataFrame) -> TrendRegime:
    if len(df) < 20:
        return TrendRegime.CHOPPY
    
    high = get_column(df, 'High')
    low = get_column(df, 'Low')
    close = get_column(df, 'Close')
    
    if high is None or low is None or close is None:
        return TrendRegime.CHOPPY
    
    adx = calculate_adx(high, low, close, 14)
    current_adx = robust_scalar(adx.iloc[-1])
    
    atr = calculate_atr(high, low, close, 14)
    current_atr = robust_scalar(atr.iloc[-1])
    avg_atr = robust_scalar(atr.rolling(window=50).mean().iloc[-1])
    atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1
    
    if current_adx > 25 and atr_ratio > 1.1:
        return TrendRegime.STRONG_TRENDING
    elif current_adx > 20:
        return TrendRegime.WEAK_TRENDING
    elif current_adx < 20 and atr_ratio < 0.9:
        return TrendRegime.REVERSAL
    else:
        return TrendRegime.CHOPPY

# ============================================================================
# ENTRY SIGNAL GENERATOR (SIMPLIFIED FOR DEBUGGING)
# ============================================================================

class EntrySignalGenerator:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.df_daily = None
        
    def fetch_data(self) -> bool:
        try:
            self.df_daily = yf.download(self.ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
            return not self.df_daily.empty and len(self.df_daily) > 30
        except:
            return False
    
    def analyze(self, entry_price: float, direction: str = "LONG") -> Dict:
        if not self.fetch_data():
            return {
                "signal": EntrySignal.AVOID,
                "score": 0,
                "reasons": [],
                "warnings": ["Could not fetch data"],
                "current_price": entry_price,
                "regime": "Unknown",
                "rsi": 50.0,
                "volume_ratio": 1.0,
                "sma_20": entry_price,
                "sma_50": entry_price,
                "support": entry_price * 0.95,
                "resistance": entry_price * 1.05,
                "dist_support": 5.0,
                "dist_resistance": 5.0,
                "macd_histogram": 0.0
            }
        
        close_daily = get_column(self.df_daily, 'Close')
        if close_daily is None:
            return {
                "signal": EntrySignal.AVOID,
                "score": 0,
                "reasons": [],
                "warnings": ["No price data"],
                "current_price": entry_price,
                "regime": "Unknown",
                "rsi": 50.0,
                "volume_ratio": 1.0,
                "sma_20": entry_price,
                "sma_50": entry_price,
                "support": entry_price * 0.95,
                "resistance": entry_price * 1.05,
                "dist_support": 5.0,
                "dist_resistance": 5.0,
                "macd_histogram": 0.0
            }
        
        current_price = robust_scalar(close_daily.iloc[-1])
        regime = detect_trend_regime(self.df_daily)
        
        sma_20 = robust_scalar(close_daily.rolling(20).mean().iloc[-1])
        sma_50 = robust_scalar(close_daily.rolling(50).mean().iloc[-1])
        rsi = robust_scalar(calculate_rsi(close_daily, 14).iloc[-1])
        
        volume = get_column(self.df_daily, 'Volume')
        avg_volume = robust_scalar(volume.rolling(20).mean().iloc[-1]) if volume is not None else 0
        current_volume = robust_scalar(volume.iloc[-1]) if volume is not None else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        high = get_column(self.df_daily, 'High')
        low = get_column(self.df_daily, 'Low')
        recent_high = robust_scalar(high.tail(20).max()) if high is not None else current_price * 1.05
        recent_low = robust_scalar(low.tail(20).min()) if low is not None else current_price * 0.95
        
        dist_from_support = (current_price - recent_low) / current_price * 100
        dist_from_resistance = (recent_high - current_price) / current_price * 100
        
        # Simple scoring for debugging
        score = 75  # Force Strong Buy for testing
        reasons = ["✅ Debug: Forced Strong Buy for testing"]
        warnings = []
        
        macd, signal, hist = calculate_macd(close_daily)
        hist_val = robust_scalar(hist.iloc[-1])
        
        return {
            "signal": EntrySignal.STRONG_BUY,  # Force Strong Buy
            "score": 75,
            "reasons": reasons,
            "warnings": warnings,
            "current_price": current_price,
            "regime": regime.value,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "support": recent_low,
            "resistance": recent_high,
            "dist_support": dist_from_support,
            "dist_resistance": dist_from_resistance,
            "macd_histogram": hist_val
        }

# ============================================================================
# UI STYLING
# ============================================================================

st.markdown("""
<style>
    .strategy-header {
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .signal-strong { background: #22c55e; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; display: inline-block; }
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE - CRITICAL
# ============================================================================

if 'trades' not in st.session_state:
    st.session_state.trades = []
    st.write("DEBUG: Initialized empty trades list")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📈 Entry Configuration")
    
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True)
    entry_price = st.number_input("Entry Price ($)", value=180.00, step=0.01, format="%.2f")
    
    st.divider()
    analyze_button = st.button("🔍 Analyze Entry", type="primary", use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="strategy-header">📈 Entry Strategy Manager (DEBUG)</h1>', unsafe_allow_html=True)

# DEBUG: Show current session state
st.markdown("### 🔍 Debug Info")
st.write(f"Number of trades in session state: **{len(st.session_state.trades)}**")
if len(st.session_state.trades) > 0:
    st.write("Trades:")
    for i, t in enumerate(st.session_state.trades):
        st.write(f"  {i}: {t.get('ticker')} - {t.get('direction')} - Status: {t.get('status')}")

tab1, tab2, tab3 = st.tabs(["🔍 Entry Analysis", "📊 Active Trades", "📈 Performance"])

# ============================================================================
# TAB 1: ENTRY ANALYSIS
# ============================================================================

with tab1:
    if analyze_button:
        with st.spinner(f"Analyzing {ticker}..."):
            generator = EntrySignalGenerator(ticker)
            result = generator.analyze(entry_price, direction)
            st.session_state.analysis_result = result
    
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        signal = result.get('signal', EntrySignal.AVOID)
        score_value = result.get('score', 0)
        
        st.success(f"Signal: {signal.value} | Score: {score_value}")
        st.metric("Current Price", f"${result.get('current_price', 0):.2f}")
        
        # SIMPLE ADD TRADE BUTTON
        if st.button("📝 ADD TO TRADE JOURNAL", use_container_width=True, type="primary"):
            new_trade = {
                "ticker": ticker,
                "direction": direction,
                "entry_price": entry_price,
                "entry_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "entry_score": score_value,
                "status": "Active",  # Plain string, not enum
                "stop_loss": entry_price * 0.98 if direction == "LONG" else entry_price * 1.02,
                "take_profit": entry_price * 1.04 if direction == "LONG" else entry_price * 0.96,
                "position_size": 100,
                "pnl": 0.0
            }
            st.session_state.trades.append(new_trade)
            st.success(f"✅ Added {ticker} to trades! Total trades: {len(st.session_state.trades)}")
            st.write("DEBUG: Trade added:", new_trade)
            st.rerun()
    else:
        st.info("👈 Click 'Analyze Entry' to begin")

# ============================================================================
# TAB 2: ACTIVE TRADES
# ============================================================================

with tab2:
    st.subheader("📊 Active Trades")
    
    # SIMPLE FILTER - just check if status contains "Active"
    active_trades = [t for t in st.session_state.trades if "Active" in str(t.get('status', ''))]
    
    st.write(f"DEBUG: Found {len(active_trades)} active trades out of {len(st.session_state.trades)} total")
    
    if active_trades:
        for i, trade in enumerate(active_trades):
            with st.expander(f"✅ {trade['ticker']} - {trade['direction']} @ ${trade['entry_price']:.2f}"):
                st.write(f"**Entry Date:** {trade.get('entry_date', 'N/A')}")
                st.write(f"**Status:** {trade.get('status', 'N/A')}")
                st.write(f"**Score:** {trade.get('entry_score', 'N/A')}")
                st.write(f"**Stop:** ${trade.get('stop_loss', 0):.2f}")
                st.write(f"**Target:** ${trade.get('take_profit', 0):.2f}")
                
                # Update status
                new_status = st.selectbox(
                    "Change Status",
                    ["Active", "Partial Exit", "Closed (Win)", "Closed (Loss)", "Closed (Breakeven)"],
                    key=f"status_{i}"
                )
                
                if st.button("Update Status", key=f"update_{i}"):
                    # Find and update the trade in the main list
                    for t in st.session_state.trades:
                        if t.get('ticker') == trade['ticker'] and t.get('entry_date') == trade['entry_date']:
                            t['status'] = new_status
                            break
                    st.success(f"Status updated to: {new_status}")
                    st.rerun()
                
                # Delete trade button
                if st.button("🗑️ Delete Trade", key=f"delete_{i}"):
                    st.session_state.trades = [t for t in st.session_state.trades if not (t.get('ticker') == trade['ticker'] and t.get('entry_date') == trade['entry_date'])]
                    st.success("Trade deleted!")
                    st.rerun()
    else:
        st.warning("No active trades found.")
        st.markdown("---")
        st.markdown("**All trades in session state:**")
        for i, t in enumerate(st.session_state.trades):
            st.write(f"{i}: {t.get('ticker')} - Status: '{t.get('status')}'")

# ============================================================================
# TAB 3: PERFORMANCE
# ============================================================================

with tab3:
    st.subheader("📈 Performance")
    
    if len(st.session_state.trades) > 0:
        st.metric("Total Trades", len(st.session_state.trades))
        
        # Export
        if st.button("Export Trades as JSON"):
            st.download_button(
                "Download",
                json.dumps(st.session_state.trades, indent=2, default=str),
                file_name="trades.json",
                mime="application/json"
            )
        
        # Clear all trades
        if st.button("🗑️ Clear All Trades", type="secondary"):
            st.session_state.trades = []
            st.success("All trades cleared!")
            st.rerun()
    else:
        st.info("No trades yet")

# Footer
st.divider()
st.caption("DEBUG VERSION - Check the debug info at the top of the page")