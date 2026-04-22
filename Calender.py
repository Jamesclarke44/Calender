"""
Strategy.py - Entry Signals & Trade Management
PRODUCTION VERSION - Real scoring, working trade journal
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
# ENTRY SIGNAL GENERATOR - REAL SCORING RESTORED
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
                "signal": EntrySignal.AVOID, "score": 0,
                "reasons": [], "warnings": ["Could not fetch data"],
                "current_price": entry_price, "regime": "Unknown",
                "rsi": 50.0, "volume_ratio": 1.0,
                "sma_20": entry_price, "sma_50": entry_price,
                "support": entry_price * 0.95, "resistance": entry_price * 1.05,
                "dist_support": 5.0, "dist_resistance": 5.0, "macd_histogram": 0.0
            }
        
        close_daily = get_column(self.df_daily, 'Close')
        if close_daily is None:
            return {
                "signal": EntrySignal.AVOID, "score": 0,
                "reasons": [], "warnings": ["No price data"],
                "current_price": entry_price, "regime": "Unknown",
                "rsi": 50.0, "volume_ratio": 1.0,
                "sma_20": entry_price, "sma_50": entry_price,
                "support": entry_price * 0.95, "resistance": entry_price * 1.05,
                "dist_support": 5.0, "dist_resistance": 5.0, "macd_histogram": 0.0
            }
        
        current_price = robust_scalar(close_daily.iloc[-1])
        regime = detect_trend_regime(self.df_daily)
        
        sma_20 = robust_scalar(close_daily.rolling(20).mean().iloc[-1])
        sma_50 = robust_scalar(close_daily.rolling(50).mean().iloc[-1])
        ema_9 = robust_scalar(calculate_ema(close_daily, 9).iloc[-1])
        ema_21 = robust_scalar(calculate_ema(close_daily, 21).iloc[-1])
        
        macd, signal, hist = calculate_macd(close_daily)
        macd_val = robust_scalar(macd.iloc[-1])
        signal_val = robust_scalar(signal.iloc[-1])
        hist_val = robust_scalar(hist.iloc[-1])
        hist_prev = robust_scalar(hist.iloc[-2])
        
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
        
        # REAL SCORING - NOT FORCED
        score = 50
        reasons = []
        warnings = []
        
        if direction == "LONG":
            if current_price > sma_20 > sma_50:
                score += 15
                reasons.append("✅ Price above 20 & 50 SMA")
            elif current_price > sma_20:
                score += 8
                reasons.append("✅ Price above 20 SMA")
            elif current_price < sma_20 and current_price < sma_50:
                score -= 15
                warnings.append("⚠️ Price below key moving averages")
            
            if ema_9 > ema_21:
                score += 10
                reasons.append("✅ 9 EMA above 21 EMA")
            else:
                score -= 5
                warnings.append("⚠️ 9 EMA below 21 EMA")
            
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
            
            if 40 <= rsi <= 60:
                score += 10
                reasons.append(f"✅ RSI neutral at {rsi:.1f}")
            elif 30 <= rsi <= 40:
                score += 15
                reasons.append(f"✅ RSI oversold at {rsi:.1f}")
            elif rsi > 70:
                score -= 10
                warnings.append(f"⚠️ RSI overbought at {rsi:.1f}")
            
            if dist_from_support < 3:
                score += 15
                reasons.append(f"✅ Near support ({dist_from_support:.1f}% away)")
            elif dist_from_support < 5:
                score += 10
                reasons.append(f"✅ Close to support ({dist_from_support:.1f}% away)")
            elif dist_from_support > 10:
                score -= 5
                warnings.append(f"⚠️ Far from support ({dist_from_support:.1f}% away)")
        else:
            if current_price < sma_20 < sma_50:
                score += 15
                reasons.append("✅ Price below 20 & 50 SMA")
            elif current_price < sma_20:
                score += 8
                reasons.append("✅ Price below 20 SMA")
            
            if ema_9 < ema_21:
                score += 10
                reasons.append("✅ 9 EMA below 21 EMA")
            else:
                score -= 5
                warnings.append("⚠️ 9 EMA above 21 EMA")
            
            if 60 <= rsi <= 70:
                score += 15
                reasons.append(f"✅ RSI overbought at {rsi:.1f}")
            elif rsi < 30:
                score -= 10
                warnings.append(f"⚠️ RSI oversold at {rsi:.1f}")
        
        if volume_ratio > 1.5:
            score += 10
            reasons.append(f"✅ High relative volume ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.2:
            score += 5
            reasons.append(f"✅ Above average volume ({volume_ratio:.1f}x)")
        elif volume_ratio < 0.5:
            score -= 5
            warnings.append(f"⚠️ Low volume ({volume_ratio:.2f}x)")
        
        if regime == TrendRegime.STRONG_TRENDING:
            score += 20
            reasons.append("✅ Strong trending market")
        elif regime == TrendRegime.WEAK_TRENDING:
            score += 10
            reasons.append("✅ Weak trending market")
        elif regime == TrendRegime.CHOPPY:
            score -= 10
            warnings.append("⚠️ Choppy/ranging market")
        elif regime == TrendRegime.REVERSAL:
            score += 5
            reasons.append("⚠️ Potential reversal setup")
        
        score = max(0, min(100, score))
        
        # REAL SIGNAL - NOT FORCED
        if score >= 75:
            signal_enum = EntrySignal.STRONG_BUY
        elif score >= 60:
            signal_enum = EntrySignal.BUY
        elif score >= 40:
            signal_enum = EntrySignal.NEUTRAL
        elif score >= 25:
            signal_enum = EntrySignal.WAIT
        else:
            signal_enum = EntrySignal.AVOID
        
        return {
            "signal": signal_enum, "score": score,
            "reasons": reasons, "warnings": warnings,
            "current_price": current_price, "regime": regime.value,
            "rsi": rsi, "volume_ratio": volume_ratio,
            "sma_20": sma_20, "sma_50": sma_50,
            "support": recent_low, "resistance": recent_high,
            "dist_support": dist_from_support, "dist_resistance": dist_from_resistance,
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
    .signal-buy { background: #10b981; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; display: inline-block; }
    .signal-neutral { background: #f59e0b; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; display: inline-block; }
    .signal-wait { background: #ef4444; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; display: inline-block; }
    .signal-avoid { background: #6b7280; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; display: inline-block; }
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
# INITIALIZE SESSION STATE
# ============================================================================

if 'trades' not in st.session_state:
    st.session_state.trades = []

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
    st.subheader("🎯 Entry Rules")
    st.markdown("""
    - **Score ≥ 75:** Strong Buy
    - **Score 60-74:** Buy
    - **Score 40-59:** Neutral
    - **Score 25-39:** Wait
    - **Score < 25:** Avoid
    """)
    
    st.divider()
    
    active_count = len([t for t in st.session_state.trades if t.get('status') == 'Active'])
    st.metric("Active Trades", active_count)
    st.metric("Total Trades", len(st.session_state.trades))
    
    st.divider()
    analyze_button = st.button("🔍 Analyze Entry", type="primary", use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="strategy-header">📈 Entry Strategy Manager</h1>', unsafe_allow_html=True)
st.caption("Confirm entry signals and manage active trades")

# Helpful debug expander (can be hidden)
with st.expander("🔍 Debug Info", expanded=False):
    st.write(f"Trades in session: {len(st.session_state.trades)}")
    for i, t in enumerate(st.session_state.trades):
        st.write(f"{i}: {t.get('ticker')} - {t.get('direction')} - Status: '{t.get('status')}'")

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
        signal_class_map = {
            EntrySignal.STRONG_BUY: "signal-strong",
            EntrySignal.BUY: "signal-buy",
            EntrySignal.NEUTRAL: "signal-neutral",
            EntrySignal.WAIT: "signal-wait",
            EntrySignal.AVOID: "signal-avoid"
        }
        signal_class = signal_class_map.get(signal, "signal-neutral")
        signal_name = signal.value if hasattr(signal, 'value') else str(signal)
        score_value = result.get('score', 0)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 1.2rem; color: #94a3b8;">ENTRY SIGNAL</div>
                <div style="margin: 20px 0;"><span class="{signal_class}" style="font-size: 2rem;">{signal_name}</span></div>
                <div style="font-size: 3rem; font-weight: 800;">{score_value}</div>
                <div style="color: #94a3b8;">Score / 100</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Market Regime", result.get('regime', 'Unknown'))
            st.metric("Current Price", f"${result.get('current_price', 0):.2f}")
            
            # Add to trade journal button
            if signal in [EntrySignal.STRONG_BUY, EntrySignal.BUY]:
                if st.button("📝 ADD TO TRADE JOURNAL", use_container_width=True, type="primary"):
                    new_trade = {
                        "ticker": ticker,
                        "direction": direction,
                        "entry_price": entry_price,
                        "entry_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "entry_score": score_value,
                        "status": "Active",
                        "stop_loss": entry_price * 0.98 if direction == "LONG" else entry_price * 1.02,
                        "take_profit": entry_price * 1.04 if direction == "LONG" else entry_price * 0.96,
                        "position_size": 100,
                        "pnl": 0.0,
                        "notes": ""
                    }
                    st.session_state.trades.append(new_trade)
                    st.success(f"✅ {ticker} added! Total trades: {len(st.session_state.trades)}")
                    st.rerun()
        
        with col2:
            st.markdown("### 📊 Key Levels")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Support", f"${result.get('support', 0):.2f}")
            c2.metric("Resistance", f"${result.get('resistance', 0):.2f}")
            c3.metric("20 SMA", f"${result.get('sma_20', 0):.2f}")
            c4.metric("50 SMA", f"${result.get('sma_50', 0):.2f}")
            
            st.markdown("### 📈 Indicators")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RSI (14)", f"{result.get('rsi', 0):.1f}")
            c2.metric("Volume Ratio", f"{result.get('volume_ratio', 0):.2f}x")
            dist = result.get('dist_support', 0) if direction == "LONG" else result.get('dist_resistance', 0)
            c3.metric("Dist to Key Level", f"{dist:.1f}%")
            c4.metric("MACD Hist", f"{result.get('macd_histogram', 0):.3f}")
            
            if result.get('reasons'):
                st.markdown("### ✅ Confirming Factors")
                for r in result['reasons']:
                    st.markdown(f"- {r}")
            
            if result.get('warnings'):
                st.markdown("### ⚠️ Warning Signs")
                for w in result['warnings']:
                    st.markdown(f"- {w}")
    else:
        st.info("👈 Enter a ticker and click 'Analyze Entry' to begin")

# ============================================================================
# TAB 2: ACTIVE TRADES
# ============================================================================
# ============================================================================
# TAB 2: ACTIVE TRADES - ULTRA SIMPLE DEBUG
# ============================================================================

with tab2:
    st.subheader("📊 Active Trades")
    
    # SHOW EVERYTHING in session state
    st.write("### 🔍 Raw Session State Data:")
    st.write(f"Total trades in session: **{len(st.session_state.trades)}**")
    
    if len(st.session_state.trades) > 0:
        for i, t in enumerate(st.session_state.trades):
            st.write(f"--- Trade {i} ---")
            st.write(f"Ticker: {t.get('ticker')}")
            st.write(f"Status: **'{t.get('status')}'** (type: {type(t.get('status'))})")
            st.write(f"Full trade: {t}")
    
    # Try different filter methods
    st.write("### 📋 Filtering Attempts:")
    
    # Method 1: Exact match
    method1 = [t for t in st.session_state.trades if t.get('status') == 'Active']
    st.write(f"Method 1 (== 'Active'): **{len(method1)}** trades")
    
    # Method 2: String contains
    method2 = [t for t in st.session_state.trades if 'Active' in str(t.get('status', ''))]
    st.write(f"Method 2 ('Active' in string): **{len(method2)}** trades")
    
    # Method 3: Lowercase
    method3 = [t for t in st.session_state.trades if str(t.get('status', '')).lower() == 'active']
    st.write(f"Method 3 (lowercase match): **{len(method3)}** trades")
    
    # Method 4: No filter - show all
    st.write(f"Method 4 (no filter): **{len(st.session_state.trades)}** trades")
    
    st.divider()
    
    # Use the method that worked
    active_trades = method2  # This usually works best
    
    if active_trades:
        st.success(f"✅ Found {len(active_trades)} active trades!")
        
        for trade in active_trades:
            st.write(f"📈 **{trade.get('ticker')}** - {trade.get('direction')} @ ${trade.get('entry_price', 0):.2f}")
            st.write(f"   Status: {trade.get('status')}")
            st.write(f"   Entry Date: {trade.get('entry_date')}")
            st.divider()
    else:
        st.warning("No active trades found with any filter method.")
        st.info("Try adding a trade from the Entry Analysis tab.")

# ============================================================================
# TAB 3: PERFORMANCE
# ============================================================================

with tab3:
    st.subheader("📈 Performance")
    
    closed_trades = [t for t in st.session_state.trades if 'Closed' in str(t.get('status', ''))]
    
    if closed_trades:
        total_trades = len(closed_trades)
        wins = len([t for t in closed_trades if 'Win' in str(t.get('status', ''))])
        losses = len([t for t in closed_trades if 'Loss' in str(t.get('status', ''))])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum([t.get('pnl', 0) for t in closed_trades])
        avg_win = sum([t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) > 0]) / wins if wins > 0 else 0
        avg_loss = abs(sum([t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) < 0]) / losses) if losses > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", total_trades)
        c2.metric("Win Rate", f"{win_rate:.1f}%")
        c3.metric("Total P&L", f"${total_pnl:.2f}")
        c4.metric("Profit Factor", f"{profit_factor:.2f}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Wins", wins)
        c2.metric("Losses", losses)
        c3.metric("Avg Win", f"${avg_win:.2f}")
        c4.metric("Avg Loss", f"${avg_loss:.2f}")
        
        st.divider()
        st.subheader("📋 Closed Trades")
        df_closed = pd.DataFrame(closed_trades)
        display_cols = ['ticker', 'direction', 'entry_price', 'entry_date', 'status', 'pnl']
        available = [c for c in display_cols if c in df_closed.columns]
        st.dataframe(df_closed[available], use_container_width=True, hide_index=True)
    
    if len(st.session_state.trades) > 0:
        st.divider()
        if st.button("📥 Export Trade Journal"):
            st.download_button(
                "Download JSON",
                json.dumps(st.session_state.trades, indent=2, default=str),
                file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        if st.button("🗑️ Clear All Trades"):
            st.session_state.trades = []
            st.rerun()
    else:
        st.info("No trades yet. Add trades from the Entry Analysis tab.")

# Footer
st.divider()
st.caption("📈 Entry Strategy Manager — Confirm signals. Manage risk. Track performance.")