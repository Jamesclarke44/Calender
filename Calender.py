"""
Strategy.py - Entry Signals & Trade Management
FIXED: Trade journal persistence
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
import os

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
# ENTRY SIGNAL GENERATOR
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
                "warnings": ["Could not fetch data for this ticker"],
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
                "warnings": ["No price data available"],
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
        
        score = 50
        reasons = []
        warnings = []
        
        # Scoring logic
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
            # SHORT logic
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
            "signal": signal_enum,
            "score": score,
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

# CRITICAL: Initialize trades list in session state
if 'trades' not in st.session_state:
    st.session_state.trades = []

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Track if we just added a trade
if 'trade_added' not in st.session_state:
    st.session_state.trade_added = False

# ============================================================================
# HELPER FUNCTIONS FOR TRADE JOURNAL
# ============================================================================

def add_trade(trade: Dict):
    """Add a new trade to the journal"""
    st.session_state.trades.append(trade)
    st.session_state.trade_added = True

def update_trade(index: int, updates: Dict):
    """Update an existing trade"""
    if index < len(st.session_state.trades):
        st.session_state.trades[index].update(updates)

def get_active_trades() -> List[Dict]:
    """Get all active trades"""
    return [t for t in st.session_state.trades if t.get('status') in [TradeStatus.ACTIVE.value, TradeStatus.PARTIAL_EXIT.value]]

def get_closed_trades() -> List[Dict]:
    """Get all closed trades"""
    return [t for t in st.session_state.trades if t.get('status') in [TradeStatus.CLOSED_WIN.value, TradeStatus.CLOSED_LOSS.value, TradeStatus.CLOSED_BREAKEVEN.value]]

def calculate_metrics() -> Dict:
    """Calculate performance metrics"""
    closed = get_closed_trades()
    total = len(closed)
    wins = len([t for t in closed if t.get('status') == TradeStatus.CLOSED_WIN.value])
    losses = len([t for t in closed if t.get('status') == TradeStatus.CLOSED_LOSS.value])
    win_rate = (wins / total * 100) if total > 0 else 0
    total_pnl = sum([t.get('pnl', 0) for t in closed])
    avg_win = sum([t.get('pnl', 0) for t in closed if t.get('pnl', 0) > 0]) / wins if wins > 0 else 0
    avg_loss = abs(sum([t.get('pnl', 0) for t in closed if t.get('pnl', 0) < 0]) / losses) if losses > 0 else 0
    
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": avg_win / avg_loss if avg_loss > 0 else 0,
        "active_trades": len(get_active_trades())
    }

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
    analyze_button = st.button("🔍 Analyze Entry", type="primary", use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="strategy-header">📈 Entry Strategy Manager</h1>', unsafe_allow_html=True)
st.caption("Confirm entry signals and manage active trades")

# Show success message if trade was just added
if st.session_state.trade_added:
    st.success("✅ Trade added to journal successfully!")
    st.session_state.trade_added = False

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
            
            # FIXED: Add to trade journal button
            if signal in [EntrySignal.STRONG_BUY, EntrySignal.BUY]:
                if st.button("📝 Add to Trade Journal", use_container_width=True, key="add_trade_btn"):
                    new_trade = {
                        "id": len(st.session_state.trades),
                        "ticker": ticker,
                        "direction": direction,
                        "entry_price": entry_price,
                        "entry_date": datetime.now().strftime("%Y-%m-%d"),
                        "entry_score": score_value,
                        "status": TradeStatus.ACTIVE.value,
                        "stop_loss": entry_price * 0.98 if direction == "LONG" else entry_price * 1.02,
                        "take_profit": entry_price * 1.04 if direction == "LONG" else entry_price * 0.96,
                        "position_size": 100,
                        "risk_amount": entry_price * 0.02 * 100,
                        "notes": f"Entry score: {score_value}/100",
                        "pnl": 0.0
                    }
                    add_trade(new_trade)
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
            
            st.markdown("### 🎯 Trade Management")
            if signal == EntrySignal.STRONG_BUY:
                st.success("**Full Position** - 100% size, 2x ATR stop")
            elif signal == EntrySignal.BUY:
                st.info("**Standard Position** - 75-100% size")
            elif signal == EntrySignal.NEUTRAL:
                st.warning("**Reduced Size** - 25-50% size")
            elif signal == EntrySignal.WAIT:
                st.error("**Paper Trade Only**")
            else:
                st.error("**Avoid**")
    else:
        st.info("👈 Enter a ticker and click 'Analyze Entry' to begin")
        
        # Show sample of existing trades
        if len(st.session_state.trades) > 0:
            st.markdown(f"---")
            st.markdown(f"📊 **You have {len(st.session_state.trades)} trades in your journal**")

# ============================================================================
# TAB 2: ACTIVE TRADES
# ============================================================================

with tab2:
    st.subheader("📊 Active Trade Management")
    
    active_trades = get_active_trades()
    
    if active_trades:
        st.markdown(f"**{len(active_trades)} active trades**")
        
        for i, trade in enumerate(active_trades):
            # Find the original index in the full trades list
            original_index = next((idx for idx, t in enumerate(st.session_state.trades) if t.get('id') == trade.get('id')), i)
            
            with st.expander(f"{trade['ticker']} - {trade['direction']} @ ${trade['entry_price']:.2f} (Score: {trade.get('entry_score', 'N/A')})"):
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown(f"**Entry Date:** {trade.get('entry_date', 'N/A')}")
                    st.markdown(f"**Status:** {trade.get('status', 'Active')}")
                    st.markdown(f"**Position:** {trade.get('position_size', 100)} shares")
                    st.markdown(f"**Stop Loss:** ${trade.get('stop_loss', 0):.2f}")
                    st.markdown(f"**Take Profit:** ${trade.get('take_profit', 0):.2f}")
                
                with c2:
                    current = st.number_input("Current Price", value=trade['entry_price'], step=0.01, key=f"px_{trade.get('id', i)}")
                    
                    if trade['direction'] == "LONG":
                        pnl = (current - trade['entry_price']) * trade.get('position_size', 100)
                        pnl_pct = (current - trade['entry_price']) / trade['entry_price'] * 100
                    else:
                        pnl = (trade['entry_price'] - current) * trade.get('position_size', 100)
                        pnl_pct = (trade['entry_price'] - current) / trade['entry_price'] * 100
                    
                    color = "#22c55e" if pnl >= 0 else "#ef4444"
                    st.markdown(f"**Current P&L:** <span style='color:{color}'>${pnl:.2f} ({pnl_pct:+.2f}%)</span>", unsafe_allow_html=True)
                    
                    new_status = st.selectbox("Update Status", [s.value for s in TradeStatus], 
                                             index=[s.value for s in TradeStatus].index(trade.get('status', 'Active')),
                                             key=f"st_{trade.get('id', i)}")
                    
                    if st.button("Update Trade", key=f"upd_{trade.get('id', i)}"):
                        update_trade(original_index, {"status": new_status, "pnl": pnl, "pnl_pct": pnl_pct})
                        st.success("Trade updated!")
                        st.rerun()
                
                # Notes
                notes = st.text_area("Notes", value=trade.get('notes', ''), key=f"notes_{trade.get('id', i)}")
                if st.button("Save Notes", key=f"save_notes_{trade.get('id', i)}"):
                    update_trade(original_index, {"notes": notes})
                    st.success("Notes saved!")
                    st.rerun()
    else:
        st.info("No active trades. Go to the Entry Analysis tab, analyze a ticker, and click 'Add to Trade Journal'.")
        
        # Show all trades for debugging
        if len(st.session_state.trades) > 0:
            st.markdown("---")
            st.markdown(f"**All trades in journal ({len(st.session_state.trades)}):**")
            for t in st.session_state.trades:
                st.markdown(f"- {t.get('ticker')} - {t.get('direction')} @ ${t.get('entry_price', 0):.2f} - Status: {t.get('status', 'Unknown')}")

# ============================================================================
# TAB 3: PERFORMANCE
# ============================================================================

with tab3:
    st.subheader("📈 Trading Performance")
    
    metrics = calculate_metrics()
    closed_trades = get_closed_trades()
    
    if metrics['total_trades'] > 0 or len(st.session_state.trades) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", metrics['total_trades'])
        c2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        c3.metric("Total P&L", f"${metrics['total_pnl']:.2f}")
        c4.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Wins", metrics['wins'])
        c2.metric("Losses", metrics['losses'])
        c3.metric("Avg Win", f"${metrics['avg_win']:.2f}")
        c4.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")
        
        st.divider()
        
        if closed_trades:
            st.subheader("📋 Closed Trades")
            df_closed = pd.DataFrame(closed_trades)
            if not df_closed.empty:
                display_cols = ['ticker', 'direction', 'entry_price', 'entry_date', 'status', 'pnl']
                available_cols = [c for c in display_cols if c in df_closed.columns]
                st.dataframe(df_closed[available_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades yet. Update trade status in the Active Trades tab.")
        
        # Export button
        if st.button("📥 Export Trade Journal (JSON)"):
            export_data = json.dumps(st.session_state.trades, indent=2, default=str)
            st.download_button(
                "Download Journal",
                export_data,
                file_name=f"trade_journal_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    else:
        st.info("No trades in journal yet. Add trades from the Entry Analysis tab.")

# Footer
st.divider()
st.caption("📈 Entry Strategy Manager — Confirm signals. Manage risk. Track performance.")