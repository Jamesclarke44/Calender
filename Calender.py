"""
Strategy.py - Entry Signals & Trade Management
Run with: streamlit run Strategy.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
from enum import Enum

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Entry Strategy Manager",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# ENUMS & DATA CLASSES
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
# ROBUST SCALAR
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
# TECHNICAL INDICATORS FOR ENTRY
# ============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = calculate_ema(series, 12)
    ema_26 = calculate_ema(series, 26)
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

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
    """Determine if market is trending or choppy"""
    if len(df) < 20:
        return TrendRegime.CHOPPY
    
    high = get_column(df, 'High')
    low = get_column(df, 'Low')
    close = get_column(df, 'Close')
    
    if high is None or low is None or close is None:
        return TrendRegime.CHOPPY
    
    adx = calculate_adx(high, low, close, 14)
    current_adx = robust_scalar(adx.iloc[-1])
    
    # ATR ratio (current vs average)
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
    """Generate entry signals based on multiple timeframes and confluences"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.df_daily = None
        self.df_hourly = None
        
    def fetch_data(self) -> bool:
        """Fetch daily and hourly data"""
        try:
            self.df_daily = yf.download(self.ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
            self.df_hourly = yf.download(self.ticker, period="1mo", interval="1h", progress=False, auto_adjust=True)
            return not self.df_daily.empty and len(self.df_daily) > 50
        except:
            return False
    
    def analyze(self, entry_price: float, direction: str = "LONG") -> Dict:
        """Generate complete entry analysis"""
        if not self.fetch_data():
            return {"signal": EntrySignal.AVOID, "score": 0, "reasons": ["Data fetch failed"]}
        
        close_daily = get_column(self.df_daily, 'Close')
        close_hourly = get_column(self.df_hourly, 'Close')
        
        if close_daily is None:
            return {"signal": EntrySignal.AVOID, "score": 0, "reasons": ["No price data"]}
        
        current_price = robust_scalar(close_daily.iloc[-1])
        
        # Calculate all indicators
        regime = detect_trend_regime(self.df_daily)
        
        # Moving averages
        sma_20 = robust_scalar(close_daily.rolling(20).mean().iloc[-1])
        sma_50 = robust_scalar(close_daily.rolling(50).mean().iloc[-1])
        ema_9 = robust_scalar(calculate_ema(close_daily, 9).iloc[-1])
        ema_21 = robust_scalar(calculate_ema(close_daily, 21).iloc[-1])
        
        # MACD
        macd, signal, hist = calculate_macd(close_daily)
        macd_val = robust_scalar(macd.iloc[-1])
        signal_val = robust_scalar(signal.iloc[-1])
        hist_val = robust_scalar(hist.iloc[-1])
        hist_prev = robust_scalar(hist.iloc[-2])
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(close_daily)
        bb_upper = robust_scalar(upper.iloc[-1])
        bb_lower = robust_scalar(lower.iloc[-1])
        bb_middle = robust_scalar(middle.iloc[-1])
        
        # RSI
        from Strategy import calculate_rsi
        rsi = robust_scalar(calculate_rsi(close_daily, 14).iloc[-1])
        
        # Volume
        volume = get_column(self.df_daily, 'Volume')
        avg_volume = robust_scalar(volume.rolling(20).mean().iloc[-1]) if volume is not None else 0
        current_volume = robust_scalar(volume.iloc[-1]) if volume is not None else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Support/Resistance
        high = get_column(self.df_daily, 'High')
        low = get_column(self.df_daily, 'Low')
        recent_high = robust_scalar(high.tail(20).max()) if high is not None else current_price * 1.05
        recent_low = robust_scalar(low.tail(20).min()) if low is not None else current_price * 0.95
        
        # Distance from key levels
        dist_from_support = (current_price - recent_low) / current_price * 100
        dist_from_resistance = (recent_high - current_price) / current_price * 100
        
        # ======================================================================
        # SCORING SYSTEM (0-100)
        # ======================================================================
        
        score = 50  # Start neutral
        reasons = []
        warnings = []
        
        # Trend Alignment (25 points)
        if direction == "LONG":
            if current_price > sma_20 > sma_50:
                score += 15
                reasons.append("✅ Price above 20 & 50 SMA (bullish alignment)")
            elif current_price > sma_20:
                score += 8
                reasons.append("✅ Price above 20 SMA")
            elif current_price < sma_20 and current_price < sma_50:
                score -= 15
                warnings.append("⚠️ Price below key moving averages")
            
            if ema_9 > ema_21:
                score += 10
                reasons.append("✅ 9 EMA above 21 EMA (short-term bullish)")
            else:
                score -= 5
                warnings.append("⚠️ 9 EMA below 21 EMA")
        else:  # SHORT
            if current_price < sma_20 < sma_50:
                score += 15
                reasons.append("✅ Price below 20 & 50 SMA (bearish alignment)")
            elif current_price < sma_20:
                score += 8
                reasons.append("✅ Price below 20 SMA")
            
            if ema_9 < ema_21:
                score += 10
                reasons.append("✅ 9 EMA below 21 EMA (short-term bearish)")
            else:
                score -= 5
                warnings.append("⚠️ 9 EMA above 21 EMA")
        
        # MACD Signal (15 points)
        if direction == "LONG":
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
        else:
            if hist_val < 0 and hist_val < hist_prev:
                score += 15
                reasons.append("✅ MACD histogram negative and decreasing")
            elif hist_val < 0:
                score += 8
                reasons.append("✅ MACD histogram negative")
        
        # RSI (15 points)
        if direction == "LONG":
            if 40 <= rsi <= 60:
                score += 10
                reasons.append(f"✅ RSI neutral at {rsi:.1f}")
            elif 30 <= rsi <= 40:
                score += 15
                reasons.append(f"✅ RSI oversold at {rsi:.1f} (potential bounce)")
            elif rsi > 70:
                score -= 10
                warnings.append(f"⚠️ RSI overbought at {rsi:.1f}")
            elif rsi < 30:
                score += 5
                reasons.append(f"✅ RSI deeply oversold at {rsi:.1f}")
        else:
            if 40 <= rsi <= 60:
                score += 10
            elif 60 <= rsi <= 70:
                score += 15
                reasons.append(f"✅ RSI overbought at {rsi:.1f}")
            elif rsi < 30:
                score -= 10
                warnings.append(f"⚠️ RSI oversold at {rsi:.1f}")
        
        # Volume Confirmation (10 points)
        if volume_ratio > 1.5:
            score += 10
            reasons.append(f"✅ High relative volume ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.2:
            score += 5
            reasons.append(f"✅ Above average volume ({volume_ratio:.1f}x)")
        elif volume_ratio < 0.5:
            score -= 5
            warnings.append(f"⚠️ Low volume ({volume_ratio:.2f}x)")
        
        # Distance from Support/Resistance (15 points)
        if direction == "LONG":
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
            if dist_from_resistance < 3:
                score += 15
                reasons.append(f"✅ Near resistance ({dist_from_resistance:.1f}% away)")
            elif dist_from_resistance < 5:
                score += 10
                reasons.append(f"✅ Close to resistance ({dist_from_resistance:.1f}% away)")
        
        # Trend Regime (20 points)
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
        
        # Determine signal
        if score >= 75:
            signal = EntrySignal.STRONG_BUY
        elif score >= 60:
            signal = EntrySignal.BUY
        elif score >= 40:
            signal = EntrySignal.NEUTRAL
        elif score >= 25:
            signal = EntrySignal.WAIT
        else:
            signal = EntrySignal.AVOID
        
        return {
            "signal": signal,
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
# TRADE JOURNAL MANAGER
# ============================================================================

class TradeJournal:
    """Manage active and closed trades"""
    
    def __init__(self):
        if 'trades' not in st.session_state:
            st.session_state.trades = []
    
    def add_trade(self, trade: Dict):
        st.session_state.trades.append(trade)
    
    def update_trade(self, index: int, updates: Dict):
        if index < len(st.session_state.trades):
            st.session_state.trades[index].update(updates)
    
    def get_active_trades(self) -> List[Dict]:
        return [t for t in st.session_state.trades if t['status'] in [TradeStatus.ACTIVE.value, TradeStatus.PARTIAL_EXIT.value]]
    
    def get_closed_trades(self) -> List[Dict]:
        return [t for t in st.session_state.trades if t['status'] in [TradeStatus.CLOSED_WIN.value, TradeStatus.CLOSED_LOSS.value, TradeStatus.CLOSED_BREAKEVEN.value]]
    
    def calculate_portfolio_metrics(self) -> Dict:
        closed = self.get_closed_trades()
        active = self.get_active_trades()
        
        total_trades = len(closed)
        wins = len([t for t in closed if t['status'] == TradeStatus.CLOSED_WIN.value])
        losses = len([t for t in closed if t['status'] == TradeStatus.CLOSED_LOSS.value])
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum([t.get('pnl', 0) for t in closed])
        avg_win = sum([t.get('pnl', 0) for t in closed if t.get('pnl', 0) > 0]) / wins if wins > 0 else 0
        avg_loss = abs(sum([t.get('pnl', 0) for t in closed if t.get('pnl', 0) < 0]) / losses) if losses > 0 else 0
        
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": avg_win / avg_loss if avg_loss > 0 else 0,
            "active_trades": len(active),
            "active_risk": sum([t.get('risk_amount', 0) for t in active])
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
    .signal-strong { background: #22c55e; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; }
    .signal-buy { background: #10b981; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; }
    .signal-neutral { background: #f59e0b; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; }
    .signal-wait { background: #ef4444; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; }
    .signal-avoid { background: #6b7280; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700; }
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
# SESSION STATE INIT
# ============================================================================

if 'journal' not in st.session_state:
    st.session_state.journal = TradeJournal()
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# ============================================================================
# SIDEBAR - ENTRY CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("📈 Entry Configuration")
    
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True)
    entry_price = st.number_input("Entry Price ($)", value=180.00, step=0.01, format="%.2f")
    
    st.divider()
    
    st.subheader("🎯 Entry Rules")
    st.markdown("""
    - **Score ≥ 75:** Strong Buy - Full position
    - **Score 60-74:** Buy - Standard position
    - **Score 40-59:** Neutral - Reduce size
    - **Score 25-39:** Wait - Paper trade only
    - **Score < 25:** Avoid - No trade
    """)
    
    st.divider()
    
    analyze_button = st.button("🔍 Analyze Entry", type="primary", use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="strategy-header">📈 Entry Strategy Manager</h1>', unsafe_allow_html=True)
st.caption("Confirm entry signals and manage active trades")

# Tabs for different views
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
        
        # Signal Display
        signal = result['signal']
        signal_class = {
            EntrySignal.STRONG_BUY: "signal-strong",
            EntrySignal.BUY: "signal-buy",
            EntrySignal.NEUTRAL: "signal-neutral",
            EntrySignal.WAIT: "signal-wait",
            EntrySignal.AVOID: "signal-avoid"
        }[signal]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 1.2rem; color: #94a3b8;">ENTRY SIGNAL</div>
                <div class="{signal_class}" style="font-size: 2rem; margin: 20px 0;">{signal.value}</div>
                <div style="font-size: 3rem; font-weight: 800;">{result['score']}</div>
                <div style="color: #94a3b8;">Score / 100</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Market Regime
            st.metric("Market Regime", result['regime'])
            st.metric("Current Price", f"${result['current_price']:.2f}")
            
            # Add to journal button
            if signal in [EntrySignal.STRONG_BUY, EntrySignal.BUY]:
                if st.button("📝 Add to Trade Journal", use_container_width=True):
                    new_trade = {
                        "ticker": ticker,
                        "direction": direction,
                        "entry_price": entry_price,
                        "entry_date": datetime.now().strftime("%Y-%m-%d"),
                        "entry_score": result['score'],
                        "status": TradeStatus.ACTIVE.value,
                        "stop_loss": entry_price * 0.98 if direction == "LONG" else entry_price * 1.02,
                        "take_profit": entry_price * 1.04 if direction == "LONG" else entry_price * 0.96,
                        "position_size": 100,
                        "risk_amount": entry_price * 0.02 * 100,
                        "notes": f"Entry score: {result['score']}/100"
                    }
                    st.session_state.journal.add_trade(new_trade)
                    st.success(f"✅ {ticker} added to active trades!")
        
        with col2:
            # Key Levels
            st.markdown("### 📊 Key Levels")
            level_col1, level_col2, level_col3, level_col4 = st.columns(4)
            with level_col1:
                st.metric("Support", f"${result['support']:.2f}")
            with level_col2:
                st.metric("Resistance", f"${result['resistance']:.2f}")
            with level_col3:
                st.metric("20 SMA", f"${result['sma_20']:.2f}")
            with level_col4:
                st.metric("50 SMA", f"${result['sma_50']:.2f}")
            
            # Indicators
            st.markdown("### 📈 Technical Indicators")
            ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
            with ind_col1:
                st.metric("RSI (14)", f"{result['rsi']:.1f}")
            with ind_col2:
                st.metric("Volume Ratio", f"{result['volume_ratio']:.2f}x")
            with ind_col3:
                dist = result['dist_support'] if direction == "LONG" else result['dist_resistance']
                st.metric("Dist to Key Level", f"{dist:.1f}%")
            with ind_col4:
                st.metric("MACD Hist", f"{result['macd_histogram']:.3f}")
            
            # Reasons & Warnings
            st.markdown("### ✅ Confirming Factors")
            for reason in result['reasons']:
                st.markdown(f"- {reason}")
            
            if result['warnings']:
                st.markdown("### ⚠️ Warning Signs")
                for warning in result['warnings']:
                    st.markdown(f"- {warning}")
            
            # Trade Management Suggestions
            st.markdown("### 🎯 Trade Management")
            
            if signal == EntrySignal.STRONG_BUY:
                st.success("**Full Position Size** - Consider scaling in with 100% of planned position")
                st.info("**Wider Stop** - Use 2x ATR stop to allow for normal volatility")
            elif signal == EntrySignal.BUY:
                st.info("**Standard Position** - Use 75-100% of planned position")
                st.info("**Standard Stop** - Use 1.5x ATR stop")
            elif signal == EntrySignal.NEUTRAL:
                st.warning("**Reduced Size** - Consider 25-50% of planned position")
                st.warning("**Tighter Stop** - Use 1x ATR stop, take profits quickly")
            elif signal == EntrySignal.WAIT:
                st.error("**Paper Trade Only** - Track but do not commit capital")
                st.error("**Wait for Confirmation** - Look for specific trigger (e.g., break above resistance)")
            else:
                st.error("**Avoid This Setup** - Capital preservation is priority")
    
    else:
        st.info("👈 Enter a ticker and click 'Analyze Entry' to generate entry signals")

# ============================================================================
# TAB 2: ACTIVE TRADES
# ============================================================================

with tab2:
    st.subheader("📊 Active Trade Management")
    
    journal = st.session_state.journal
    active_trades = journal.get_active_trades()
    
    if active_trades:
        for i, trade in enumerate(active_trades):
            with st.expander(f"{trade['ticker']} - {trade['direction']} @ ${trade['entry_price']:.2f} (Score: {trade['entry_score']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Entry Date:** {trade['entry_date']}")
                    st.markdown(f"**Status:** {trade['status']}")
                    st.markdown(f"**Position:** {trade['position_size']} shares")
                
                with col2:
                    current_price = st.number_input("Current Price", value=trade['entry_price'], step=0.01, key=f"price_{i}")
                    
                    # Calculate current P&L
                    if trade['direction'] == "LONG":
                        pnl = (current_price - trade['entry_price']) * trade['position_size']
                        pnl_pct = (current_price - trade['entry_price']) / trade['entry_price'] * 100
                    else:
                        pnl = (trade['entry_price'] - current_price) * trade['position_size']
                        pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price'] * 100
                    
                    pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
                    st.markdown(f"**Current P&L:** <span style='color:{pnl_color}'>${pnl:.2f} ({pnl_pct:+.2f}%)</span>", unsafe_allow_html=True)
                
                with col3:
                    new_status = st.selectbox(
                        "Update Status",
                        [s.value for s in TradeStatus],
                        key=f"status_{i}"
                    )
                    
                    if st.button("Update Trade", key=f"update_{i}"):
                        journal.update_trade(i, {
                            "status": new_status,
                            "current_price": current_price,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct
                        })
                        st.success("Trade updated!")
                        st.rerun()
                
                # Trade notes
                st.text_area("Notes", value=trade.get('notes', ''), key=f"notes_{i}")
    else:
        st.info("No active trades. Use the Entry Analysis tab to add trades to your journal.")

# ============================================================================
# TAB 3: PERFORMANCE
# ============================================================================

with tab3:
    st.subheader("📈 Trading Performance")
    
    journal = st.session_state.journal
    metrics = journal.calculate_portfolio_metrics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", metrics['total_trades'])
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    with col3:
        st.metric("Total P&L", f"${metrics['total_pnl']:.2f}")
    with col4:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Wins", metrics['wins'])
    with col2:
        st.metric("Losses", metrics['losses'])
    with col3:
        st.metric("Avg Win", f"${metrics['avg_win']:.2f}")
    with col4:
        st.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")
    
    st.divider()
    
    # Closed trades table
    closed_trades = journal.get_closed_trades()
    
    if closed_trades:
        st.subheader("📋 Closed Trades")
        
        df_closed = pd.DataFrame(closed_trades)
        display_cols = ['ticker', 'direction', 'entry_price', 'entry_date', 'status', 'pnl', 'pnl_pct']
        if all(col in df_closed.columns for col in display_cols):
            display_df = df_closed[display_cols].copy()
            display_df.columns = ['Ticker', 'Dir', 'Entry', 'Date', 'Status', 'P&L', 'P&L %']
            
            def color_pnl(val):
                if isinstance(val, (int, float)):
                    return 'color: #22c55e' if val >= 0 else 'color: #ef4444'
                return ''
            
            styled = display_df.style.map(color_pnl, subset=['P&L', 'P&L %'])\
                                    .format({'Entry': '${:.2f}', 'P&L': '${:.2f}', 'P&L %': '{:.2f}%'})
            st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades yet. Update trade status in the Active Trades tab.")

# Footer
st.divider()
st.caption("📈 Entry Strategy Manager — Confirm signals. Manage risk. Track performance.")