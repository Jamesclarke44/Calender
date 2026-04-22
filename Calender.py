"""
Calender.py - Entry Signal Analyzer
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Entry Analyzer", page_icon="📈", layout="wide")

st.title("📈 Entry Signal Analyzer")
st.caption("Confirm entry signals before trading")

# Sidebar
with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").upper()
    direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True)
    entry_price = st.number_input("Entry Price ($)", value=180.00, step=0.01)
    analyze_btn = st.button("Analyze Entry", type="primary", use_container_width=True)

# Main
if analyze_btn:
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
            
            if df.empty:
                st.error("No data found")
            else:
                # Handle MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                close = df['Close']
                current = float(close.iloc[-1])
                
                # Simple moving averages
                sma20 = float(close.rolling(20).mean().iloc[-1])
                sma50 = float(close.rolling(50).mean().iloc[-1])
                
                # Simple RSI
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = float(rsi.iloc[-1])
                
                # Simple score
                score = 50
                if direction == "LONG":
                    if current > sma20 > sma50:
                        score += 20
                    if 30 < rsi_val < 70:
                        score += 10
                else:
                    if current < sma20 < sma50:
                        score += 20
                    if rsi_val > 60:
                        score += 10
                
                # Results
                col1, col2 = st.columns(2)
                
                with col1:
                    if score >= 70:
                        st.success(f"### ✅ Strong Buy\nScore: {score}/100")
                    elif score >= 60:
                        st.info(f"### 📈 Buy\nScore: {score}/100")
                    elif score >= 50:
                        st.warning(f"### ⚠️ Neutral\nScore: {score}/100")
                    else:
                        st.error(f"### ❌ Avoid\nScore: {score}/100")
                    
                    st.metric("Current Price", f"${current:.2f}")
                    st.metric("RSI", f"{rsi_val:.1f}")
                
                with col2:
                    st.metric("20 SMA", f"${sma20:.2f}")
                    st.metric("50 SMA", f"${sma50:.2f}")
                    
                    if direction == "LONG":
                        stop_suggestion = sma50 if sma50 < current else current * 0.95
                    else:
                        stop_suggestion = sma50 if sma50 > current else current * 1.05
                    
                    st.metric("Suggested Stop", f"${stop_suggestion:.2f}")
                
                st.divider()
                st.markdown("### 🚀 Next Step")
                st.markdown(f"Use **Exit Planner** with entry **${entry_price:.2f}** and stop **${stop_suggestion:.2f}**")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Try a different ticker")
else:
    st.info("👈 Enter a ticker and click 'Analyze Entry'")

st.divider()
st.caption("Entry Analyzer — Simple and reliable")