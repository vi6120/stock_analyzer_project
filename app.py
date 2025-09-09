#!/usr/bin/env python3
"""
Simple Streamlit Stock Analyzer
Author: Vikas Ramaswamy
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# Test if dependencies work
try:
    from sentiment_analyzer import SentimentStockAnalyzer
    analyzer = SentimentStockAnalyzer()
    deps_ok = True
except Exception as e:
    deps_ok = False
    error_msg = str(e)

# Header
st.title("📈 Stock Analyzer")
st.write("*Created by Vikas Ramaswamy*")

if not deps_ok:
    st.error(f"Dependency Error: {error_msg}")
    st.write("Please check that all required packages are installed.")
    st.stop()

# Simple stock input
symbol = st.text_input("Enter Stock Symbol:", value="AAPL").upper()

if st.button("Analyze Stock"):
    if symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                result = analyzer.analyze_stock(symbol)
                
                if result:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${result['current_price']:.2f}")
                    
                    with col2:
                        if result['predicted_price']:
                            st.metric("Predicted Price", f"${result['predicted_price']:.2f}")
                        else:
                            st.metric("Predicted Price", "N/A")
                    
                    with col3:
                        st.metric("Score", f"{result['score']}/{result.get('max_score', 7)}")
                    
                    # Recommendation
                    rec = result['recommendation']
                    if rec == "STRONG BUY":
                        st.success(f"🚀 Recommendation: {rec}")
                    elif rec == "BUY":
                        st.success(f"📈 Recommendation: {rec}")
                    elif rec == "HOLD":
                        st.warning(f"⏸️ Recommendation: {rec}")
                    else:
                        st.error(f"📉 Recommendation: {rec}")
                    
                    # Details
                    st.subheader("Analysis Details")
                    st.write(f"**RSI:** {result['rsi']:.1f}")
                    st.write(f"**Sentiment Score:** {result['sentiment_score']:.3f}")
                    st.write(f"**Model Accuracy:** {result['model_accuracy']:.1%}")
                    
                    # Factors
                    st.write("**Analysis Factors:**")
                    for reason in result['reasons']:
                        st.write(f"• {reason}")
                        
                else:
                    st.error("Unable to analyze stock. Please check the symbol.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a stock symbol.")

st.markdown("---")
st.write("⚠️ **Disclaimer:** Educational purposes only. Not financial advice.")