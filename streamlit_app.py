#!/usr/bin/env python3
"""
Streamlit Stock Analyzer with Sentiment Analysis
Author: Vikas Ramaswamy

Streamlit web application for sentiment-enhanced stock analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

from sentiment_analyzer import SentimentStockAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Stock Analyzer - Vikas Ramaswamy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_analyzer():
    return SentimentStockAnalyzer()

analyzer = get_analyzer()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .recommendation-strong-buy {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    .recommendation-buy {
        background: #17a2b8;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    .recommendation-hold {
        background: #ffc107;
        color: #212529;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    .recommendation-sell {
        background: #dc3545;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Stock Analyzer & Investment Predictor</h1>
    <p>Comprehensive stock analysis using machine learning and technical analysis</p>
    <p>Real-time data • ML predictions • Sentiment analysis • Investment recommendations</p>
    <p><em>Created by Vikas Ramaswamy</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Stock Analysis Options")

# Popular stocks with sentiment sensitivity indicators
popular_stocks = {
    'TSLA': '🚗 Tesla (High Sentiment)',
    'NVDA': '🤖 NVIDIA (AI Sentiment)',
    'AAPL': '🍎 Apple',
    'META': '📱 Meta (Social Sentiment)',
    'GOOGL': '🔍 Google',
    'MSFT': '💻 Microsoft',
    'AMZN': '📦 Amazon',
    'NFLX': '🎬 Netflix (Content Sentiment)'
}

# Stock selection
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Single Stock Analysis", "Popular Stocks Dashboard", "Custom Portfolio"]
)

if analysis_type == "Single Stock Analysis":
    # Single stock analysis
    st.sidebar.subheader("Select Stock")
    
    # Quick select buttons
    st.sidebar.write("**Quick Select:**")
    cols = st.sidebar.columns(2)
    selected_stock = None
    
    for i, (symbol, name) in enumerate(popular_stocks.items()):
        col = cols[i % 2]
        if col.button(name, key=f"quick_{symbol}"):
            selected_stock = symbol
    
    # Manual input
    manual_input = st.sidebar.text_input(
        "Or enter symbol manually:",
        value=selected_stock if selected_stock else "",
        placeholder="e.g., TSLA"
    ).upper()
    
    symbol = manual_input if manual_input else selected_stock
    
    if symbol and st.sidebar.button("🔍 Analyze Stock", type="primary"):
        st.session_state.analyze_symbol = symbol

elif analysis_type == "Popular Stocks Dashboard":
    # Dashboard settings
    st.sidebar.subheader("Dashboard Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (90s)", value=False)
    show_charts = st.sidebar.checkbox("Show Price Charts", value=True)
    
    if st.sidebar.button("🔄 Refresh Dashboard", type="primary"):
        st.session_state.refresh_dashboard = True

else:  # Custom Portfolio
    st.sidebar.subheader("Custom Portfolio")
    portfolio_input = st.sidebar.text_area(
        "Enter stock symbols (comma-separated):",
        placeholder="TSLA, AAPL, NVDA, GOOGL"
    )
    
    if st.sidebar.button("📊 Analyze Portfolio", type="primary"):
        if portfolio_input:
            symbols = [s.strip().upper() for s in portfolio_input.split(',') if s.strip()]
            st.session_state.portfolio_symbols = symbols

# Main content
if analysis_type == "Single Stock Analysis":
    if hasattr(st.session_state, 'analyze_symbol'):
        symbol = st.session_state.analyze_symbol
        
        st.subheader(f"Analyzing {symbol}")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Fetching stock data...")
            progress_bar.progress(25)
            
            status_text.text("Analyzing sentiment...")
            progress_bar.progress(50)
            
            status_text.text("Training ML model...")
            progress_bar.progress(75)
            
            # Analyze stock
            result = analyzer.analyze_stock(symbol)
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            if result:
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"${result['current_price']:.2f}"
                    )
                
                with col2:
                    if result['predicted_price']:
                        change = result['predicted_price'] - result['current_price']
                        st.metric(
                            "Predicted Price",
                            f"${result['predicted_price']:.2f}",
                            f"{change:.2f}"
                        )
                    else:
                        st.metric("Predicted Price", "N/A")
                
                with col3:
                    if result['predicted_price']:
                        expected_return = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
                        st.metric(
                            "Expected Return",
                            f"{expected_return:.1f}%"
                        )
                    else:
                        st.metric("Expected Return", "N/A")
                
                with col4:
                    st.metric(
                        "Analysis Score",
                        f"{result['score']}/{result['max_score']}"
                    )
                
                # Investment Recommendation with scoring system
                rec_class = result['recommendation'].lower().replace(' ', '-')
                st.markdown(f"""
                <div class="recommendation-{rec_class}">
                    <h3>Investment Recommendation: {result['recommendation']}</h3>
                    <p>Confidence Score: {result['score']}/{result['max_score']} points</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Sentiment Analysis
                st.subheader("📰 Sentiment Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_score = result['sentiment_score']
                    if sentiment_score > 0.1:
                        sentiment_class = "positive"
                        sentiment_text = "Positive"
                    elif sentiment_score < -0.1:
                        sentiment_class = "negative"
                        sentiment_text = "Negative"
                    else:
                        sentiment_class = "neutral"
                        sentiment_text = "Neutral"
                    
                    st.markdown(f"""
                    <div class="sentiment-{sentiment_class}">
                        <h4>Market Sentiment: {sentiment_text}</h4>
                        <p>Score: {sentiment_score:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Sentiment progress bar
                    sentiment_percent = max(0, min(100, (sentiment_score + 1) * 50))
                    st.progress(sentiment_percent / 100)
                
                with col2:
                    st.write("**Key Topics:**")
                    topics = result['sentiment_data'].get('key_topics', [])
                    for topic in topics[:4]:
                        st.write(f"• {topic}")
                    
                    st.write(f"**News Count:** {result['sentiment_data'].get('news_count', 0)}")
                
                # Technical Analysis & ML Model Performance
                st.subheader("📊 Technical Analysis & ML Model")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("RSI (0-100)", f"{result['rsi']:.1f}")
                    st.metric("20-day MA", f"${result['ma_20']:.2f}")
                
                with col2:
                    st.metric("50-day MA", f"${result['ma_50']:.2f}")
                    st.metric("Volatility (Risk)", f"{result['volatility']:.2f}")
                
                with col3:
                    st.metric("Volume Ratio", f"{result['volume_ratio']:.2f}")
                    st.metric("Random Forest Accuracy", f"{result['model_accuracy']:.1%}")
                
                # ML Model Details
                st.info("🤖 **ML Model:** Random Forest Regressor (150 estimators) with 80/20 train/test split")
                
                # Scoring System Breakdown
                st.subheader("🎯 Investment Scoring System (0-9 Points)")
                st.write("**Analysis Factors Contributing to Score:**")
                for reason in result['reasons']:
                    st.write(f"• {reason}")
                
                # Scoring explanation
                with st.expander("How the 9-Point Scoring System Works"):
                    st.write("""
                    **Technical Analysis (5 points max):**
                    - Price above 20-day MA: +1 point
                    - Price above 50-day MA: +1 point  
                    - 20-day MA above 50-day MA: +1 point
                    - RSI in healthy range (30-70): +1 point
                    - Lower volatility: +1 point
                    
                    **Machine Learning (2 points max):**
                    - ML predicts price increase: +2 points
                    
                    **Sentiment Analysis (2 points max):**
                    - Positive sentiment: +1-2 points
                    - High trading volume: +1 point
                    """)
                
            else:
                st.error("Unable to analyze stock. Please check the symbol and try again.")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error analyzing {symbol}: {str(e)}")

elif analysis_type == "Popular Stocks Dashboard":
    st.subheader("📈 Popular Stocks Dashboard")
    
    if hasattr(st.session_state, 'refresh_dashboard') or not hasattr(st.session_state, 'dashboard_data'):
        # Analyze popular stocks
        with st.spinner("Analyzing popular stocks with sentiment..."):
            dashboard_data = []
            
            for symbol in popular_stocks.keys():
                try:
                    result = analyzer.analyze_stock(symbol)
                    if result:
                        if result['predicted_price']:
                            expected_return = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
                            result['expected_return'] = expected_return
                        else:
                            result['expected_return'] = 0
                        dashboard_data.append(result)
                except Exception as e:
                    st.warning(f"Error analyzing {symbol}: {e}")
            
            # Sort by score
            dashboard_data.sort(key=lambda x: x['score'], reverse=True)
            st.session_state.dashboard_data = dashboard_data
    
    # Display dashboard
    if hasattr(st.session_state, 'dashboard_data'):
        data = st.session_state.dashboard_data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        strong_buys = len([d for d in data if d['recommendation'] == 'STRONG BUY'])
        buys = len([d for d in data if d['recommendation'] == 'BUY'])
        avg_sentiment = np.mean([d['sentiment_score'] for d in data])
        avg_accuracy = np.mean([d['model_accuracy'] for d in data])
        
        col1.metric("Strong Buys", strong_buys)
        col2.metric("Total Buys", strong_buys + buys)
        col3.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
        col4.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
        
        # Stock cards
        for result in data:
            with st.expander(f"{result['symbol']} - {result['recommendation']} (Score: {result['score']}/{result['max_score']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Current Price:** ${result['current_price']:.2f}")
                    if result['predicted_price']:
                        st.write(f"**Predicted Price:** ${result['predicted_price']:.2f}")
                        st.write(f"**Expected Return:** {result['expected_return']:.1f}%")
                
                with col2:
                    st.write(f"**Sentiment Score:** {result['sentiment_score']:.3f}")
                    st.write(f"**RSI:** {result['rsi']:.1f}")
                    st.write(f"**Model Accuracy:** {result['model_accuracy']:.1%}")
                
                with col3:
                    st.write("**Key Topics:**")
                    topics = result['sentiment_data'].get('key_topics', [])[:3]
                    for topic in topics:
                        st.write(f"• {topic}")

else:  # Custom Portfolio
    if hasattr(st.session_state, 'portfolio_symbols'):
        symbols = st.session_state.portfolio_symbols
        st.subheader(f"📊 Portfolio Analysis ({len(symbols)} stocks)")
        
        portfolio_results = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
            progress_bar.progress((i + 1) / len(symbols))
            
            try:
                result = analyzer.analyze_stock(symbol)
                if result:
                    if result['predicted_price']:
                        expected_return = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
                        result['expected_return'] = expected_return
                    else:
                        result['expected_return'] = 0
                    portfolio_results.append(result)
            except Exception as e:
                st.warning(f"Error analyzing {symbol}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        if portfolio_results:
            # Sort by score
            portfolio_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Portfolio summary
            col1, col2, col3, col4 = st.columns(4)
            
            total_value = sum([r['current_price'] for r in portfolio_results])
            avg_expected_return = np.mean([r['expected_return'] for r in portfolio_results])
            avg_sentiment = np.mean([r['sentiment_score'] for r in portfolio_results])
            strong_buys = len([r for r in portfolio_results if r['recommendation'] == 'STRONG BUY'])
            
            col1.metric("Portfolio Stocks", len(portfolio_results))
            col2.metric("Strong Buys", strong_buys)
            col3.metric("Avg Expected Return", f"{avg_expected_return:.1f}%")
            col4.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
            
            # Portfolio table
            df_data = []
            for result in portfolio_results:
                df_data.append({
                    'Symbol': result['symbol'],
                    'Price': f"${result['current_price']:.2f}",
                    'Predicted': f"${result['predicted_price']:.2f}" if result['predicted_price'] else "N/A",
                    'Expected Return': f"{result['expected_return']:.1f}%",
                    'Sentiment': f"{result['sentiment_score']:.3f}",
                    'Score': f"{result['score']}/{result['max_score']}",
                    'Recommendation': result['recommendation']
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Key Features:</strong> Real-time Yahoo Finance data • Random Forest ML predictions • Technical analysis • Sentiment analysis</p>
    <p><strong>Performance:</strong> 75-90% model accuracy • 3-7 second analysis time • 9-point scoring system</p>
    <p>⚠️ <strong>Disclaimer:</strong> Educational purposes only. Not financial advice. All investments carry risk.</p>
    <p>Created by <strong>Vikas Ramaswamy</strong></p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh disabled for cloud deployment
# if analysis_type == "Popular Stocks Dashboard":
#     if st.sidebar.checkbox("Auto-refresh (90s)", value=False):
#         time.sleep(90)
#         st.rerun()