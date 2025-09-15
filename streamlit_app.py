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

from stock_analyzer_unified import UnifiedStockAnalyzer

# Initialize analyzer with automatic sentiment detection
analyzer = UnifiedStockAnalyzer(use_realtime_sentiment=True)
if analyzer.use_realtime_sentiment:
    st.info("🔴 **LIVE**: Using real-time sentiment analysis from news sources")
else:
    st.warning("⚠️ Using simulated sentiment. Run setup_apis.py for real-time data")
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Stock Analyzer - Vikas Ramaswamy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def calculate_expected_return(predicted_price, current_price):
    """Calculate expected return percentage."""
    if predicted_price and current_price > 0:
        return ((predicted_price - current_price) / current_price) * 100
    return 0

# Custom CSS for dark/light compatibility
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
        background: var(--background-color);
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
    .news-ticker {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        overflow: hidden;
        white-space: nowrap;
        position: relative;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .news-ticker::before {
        content:'Latest Trending NEWS';
        position: absolute;
        left: 15px;
        top: 50%;
        transform: translateY(-50%);
        background: #dc3545;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        z-index: 2;
    }
    .news-item {
        display: inline-block;
        animation: scroll 05s linear infinite;
        padding-left: 120px;
        padding-right: 50px;
        font-weight: 500;
        color: #2c3e50;
    }
    @keyframes scroll {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>Stock Analyzer & Investment Predictor</h1>
    <p>Professional stock analysis using machine learning and technical analysis</p>
    <p>Real-time data | ML predictions | Sentiment analysis | Investment recommendations</p>
    <p><em>Created by Vikas Ramaswamy</em></p>
</div>
""", unsafe_allow_html=True)



# Sidebar
st.sidebar.title("Stock Analysis Options")

# Popular stocks with sentiment sensitivity indicators
popular_stocks = {
    'TSLA': 'Tesla (High Sentiment)',
    'NVDA': 'NVIDIA (AI Sentiment)',
    'AAPL': 'Apple',
    'META': 'Meta (Social Sentiment)',
    'GOOGL': 'Google',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'NFLX': 'Netflix (Content Sentiment)'
}

# Stock selection
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Homepage", "Single Stock Analysis", "Custom Portfolio"],
    index=0
)

if analysis_type == "Single Stock Analysis":
    # Single stock analysis
    st.sidebar.subheader("Select Stock")
    
    # Quick select buttons
    st.sidebar.write("**Quick Select:**")
    cols = st.sidebar.columns(2)
    
    for i, (symbol, name) in enumerate(popular_stocks.items()):
        col = cols[i % 2]
        if col.button(name, key=f"quick_{symbol}"):
            st.session_state.analyze_symbol = symbol
            st.rerun()
    
    # Manual input
    manual_input = st.sidebar.text_input(
        "Or enter symbol manually:",
        placeholder="e.g., TSLA",
        key="stock_input"
    ).upper()
    
    if st.sidebar.button("Analyze Stock", type="primary") and manual_input:
        st.session_state.analyze_symbol = manual_input
        st.rerun()
    
    # Enter key support
    if manual_input and manual_input != st.session_state.get('last_input', ''):
        st.session_state.last_input = manual_input
        st.session_state.analyze_symbol = manual_input
        st.rerun()

elif analysis_type == "Custom Portfolio":
    st.sidebar.subheader("Custom Portfolio")
    portfolio_input = st.sidebar.text_area(
        "Enter stock symbols (comma-separated):",
        placeholder="TSLA, AAPL, NVDA, GOOGL"
    )
    
    if st.sidebar.button("Analyze Portfolio", type="primary"):
        if portfolio_input:
            symbols = [s.strip().upper() for s in portfolio_input.split(',') if s.strip()]
            st.session_state.portfolio_symbols = symbols

# Main content
if analysis_type == "Homepage":
    # Professional header info
    st.markdown("""
    <div style="background: var(--background-color); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #667eea;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>Author:</strong> Vikas Ramaswamy | <strong>Version:</strong> 1.0 | <strong>Technology:</strong> Python, Random Forest, yfinance, Sentiment Analysis
            </div>
            <div style="color: #6c757d; font-size: 0.9rem;">
                Professional Stock Analysis & Investment Prediction Platform
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Features Slider
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; color: white; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; align-items: center;">
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; backdrop-filter: blur(10px);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">NEWS</div>
                <div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem;">Analyze Breaking News</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Track live sentiment from 20+ sources and discover what's moving markets</div>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; backdrop-filter: blur(10px);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">SCORE</div>
                <div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem;">Score Investment Potential</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Get clear buy/sell signals using our 9-point system that combines technical + AI + sentiment</div>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; backdrop-filter: blur(10px);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">INSIGHTS</div>
                <div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem;">Discover Market Drivers</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Uncover why stocks move with AI that spots earnings, partnerships, and regulatory shifts</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    


elif analysis_type == "Single Stock Analysis":
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
                    expected_return = calculate_expected_return(result['predicted_price'], result['current_price'])
                    if expected_return != 0:
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
                st.subheader("Sentiment Analysis")
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
                
                # Top News Headlines Section with Ticker
                if result['sentiment_data'].get('top_headlines'):
                    source_type = "Latest News Headlines" if result['sentiment_data'].get('source') == 'news_api' else "Sample News Headlines (Demo)"
                    st.subheader(source_type)
                    headlines = result['sentiment_data']['top_headlines'][:5]
                    ticker_text = " | ".join(headlines)
                    
                    st.markdown(f"""
                    <div class="news-ticker">
                        <div class="news-item">{ticker_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result['sentiment_data'].get('source') != 'news_api':
                        st.info("These are sample headlines for demonstration. Set NEWS_API_KEY for real-time news data.")
                
                # Technical Analysis & ML Model Performance
                st.subheader("Technical Analysis & ML Model")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("RSI (0-100)", f"{result['rsi']:.1f}")
                    st.metric("20-day MA", f"${result['ma_20']:.2f}")
                
                with col2:
                    st.metric("50-day MA", f"${result['ma_50']:.2f}")
                    st.metric("Volatility (Risk)", f"{result['volatility']:.2f}")
                
                with col3:
                    st.metric("Volume Ratio", f"{result['volume_ratio']:.2f}")
                    st.metric("Volatility (Risk)", f"{result['volatility']:.2f}")
                
                # ML Model Details
                st.info("**ML Model:** Random Forest Regressor for educational demonstration purposes")
                
                # Scoring System Breakdown
                st.subheader("Investment Scoring System (0-9 Points)")
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

else:  # Custom Portfolio
    if hasattr(st.session_state, 'portfolio_symbols'):
        symbols = st.session_state.portfolio_symbols
        st.subheader(f"Portfolio Analysis ({len(symbols)} stocks)")
        
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
                    result['expected_return'] = calculate_expected_return(result['predicted_price'], result['current_price'])
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
            col3.metric("Avg Score", f"{np.mean([r['score'] for r in portfolio_results]):.1f}/9")
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

# Professional Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px; margin-top: 2rem;">
    <div style="margin-bottom: 1rem;">
        <strong>Stock Analyzer & Investment Predictor</strong>
    </div>
    <div style="color: #6c757d; margin-bottom: 1rem;">
        Professional stock analysis platform with machine learning predictions and sentiment-enhanced recommendations
    </div>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem; flex-wrap: wrap;">
        <div><strong>Features:</strong> Real-time Data | ML Predictions | Sentiment Analysis</div>
        <div><strong>Performance:</strong> 9-Point Scoring System</div>
    </div>
    <div style="color: #6c757d; font-size: 0.9rem;">
        © 2024 Vikas Ramaswamy | Professional Analytics Portfolio | Educational Purpose Only
    </div>
</div>
""", unsafe_allow_html=True)

