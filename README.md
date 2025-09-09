# Stock Analyzer & Investment Predictor

**Author: Vikas Ramaswamy**

A comprehensive Python application that analyzes stocks and provides investment recommendations using machine learning and technical analysis. Features both command-line and web interfaces with real-time data streaming.

## Features

- **Real-time Stock Data**: Fetches live stock data using Yahoo Finance API
- **Technical Analysis**: Calculates moving averages, RSI, volatility indicators
- **Machine Learning Predictions**: Uses Random Forest algorithm to predict next day's price
- **Investment Recommendations**: Provides BUY/SELL/HOLD recommendations with 7-point scoring system
- **Web Interface**: Beautiful, responsive web application with live streaming
- **Command-line Interface**: Interactive terminal-based analysis tool
- **Popular Stock Streaming**: Auto-updates analysis for major stocks every minute
- **Sentiment Analysis**: Enhanced predictions using news and social media sentiment

## Installation

1. **Clone or download the project**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Streamlit Web Application (Recommended)
🌐 **Live Demo:** https://my-stock-analyzer.streamlit.app/

Or run locally:
```bash
streamlit run streamlit_app.py
```
Or use the runner script:
```bash
python run_streamlit.py
```
Then open your browser to: **http://localhost:8501**

**Streamlit Features:**
- Interactive dashboard with real-time analysis
- Single stock analysis with detailed metrics
- Popular stocks dashboard with sentiment tracking
- Custom portfolio analysis
- Beautiful charts and responsive design
- Auto-refresh capabilities

### Enhanced Web Application (Flask)
Launch the sentiment-enhanced Flask interface:
```bash
python enhanced_web_app.py
```
Then open your browser to: **http://localhost:5002**

### Command Line Analysis
Run default analysis on popular stocks:
```bash
python stock_analyzer.py
```

### Sentiment-Enhanced Analysis
Run sentiment-enhanced analysis:
```bash
python sentiment_analyzer.py
```

### Interactive Custom Analysis
Analyze specific stocks interactively:
```bash
python custom_analyzer.py
```

## How It Works

### Technical Indicators
- **Moving Averages (MA20, MA50)**: Trend analysis and momentum
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100)
- **Volatility**: 20-day rolling standard deviation for risk assessment
- **Price Trends**: Direction and strength analysis

### Machine Learning Model
- **Algorithm**: Random Forest Regressor (150 estimators for enhanced version)
- **Features**: Open, High, Low, Volume, MA20, MA50, RSI, Volatility, Sentiment
- **Training**: 80/20 train/test split with StandardScaler normalization
- **Prediction**: Next day's closing price with accuracy metrics

### Sentiment Analysis
- **News Sentiment**: Analysis of news headlines and articles
- **Social Media Sentiment**: Social media buzz and sentiment tracking
- **Stock-Specific Weighting**: Different sentiment sensitivity for different stocks
- **Tesla Enhancement**: 40% sentiment weight for highly volatile stocks

### Enhanced Scoring System (0-9 points)
| Criteria | Points | Description |
|----------|--------|-------------|
| Price above 20-day MA | +1 | Short-term bullish trend |
| Price above 50-day MA | +1 | Medium-term bullish trend |
| 20-day MA above 50-day MA | +1 | Moving average crossover signal |
| RSI in healthy range (30-70) | +1 | Not overbought/oversold |
| Lower than average volatility | +1 | Reduced risk |
| ML predicts price increase | +2 | Model confidence boost |
| Positive sentiment | +1-2 | News and social sentiment |
| High trading volume | +1 | Volume confirmation |

### Smart Recommendation Logic
The system prioritizes ML predictions and sentiment over technical scores:

```python
if predicted_price < current_price:
    price_drop = ((current_price - predicted_price) / current_price) * 100
    # Factor in sentiment for volatile stocks
    if symbol in sentiment_sensitive_stocks and sentiment_score < -0.1:
        price_drop *= 1.2  # Amplify negative sentiment impact
    
    if price_drop > 5: recommendation = "STRONG SELL"
    elif price_drop > 2: recommendation = "SELL"
    else: recommendation = "HOLD" if score >= 3 else "SELL"
else:
    # Positive sentiment boost for sentiment-sensitive stocks
    if symbol in sentiment_sensitive_stocks and sentiment_score > 0.15:
        score += 1
    
    if score >= 7: recommendation = "STRONG BUY"
    elif score >= 5: recommendation = "BUY"
    elif score >= 3: recommendation = "HOLD"
    else: recommendation = "SELL"
```

### Investment Recommendations
- **STRONG BUY**: Score 7-9 points - High confidence buy signal
- **BUY**: Score 5-6 points - Moderate buy signal  
- **HOLD**: Score 3-4 points - Neutral, maintain position
- **SELL**: Score 1-2 points - Sell signal or avoid
- **STRONG SELL**: Score 0 points or significant predicted drop

## Example Output

```
=== Analyzing TSLA with Sentiment Analysis ===
Current Price: $185.25
Predicted Price: $187.40
Expected Change: +1.2%
Sentiment Score: 0.076
News Count: 45
Key Topics: EV adoption, Autopilot updates, Elon Musk tweets
Model Accuracy: 82.5%
Recommendation: STRONG BUY (Score: 8/9)
Analysis Factors: Price above 20-day MA, RSI in healthy range, Model predicts price increase, Positive market sentiment
```

## Project Structure

```
stock_analyzer_project/
├── stock_analyzer.py         # Core analysis engine with ML model
├── sentiment_analyzer.py     # Enhanced analyzer with sentiment
├── streamlit_app.py         # Modern Streamlit web application (Recommended)
├── enhanced_web_app.py      # Flask web application
├── custom_analyzer.py       # Interactive command-line interface
├── run_streamlit.py         # Streamlit runner script
├── requirements.txt         # Python dependencies
└── README.md               # This documentation
```

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **yfinance**: Yahoo Finance API for stock data
- **Flask**: Web application framework
- **textblob**: Sentiment analysis
- **requests**: HTTP requests for news data

### Data Sources
- **Yahoo Finance**: Real-time and historical stock data
- **Technical Indicators**: Calculated from price/volume data
- **ML Features**: Derived from technical analysis
- **Sentiment Data**: News headlines and social media sentiment

### Performance
- **Model Accuracy**: Typically 75-90% on test data (improved with sentiment)
- **Update Frequency**: Web app updates popular stocks every 90 seconds
- **Response Time**: Analysis typically completes in 3-7 seconds per stock
- **Tesla Accuracy**: Significantly improved with sentiment analysis

## Important Disclaimers

1. **Educational Purpose**: This tool is designed for learning and research purposes only
2. **Not Financial Advice**: Do not use as sole basis for investment decisions
3. **Market Risk**: All investments carry risk of loss
4. **Data Accuracy**: Relies on third-party data sources
5. **Model Limitations**: Past performance doesn't guarantee future results
6. **Sentiment Limitations**: Sentiment analysis is based on simulated data for demo purposes

## Contributing

This project was created by **Vikas Ramaswamy**. Feel free to fork, modify, and enhance the codebase for your own learning and research purposes.

## License

This project is open source and available for educational and research use.

---

⚠️ **Disclaimer**: This tool is for educational purposes only. Not financial advice. Always do your own research before investing.

**Created by Vikas Ramaswamy**