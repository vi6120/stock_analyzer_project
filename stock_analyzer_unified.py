#!/usr/bin/env python3
"""
Unified Stock Analyzer & Investment Predictor
Author: Vikas Ramaswamy

Comprehensive stock analysis tool that combines technical indicators, machine learning,
and sentiment analysis (both simulated and real-time) for investment recommendations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import requests
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system environment variables

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

class UnifiedStockAnalyzer:
    """
    Unified Stock Analyzer with comprehensive analysis capabilities:
    - Technical indicators (Moving Averages, RSI, Volatility)
    - Machine Learning price predictions
    - Sentiment analysis (simulated and real-time)
    - Investment recommendations with 9-point scoring system
    """
    
    # Feature list for ML model
    FEATURES = [
        'Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Volatility',
        'Price_Momentum', 'Volume_Ratio', 'High_Low_Ratio', 'MACD', 'Sentiment'
    ]
    
    # Simulated sentiment data for demo purposes
    SENTIMENT_DATA = {
        'TSLA': {
            'news_sentiment': 0.15, 'social_sentiment': 0.25, 'news_count': 45,
            'key_topics': ['EV adoption', 'Autopilot updates', 'Elon Musk tweets', 'Production numbers'],
            'top_headlines': [
                'Tesla Reports Record Q4 Deliveries Beating Analyst Expectations',
                'Elon Musk Announces Major Autopilot Software Update Coming Next Month',
                'Tesla Gigafactory Production Ramps Up to Meet Growing EV Demand',
                'Tesla Stock Surges on Strong China Sales Data',
                'New Tesla Model Y Refresh Features Enhanced Battery Technology'
            ]
        },
        'AAPL': {
            'news_sentiment': 0.1, 'social_sentiment': 0.05, 'news_count': 32,
            'key_topics': ['iPhone sales', 'Services growth', 'China market', 'AI features'],
            'top_headlines': [
                'Apple Reports Strong iPhone 15 Sales Despite Economic Headwinds',
                'Apple Services Revenue Hits New Record High in Q4',
                'Apple Expands AI Features Across iOS Ecosystem',
                'Apple China Sales Show Signs of Recovery',
                'New Apple Vision Pro Pre-Orders Exceed Expectations'
            ]
        },
        'NVDA': {
            'news_sentiment': 0.3, 'social_sentiment': 0.35, 'news_count': 38,
            'key_topics': ['AI boom', 'Data center demand', 'Gaming market', 'Chip shortage'],
            'top_headlines': [
                'NVIDIA AI Chip Demand Continues to Surge in Data Centers',
                'NVIDIA Partners with Major Cloud Providers for AI Infrastructure',
                'Gaming Revenue Shows Strong Recovery for NVIDIA',
                'NVIDIA Stock Hits New All-Time High on AI Optimism',
                'New NVIDIA GPU Architecture Promises 40% Performance Boost'
            ]
        },
        'META': {
            'news_sentiment': 0.05, 'social_sentiment': 0.1, 'news_count': 28,
            'key_topics': ['Metaverse investment', 'Ad revenue', 'Privacy concerns', 'VR adoption'],
            'top_headlines': [
                'Meta Reports Improved Ad Revenue Growth in Latest Quarter',
                'Meta VR Headset Sales Show Steady Improvement',
                'Meta Announces New AI-Powered Advertising Tools',
                'Meta Faces New Privacy Regulations in European Markets',
                'Meta Reality Labs Division Reduces Losses Significantly'
            ]
        },
        'GOOGL': {
            'news_sentiment': 0.08, 'social_sentiment': 0.02, 'news_count': 25,
            'key_topics': ['Search dominance', 'Cloud growth', 'AI integration', 'Regulatory issues'],
            'top_headlines': [
                'Google Cloud Revenue Growth Accelerates in Q4',
                'Google Integrates Advanced AI into Search Results',
                'Google Faces Antitrust Scrutiny in Multiple Jurisdictions',
                'YouTube Ad Revenue Shows Strong Recovery Trends',
                'Google Announces Major AI Research Breakthrough'
            ]
        },
        'MSFT': {
            'news_sentiment': 0.12, 'social_sentiment': 0.05, 'news_count': 22,
            'key_topics': ['Azure growth', 'AI partnerships', 'Office 365', 'Gaming division'],
            'top_headlines': [
                'Microsoft Azure Continues Double-Digit Growth Streak',
                'Microsoft Copilot AI Assistant Gains Enterprise Adoption',
                'Office 365 Subscriber Base Reaches New Milestone',
                'Xbox Game Pass Subscription Numbers Hit Record High',
                'Microsoft Partners with OpenAI for Next-Gen AI Tools'
            ]
        },
        'AMZN': {
            'news_sentiment': 0.06, 'social_sentiment': 0.03, 'news_count': 30,
            'key_topics': ['E-commerce growth', 'AWS expansion', 'Logistics efficiency', 'Prime membership'],
            'top_headlines': [
                'Amazon AWS Revenue Growth Beats Analyst Expectations',
                'Amazon Prime Membership Reaches 200 Million Globally',
                'Amazon Logistics Network Expansion Continues Globally',
                'Amazon E-commerce Sales Show Holiday Season Strength',
                'Amazon Invests Heavily in AI and Machine Learning Infrastructure'
            ]
        },
        'NFLX': {
            'news_sentiment': 0.02, 'social_sentiment': 0.08, 'news_count': 20,
            'key_topics': ['Content strategy', 'Subscriber growth', 'Competition', 'International expansion'],
            'top_headlines': [
                'Netflix Subscriber Growth Exceeds Expectations in Q4',
                'Netflix Original Content Strategy Pays Off with Awards',
                'Netflix Expands Gaming Portfolio with New Titles',
                'Netflix International Markets Drive Revenue Growth',
                'Netflix Ad-Supported Tier Gains Significant Traction'
            ]
        }
    }
    
    def __init__(self, use_realtime_sentiment=True):
        """Initialize the unified analyzer."""
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42
        )
        
        # Sentiment configuration
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.use_realtime_sentiment = use_realtime_sentiment and VADER_AVAILABLE and bool(self.news_api_key)
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Sentiment-sensitive stocks with weights
        self.sentiment_weights = {
            'TSLA': 0.4, 'NVDA': 0.3, 'META': 0.3, 'NFLX': 0.25,
            'AAPL': 0.2, 'AMZN': 0.15, 'GOOGL': 0.15, 'MSFT': 0.1
        }
    
    def fetch_data(self, symbol, period="1y"):
        """Fetch historical stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators for stock analysis."""
        # Moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI with zero-division protection
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility and enhanced features
        data['Volatility'] = data['Close'].rolling(window=20).std()
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Momentum'] = data['Close'].pct_change(periods=5)
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        
        return data
    
    def get_sentiment_data(self, symbol):
        """Get sentiment data (real-time or simulated)."""
        if self.use_realtime_sentiment and self.news_api_key:
            return self._get_realtime_sentiment(symbol)
        else:
            return self._get_simulated_sentiment(symbol)
    
    def _get_realtime_sentiment(self, symbol):
        """Get real-time sentiment using News API."""
        try:
            company_names = {
                'TSLA': 'Tesla', 'AAPL': 'Apple', 'NVDA': 'NVIDIA',
                'META': 'Meta Facebook', 'GOOGL': 'Google Alphabet',
                'MSFT': 'Microsoft', 'AMZN': 'Amazon', 'NFLX': 'Netflix'
            }
            
            search_term = company_names.get(symbol, symbol)
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{search_term}" OR "{symbol}"',
                'from': yesterday, 'sortBy': 'relevancy',
                'language': 'en', 'pageSize': 20,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                news_data = response.json()
                return self._analyze_news_sentiment(news_data, symbol)
            else:
                return self._get_fallback_sentiment(symbol)
                
        except Exception as e:
            print(f"Error fetching real-time sentiment: {e}")
            return self._get_fallback_sentiment(symbol)
    
    def _analyze_news_sentiment(self, news_data, symbol):
        """Analyze sentiment from news articles."""
        articles = news_data.get('articles', [])
        
        if not articles:
            return self._get_fallback_sentiment(symbol)
        
        sentiments = []
        topics = set()
        
        for article in articles[:15]:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title}. {description}".strip()
            
            if text and len(text) > 10:
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                sentiments.append(sentiment['compound'])
                self._extract_topics(text, topics, symbol)
        
        if not sentiments:
            return self._get_fallback_sentiment(symbol)
        
        avg_sentiment = np.mean(sentiments)
        
        return {
            'sentiment_score': avg_sentiment,
            'news_count': len(articles),
            'key_topics': list(topics)[:6],
            'top_headlines': [article.get('title', '') for article in articles[:5] if article.get('title')],
            'source': 'news_api'
        }
    
    def _extract_topics(self, text, topics, symbol):
        """Extract key topics from news text."""
        topic_keywords = {
            'TSLA': ['autopilot', 'electric', 'EV', 'battery', 'charging', 'production'],
            'NVDA': ['AI', 'artificial intelligence', 'GPU', 'gaming', 'data center'],
            'AAPL': ['iPhone', 'iPad', 'Mac', 'services', 'App Store', 'China'],
            'META': ['metaverse', 'VR', 'advertising', 'social media', 'privacy'],
            'GOOGL': ['search', 'cloud', 'YouTube', 'advertising', 'AI'],
            'MSFT': ['Azure', 'Office', 'cloud', 'AI', 'gaming', 'Teams'],
            'AMZN': ['AWS', 'e-commerce', 'Prime', 'logistics', 'cloud'],
            'NFLX': ['streaming', 'content', 'subscribers', 'competition']
        }
        
        keywords = topic_keywords.get(symbol, [])
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                topics.add(keyword.title())
    
    def _get_simulated_sentiment(self, symbol):
        """Get simulated sentiment data for demo purposes."""
        default_sentiment = {
            'sentiment_score': 0.0, 'news_count': 15,
            'key_topics': ['Earnings reports', 'Market trends', 'Industry news'],
            'source': 'simulated'
        }
        
        data = self.SENTIMENT_DATA.get(symbol, default_sentiment.copy())
        
        # Calculate combined sentiment score
        news_sentiment = data.get('news_sentiment', 0)
        social_sentiment = data.get('social_sentiment', 0)
        news_count = data.get('news_count', 0)
        
        volume_weight = min(news_count / 30, 1.0)
        combined_sentiment = (news_sentiment * 0.6 + social_sentiment * 0.4) * volume_weight
        sensitivity = self.sentiment_weights.get(symbol, 0.1)
        
        return {
            'sentiment_score': max(-1.0, min(1.0, combined_sentiment * sensitivity)),
            'news_count': news_count,
            'key_topics': data.get('key_topics', []),
            'top_headlines': data.get('top_headlines', []),
            'source': 'simulated'
        }
    
    def _get_fallback_sentiment(self, symbol):
        """Fallback sentiment based on recent price performance."""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='5d')
            
            if len(hist) >= 2:
                recent_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
                sentiment = np.tanh(recent_change * 10)
            else:
                sentiment = 0.0
            
            return {
                'sentiment_score': sentiment,
                'news_count': 0,
                'key_topics': ['Market Performance'],
                'source': 'fallback'
            }
        except:
            return {
                'sentiment_score': 0.0, 'news_count': 0,
                'key_topics': [], 'source': 'fallback'
            }
    
    def prepare_features(self, data, sentiment_score=0):
        """Prepare features for ML model including sentiment."""
        # Add sentiment as time-varying feature
        time_factor = np.linspace(-0.02, 0.02, len(data))
        data['Sentiment'] = np.clip(sentiment_score + time_factor, -1, 1)
        
        # Create target
        data['Target'] = data['Close'].shift(-1)
        
        # Clean data
        clean_data = data[self.FEATURES + ['Target']].dropna()
        
        X = clean_data[self.FEATURES]
        y = clean_data['Target']
        
        return X, y
    
    def train_model(self, X, y):
        """Train model with TimeSeriesSplit validation."""
        tscv = TimeSeriesSplit(n_splits=3)
        test_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_test_scaled = fold_scaler.transform(X_test)
            
            fold_model = RandomForestRegressor(n_estimators=200, random_state=42)
            fold_model.fit(X_train_scaled, y_train)
            test_scores.append(fold_model.score(X_test_scaled, y_test))
        
        # Final training
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        train_score = self.model.score(X_scaled, y)
        test_score = np.mean(test_scores)
        
        return train_score, test_score
    
    def predict_price(self, data, sentiment_score=0):
        """Predict next day's price with sentiment consideration."""
        latest_data = data[self.FEATURES[:-1]].iloc[-1:].copy()
        latest_data['Sentiment'] = sentiment_score
        
        if latest_data.isna().any().any():
            return None
        
        scaled_data = self.scaler.transform(latest_data)
        prediction = self.model.predict(scaled_data)[0]
        
        return prediction
    
    def analyze_stock(self, symbol):
        """Complete unified stock analysis with sentiment enhancement."""
        print(f"\n=== Analyzing {symbol} ===")
        
        # Fetch stock data
        data = self.fetch_data(symbol)
        if data is None:
            return None
        
        # Get sentiment data
        sentiment_data = self.get_sentiment_data(symbol)
        sentiment_score = sentiment_data['sentiment_score']
        
        # Calculate technical indicators
        data = self.calculate_indicators(data)
        
        # Extract current metrics
        last_row = data.iloc[-1]
        current_price = last_row['Close']
        ma_20 = last_row['MA_20']
        ma_50 = last_row['MA_50']
        rsi = last_row['RSI']
        volatility = last_row['Volatility']
        volume_ratio = last_row['Volume_Ratio']
        
        # Prepare and train model
        X, y = self.prepare_features(data, sentiment_score)
        if len(X) < 50:
            print("Insufficient data for analysis")
            return None
        
        train_score, test_score = self.train_model(X, y)
        
        # Predict price
        predicted_price = self.predict_price(data, sentiment_score)
        
        # Enhanced scoring system (0-9 points)
        score = 0
        reasons = []
        
        # Technical analysis (5 points)
        if current_price > ma_20:
            score += 1
            reasons.append("Price above 20-day MA")
        if current_price > ma_50:
            score += 1
            reasons.append("Price above 50-day MA")
        if ma_20 > ma_50:
            score += 1
            reasons.append("20-day MA above 50-day MA")
        if 30 <= rsi <= 70:
            score += 1
            reasons.append("RSI in healthy range")
        if volatility < data['Volatility'].mean():
            score += 1
            reasons.append("Lower volatility")
        
        # ML prediction (2 points)
        if predicted_price and predicted_price > current_price:
            score += 2
            reasons.append("Model predicts price increase")
        
        # Sentiment analysis (2 points)
        if sentiment_score > 0.1:
            score += 2
            reasons.append("Positive market sentiment")
        elif sentiment_score > 0.05:
            score += 1
            reasons.append("Neutral-positive sentiment")
        elif sentiment_score < -0.1:
            score -= 1
            reasons.append("Negative market sentiment")
        
        # Volume confirmation
        if volume_ratio > 1.2:
            score += 1
            reasons.append("High trading volume")
        
        # Generate recommendation
        if predicted_price and predicted_price < current_price:
            price_drop = ((current_price - predicted_price) / current_price) * 100
            if symbol in self.sentiment_weights and sentiment_score < -0.1:
                price_drop *= 1.2
            
            if price_drop > 5:
                recommendation = "STRONG SELL"
            elif price_drop > 2:
                recommendation = "SELL"
            else:
                recommendation = "HOLD" if score >= 3 else "SELL"
        else:
            if symbol in self.sentiment_weights and sentiment_score > 0.15:
                score += 1
                reasons.append("High sentiment boost")
            
            if score >= 7:
                recommendation = "STRONG BUY"
            elif score >= 5:
                recommendation = "BUY"
            elif score >= 3:
                recommendation = "HOLD"
            else:
                recommendation = "SELL"
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'rsi': rsi,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'sentiment_score': sentiment_score,
            'sentiment_data': sentiment_data,
            'model_accuracy': test_score,
            'recommendation': recommendation,
            'score': max(0, min(9, score)),
            'max_score': 9,
            'reasons': reasons
        }

def main():
    """Main function to demonstrate unified analysis."""
    # Try real-time sentiment first, fallback to simulated
    try:
        analyzer = UnifiedStockAnalyzer(use_realtime_sentiment=True)
        if analyzer.use_realtime_sentiment:
            print("Using real-time sentiment analysis")
        else:
            print("Using simulated sentiment analysis")
    except:
        analyzer = UnifiedStockAnalyzer(use_realtime_sentiment=False)
        print("Using simulated sentiment analysis")
    
    stocks = ['TSLA', 'NVDA', 'AAPL', 'META', 'GOOGL']
    
    print("Unified Stock Analyzer & Investment Predictor")
    print("Author: Vikas Ramaswamy")
    print("=" * 60)
    
    results = []
    
    for symbol in stocks:
        try:
            result = analyzer.analyze_stock(symbol)
            if result:
                results.append(result)
                
                print(f"Current Price: ${result['current_price']:.2f}")
                if result['predicted_price']:
                    change = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
                    print(f"Predicted Price: ${result['predicted_price']:.2f} ({change:+.1f}%)")
                
                print(f"Sentiment: {result['sentiment_score']:.3f} (Source: {result['sentiment_data']['source']})")
                print(f"News Count: {result['sentiment_data']['news_count']}")
                print(f"Key Topics: {', '.join(result['sentiment_data']['key_topics'][:3])}")
                print(f"Recommendation: {result['recommendation']} (Score: {result['score']}/9)")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # Sort and display top recommendations
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n=== TOP RECOMMENDATIONS ===")
    for i, result in enumerate(results[:3], 1):
        expected_return = 0
        if result['predicted_price']:
            expected_return = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
        
        print(f"{i}. {result['symbol']} - {result['recommendation']}")
        print(f"   Score: {result['score']}/9 | Expected Return: {expected_return:+.1f}%")
        print(f"   Sentiment: {result['sentiment_score']:.3f} | Accuracy: {result['model_accuracy']:.1%}")

if __name__ == "__main__":
    main()