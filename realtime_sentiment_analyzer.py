#!/usr/bin/env python3
"""
Real-time Sentiment-Enhanced Stock Analyzer
Author: Vikas Ramaswamy

Uses free APIs for real-time sentiment analysis:
- News API (free tier: 1000 requests/day)
- VADER Sentiment Analysis (offline)
- Reddit API (free)
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

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Installing vaderSentiment...")
    import subprocess
    subprocess.check_call(["pip", "install", "vaderSentiment"])
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class RealtimeSentimentStockAnalyzer:
    """
    Real-time sentiment-enhanced stock analyzer using free APIs.
    """
    
    # Feature list for ML model
    FEATURES = [
        'Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Volatility',
        'Price_Momentum', 'Volume_Ratio', 'High_Low_Ratio', 'MACD', 'Sentiment'
    ]
    
    def __init__(self):
        """Initialize with ML model and sentiment analyzer."""
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Get API keys from environment variables
        self.news_api_key = os.getenv('NEWS_API_KEY')  # Get free key from newsapi.org
        
        # Sentiment-sensitive stocks
        self.sentiment_weights = {
            'TSLA': 0.4, 'NVDA': 0.3, 'META': 0.3, 'NFLX': 0.25,
            'AAPL': 0.2, 'AMZN': 0.15, 'GOOGL': 0.15, 'MSFT': 0.1
        }
    
    def get_news_sentiment(self, symbol, company_name=None):
        """
        Get real-time news sentiment using News API.
        
        Args:
            symbol (str): Stock symbol
            company_name (str): Company name for better search
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            if not self.news_api_key:
                print("NEWS_API_KEY not found. Using fallback sentiment.")
                return self._get_fallback_sentiment(symbol)
            
            # Company names for better news search
            company_names = {
                'TSLA': 'Tesla', 'AAPL': 'Apple', 'NVDA': 'NVIDIA',
                'META': 'Meta Facebook', 'GOOGL': 'Google Alphabet',
                'MSFT': 'Microsoft', 'AMZN': 'Amazon', 'NFLX': 'Netflix'
            }
            
            search_term = company_names.get(symbol, symbol)
            
            # Get news from last 24 hours
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f'"{search_term}" OR "{symbol}"',
                'from': yesterday,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 20,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                news_data = response.json()
                return self._analyze_news_sentiment(news_data, symbol)
            else:
                print(f"News API error: {response.status_code}")
                return self._get_fallback_sentiment(symbol)
                
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return self._get_fallback_sentiment(symbol)
    
    def _analyze_news_sentiment(self, news_data, symbol):
        """Analyze sentiment from news articles."""
        articles = news_data.get('articles', [])
        
        if not articles:
            return self._get_fallback_sentiment(symbol)
        
        sentiments = []
        topics = set()
        
        for article in articles[:15]:  # Analyze top 15 articles
            title = article.get('title', '')
            description = article.get('description', '')
            
            # Combine title and description
            text = f"{title}. {description}".strip()
            
            if text and len(text) > 10:
                # Use VADER sentiment analysis
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                sentiments.append(sentiment['compound'])
                
                # Extract key topics (simple keyword extraction)
                self._extract_topics(text, topics, symbol)
        
        if not sentiments:
            return self._get_fallback_sentiment(symbol)
        
        # Calculate average sentiment
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        
        return {
            'sentiment_score': avg_sentiment,
            'sentiment_std': sentiment_std,
            'news_count': len(articles),
            'analyzed_count': len(sentiments),
            'key_topics': list(topics)[:6],
            'source': 'news_api'
        }
    
    def _extract_topics(self, text, topics, symbol):
        """Extract key topics from news text."""
        # Define topic keywords for different stocks
        topic_keywords = {
            'TSLA': ['autopilot', 'electric', 'EV', 'battery', 'charging', 'production', 'delivery'],
            'NVDA': ['AI', 'artificial intelligence', 'GPU', 'gaming', 'data center', 'chip'],
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
        
        # General financial keywords
        financial_terms = ['earnings', 'revenue', 'profit', 'growth', 'market', 'stock']
        for term in financial_terms:
            if term in text_lower:
                topics.add(term.title())
    
    def _get_fallback_sentiment(self, symbol):
        """Fallback sentiment when API is unavailable."""
        # Use recent market performance as sentiment proxy
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='5d')
            
            if len(hist) >= 2:
                recent_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
                sentiment = np.tanh(recent_change * 10)  # Scale to [-1, 1]
            else:
                sentiment = 0.0
            
            return {
                'sentiment_score': sentiment,
                'sentiment_std': 0.1,
                'news_count': 0,
                'analyzed_count': 0,
                'key_topics': ['Market Performance'],
                'source': 'fallback'
            }
        except:
            return {
                'sentiment_score': 0.0,
                'sentiment_std': 0.1,
                'news_count': 0,
                'analyzed_count': 0,
                'key_topics': [],
                'source': 'fallback'
            }
    
    def fetch_data(self, symbol, period="1y"):
        """Fetch historical stock data."""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators."""
        # Moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI with zero-division protection
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        # Enhanced features
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Momentum'] = data['Close'].pct_change(periods=5)
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        
        return data
    
    def prepare_features(self, data, sentiment_score=0):
        """Prepare features for ML model."""
        # Add sentiment as feature
        data['Sentiment'] = sentiment_score
        
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
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            test_scores.append(self.model.score(X_test_scaled, y_test))
        
        # Final training on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        train_score = self.model.score(X_scaled, y)
        test_score = np.mean(test_scores)
        
        return train_score, test_score
    
    def predict_price(self, data):
        """Predict next day's price."""
        latest_data = data[self.FEATURES].iloc[-1:].values
        
        if np.any(pd.isnull(latest_data)):
            print("Warning: NaN values detected in features")
            return None
        
        scaled_data = self.scaler.transform(latest_data)
        prediction = self.model.predict(scaled_data)[0]
        
        return prediction
    
    def analyze_stock(self, symbol):
        """Complete real-time sentiment-enhanced analysis."""
        print(f"\n=== Real-time Analysis: {symbol} ===")
        
        # Fetch stock data
        data = self.fetch_data(symbol)
        if data is None:
            return None
        
        # Get real-time sentiment
        sentiment_data = self.get_news_sentiment(symbol)
        sentiment_score = sentiment_data['sentiment_score']
        
        # Apply sentiment weight
        weight = self.sentiment_weights.get(symbol, 0.1)
        weighted_sentiment = sentiment_score * weight
        
        # Calculate indicators
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
        X, y = self.prepare_features(data, weighted_sentiment)
        if len(X) < 50:
            print("Insufficient data for analysis")
            return None
        
        train_score, test_score = self.train_model(X, y)
        
        # Predict price
        predicted_price = self.predict_price(data)
        
        # Enhanced scoring (0-9 points)
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
        
        # Real-time sentiment (2 points)
        if sentiment_score > 0.1:
            score += 2
            reasons.append("Positive news sentiment")
        elif sentiment_score > 0.05:
            score += 1
            reasons.append("Neutral-positive sentiment")
        elif sentiment_score < -0.1:
            score -= 1
            reasons.append("Negative news sentiment")
        
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
            'weighted_sentiment': weighted_sentiment,
            'sentiment_data': sentiment_data,
            'model_accuracy': test_score,
            'recommendation': recommendation,
            'score': max(0, min(9, score)),
            'max_score': 9,
            'reasons': reasons
        }

def main():
    """Demo real-time sentiment analysis."""
    analyzer = RealtimeSentimentStockAnalyzer()
    
    print("Real-time Sentiment-Enhanced Stock Analyzer")
    print("Author: Vikas Ramaswamy")
    print("=" * 60)
    
    if not analyzer.news_api_key:
        print("⚠️  NEWS_API_KEY not set. Get free key from https://newsapi.org")
        print("   Set environment variable: export NEWS_API_KEY='your_key'")
        print("   Using fallback sentiment analysis...\n")
    
    stocks = ['TSLA', 'NVDA', 'AAPL', 'META']
    
    for symbol in stocks:
        try:
            result = analyzer.analyze_stock(symbol)
            if result:
                print(f"Current Price: ${result['current_price']:.2f}")
                if result['predicted_price']:
                    change = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
                    print(f"Predicted Price: ${result['predicted_price']:.2f} ({change:+.1f}%)")
                
                print(f"Sentiment: {result['sentiment_score']:.3f} (Source: {result['sentiment_data']['source']})")
                print(f"News Articles: {result['sentiment_data']['news_count']}")
                print(f"Key Topics: {', '.join(result['sentiment_data']['key_topics'][:3])}")
                print(f"Recommendation: {result['recommendation']} (Score: {result['score']}/9)")
                print("-" * 50)
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

if __name__ == "__main__":
    main()