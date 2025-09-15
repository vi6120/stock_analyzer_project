#!/usr/bin/env python3
"""
Stock Analyzer & Investment Predictor
Author: Vikas Ramaswamy

A comprehensive stock analysis tool that combines technical indicators with machine learning
to provide investment recommendations and price predictions.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class StockAnalyzer:
    """
    Stock Analyzer class that provides comprehensive stock analysis including:
    - Technical indicators (Moving Averages, RSI, Volatility)
    - Machine Learning price predictions
    - Investment recommendations with scoring system
    """
    
    # Feature list for ML model
    FEATURES = [
        'Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Volatility',
        'Price_Momentum', 'Volume_Ratio', 'High_Low_Ratio', 'MACD'
    ]
    
    def __init__(self):
        """Initialize the analyzer with enhanced ML model and scaler."""
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    def fetch_data(self, symbol, period="1y"):
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period for data ('1y', '6mo', '3mo', etc.)
            
        Returns:
            pandas.DataFrame: Historical stock data or None if error
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, data):
        """
        Calculate technical indicators for stock analysis.
        
        Args:
            data (pandas.DataFrame): Stock price data
            
        Returns:
            pandas.DataFrame: Data with added technical indicators
        """
        # Moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI (Relative Strength Index) with zero-division protection
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)  # Add epsilon to prevent division by zero
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility (20-day rolling standard deviation)
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        # Enhanced features for better prediction
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Momentum'] = data['Close'].pct_change(periods=5)
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        
        return data
    
    def prepare_features(self, data):
        """
        Prepare features for machine learning model.
        
        Args:
            data (pandas.DataFrame): Stock data with indicators
            
        Returns:
            tuple: (X, y) features and target arrays
        """
        features = self.FEATURES
        
        # Create target (next day's closing price)
        data['Target'] = data['Close'].shift(-1)
        
        # Drop rows with NaN values
        clean_data = data[features + ['Target']].dropna()
        
        X = clean_data[features]
        y = clean_data['Target']
        
        return X, y
    
    def train_model(self, X, y):
        """
        Train the Random Forest prediction model.
        
        Args:
            X (pandas.DataFrame): Feature data
            y (pandas.Series): Target data
            
        Returns:
            tuple: (train_score, test_score) model accuracy scores
        """
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=3)
        test_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and evaluate model
            self.model.fit(X_train_scaled, y_train)
            test_scores.append(self.model.score(X_test_scaled, y_test))
        
        # Final training on all data for prediction
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        train_score = self.model.score(X_scaled, y)
        test_score = np.mean(test_scores)
        
        return train_score, test_score
    
    def predict_price(self, data):
        """
        Predict next day's stock price using trained model.
        
        Args:
            data (pandas.DataFrame): Stock data with indicators
            
        Returns:
            float: Predicted price or None if error
        """
        latest_data = data[self.FEATURES].iloc[-1:].values
        
        if np.any(np.isnan(latest_data)):
            return None
        
        scaled_data = self.scaler.transform(latest_data)
        prediction = self.model.predict(scaled_data)[0]
        
        return prediction
    
    def analyze_stock(self, symbol):
        """
        Perform complete stock analysis with recommendation.
        
        Args:
            symbol (str): Stock symbol to analyze
            
        Returns:
            dict: Complete analysis results including recommendation
        """
        print(f"\n=== Analyzing {symbol} ===")
        
        # Fetch and validate data
        data = self.fetch_data(symbol)
        if data is None:
            return None
        
        # Calculate technical indicators
        data = self.calculate_indicators(data)
        
        # Extract current metrics (optimize by getting last row once)
        last_row = data.iloc[-1]
        current_price = last_row['Close']
        ma_20 = last_row['MA_20']
        ma_50 = last_row['MA_50']
        rsi = last_row['RSI']
        volatility = last_row['Volatility']
        
        # Prepare and train ML model
        X, y = self.prepare_features(data)
        if len(X) < 50:  # Need sufficient data
            print("Insufficient data for analysis")
            return None
        
        train_score, test_score = self.train_model(X, y)
        
        # Predict next day's price
        predicted_price = self.predict_price(data)
        
        # Calculate investment score (0-7 points)
        score = 0
        reasons = []
        
        # Technical analysis scoring
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
        
        # ML prediction scoring
        if predicted_price and predicted_price > current_price:
            score += 2
            reasons.append("Model predicts price increase")
        
        # Generate recommendation with price prediction priority
        if predicted_price and predicted_price < current_price:
            price_drop = ((current_price - predicted_price) / current_price) * 100
            if price_drop > 5:  # >5% drop predicted
                recommendation = "SELL"
            elif price_drop > 2:  # 2-5% drop predicted
                recommendation = "HOLD"
            else:  # <2% drop
                recommendation = "HOLD" if score >= 3 else "SELL"
        else:
            # Standard scoring when price increase predicted
            if score >= 5:
                recommendation = "STRONG BUY"
            elif score >= 3:
                recommendation = "BUY"
            elif score >= 2:
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
            'model_accuracy': test_score,
            'recommendation': recommendation,
            'score': score,
            'reasons': reasons
        }

def main():
    """Main function to run stock analysis on popular stocks."""
    analyzer = StockAnalyzer()
    
    # Popular stocks to analyze
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    
    print("Stock Analysis and Investment Recommendations")
    print("Author: Vikas Ramaswamy")
    print("=" * 50)
    
    results = []
    
    for symbol in stocks:
        try:
            result = analyzer.analyze_stock(symbol)
            if result:
                results.append(result)
                
                print(f"Current Price: ${result['current_price']:.2f}")
                if result['predicted_price']:
                    print(f"Predicted Price: ${result['predicted_price']:.2f}")
                    change = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
                    print(f"Expected Change: {change:.1f}%")
                print(f"RSI: {result['rsi']:.1f}")
                print(f"Model Accuracy: {result['model_accuracy']:.2f}")
                print(f"Recommendation: {result['recommendation']} (Score: {result['score']}/7)")
                print("Reasons:", ", ".join(result['reasons']))
                print("-" * 30)
                
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # Sort by recommendation score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n=== TOP INVESTMENT RECOMMENDATIONS ===")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result['symbol']} - {result['recommendation']} (Score: {result['score']}/7)")
        if result['predicted_price']:
            expected_return = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
            print(f"   Expected Return: {expected_return:.1f}%")

if __name__ == "__main__":
    main()