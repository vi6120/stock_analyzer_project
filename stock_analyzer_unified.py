#!/usr/bin/env python3
"""
Unified Stock Analyzer & Investment Predictor
Author: Vikas Ramaswamy

Comprehensive stock analysis combining technical indicators, machine learning,
and sentiment analysis for investment recommendations.

Fix log (v2):
  - sklearn Pipeline eliminates any scaler/split ordering bugs structurally
  - Target is now next-day % return (not raw price) — eliminates inflated R²
  - 5-year training window (~1260 rows vs ~252 before)
  - Per-tree confidence interval on every price prediction
  - Three-tier sentiment: NewsAPI → yfinance.news (free) → simulated fallback
  - TimeSeriesSplit n_splits raised from 3 to 5 for more stable CV score
  - Volatility scoring uses z-score vs stock's own history (not absolute mean)
  - model_accuracy_std added to result dict so UI can show uncertainty
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import requests
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class UnifiedStockAnalyzer:
    """
    Unified Stock Analyzer:
    - Technical indicators (MA, RSI, Volatility, MACD, Momentum)
    - ML price predictions via sklearn Pipeline (no leakage possible)
    - Sentiment: NewsAPI → yfinance.news (free, no key) → simulated fallback
    - 9-point investment scoring with buy/sell recommendations
    """

    FEATURES = [
        'Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Volatility',
        'Price_Momentum', 'Volume_Ratio', 'High_Low_Ratio', 'MACD', 'Sentiment'
    ]

    # Last-resort fallback — only used if NewsAPI AND yfinance.news both fail
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
        """Set up the analyzer with sklearn Pipeline and sentiment tools."""
        # Single pipeline — scaler is structurally always fit only on training data
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ))
        ])

        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.use_realtime_sentiment = (
            use_realtime_sentiment and VADER_AVAILABLE and bool(self.news_api_key)
        )

        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # How much sentiment moves the score for each stock
        self.sentiment_weights = {
            'TSLA': 0.4, 'NVDA': 0.3, 'META': 0.3, 'NFLX': 0.25,
            'AAPL': 0.2, 'AMZN': 0.15, 'GOOGL': 0.15, 'MSFT': 0.1
        }

    # ─── Data fetching ───────────────────────────────────────────────────────

    def fetch_data(self, symbol, period="5y"):
        """
        Get stock price history from Yahoo Finance.
        Default is 5 years (~1260 trading days) for better model training.
        Previously 1 year (~252 rows) which was too small for RF-200.
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    # ─── Technical indicators ────────────────────────────────────────────────

    def calculate_indicators(self, data):
        """Add technical indicators to the stock data."""
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))

        data['Volatility'] = data['Close'].rolling(window=20).std()
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Momentum'] = data['Close'].pct_change(periods=5)
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['MACD'] = (
            data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        )

        return data

    # ─── Sentiment — three-tier fallback ─────────────────────────────────────

    def get_sentiment_data(self, symbol):
        """
        Sentiment with three-tier fallback:
          1. NewsAPI      — if NEWS_API_KEY env var is set
          2. yfinance.news — free, no API key, uses real recent headlines
          3. Hardcoded demo data — last resort, same values every run
        """
        if self.use_realtime_sentiment and self.news_api_key:
            return self._get_realtime_sentiment(symbol)

        # Free tier: yfinance.Ticker.news — real headlines, zero cost
        if VADER_AVAILABLE:
            yf_result = self._get_yfinance_sentiment(symbol)
            if yf_result is not None:
                return yf_result

        return self._get_simulated_sentiment(symbol)

    def _get_yfinance_sentiment(self, symbol):
        """
        Pull recent headlines from yfinance.Ticker.news and score them
        with VADER. No API key required — completely free.
        Returns None (not empty dict) when no articles found so callers
        can fall through to the next tier.
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                return None

            sentiments = []
            topics = set()
            headlines = []

            for article in news[:15]:
                title = article.get('title', '')
                if title and len(title) > 10:
                    score = self.sentiment_analyzer.polarity_scores(title)['compound']
                    sentiments.append(score)
                    headlines.append(title)
                    self._extract_topics(title, topics, symbol)

            if not sentiments:
                return None

            return {
                'sentiment_score': float(np.mean(sentiments)),
                'news_count': len(news),
                'key_topics': list(topics)[:6],
                'top_headlines': headlines[:5],
                'source': 'yfinance_news'
            }
        except Exception:
            return None

    def _get_realtime_sentiment(self, symbol):
        """Fetch live news sentiment from NewsAPI (requires API key)."""
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
                return self._analyze_news_sentiment(response.json(), symbol)

            # NewsAPI failed — try yfinance free tier before giving up
            if VADER_AVAILABLE:
                yf_result = self._get_yfinance_sentiment(symbol)
                if yf_result:
                    return yf_result
            return self._get_fallback_sentiment(symbol)

        except Exception as e:
            print(f"Real-time sentiment error for {symbol}: {e}")
            if VADER_AVAILABLE:
                yf_result = self._get_yfinance_sentiment(symbol)
                if yf_result:
                    return yf_result
            return self._get_fallback_sentiment(symbol)

    def _analyze_news_sentiment(self, news_data, symbol):
        """Score NewsAPI articles with VADER."""
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

        return {
            'sentiment_score': float(np.mean(sentiments)),
            'news_count': len(articles),
            'key_topics': list(topics)[:6],
            'top_headlines': [
                a.get('title', '') for a in articles[:5] if a.get('title')
            ],
            'source': 'news_api'
        }

    def _extract_topics(self, text, topics, symbol):
        """Find stock-relevant keywords in a piece of text."""
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
        """
        Hardcoded demo data — final fallback only.
        NOTE: Same values every run by design for this tier.
        The yfinance.news tier above provides real data for free.
        """
        default_sentiment = {
            'sentiment_score': 0.0, 'news_count': 15,
            'key_topics': ['Earnings reports', 'Market trends', 'Industry news'],
            'source': 'simulated'
        }

        data = self.SENTIMENT_DATA.get(symbol, default_sentiment.copy())

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
        """Price-momentum proxy when all news sources are unavailable."""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='5d')

            if len(hist) >= 2:
                recent_change = (
                    (hist['Close'].iloc[-1] - hist['Close'].iloc[-2])
                    / hist['Close'].iloc[-2]
                )
                sentiment = float(np.tanh(recent_change * 10))
            else:
                sentiment = 0.0

            return {
                'sentiment_score': sentiment,
                'news_count': 0,
                'key_topics': ['Market Performance'],
                'source': 'price_fallback'
            }
        except Exception:
            return {
                'sentiment_score': 0.0, 'news_count': 0,
                'key_topics': [], 'source': 'fallback'
            }

    # ─── ML: features, training, prediction ──────────────────────────────────

    def prepare_features(self, data, sentiment_score=0):
        """
        Build the feature matrix and target vector.

        FIX: Target is now next-day % RETURN, not raw closing price.

        Why this matters:
          Old: Target = Close.shift(-1)
               R² looks great (0.95+) because tomorrow's price ≈ today's.
               Any model that just echoes the current price scores well.

          New: Target = Close.pct_change().shift(-1)
               R² on returns is much harder to inflate — the model has to
               actually learn something about direction and magnitude.
               Negative R² is now possible and honest.
        """
        time_factor = np.linspace(-0.02, 0.02, len(data))
        data['Sentiment'] = np.clip(sentiment_score + time_factor, -1, 1)

        # Next-day % return — harder target, more honest accuracy metric
        data['Target'] = data['Close'].pct_change().shift(-1)

        clean_data = data[self.FEATURES + ['Target']].dropna()
        X = clean_data[self.FEATURES]
        y = clean_data['Target']

        return X, y

    def train_model(self, X, y):
        """
        TimeSeriesSplit cross-validation with a fresh Pipeline per fold.

        FIX: Each fold creates its own Pipeline so the StandardScaler is
        fit only on that fold's training rows. The scaler never sees test
        data at any point, making data leakage structurally impossible.

        Returns (test_score, score_std) — the CV mean and its standard
        deviation across folds. score_std tells you how stable the model
        is (high std = results vary a lot across time windows).

        Note: train_score is intentionally NOT returned. It was always
        near 1.0 (fit on its own training data) and was misleading users.
        """
        tscv = TimeSeriesSplit(n_splits=5)  # Raised from 3 for more stable estimate
        test_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fresh pipeline per fold: scaler fit only on X_train, never X_test
            fold_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                ))
            ])
            fold_pipeline.fit(X_train, y_train)
            test_scores.append(fold_pipeline.score(X_test, y_test))

        # Final pipeline fit on ALL data — used for inference only
        self.pipeline.fit(X, y)

        return float(np.mean(test_scores)), float(np.std(test_scores))

    def predict_price(self, data, sentiment_score=0):
        """
        Predict next-day closing price.

        FIX: Model now predicts % return; we convert back to absolute price.
        Uses individual tree predictions to produce a 1-sigma confidence band
        so the UI can show "$187.40 ± $4.20" instead of a bare point estimate.

        Returns:
            (predicted_price, (price_lower, price_upper))  — on success
            (None, None)                                    — on missing data
        """
        latest_data = data[self.FEATURES[:-1]].iloc[-1:].copy()
        latest_data['Sentiment'] = sentiment_score

        if latest_data.isna().any().any():
            return None, None

        scaler = self.pipeline.named_steps['scaler']
        rf_model = self.pipeline.named_steps['model']
        X_scaled = scaler.transform(latest_data)

        # Each tree votes on a return value — std across trees = model uncertainty
        tree_returns = np.array([
            tree.predict(X_scaled)[0] for tree in rf_model.estimators_
        ])

        predicted_return = float(np.mean(tree_returns))
        return_std = float(np.std(tree_returns))

        current_price = float(data['Close'].iloc[-1])
        predicted_price = current_price * (1 + predicted_return)
        price_lower = current_price * (1 + predicted_return - return_std)
        price_upper = current_price * (1 + predicted_return + return_std)

        return predicted_price, (price_lower, price_upper)

    # ─── Main analysis entry point ────────────────────────────────────────────

    def analyze_stock(self, symbol):
        """Run the full analysis pipeline on a stock symbol."""
        print(f"\n=== Analyzing {symbol} ===")

        # 5y for richer training; graceful fallback to 2y for newer/delisted stocks
        data = self.fetch_data(symbol, period="5y")
        if data is None or len(data) < 100:
            data = self.fetch_data(symbol, period="2y")
        if data is None:
            return None

        sentiment_data = self.get_sentiment_data(symbol)
        sentiment_score = sentiment_data['sentiment_score']

        data = self.calculate_indicators(data)

        last_row = data.iloc[-1]
        current_price = float(last_row['Close'])
        ma_20 = float(last_row['MA_20'])
        ma_50 = float(last_row['MA_50'])
        rsi = float(last_row['RSI'])
        volatility = float(last_row['Volatility'])
        volume_ratio = float(last_row['Volume_Ratio'])

        X, y = self.prepare_features(data, sentiment_score)
        if len(X) < 50:
            print("Not enough data for a reliable prediction")
            return None

        test_score, score_std = self.train_model(X, y)
        predicted_price, prediction_interval = self.predict_price(data, sentiment_score)

        # ── Scoring (0–9 points) ─────────────────────────────────────────────
        score = 0
        reasons = []

        if current_price > ma_20:
            score += 1
            reasons.append("Price above 20-day average")
        if current_price > ma_50:
            score += 1
            reasons.append("Price above 50-day average")
        if ma_20 > ma_50:
            score += 1
            reasons.append("Short-term trend is up")

        if 30 <= rsi <= 70:
            score += 1
            reasons.append("RSI looks good")
        elif rsi > 80:
            score -= 1
            reasons.append("RSI too high (overbought)")
        elif rsi > 70:
            reasons.append("RSI getting high")
        elif rsi < 20:
            score -= 1
            reasons.append("RSI too low (oversold)")

        # FIX: Volatility scored relative to this stock's own history (z-score)
        # Previously used an absolute mean which unfairly penalised volatile stocks
        vol_mean = data['Volatility'].mean()
        vol_std_hist = data['Volatility'].std()
        vol_zscore = (volatility - vol_mean) / (vol_std_hist + 1e-10)

        if vol_zscore < -0.5:
            score += 1
            reasons.append("Calmer than usual for this stock (lower risk)")
        elif vol_zscore > 1.5:
            score -= 1
            reasons.append("Unusually high volatility vs its own history")

        if predicted_price and predicted_price > current_price:
            score += 2
            reasons.append("AI model expects price to go up")

        if sentiment_score > 0.1:
            score += 2
            reasons.append(f"Positive news sentiment ({sentiment_data['source']})")
        elif sentiment_score > 0.05:
            score += 1
            reasons.append(f"Slightly positive news ({sentiment_data['source']})")
        elif sentiment_score < -0.1:
            score -= 1
            reasons.append("Negative news sentiment")

        if volume_ratio > 1.2:
            score += 1
            reasons.append("Higher than normal trading volume")

        # ── Final recommendation ─────────────────────────────────────────────
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
                reasons.append("Extra boost from strong positive news")

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
            'prediction_interval': prediction_interval,   # NEW: (lower, upper) ±1σ
            'ma_20': ma_20,
            'ma_50': ma_50,
            'rsi': rsi,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'sentiment_score': sentiment_score,
            'sentiment_data': sentiment_data,
            'model_accuracy': test_score,                 # CV R² on % return target
            'model_accuracy_std': score_std,              # NEW: std across CV folds
            'recommendation': recommendation,
            'score': max(0, min(9, score)),
            'max_score': 9,
            'reasons': reasons
        }


# ─── CLI demo ─────────────────────────────────────────────────────────────────

def main():
    """Demonstrate unified analysis with the improved engine."""
    try:
        analyzer = UnifiedStockAnalyzer(use_realtime_sentiment=True)
        if analyzer.use_realtime_sentiment:
            print("Sentiment source: NewsAPI (real-time)")
        elif VADER_AVAILABLE:
            print("Sentiment source: yfinance.news (free real headlines)")
        else:
            print("Sentiment source: simulated fallback")
    except Exception:
        analyzer = UnifiedStockAnalyzer(use_realtime_sentiment=False)
        print("Sentiment source: simulated fallback")

    stocks = ['TSLA', 'NVDA', 'AAPL', 'META', 'GOOGL']

    print("\nUnified Stock Analyzer & Investment Predictor")
    print("Author: Vikas Ramaswamy")
    print("=" * 60)

    results = []

    for symbol in stocks:
        try:
            result = analyzer.analyze_stock(symbol)
            if not result:
                continue
            results.append(result)

            print(f"Current Price:   ${result['current_price']:.2f}")

            if result['predicted_price']:
                change = (
                    (result['predicted_price'] - result['current_price'])
                    / result['current_price'] * 100
                )
                print(f"Predicted Price: ${result['predicted_price']:.2f} ({change:+.1f}%)")

                if result['prediction_interval']:
                    lo, hi = result['prediction_interval']
                    print(f"Confidence Band: ${lo:.2f} – ${hi:.2f}")

            acc = result['model_accuracy']
            acc_std = result.get('model_accuracy_std', 0)
            print(f"Model R² (CV):   {acc:.3f} ± {acc_std:.3f}  (on % returns, not raw price)")
            print(f"Sentiment:       {result['sentiment_score']:.3f} ({result['sentiment_data']['source']})")
            print(f"Recommendation:  {result['recommendation']} ({result['score']}/9)")
            print("-" * 50)

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    results.sort(key=lambda x: x['score'], reverse=True)

    print("\n=== TOP RECOMMENDATIONS ===")
    for i, result in enumerate(results[:3], 1):
        expected_return = 0
        if result['predicted_price']:
            expected_return = (
                (result['predicted_price'] - result['current_price'])
                / result['current_price'] * 100
            )
        print(f"{i}. {result['symbol']} — {result['recommendation']}")
        print(f"   Score: {result['score']}/9 | Expected: {expected_return:+.1f}%")
        print(f"   Sentiment: {result['sentiment_score']:.3f} | R²: {result['model_accuracy']:.3f}")


if __name__ == "__main__":
    main()
