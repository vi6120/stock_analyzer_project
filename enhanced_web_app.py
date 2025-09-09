#!/usr/bin/env python3
"""
Enhanced Stock Analyzer Web Application with Sentiment Analysis
Author: Vikas Ramaswamy

Advanced Flask web application that incorporates sentiment analysis for better
stock predictions, especially for sentiment-driven stocks like Tesla.
"""

from flask import Flask, render_template_string, request, jsonify
from sentiment_analyzer import SentimentStockAnalyzer
import threading
import time
import json

app = Flask(__name__)
analyzer = SentimentStockAnalyzer()

# Global storage for streaming data
stock_data = {}
popular_stocks = ['TSLA', 'AAPL', 'NVDA', 'GOOGL', 'MSFT', 'META', 'AMZN', 'NFLX']

def update_popular_stocks():
    """Background thread to update popular stocks with sentiment analysis."""
    while True:
        print("Updating stocks with sentiment analysis...")
        for symbol in popular_stocks:
            try:
                result = analyzer.analyze_stock(symbol)
                if result:
                    # Calculate expected return
                    if result['predicted_price']:
                        expected_return = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
                        result['expected_return'] = expected_return
                    else:
                        result['expected_return'] = 0
                    stock_data[symbol] = result
                    print(f"Updated {symbol}: {result['recommendation']} (Sentiment: {result['sentiment_score']:.3f})")
            except Exception as e:
                print(f"Error updating {symbol}: {e}")
        time.sleep(90)  # Update every 90 seconds (sentiment analysis takes longer)

# Enhanced HTML Template with sentiment visualization
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment-Enhanced Stock Analyzer - Vikas Ramaswamy</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { 
            background: rgba(255,255,255,0.95); 
            color: #2c3e50; 
            padding: 30px; 
            border-radius: 15px; 
            margin-bottom: 30px; 
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; color: #666; margin-bottom: 5px; }
        .author { font-size: 0.9em; color: #888; margin-top: 10px; }
        .enhancement-badge {
            display: inline-block;
            background: linear-gradient(45deg, #ff6b6b, #feca57);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .controls { 
            background: rgba(255,255,255,0.95); 
            padding: 25px; 
            border-radius: 15px; 
            margin-bottom: 30px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .input-group { display: flex; gap: 15px; margin-bottom: 20px; align-items: center; }
        input[type="text"] { 
            flex: 1; 
            padding: 15px; 
            border: 2px solid #ddd; 
            border-radius: 10px; 
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus { border-color: #667eea; outline: none; }
        
        .btn { 
            padding: 15px 25px; 
            border: none; 
            border-radius: 10px; 
            cursor: pointer; 
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        .btn-primary { background: #667eea; color: white; }
        .btn-primary:hover { background: #5a6fd8; transform: translateY(-2px); }
        .btn-secondary { background: #6c757d; color: white; }
        .btn-secondary:hover { background: #5a6268; }
        
        .quick-stocks { margin-bottom: 20px; }
        .quick-stocks h3 { margin-bottom: 15px; color: #2c3e50; }
        .quick-stock { 
            display: inline-block; 
            background: #f8f9fa; 
            padding: 10px 20px; 
            margin: 5px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        .quick-stock:hover { 
            background: #667eea; 
            color: white; 
            transform: translateY(-2px);
        }
        .sentiment-sensitive { border: 2px solid #ff6b6b; }
        
        .streaming-section {
            background: rgba(255,255,255,0.95);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .streaming-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .live-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #28a745;
            font-weight: 600;
        }
        .live-dot {
            width: 12px;
            height: 12px;
            background: #28a745;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading { 
            display: none; 
            text-align: center; 
            padding: 30px; 
            color: #666;
            background: rgba(255,255,255,0.9);
            border-radius: 15px;
            margin: 20px 0;
        }
        .loading h3 { font-size: 1.5em; }
        
        .results { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 25px; 
        }
        .stock-card { 
            background: rgba(255,255,255,0.95); 
            border-radius: 15px; 
            padding: 25px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
            position: relative;
        }
        .stock-card:hover { transform: translateY(-5px); }
        .sentiment-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .sentiment-positive { background: #28a745; color: white; }
        .sentiment-neutral { background: #ffc107; color: #212529; }
        .sentiment-negative { background: #dc3545; color: white; }
        
        .stock-header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 20px; 
        }
        .symbol { 
            font-size: 28px; 
            font-weight: bold; 
            color: #2c3e50; 
        }
        .recommendation { 
            padding: 8px 16px; 
            border-radius: 25px; 
            font-weight: bold; 
            font-size: 14px;
            text-transform: uppercase;
        }
        .strong-buy { background: #28a745; color: white; }
        .buy { background: #17a2b8; color: white; }
        .hold { background: #ffc107; color: #212529; }
        .sell { background: #dc3545; color: white; }
        .strong-sell { background: #8b0000; color: white; }
        
        .metrics { 
            display: grid; 
            grid-template-columns: 1fr 1fr 1fr; 
            gap: 15px; 
            margin-bottom: 20px; 
        }
        .metric { 
            text-align: center; 
            padding: 15px; 
            background: #f8f9fa; 
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .metric-label { 
            font-size: 12px; 
            color: #666; 
            margin-bottom: 8px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .metric-value { 
            font-size: 18px; 
            font-weight: bold; 
            color: #2c3e50; 
        }
        
        .sentiment-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .sentiment-score {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .sentiment-bar {
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        .sentiment-fill {
            height: 100%;
            transition: width 0.3s;
        }
        
        .news-topics {
            margin-top: 10px;
        }
        .topic-tag {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 11px;
        }
        
        .reasons { margin-top: 20px; }
        .reasons h4 { 
            color: #2c3e50; 
            margin-bottom: 15px;
            font-size: 16px;
        }
        .reason-tag { 
            display: inline-block; 
            background: #e9ecef; 
            padding: 6px 12px; 
            margin: 3px; 
            border-radius: 20px; 
            font-size: 12px;
            color: #495057;
            font-weight: 500;
        }
        
        .error { 
            background: #dc3545; 
            color: white; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 15px 0;
            text-align: center;
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            color: rgba(255,255,255,0.8);
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sentiment-Enhanced Stock Analyzer</h1>
            <p>Advanced stock analysis with ML predictions and sentiment analysis</p>
            <div class="enhancement-badge">Now with News & Social Sentiment!</div>
            <div class="author">Created by Vikas Ramaswamy</div>
        </div>

        <div class="streaming-section">
            <div class="streaming-header">
                <h2>Live Sentiment-Enhanced Analysis</h2>
                <div class="live-indicator">
                    <div class="live-dot"></div>
                    <span>Live Updates with Sentiment</span>
                </div>
            </div>
            <div id="popularStocks" class="results"></div>
        </div>

        <div class="controls">
            <div class="quick-stocks">
                <h3>Quick Select Stocks:</h3>
                <span class="quick-stock sentiment-sensitive" onclick="addStock('TSLA')" title="High sentiment sensitivity">TSLA</span>
                <span class="quick-stock sentiment-sensitive" onclick="addStock('NVDA')" title="AI sentiment driven">NVDA</span>
                <span class="quick-stock" onclick="addStock('AAPL')">AAPL</span>
                <span class="quick-stock sentiment-sensitive" onclick="addStock('META')" title="Social media sensitive">META</span>
                <span class="quick-stock" onclick="addStock('GOOGL')">GOOGL</span>
                <span class="quick-stock" onclick="addStock('MSFT')">MSFT</span>
                <span class="quick-stock" onclick="addStock('AMZN')">AMZN</span>
                <span class="quick-stock sentiment-sensitive" onclick="addStock('NFLX')" title="Content sentiment driven">NFLX</span>
            </div>
            
            <div class="input-group">
                <input type="text" id="stockInput" placeholder="Enter stock symbols (e.g., TSLA, NVDA, AAPL)" />
                <button class="btn btn-primary" onclick="analyzeStocks()">Analyze with Sentiment</button>
                <button class="btn btn-secondary" onclick="clearResults()">Clear</button>
            </div>
        </div>

        <div class="loading" id="loading">
            <h3>Analyzing stocks with sentiment data...</h3>
            <p>Processing news sentiment, social media buzz, and ML predictions</p>
        </div>

        <div id="results" class="results"></div>
        
        <div class="footer">
            <p>Enhanced with sentiment analysis for better predictions on volatile stocks like Tesla</p>
            <p>⚠️ Disclaimer: This tool is for educational purposes only. Not financial advice.</p>
            <p>© 2024 Vikas Ramaswamy - Sentiment-Enhanced Stock Analyzer</p>
        </div>
    </div>

    <script>
        // Load popular stocks on page load
        window.onload = function() {
            loadPopularStocks();
            setInterval(loadPopularStocks, 90000); // Update every 90 seconds
        };

        function loadPopularStocks() {
            fetch('/api/popular-stocks')
            .then(response => response.json())
            .then(data => {
                displayPopularStocks(data.results);
            })
            .catch(error => {
                console.error('Error loading popular stocks:', error);
            });
        }

        function displayPopularStocks(results) {
            const container = document.getElementById('popularStocks');
            container.innerHTML = '';
            
            results.forEach(result => {
                const card = createStockCard(result);
                container.innerHTML += card;
            });
        }

        function addStock(symbol) {
            const input = document.getElementById('stockInput');
            const current = input.value.trim();
            if (current && !current.includes(symbol)) {
                input.value = current + ', ' + symbol;
            } else if (!current) {
                input.value = symbol;
            }
        }

        function analyzeStocks() {
            const symbols = document.getElementById('stockInput').value.trim();
            if (!symbols) {
                alert('Please enter stock symbols');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';

            fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbols: symbols })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                displayResults(data.results);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').innerHTML = '<div class="error">Error: ' + error + '</div>';
            });
        }

        function displayResults(results) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';

            results.forEach(result => {
                if (result.error) {
                    resultsDiv.innerHTML += `<div class="error">Error analyzing ${result.symbol}: ${result.error}</div>`;
                    return;
                }

                const card = createStockCard(result);
                resultsDiv.innerHTML += card;
            });
        }

        function createStockCard(result) {
            const expectedReturn = result.expected_return ? result.expected_return.toFixed(1) : '0.0';
            const recClass = result.recommendation.toLowerCase().replace(' ', '-');
            const sentimentScore = result.sentiment_score || 0;
            
            // Sentiment badge
            let sentimentBadge = '';
            let sentimentClass = '';
            if (sentimentScore > 0.1) {
                sentimentBadge = 'Positive';
                sentimentClass = 'sentiment-positive';
            } else if (sentimentScore < -0.1) {
                sentimentBadge = 'Negative';
                sentimentClass = 'sentiment-negative';
            } else {
                sentimentBadge = 'Neutral';
                sentimentClass = 'sentiment-neutral';
            }
            
            // Sentiment bar
            const sentimentPercent = Math.max(0, Math.min(100, (sentimentScore + 1) * 50));
            const sentimentColor = sentimentScore > 0 ? '#28a745' : sentimentScore < 0 ? '#dc3545' : '#ffc107';
            
            // News topics
            const topics = result.sentiment_data?.key_topics || [];
            const topicTags = topics.slice(0, 4).map(topic => `<span class="topic-tag">${topic}</span>`).join('');
            
            return `
                <div class="stock-card">
                    <div class="sentiment-badge ${sentimentClass}">${sentimentBadge}</div>
                    <div class="stock-header">
                        <div class="symbol">${result.symbol}</div>
                        <div class="recommendation ${recClass}">${result.recommendation}</div>
                    </div>
                    
                    <div class="sentiment-section">
                        <div class="sentiment-score">
                            <strong>Market Sentiment</strong>
                            <span>${sentimentScore.toFixed(3)}</span>
                        </div>
                        <div class="sentiment-bar">
                            <div class="sentiment-fill" style="width: ${sentimentPercent}%; background: ${sentimentColor};"></div>
                        </div>
                        <div class="news-topics">
                            <strong>Key Topics:</strong> ${topicTags}
                        </div>
                    </div>
                    
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Current Price</div>
                            <div class="metric-value">$${result.current_price.toFixed(2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Predicted Price</div>
                            <div class="metric-value">$${result.predicted_price ? result.predicted_price.toFixed(2) : 'N/A'}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Expected Return</div>
                            <div class="metric-value" style="color: ${expectedReturn >= 0 ? '#28a745' : '#dc3545'}">${expectedReturn}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Score</div>
                            <div class="metric-value">${result.score}/${result.max_score || 7}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">RSI</div>
                            <div class="metric-value">${result.rsi.toFixed(1)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Model Accuracy</div>
                            <div class="metric-value">${(result.model_accuracy * 100).toFixed(1)}%</div>
                        </div>
                    </div>

                    <div class="reasons">
                        <h4>Analysis Factors:</h4>
                        ${result.reasons.map(reason => `<span class="reason-tag">${reason}</span>`).join('')}
                    </div>
                </div>
            `;
        }

        function clearResults() {
            document.getElementById('stockInput').value = '';
            document.getElementById('results').innerHTML = '';
        }

        // Allow Enter key to trigger analysis
        document.getElementById('stockInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                analyzeStocks();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the enhanced web application."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/popular-stocks')
def get_popular_stocks():
    """API endpoint to get popular stocks with sentiment data."""
    results = []
    for symbol in popular_stocks:
        if symbol in stock_data:
            results.append(stock_data[symbol])
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'results': results})

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze custom stocks with sentiment."""
    data = request.get_json()
    symbols = [s.strip().upper() for s in data.get('symbols', '').split(',') if s.strip()]
    
    if not symbols:
        return jsonify({'error': 'No symbols provided'})
    
    results = []
    for symbol in symbols:
        try:
            result = analyzer.analyze_stock(symbol)
            if result:
                # Calculate expected return
                if result['predicted_price']:
                    expected_return = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
                    result['expected_return'] = expected_return
                else:
                    result['expected_return'] = 0
                results.append(result)
        except Exception as e:
            results.append({
                'symbol': symbol,
                'error': str(e)
            })
    
    # Sort by score
    valid_results = [r for r in results if 'error' not in r]
    valid_results.sort(key=lambda x: x['score'], reverse=True)
    
    return jsonify({'results': results})

if __name__ == '__main__':
    print("Starting Sentiment-Enhanced Stock Analyzer")
    print("Author: Vikas Ramaswamy")
    print("Open your browser and go to: http://localhost:5002")
    print("Enhanced Features:")
    print("   • News sentiment analysis")
    print("   • Social media sentiment tracking")
    print("   • Improved predictions for volatile stocks")
    print("   • Enhanced scoring system (0-9 points)")
    print("   • Real-time sentiment updates")
    print("Press Ctrl+C to stop the server\n")
    
    # Start background thread for sentiment-enhanced updates
    thread = threading.Thread(target=update_popular_stocks, daemon=True)
    thread.start()
    
    # Initial data load
    print("Loading initial sentiment-enhanced stock data...")
    
    app.run(debug=True, host='0.0.0.0', port=5002)