#!/usr/bin/env python3
"""
Custom Stock Analyzer
Author: Vikas Ramaswamy

Interactive command-line tool for analyzing custom stocks with detailed output.
"""

from stock_analyzer_unified import UnifiedStockAnalyzer

def main():
    """Interactive custom stock analysis."""
    analyzer = UnifiedStockAnalyzer(use_realtime_sentiment=True)
    
    print("=" * 60)
    print("Custom Stock Analyzer & Investment Predictor")
    print("Author: Vikas Ramaswamy")
    print("=" * 60)
    print("\nEnter stock symbols to analyze (e.g., AAPL, GOOGL, TSLA)")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for more information\n")
    
    while True:
        try:
            user_input = input("Enter stock symbols (comma-separated): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Stock Analyzer!")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if not user_input:
                print("Please enter at least one stock symbol")
                continue
            
            # Parse symbols
            symbols = [s.strip().upper() for s in user_input.split(',') if s.strip()]
            
            if not symbols:
                print("No valid symbols found")
                continue
            
            print(f"\nAnalyzing {len(symbols)} stock(s)...")
            print("-" * 50)
            
            results = []
            
            # Analyze each stock
            for symbol in symbols:
                try:
                    result = analyzer.analyze_stock(symbol)
                    if result:
                        results.append(result)
                        display_stock_analysis(result)
                    else:
                        print(f"Unable to analyze {symbol} - insufficient data or invalid symbol")
                        
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
            
            # Display summary if multiple stocks
            if len(results) > 1:
                display_summary(results)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

def display_stock_analysis(result):
    """Display detailed analysis for a single stock."""
    try:
        symbol = result.get('symbol', 'Unknown')
        current_price = result.get('current_price', 0)
        predicted_price = result.get('predicted_price')
        recommendation = result.get('recommendation', 'N/A')
        score = result.get('score', 0)
        
        print(f"\n{symbol} Analysis Results:")
        print(f"   Current Price: ${current_price:.2f}")
        
        if predicted_price and current_price > 0:
            expected_change = ((predicted_price - current_price) / current_price) * 100
            print(f"   Predicted Price: ${predicted_price:.2f}")
            print(f"   Expected Change: {expected_change:+.1f}%")
        else:
            print("   Predicted Price: N/A")
        
        print(f"   RSI: {result.get('rsi', 0):.1f}")
        print(f"   20-day MA: ${result.get('ma_20', 0):.2f}")
        print(f"   50-day MA: ${result.get('ma_50', 0):.2f}")
        print(f"   Volatility: {result.get('volatility', 0):.2f}")
        print(f"   Model Accuracy: {result.get('model_accuracy', 0):.1%}")
        
        print(f"   Recommendation: {recommendation} (Score: {score}/{result.get('max_score', 9)})")
        
        reasons = result.get('reasons', [])
        if reasons:
            print(f"   Analysis Factors:")
            for reason in reasons:
                print(f"     * {reason}")
        
        print("-" * 50)
        except Exception as e:
            print(f"Error displaying analysis for {result.get('symbol', 'Unknown')}: {e}")
            print("-" * 50)

def display_summary(results):
    """Display summary of multiple stock analyses."""
    print("\nINVESTMENT SUMMARY")
    print("=" * 50)
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("Top Recommendations:")
    for i, result in enumerate(results[:3], 1):
        expected_return = 0
        predicted_price = result.get('predicted_price')
        current_price = result.get('current_price', 0)
        if predicted_price and current_price > 0:
            expected_return = ((predicted_price - current_price) / current_price) * 100
        
        print(f"   {i}. {result.get('symbol', 'Unknown')} - {result.get('recommendation', 'N/A')}")
        print(f"      Score: {result.get('score', 0)}/7 | Expected Return: {expected_return:+.1f}%")
    
    # Statistics
    strong_buys = len([r for r in results if r['recommendation'] == 'STRONG BUY'])
    buys = len([r for r in results if r['recommendation'] == 'BUY'])
    holds = len([r for r in results if r['recommendation'] == 'HOLD'])
    sells = len([r for r in results if r['recommendation'] == 'SELL'])
    
    print(f"\nPortfolio Breakdown:")
    print(f"   Strong Buy: {strong_buys}")
    print(f"   Buy: {buys}")
    print(f"   Hold: {holds}")
    print(f"   Sell: {sells}")
    
    print("\n" + "=" * 50)

def print_help():
    """Display help information."""
    print("\n" + "=" * 60)
    print("HELP - Stock Analyzer Usage")
    print("=" * 60)
    print("How to use:")
    print("   • Enter stock symbols separated by commas")
    print("   • Examples: AAPL, GOOGL, MSFT, TSLA")
    print("   • Use official ticker symbols from major exchanges")
    print("")
    print("Analysis includes:")
    print("   • Current and predicted stock prices")
    print("   • Technical indicators (RSI, Moving Averages)")
    print("   • Machine Learning price predictions")
    print("   • Investment recommendations (STRONG BUY/BUY/HOLD/SELL)")
    print("   • 9-point maximum scoring system with sentiment analysis")
    print("")
    print("Scoring System (0-9 points):")
    print("   • Price above 20-day MA: +1 point")
    print("   • Price above 50-day MA: +1 point") 
    print("   • 20-day MA above 50-day MA: +1 point")
    print("   • RSI in healthy range (30-70): +1 point")
    print("   • Lower volatility: +1 point")
    print("   • ML model predicts price increase: +2 points")
    print("   • Positive sentiment analysis: +1-2 points")
    print("   • High trading volume: +1 point")
    print("")
    print("Recommendations:")
    print("   • STRONG BUY: Score 7-9 points")
    print("   • BUY: Score 5-6 points")
    print("   • HOLD: Score 3-4 points")
    print("   • SELL: Score 0-2 points")
    print("")
    print("⚠️  Disclaimer: For educational purposes only. Not financial advice.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()