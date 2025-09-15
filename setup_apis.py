#!/usr/bin/env python3
"""
API Setup Helper
Author: Vikas Ramaswamy

Helper script to set up free APIs for real-time sentiment analysis.
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages for sentiment analysis."""
    packages = [
        'vaderSentiment',
        'requests',
        'python-dotenv'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"{package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully")

def setup_environment():
    """Set up environment variables for API keys."""
    print("\n" + "="*60)
    print("API SETUP FOR REAL-TIME SENTIMENT ANALYSIS")
    print("="*60)
    
    print("\n1. NEWS API (Free Tier: 1000 requests/day)")
    print("Visit: https://newsapi.org/register")
    print("Sign up for free account")
    print("Copy your API key")
    
    news_api_key = input("\n   Enter your News API key (or press Enter to skip): ").strip()
    
    if news_api_key:
        # Create .env file
        env_content = f"NEWS_API_KEY={news_api_key}\n"
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("API key saved to .env file")
        
        # Also set environment variable for current session
        os.environ['NEWS_API_KEY'] = news_api_key
        print("Environment variable set for current session")
        
        print("\nTo make permanent, add to your shell profile:")
        print(f"export NEWS_API_KEY='{news_api_key}'")
    else:
        print("Skipped - will use fallback sentiment analysis")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    
    if news_api_key:
        print("Real-time sentiment analysis enabled")
        print("You can now get live news sentiment for stocks")
    else:
        print("Using fallback sentiment (based on price momentum)")
        print("Get News API key later for real-time sentiment")
    
    print("\n Next steps:")
    print("   1. Run: python realtime_sentiment_analyzer.py")
    print("   2. Or use in your applications:")
    print("      from realtime_sentiment_analyzer import RealtimeSentimentStockAnalyzer")

def test_setup():
    """Test the API setup."""
    print("\n Testing setup...")
    
    try:
        from realtime_sentiment_analyzer import RealtimeSentimentStockAnalyzer
        analyzer = RealtimeSentimentStockAnalyzer()
        
        # Test sentiment analysis
        sentiment_data = analyzer.get_news_sentiment('AAPL')
        
        if sentiment_data['source'] == 'news_api':
            print("News API working correctly")
            print(f"Found {sentiment_data['news_count']} articles")
        else:
            print("Using fallback sentiment analysis")
        
        print("Sentiment analyzer initialized successfully")
        
    except Exception as e:
        print(f" Error testing setup: {e}")
        return False
    
    return True

def main():
    """Main setup function."""
    print(" Setting up Real-time Sentiment Analysis")
    print("Author: Vikas Ramaswamy\n")
    
    # Install requirements
    print(" Installing required packages...")
    install_requirements()
    
    # Setup APIs
    setup_environment()
    
    # Test setup
    if test_setup():
        print("\n Setup completed successfully!")
        
        # Ask if user wants to run demo
        run_demo = input("\n Run demo analysis? (y/n): ").lower().strip()
        if run_demo in ['y', 'yes']:
            print("\n" + "="*60)
            try:
                from realtime_sentiment_analyzer import RealtimeSentimentStockAnalyzer
                analyzer = RealtimeSentimentStockAnalyzer()
                
                # Quick demo
                result = analyzer.analyze_stock('AAPL')
                if result:
                    print(f"AAPL Analysis:")
                    print(f"   Price: ${result['current_price']:.2f}")
                    print(f"   Sentiment: {result['sentiment_score']:.3f}")
                    print(f"   Recommendation: {result['recommendation']}")
                    print(f"   News Source: {result['sentiment_data']['source']}")
                
            except Exception as e:
                print(f"Demo failed: {e}")
    else:
        print("\n Setup incomplete. Please check errors above.")

if __name__ == "__main__":
    main()