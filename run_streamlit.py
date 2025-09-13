#!/usr/bin/env python3
"""
Streamlit App Runner
Author: Vikas Ramaswamy

Simple script to run the Streamlit stock analyzer application.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application."""
    print("Starting Streamlit Stock Analyzer")
    print("Author: Vikas Ramaswamy")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed")
    except ImportError:
        print("Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0"])
        print("Streamlit installed successfully")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app_path = os.path.join(script_dir, "streamlit_app.py")
    
    print(f"Launching Streamlit app from: {streamlit_app_path}")
    print("Your app will open in your default browser")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", streamlit_app_path,
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nStreamlit app stopped")

if __name__ == "__main__":
    main()