#!/usr/bin/env python3
"""
Launcher script for the LSTM Stock Price Predictor Streamlit App
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'tensorflow',
        'yfinance',
        'plotly',
        'pandas',
        'numpy',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_packages(packages):
    """Install missing packages"""
    print(f"Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        return True
    except subprocess.CalledProcessError:
        print("Failed to install packages. Please install them manually:")
        print(f"pip install {' '.join(packages)}")
        return False

def main():
    print("🚀 LSTM Stock Price Predictor Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        response = input("Would you like to install them automatically? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            if not install_packages(missing_packages):
                sys.exit(1)
        else:
            print("Please install the missing packages manually:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)
    
    print("✅ All dependencies are installed!")
    
    # Check if Streamlit app exists
    if not os.path.exists('streamlit_stock_predictor.py'):
        print("❌ streamlit_stock_predictor.py not found!")
        print("Please make sure you're in the correct directory.")
        sys.exit(1)
    
    print("🎯 Starting Streamlit app...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run Streamlit app
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_stock_predictor.py'])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main() 