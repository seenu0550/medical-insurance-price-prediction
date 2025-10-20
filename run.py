#!/usr/bin/env python3
"""
Medical Insurance Price Prediction System
Run this script to train the model and start the web application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def train_model():
    """Train the ML model"""
    print("Training the machine learning model...")
    subprocess.check_call([sys.executable, "model.py"])

def start_app():
    """Start the Flask web application"""
    print("Starting the web application...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    subprocess.check_call([sys.executable, "app.py"])

if __name__ == "__main__":
    try:
        # Install requirements
        install_requirements()
        
        # Train model
        train_model()
        
        # Start web app
        start_app()
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure you have Python and pip installed.")