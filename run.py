#!/usr/bin/env python3
"""
Main entry point for the Ball Collection Robot application.

This script sets up the Python path and runs the main application from the src package.
"""

import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import and run the main application
from src.main import BallCollector

if __name__ == "__main__":
    collector = BallCollector()
    collector.run() 