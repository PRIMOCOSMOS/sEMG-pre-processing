#!/bin/bash
# sEMG Preprocessing Toolkit - Unix/Mac Launcher
# Run this script to start the application: ./run_app.sh

echo "============================================================"
echo "  sEMG Preprocessing Toolkit v0.4.0"
echo "============================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

# Run the launcher script
python3 run_app.py
