#!/bin/bash
# sEMG Preprocessing Toolkit - Unix/Mac Launcher
# Run this script to start the application: ./run_app.sh

echo "============================================================"
echo "  sEMG Preprocessing Toolkit v0.4.0"
echo "============================================================"
echo

# Function to find Python
find_python() {
    # Try python3 first (preferred on Unix/Mac)
    if command -v python3 &> /dev/null; then
        echo "python3"
        return 0
    fi
    
    # Try python
    if command -v python &> /dev/null; then
        # Check if it's Python 3
        version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1)
        if [ "$version" = "3" ]; then
            echo "python"
            return 0
        fi
    fi
    
    # Try common installation paths on Mac
    if [ -f "/usr/local/bin/python3" ]; then
        echo "/usr/local/bin/python3"
        return 0
    fi
    
    # Try Homebrew Python on Mac
    if [ -f "/opt/homebrew/bin/python3" ]; then
        echo "/opt/homebrew/bin/python3"
        return 0
    fi
    
    # Try pyenv
    if [ -f "$HOME/.pyenv/shims/python3" ]; then
        echo "$HOME/.pyenv/shims/python3"
        return 0
    fi
    
    return 1
}

# Find Python
PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3 is not installed or not found"
    echo
    echo "Please install Python 3.8+ from https://www.python.org/"
    echo "Or use your system's package manager:"
    echo "  - macOS: brew install python3"
    echo "  - Ubuntu/Debian: sudo apt install python3"
    echo "  - Fedora: sudo dnf install python3"
    exit 1
fi

echo "Found Python: $PYTHON_CMD"
echo

# Change to script directory
cd "$(dirname "$0")"

# Run the launcher script
"$PYTHON_CMD" run_app.py
