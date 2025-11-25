#!/usr/bin/env python
"""
sEMG Preprocessing Toolkit - Application Launcher

This script provides a simple way to launch the GUI application.
Just run: python run_app.py

For Windows users, you can also double-click on run_app.bat
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed."""
    required = [
        'numpy',
        'scipy', 
        'pandas',
        'matplotlib',
        'scikit-learn',
        'ruptures',
        'gradio',
        'pywavelets'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg if pkg != 'pywavelets' else 'pywt')
        except ImportError:
            missing.append(pkg)
    
    return missing


def install_dependencies(packages):
    """Install missing packages using pip."""
    print(f"Installing missing packages: {', '.join(packages)}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)


def main():
    """Main entry point."""
    print("=" * 60)
    print("  sEMG Preprocessing Toolkit v0.4.0")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        response = input("Install missing packages? (y/n): ")
        if response.lower() in ['y', 'yes']:
            install_dependencies(missing)
            print("Dependencies installed successfully!")
        else:
            print("Cannot run without dependencies. Exiting.")
            sys.exit(1)
    else:
        print("All dependencies are installed.")
    
    print()
    print("Starting GUI application...")
    print("Open your browser to: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Launch the GUI
    from gui_app import create_gui
    app = create_gui()
    app.queue(default_concurrency_limit=4)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True  # Automatically open browser
    )


if __name__ == "__main__":
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Add to path
    sys.path.insert(0, script_dir)
    
    main()
