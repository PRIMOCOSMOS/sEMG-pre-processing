"""
Example: Comparing different filtering methods for sEMG preprocessing.

This script demonstrates and compares:
1. Different filter types (Butterworth vs Chebyshev)
2. Different filter orders
3. Notch filter vs DFT method for power line removal
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from semg_preprocessing import (
    load_csv_data,
    apply_highpass_filter,
    apply_lowpass_filter,
    apply_notch_filter,
    remove_powerline_dft,
)


def main():
    # Load sample data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'sample_emg.csv')
    fs = 1000.0
    
    print("Loading sEMG data...")
    signal, _ = load_csv_data(input_file, value_column=1)
    
    # Compare different high-pass filter configurations
    print("\nComparing high-pass filters...")
    hp_butter_2 = apply_highpass_filter(signal, fs, cutoff=20, order=2, filter_type='butterworth')
    hp_butter_4 = apply_highpass_filter(signal, fs, cutoff=20, order=4, filter_type='butterworth')
    hp_cheby_4 = apply_highpass_filter(signal, fs, cutoff=20, order=4, filter_type='chebyshev')
    
    # Compare power line removal methods
    print("\nComparing power line interference removal methods...")
    
    # Apply bandpass first
    from semg_preprocessing import apply_bandpass_filter
    bandpassed = apply_bandpass_filter(signal, fs, lowcut=20, highcut=450)
    
    # Method 1: Notch filter
    notch_filtered = apply_notch_filter(bandpassed, fs, freq=50, harmonics=[1, 2, 3])
    
    # Method 2: DFT method
    dft_filtered = remove_powerline_dft(bandpassed, fs, freq=50, harmonics=[1, 2, 3])
    
    # Visualize comparisons
    visualize_filter_comparison(signal, hp_butter_2, hp_butter_4, hp_cheby_4, fs)
    visualize_powerline_comparison(bandpassed, notch_filtered, dft_filtered, fs)
    
    print("\nComparison complete! Check the generated plots.")


def visualize_filter_comparison(original, butter2, butter4, cheby4, fs):
    """Compare different high-pass filter configurations."""
    time = np.arange(len(original)) / fs
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    signals = [
        (original, 'Original Signal'),
        (butter2, 'Butterworth Order 2'),
        (butter4, 'Butterworth Order 4'),
        (cheby4, 'Chebyshev Order 4')
    ]
    
    for ax, (sig, title) in zip(axes, signals):
        ax.plot(time[:5000], sig[:5000], linewidth=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_path = os.path.join(project_dir, 'data', 'filter_comparison.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Filter comparison saved to: {output_path}")


def visualize_powerline_comparison(original, notch, dft, fs):
    """Compare notch filter vs DFT method for power line removal."""
    # Calculate frequency spectra
    from scipy import signal
    
    f_orig, psd_orig = signal.welch(original, fs, nperseg=1024)
    f_notch, psd_notch = signal.welch(notch, fs, nperseg=1024)
    f_dft, psd_dft = signal.welch(dft, fs, nperseg=1024)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time domain
    time = np.arange(len(original)) / fs
    axes[0].plot(time[:2000], original[:2000], 'b-', alpha=0.7, label='With Power Line', linewidth=0.8)
    axes[0].plot(time[:2000], notch[:2000], 'r-', alpha=0.7, label='Notch Filtered', linewidth=0.8)
    axes[0].plot(time[:2000], dft[:2000], 'g-', alpha=0.7, label='DFT Filtered', linewidth=0.8)
    axes[0].set_title('Time Domain Comparison', fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Frequency domain
    axes[1].semilogy(f_orig, psd_orig, 'b-', alpha=0.7, label='With Power Line', linewidth=1.5)
    axes[1].semilogy(f_notch, psd_notch, 'r-', alpha=0.7, label='Notch Filtered', linewidth=1.5)
    axes[1].semilogy(f_dft, psd_dft, 'g-', alpha=0.7, label='DFT Filtered', linewidth=1.5)
    axes[1].axvline(50, color='k', linestyle='--', alpha=0.5, label='50Hz')
    axes[1].axvline(100, color='k', linestyle='--', alpha=0.5)
    axes[1].axvline(150, color='k', linestyle='--', alpha=0.5)
    axes[1].set_title('Frequency Domain Comparison (Power Spectral Density)', fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('PSD')
    axes[1].set_xlim([0, 200])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_path = os.path.join(project_dir, 'data', 'powerline_comparison.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Power line comparison saved to: {output_path}")


if __name__ == '__main__':
    main()
