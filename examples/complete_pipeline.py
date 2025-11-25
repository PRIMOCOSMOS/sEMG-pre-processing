"""
Complete example of sEMG signal preprocessing pipeline.

This script demonstrates:
1. Loading sEMG data from CSV
2. Applying filters (high-pass, low-pass, notch)
3. Detecting muscle activity
4. Segmenting the signal
5. Saving processed data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from semg_preprocessing import (
    load_csv_data,
    apply_bandpass_filter,
    apply_notch_filter,
    detect_muscle_activity,
    segment_signal,
    save_processed_data,
)


def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'sample_emg.csv')
    output_file = os.path.join(project_dir, 'data', 'processed_emg.csv')
    fs = 1000.0  # Sampling frequency in Hz (adjust based on your data)
    
    print("=" * 60)
    print("sEMG Signal Preprocessing Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data from CSV...")
    signal, df = load_csv_data(input_file, value_column=1, has_header=True)
    print(f"  Loaded {len(signal)} samples")
    print(f"  Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    
    # Step 2: Apply bandpass filter
    print("\n[Step 2] Applying bandpass filter (20-450 Hz)...")
    filtered_signal = apply_bandpass_filter(
        signal, 
        fs=fs,
        lowcut=20.0,    # High-pass cutoff (removes motion artifacts, baseline drift)
        highcut=450.0,  # Low-pass cutoff (removes high-frequency noise)
        order=4,
        filter_type='butterworth'
    )
    print("  Bandpass filter applied")
    
    # Step 3: Remove power line interference
    print("\n[Step 3] Removing power line interference (50 Hz + harmonics)...")
    filtered_signal = apply_notch_filter(
        filtered_signal,
        fs=fs,
        freq=50.0,  # Use 60.0 for North America
        quality_factor=30.0,
        harmonics=[1, 2, 3]  # Remove 50Hz, 100Hz, 150Hz
    )
    print("  Notch filter applied at 50Hz, 100Hz, 150Hz")
    
    # Step 4: Detect muscle activity
    print("\n[Step 4] Detecting muscle activity events...")
    activity_segments = detect_muscle_activity(
        filtered_signal,
        fs=fs,
        method='combined',  # Use combined ruptures + amplitude method
        amplitude_threshold=None,  # Auto-calculate threshold
        min_duration=0.1,  # Minimum 100ms activity duration
        pen=3  # Ruptures penalty parameter
    )
    print(f"  Detected {len(activity_segments)} muscle activity segments")
    
    # Step 5: Segment the signal
    print("\n[Step 5] Segmenting signal based on detected activity...")
    segments = segment_signal(
        filtered_signal,
        activity_segments,
        fs=fs,
        include_metadata=True
    )
    
    # Print segment information
    print("\n  Segment details:")
    for i, seg in enumerate(segments):
        print(f"    Segment {i+1}: "
              f"{seg['start_time']:.3f}s - {seg['end_time']:.3f}s "
              f"(duration: {seg['duration']:.3f}s, "
              f"peak: {seg['peak_amplitude']:.3f}, "
              f"RMS: {seg['rms']:.3f})")
    
    # Step 6: Save processed data
    print("\n[Step 6] Saving processed data...")
    save_processed_data(
        output_file,
        filtered_signal,
        fs=fs,
        include_time=True
    )
    
    # Step 7: Visualization (optional)
    print("\n[Step 7] Generating visualization...")
    visualize_results(signal, filtered_signal, activity_segments, fs)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


def visualize_results(original, filtered, segments, fs):
    """
    Create visualization of preprocessing results.
    """
    time = np.arange(len(original)) / fs
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original signal
    axes[0].plot(time, original, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_title('Original sEMG Signal', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot filtered signal with detected segments
    axes[1].plot(time, filtered, 'g-', alpha=0.7, linewidth=0.5, label='Filtered Signal')
    
    # Highlight detected muscle activity segments
    for start, end in segments:
        axes[1].axvspan(start/fs, end/fs, alpha=0.3, color='red', label='Activity' if start == segments[0][0] else '')
    
    axes[1].set_title('Filtered Signal with Detected Muscle Activity', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save to project data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_path = os.path.join(project_dir, 'data', 'preprocessing_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved to: {output_path}")
    
    # Optional: Show plot (comment out if running headless)
    # plt.show()


if __name__ == '__main__':
    main()
