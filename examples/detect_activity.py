"""
Example: Muscle activity detection using advanced PELT algorithm.

This script demonstrates:
1. Detection using the new PELT-based combined method with single detector
2. Detection using multi-detector ensemble with different fusion methods
3. Comparison of different detector configurations
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
)


def main():
    # Load and preprocess data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'sample_emg.csv')
    fs = 1000.0
    
    print("Loading and preprocessing sEMG data...")
    signal, _ = load_csv_data(input_file, value_column=1)
    
    # Preprocess signal
    filtered = apply_bandpass_filter(signal, fs, lowcut=20, highcut=450)
    filtered = apply_notch_filter(filtered, fs, freq=50, harmonics=[1, 2, 3])
    
    # Detect muscle activity using different configurations
    print("\nDetecting muscle activity with advanced PELT algorithm...")
    
    print("  1. Single detector (baseline)...")
    segments_single = detect_muscle_activity(
        filtered, fs, method='combined', 
        min_duration=0.1, sensitivity=1.5,
        use_multi_detector=False
    )
    
    print("  2. Multi-detector with confidence fusion (recommended)...")
    segments_confidence = detect_muscle_activity(
        filtered, fs, method='combined',
        min_duration=0.1, sensitivity=1.5,
        n_detectors=3, fusion_method='confidence',
        use_multi_detector=True
    )
    
    print("  3. Multi-detector with voting fusion...")
    segments_voting = detect_muscle_activity(
        filtered, fs, method='combined',
        min_duration=0.1, sensitivity=1.5,
        n_detectors=3, fusion_method='voting',
        use_multi_detector=True
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Single detector: {len(segments_single)} segments")
    print(f"  Confidence fusion: {len(segments_confidence)} segments")
    print(f"  Voting fusion: {len(segments_voting)} segments")
    
    # Get detailed segment information
    print("\nConfidence fusion segment details:")
    segments_info = segment_signal(filtered, segments_confidence, fs, include_metadata=True)
    for i, seg in enumerate(segments_info):
        print(f"  Segment {i+1}: "
              f"{seg['start_time']:.3f}s - {seg['end_time']:.3f}s "
              f"(duration: {seg['duration']:.3f}s, "
              f"peak: {seg['peak_amplitude']:.3f})")
    
    # Visualize
    visualize_detection_comparison(
        filtered, 
        segments_single, 
        segments_confidence, 
        segments_voting,
        fs
    )
    
    print("\nVisualization saved!")


def visualize_detection_comparison(signal, single_segs, conf_segs, vote_segs, fs):
    """Visualize muscle activity detection results from different configurations."""
    time = np.arange(len(signal)) / fs
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    methods = [
        (single_segs, 'Single Detector', 'blue'),
        (conf_segs, 'Multi-Detector: Confidence Fusion (Recommended)', 'red'),
        (vote_segs, 'Multi-Detector: Voting Fusion', 'green')
    ]
    
    for ax, (segments, title, color) in zip(axes, methods):
        ax.plot(time, signal, 'k-', alpha=0.5, linewidth=0.5, label='Filtered Signal')
        
        # Highlight detected segments
        for start, end in segments:
            ax.axvspan(start/fs, end/fs, alpha=0.4, color=color, 
                      label='Detected Activity' if start == segments[0][0] else '')
        
        ax.set_title(title + f' ({len(segments)} segments detected)', 
                    fontweight='bold', fontsize=11)
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_path = os.path.join(project_dir, 'data', 'detection_comparison.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Detection comparison saved to: {output_path}")


if __name__ == '__main__':
    main()
