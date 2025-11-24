"""
Example: Multi-Feature Detection and Segment Extraction

This script demonstrates the new advanced features:
1. Multi-feature fusion detection with adaptive parameters
2. Exporting detected segments as individual CSV files
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
    export_segments_to_csv,
)


def main():
    # Load and preprocess data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'sample_emg.csv')
    output_dir = os.path.join(project_dir, 'data', 'exported_segments')
    fs = 1000.0
    
    print("=" * 70)
    print("Multi-Feature Detection & Segment Extraction Demo")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading sEMG data...")
    signal, _ = load_csv_data(input_file, value_column=1)
    print(f"  Loaded {len(signal)} samples ({len(signal)/fs:.2f}s)")
    
    # Preprocess
    print("\n[2/5] Applying filters...")
    filtered = apply_bandpass_filter(signal, fs, lowcut=20, highcut=450)
    filtered = apply_notch_filter(filtered, fs, freq=50, harmonics=[1, 2, 3])
    print("  ✓ Bandpass and notch filters applied")
    
    # Detect with different methods for comparison
    print("\n[3/5] Detecting muscle activity with different methods...")
    
    methods_results = {}
    
    # Method 1: Traditional combined
    print("  Testing 'combined' method...")
    seg_combined = detect_muscle_activity(filtered, fs, method='combined')
    methods_results['combined'] = seg_combined
    print(f"    → Detected {len(seg_combined)} segments")
    
    # Method 2: Multi-feature without clustering
    print("  Testing 'multi_feature' (no clustering)...")
    seg_multi_no_clust = detect_muscle_activity(
        filtered, fs, 
        method='multi_feature',
        use_clustering=False,
        adaptive_pen=True
    )
    methods_results['multi_feature (no clustering)'] = seg_multi_no_clust
    print(f"    → Detected {len(seg_multi_no_clust)} segments")
    
    # Method 3: Multi-feature with clustering (recommended)
    print("  Testing 'multi_feature' with clustering (recommended)...")
    seg_multi_clust = detect_muscle_activity(
        filtered, fs,
        method='multi_feature',
        use_clustering=True,
        adaptive_pen=True,
        min_duration=0.1
    )
    methods_results['multi_feature (with clustering)'] = seg_multi_clust
    print(f"    → Detected {len(seg_multi_clust)} segments")
    
    # Use the best method for export
    best_segments = seg_multi_clust
    
    # Get detailed info
    print("\n[4/5] Extracting segment metadata...")
    segment_info = segment_signal(filtered, best_segments, fs, include_metadata=True)
    
    print("\n  Detected segments:")
    for i, seg in enumerate(segment_info, 1):
        print(f"    Segment {i}: {seg['start_time']:.3f}s - {seg['end_time']:.3f}s "
              f"(duration: {seg['duration']:.3f}s, peak: {seg['peak_amplitude']:.3f}, "
              f"RMS: {seg['rms']:.3f})")
    
    # Export segments
    print(f"\n[5/5] Exporting segments to {output_dir}...")
    saved_files = export_segments_to_csv(
        filtered,
        segment_info,
        fs=fs,
        output_dir=output_dir,
        prefix='muscle_activity'
    )
    print(f"  ✓ Exported {len(saved_files)} segment files")
    
    # Visualize comparison
    print("\n[Visualization] Creating comparison plot...")
    visualize_comparison(filtered, methods_results, fs, project_dir)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print(f"\nSegment files saved to: {output_dir}")
    print("Comparison plot saved to: data/multi_feature_comparison.png")


def visualize_comparison(signal, methods_results, fs, project_dir):
    """Create comparison visualization of different detection methods."""
    time = np.arange(len(signal)) / fs
    
    n_methods = len(methods_results)
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 4 * n_methods))
    
    if n_methods == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for ax, (method_name, segments), color in zip(axes, methods_results.items(), colors):
        ax.plot(time, signal, 'k-', alpha=0.4, linewidth=0.5, label='Filtered Signal')
        
        # Highlight detected segments
        for i, (start, end) in enumerate(segments):
            ax.axvspan(start/fs, end/fs, alpha=0.4, color=color,
                      label='Detected Activity' if i == 0 else '')
        
        ax.set_title(f'{method_name} ({len(segments)} segments detected)',
                    fontweight='bold', fontsize=11)
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    output_path = os.path.join(project_dir, 'data', 'multi_feature_comparison.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Comparison plot saved to: {output_path}")


if __name__ == '__main__':
    main()
