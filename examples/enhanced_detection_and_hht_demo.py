"""
Enhanced Detection and HHT Export Demo
========================================

This demo showcases the three major improvements to the sEMG processing pipeline:

1. TKEO (Teager-Kaiser Energy Operator) preprocessing for improved PELT changepoint detection
2. Energy-aware segment merging for dumbbell exercise recognition
3. Batch HHT Hilbert spectrum export for all activity segments

Author: PRIMOCOSMOS
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import sEMG preprocessing functions
from semg_preprocessing import (
    apply_bandpass_filter,
    apply_notch_filter,
    detect_muscle_activity,
    segment_signal,
    apply_tkeo,
    export_activity_segments_hht,
)


def create_synthetic_dumbbell_exercise_signal(fs=1000, duration=10):
    """
    Create a synthetic sEMG signal simulating dumbbell bicep curls.
    
    This includes:
    - Multiple complete curl actions (lift + lower)
    - Transitions within each action (lift-to-lower)
    - Rest periods between actions
    - Realistic noise
    """
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.zeros_like(t)
    
    # Add baseline noise
    signal += np.random.normal(0, 0.05, len(t))
    
    # Simulate 3 dumbbell curl actions
    # Each action has: ramp up (lift) → transition → ramp down (lower)
    
    # Action 1: t=1s to t=3s (2 seconds total)
    action1_start = int(1 * fs)
    action1_mid = int(2 * fs)
    action1_end = int(3 * fs)
    
    # Lift phase (1s-2s): increasing amplitude
    lift1 = np.linspace(0, 1.0, action1_mid - action1_start)
    signal[action1_start:action1_mid] += lift1 + np.random.normal(0, 0.1, len(lift1))
    
    # Lower phase (2s-3s): decreasing amplitude (with transition point)
    lower1 = np.linspace(1.0, 0, action1_end - action1_mid)
    signal[action1_mid:action1_end] += lower1 + np.random.normal(0, 0.1, len(lower1))
    
    # Action 2: t=4s to t=6s
    action2_start = int(4 * fs)
    action2_mid = int(5 * fs)
    action2_end = int(6 * fs)
    
    lift2 = np.linspace(0, 1.2, action2_mid - action2_start)
    signal[action2_start:action2_mid] += lift2 + np.random.normal(0, 0.1, len(lift2))
    
    lower2 = np.linspace(1.2, 0, action2_end - action2_mid)
    signal[action2_mid:action2_end] += lower2 + np.random.normal(0, 0.1, len(lower2))
    
    # Action 3: t=7s to t=9s
    action3_start = int(7 * fs)
    action3_mid = int(8 * fs)
    action3_end = int(9 * fs)
    
    lift3 = np.linspace(0, 0.9, action3_mid - action3_start)
    signal[action3_start:action3_mid] += lift3 + np.random.normal(0, 0.1, len(lift3))
    
    lower3 = np.linspace(0.9, 0, action3_end - action3_mid)
    signal[action3_mid:action3_end] += lower3 + np.random.normal(0, 0.1, len(lower3))
    
    # Add some high-frequency EMG-like components
    for i in range(3):
        freq = np.random.uniform(50, 150)
        signal += 0.15 * np.sin(2 * np.pi * freq * t) * (signal > 0.1)
    
    return signal, t


def demonstrate_tkeo_effect(signal, fs):
    """Demonstrate the effect of TKEO preprocessing."""
    print("\n" + "="*70)
    print("1. DEMONSTRATING TKEO PREPROCESSING")
    print("="*70)
    
    # Apply TKEO
    tkeo_signal = apply_tkeo(signal)
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Original signal
    t = np.arange(len(signal)) / fs
    axes[0].plot(t, signal, 'b-', linewidth=1)
    axes[0].set_title('Original sEMG Signal', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # TKEO-enhanced signal
    axes[1].plot(t, tkeo_signal, 'r-', linewidth=1)
    axes[1].set_title('TKEO-Enhanced Signal (for changepoint detection)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('TKEO Energy', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Overlay for comparison
    axes[2].plot(t, signal / np.max(np.abs(signal)), 'b-', linewidth=1, alpha=0.6, label='Original (normalized)')
    axes[2].plot(t, tkeo_signal / np.max(tkeo_signal), 'r-', linewidth=1, alpha=0.6, label='TKEO (normalized)')
    axes[2].set_title('Comparison: TKEO Enhances Transitions', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Normalized Amplitude', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_tkeo_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ TKEO preprocessing demonstration saved to: demo_tkeo_effect.png")
    print("  - TKEO emphasizes rapid amplitude and frequency changes")
    print("  - Improves changepoint detection at activity boundaries")


def demonstrate_energy_aware_merging(signal, fs):
    """Demonstrate the improved energy-aware segment merging."""
    print("\n" + "="*70)
    print("2. DEMONSTRATING ENERGY-AWARE SEGMENT MERGING")
    print("="*70)
    
    # Detect with TKEO enabled
    segments_with_tkeo = detect_muscle_activity(
        signal, 
        fs=fs,
        min_duration=0.3,
        sensitivity=1.5,
        use_tkeo=True,
        return_changepoints=True
    )
    
    print(f"\n✓ Detected {len(segments_with_tkeo['segments'])} activity segments with energy-aware merging:")
    for i, (start, end) in enumerate(segments_with_tkeo['segments']):
        duration = (end - start) / fs
        print(f"  Segment {i+1}: {start/fs:.2f}s - {end/fs:.2f}s (duration: {duration:.2f}s)")
    
    print(f"\n✓ PELT detected {len(segments_with_tkeo['changepoints'])} change points")
    
    # Visualize detection results
    fig, ax = plt.subplots(figsize=(14, 6))
    
    t = np.arange(len(signal)) / fs
    ax.plot(t, signal, 'b-', linewidth=1, alpha=0.6, label='sEMG Signal')
    
    # Mark detected segments
    for i, (start, end) in enumerate(segments_with_tkeo['segments']):
        ax.axvspan(start/fs, end/fs, alpha=0.3, color='green', label='Activity Segment' if i == 0 else '')
    
    # Mark changepoints
    for cp in segments_with_tkeo['changepoints']:
        if cp < len(signal):
            ax.axvline(cp/fs, color='red', linestyle='--', alpha=0.5, linewidth=1,
                      label='Changepoints' if cp == segments_with_tkeo['changepoints'][0] else '')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Energy-Aware Segment Merging for Dumbbell Exercise Recognition', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_energy_aware_merging.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Energy-aware merging visualization saved to: demo_energy_aware_merging.png")
    print("  - Adjacent segments with HIGH boundary energy → MERGED (same action)")
    print("  - Adjacent segments with LOW boundary energy → SEPARATE (different actions)")
    print("  - Non-adjacent segments → ALWAYS SEPARATE")
    
    return segments_with_tkeo['segments']


def demonstrate_hht_batch_export(signal, segments, fs):
    """Demonstrate batch HHT Hilbert spectrum export."""
    print("\n" + "="*70)
    print("3. DEMONSTRATING BATCH HHT HILBERT SPECTRUM EXPORT")
    print("="*70)
    
    # Create output directory
    output_dir = Path('./hht_output_demo')
    output_dir.mkdir(exist_ok=True)
    
    # Export all Hilbert spectra
    print(f"\n✓ Exporting Hilbert spectra for {len(segments)} activity segments...")
    
    export_info = export_activity_segments_hht(
        signal,
        segments,
        fs=fs,
        output_dir=str(output_dir),
        base_filename='dumbbell_curl',
        n_freq_bins=128,
        normalize_length=128,
        use_ceemdan=True,
        save_visualization=True,
        dpi=150
    )
    
    print(f"\n✓ Successfully exported {len(export_info)} Hilbert spectra!")
    print(f"\nExport summary:")
    for info in export_info:
        print(f"  Segment {info['segment_number']:03d}:")
        print(f"    - NPZ file: {Path(info['npz_path']).name}")
        if 'png_path' in info:
            print(f"    - PNG image: {Path(info['png_path']).name}")
    
    print(f"\n✓ All files saved to: {output_dir}/")
    print("  - Each segment has its own NPZ matrix file and PNG visualization")
    print("  - NPZ files contain: spectrum, time axis, frequency axis, sampling rate")
    print("  - PNG files show time-frequency Hilbert spectrum representation")
    
    return export_info


def main():
    """Main demonstration function."""
    print("\n" + "="*70)
    print("ENHANCED sEMG DETECTION AND HHT EXPORT DEMONSTRATION")
    print("="*70)
    print("\nThis demo showcases three major improvements:")
    print("1. TKEO preprocessing for better changepoint detection")
    print("2. Energy-aware segment merging for dumbbell exercises")
    print("3. Batch HHT Hilbert spectrum export for all segments")
    print("="*70)
    
    # Generate synthetic signal
    print("\n✓ Generating synthetic dumbbell exercise signal...")
    fs = 1000  # 1000 Hz sampling rate
    signal, t = create_synthetic_dumbbell_exercise_signal(fs=fs, duration=10)
    print(f"  - Duration: 10 seconds")
    print(f"  - Sampling rate: {fs} Hz")
    print(f"  - Contains 3 complete dumbbell curl actions")
    
    # Apply standard preprocessing
    print("\n✓ Applying standard preprocessing...")
    filtered = apply_bandpass_filter(signal, fs=fs, lowcut=20, highcut=450)
    filtered = apply_notch_filter(filtered, fs=fs, freq=50, harmonics=[1, 2])
    print("  - Bandpass filter: 20-450 Hz")
    print("  - Notch filter: 50 Hz + harmonics")
    
    # Demonstration 1: TKEO effect
    demonstrate_tkeo_effect(filtered, fs)
    
    # Demonstration 2: Energy-aware merging
    segments = demonstrate_energy_aware_merging(filtered, fs)
    
    # Demonstration 3: Batch HHT export
    demonstrate_hht_batch_export(filtered, segments, fs)
    
    # Final summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nGenerated outputs:")
    print("  1. demo_tkeo_effect.png - TKEO preprocessing visualization")
    print("  2. demo_energy_aware_merging.png - Segment detection visualization")
    print("  3. ./hht_output_demo/ - Directory with Hilbert spectra for all segments")
    print("\nKey improvements demonstrated:")
    print("  ✓ TKEO enhances transitions for better changepoint detection")
    print("  ✓ Energy-aware merging correctly identifies complete dumbbell actions")
    print("  ✓ Batch export creates individual HHT analyses for each activity segment")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
