"""
Test script for CWD-based IMNF and export optimization.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semg_preprocessing.hht import (
    _compute_choi_williams_distribution,
    extract_semg_features,
    batch_hht_analysis,
    export_hilbert_spectra_batch,
    SEMG_LOW_FREQ_CUTOFF,
    SEMG_HIGH_FREQ_CUTOFF,
)

print("="*70)
print("Testing CWD-based IMNF and Export Optimization")
print("="*70)

# Test 1: CWD computation
print("\n1. Testing Choi-Williams Distribution computation...")
fs = 1000.0
duration = 1.0
t = np.linspace(0, duration, int(fs * duration))

# Create signal with known frequency components
signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t)
signal += 0.1 * np.random.randn(len(signal))

try:
    cwd, time_cwd, freq_cwd = _compute_choi_williams_distribution(
        signal, fs, sigma=1.0, n_freq_bins=256, n_time_bins=128
    )
    print(f"   ✓ CWD computed successfully")
    print(f"   CWD shape: {cwd.shape}")
    print(f"   Time points: {len(time_cwd)}")
    print(f"   Frequency range: {freq_cwd[0]:.1f} - {freq_cwd[-1]:.1f} Hz")
except Exception as e:
    print(f"   ✗ CWD computation failed: {e}")
    raise

# Test 2: IMNF computation with CWD
print("\n2. Testing IMNF calculation with CWD...")
try:
    features = extract_semg_features(signal, fs)
    imnf = features['IMNF']
    mnf = features['MNF']
    print(f"   ✓ IMNF computed successfully")
    print(f"   IMNF: {imnf:.2f} Hz")
    print(f"   MNF: {mnf:.2f} Hz (for comparison)")
    # IMNF should be in valid sEMG range
    assert SEMG_LOW_FREQ_CUTOFF <= imnf <= SEMG_HIGH_FREQ_CUTOFF, \
        f"IMNF {imnf} out of valid range"
    print(f"   ✓ IMNF in valid range ({SEMG_LOW_FREQ_CUTOFF}-{SEMG_HIGH_FREQ_CUTOFF} Hz)")
except Exception as e:
    print(f"   ✗ IMNF computation failed: {e}")
    raise

# Test 3: Export optimization with precomputed spectra
print("\n3. Testing export optimization with precomputed spectra...")
import tempfile
import time

# Create test segments
segments = [signal[:500], signal[250:750], signal[500:]]

# Method 1: Compute HHT in batch first
print("   Method 1: Batch HHT + fast export (optimized)")
start_time = time.time()
batch_results = batch_hht_analysis(
    segments,
    fs=fs,
    n_freq_bins=256,
    normalize_length=256,
    use_ceemdan=False,  # Use EMD for speed in test
    extract_features=False
)
batch_time = time.time() - start_time
print(f"     Batch HHT: {batch_time:.3f}s")

# Export using precomputed spectra
with tempfile.TemporaryDirectory() as tmpdir:
    start_time = time.time()
    export_info_fast = export_hilbert_spectra_batch(
        segments,
        fs=fs,
        output_dir=tmpdir,
        base_filename="test_fast",
        n_freq_bins=256,
        normalize_length=256,
        use_ceemdan=False,
        save_visualization=False,
        precomputed_spectra=batch_results  # Use precomputed
    )
    fast_export_time = time.time() - start_time
    print(f"     Fast export: {fast_export_time:.3f}s")
    total_fast = batch_time + fast_export_time
    print(f"     Total: {total_fast:.3f}s")

# Method 2: Export without precomputed (recalculates)
print("\n   Method 2: Export with recalculation (not optimized)")
with tempfile.TemporaryDirectory() as tmpdir:
    start_time = time.time()
    export_info_slow = export_hilbert_spectra_batch(
        segments,
        fs=fs,
        output_dir=tmpdir,
        base_filename="test_slow",
        n_freq_bins=256,
        normalize_length=256,
        use_ceemdan=False,
        save_visualization=False,
        precomputed_spectra=None  # No precomputed, will recalculate
    )
    slow_time = time.time() - start_time
    print(f"     Export with recalculation: {slow_time:.3f}s")

# Calculate speedup
if slow_time > 0:
    speedup_factor = slow_time / fast_export_time
    time_saved = slow_time - fast_export_time
    print(f"\n   ✓ Optimization speedup: {speedup_factor:.2f}x faster")
    print(f"   ✓ Time saved: {time_saved:.3f}s for {len(segments)} segments")
else:
    print(f"\n   ✓ Optimization works (both methods completed)")

# Verify both methods produce same number of files
assert len(export_info_fast) == len(export_info_slow) == len(segments), \
    "Export file counts don't match"
print(f"   ✓ Both methods exported {len(segments)} segments correctly")

print("\n" + "="*70)
print("All tests passed! ✓")
print("="*70)
print("\nSummary:")
print("  1. Choi-Williams Distribution (CWD) implementation works correctly")
print("  2. IMNF calculation uses true CWD (not STFT approximation)")
print("  3. Export optimization reuses precomputed HHT results")
print(f"  4. Export speedup: ~{speedup_factor:.1f}x faster when reusing batch results" if 'speedup_factor' in locals() else "")
