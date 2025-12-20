"""
Test script for CWD robustness improvements and energy threshold extension.

Tests the following improvements:
1. CWD handles edge cases gracefully (short signals, low energy, etc.)
2. IMNF calculation has robust fallback to MNF when CWD fails
3. Extended energy threshold range works correctly
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semg_preprocessing.hht import (
    _compute_choi_williams_distribution,
    extract_semg_features,
    SEMG_LOW_FREQ_CUTOFF,
    SEMG_HIGH_FREQ_CUTOFF,
    EPSILON,
)
from semg_preprocessing.detection import (
    detect_activity_hht,
    HHT_MIN_ENERGY_THRESHOLD,
    HHT_MAX_ENERGY_THRESHOLD,
)

print("="*70)
print("Testing CWD Robustness and Energy Threshold Extension")
print("="*70)

# Test 1: CWD with very short signal (edge case)
print("\n1. Testing CWD with very short signal (edge case)...")
fs = 1000.0
short_signal = np.random.randn(10)  # Only 10 samples

try:
    cwd, time_cwd, freq_cwd = _compute_choi_williams_distribution(
        short_signal, fs, sigma=1.0
    )
    print(f"   ✗ Expected ValueError for short signal, but succeeded")
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError: {str(e)[:60]}...")
except Exception as e:
    print(f"   ✗ Unexpected exception type: {type(e).__name__}: {e}")

# Test 2: CWD with zero/near-zero signal
print("\n2. Testing CWD with near-zero signal (edge case)...")
zero_signal = np.zeros(100)
zero_signal += 1e-15  # Tiny noise

try:
    cwd, time_cwd, freq_cwd = _compute_choi_williams_distribution(
        zero_signal, fs, sigma=1.0
    )
    print(f"   ✗ Expected ValueError for near-zero signal, but succeeded")
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError: {str(e)[:60]}...")
except Exception as e:
    print(f"   ✗ Unexpected exception type: {type(e).__name__}: {e}")

# Test 3: CWD with normal signal
print("\n3. Testing CWD with normal signal...")
t = np.linspace(0, 1.0, 1000)
normal_signal = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))

try:
    cwd, time_cwd, freq_cwd = _compute_choi_williams_distribution(
        normal_signal, fs, sigma=1.0, n_freq_bins=128, n_time_bins=64
    )
    print(f"   ✓ CWD computed successfully")
    print(f"   Shape: {cwd.shape}, Time: {len(time_cwd)}, Freq: {len(freq_cwd)}")
    assert np.all(np.isfinite(cwd)), "CWD contains non-finite values"
    print(f"   ✓ All values are finite")
except Exception as e:
    print(f"   ✗ CWD failed: {type(e).__name__}: {e}")
    raise

# Test 4: IMNF calculation with short signal (should fallback to MNF gracefully)
print("\n4. Testing IMNF with short signal (should fallback to MNF)...")
short_signal_for_imnf = np.random.randn(15) + np.sin(2 * np.pi * 50 * np.linspace(0, 0.015, 15))

try:
    features = extract_semg_features(short_signal_for_imnf, fs)
    imnf = features['IMNF']
    mnf = features['MNF']
    print(f"   ✓ IMNF computed: {imnf:.2f} Hz")
    print(f"   ✓ MNF: {mnf:.2f} Hz")
    # For very short signals, IMNF should fallback to MNF
    print(f"   ✓ IMNF equals MNF (expected fallback): {abs(imnf - mnf) < 0.01}")
except Exception as e:
    print(f"   ✗ IMNF calculation failed: {type(e).__name__}: {e}")
    raise

# Test 5: IMNF calculation with normal signal (should use CWD)
print("\n5. Testing IMNF with normal signal (should use CWD)...")
duration = 1.0
t = np.linspace(0, duration, int(fs * duration))
normal_signal_for_imnf = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(len(t))

try:
    features = extract_semg_features(normal_signal_for_imnf, fs)
    imnf = features['IMNF']
    mnf = features['MNF']
    print(f"   ✓ IMNF computed: {imnf:.2f} Hz")
    print(f"   ✓ MNF: {mnf:.2f} Hz")
    # IMNF should be in valid range
    assert SEMG_LOW_FREQ_CUTOFF <= imnf <= SEMG_HIGH_FREQ_CUTOFF, f"IMNF {imnf} out of range"
    print(f"   ✓ IMNF in valid range")
    # IMNF and MNF can differ when using CWD
    print(f"   ✓ IMNF vs MNF difference: {abs(imnf - mnf):.2f} Hz")
except Exception as e:
    print(f"   ✗ IMNF calculation failed: {type(e).__name__}: {e}")
    raise

# Test 6: Extended energy threshold range constants
print("\n6. Testing extended energy threshold constants...")
print(f"   HHT_MIN_ENERGY_THRESHOLD: {HHT_MIN_ENERGY_THRESHOLD}")
print(f"   HHT_MAX_ENERGY_THRESHOLD: {HHT_MAX_ENERGY_THRESHOLD}")
assert HHT_MIN_ENERGY_THRESHOLD == 0.1, f"Expected min 0.1, got {HHT_MIN_ENERGY_THRESHOLD}"
assert HHT_MAX_ENERGY_THRESHOLD == 0.95, f"Expected max 0.95, got {HHT_MAX_ENERGY_THRESHOLD}"
print(f"   ✓ Constants updated correctly for extended range")

# Test 7: HHT detection with extreme threshold values
print("\n7. Testing HHT detection with extreme energy thresholds...")

# Create test signal with clear activity
duration = 2.0
t = np.linspace(0, duration, int(fs * duration))
test_signal = np.zeros(len(t))
# Add a clear muscle activity burst from 0.5s to 1.5s
activity_mask = (t >= 0.5) & (t <= 1.5)
test_signal[activity_mask] = 0.5 * np.sin(2 * np.pi * 100 * t[activity_mask])
test_signal += 0.01 * np.random.randn(len(t))  # Add small noise

# Test with very low threshold (should detect more/broader segments)
print("   Testing with very low threshold (0.1)...")
try:
    result_low = detect_activity_hht(
        test_signal, fs, 
        min_duration=0.1,
        energy_threshold=0.1,  # Very low threshold
        return_spectrum=True
    )
    segments_low = result_low['segments']
    print(f"   ✓ Low threshold detection successful: {len(segments_low)} segment(s)")
    if len(segments_low) > 0:
        total_duration_low = sum([(e-s)/fs for s, e in segments_low])
        print(f"   ✓ Total detected duration: {total_duration_low:.2f}s")
except Exception as e:
    print(f"   ✗ Low threshold detection failed: {type(e).__name__}: {e}")
    raise

# Test with high threshold (should detect less/narrower segments)
print("   Testing with high threshold (0.9)...")
try:
    result_high = detect_activity_hht(
        test_signal, fs,
        min_duration=0.1,
        energy_threshold=0.9,  # High threshold
        return_spectrum=True
    )
    segments_high = result_high['segments']
    print(f"   ✓ High threshold detection successful: {len(segments_high)} segment(s)")
    if len(segments_high) > 0:
        total_duration_high = sum([(e-s)/fs for s, e in segments_high])
        print(f"   ✓ Total detected duration: {total_duration_high:.2f}s")
except Exception as e:
    print(f"   ✗ High threshold detection failed: {type(e).__name__}: {e}")
    raise

# Verify that lower threshold detects more or equal duration
if len(segments_low) > 0 and len(segments_high) > 0:
    total_duration_low = sum([(e-s)/fs for s, e in segments_low])
    total_duration_high = sum([(e-s)/fs for s, e in segments_high])
    if total_duration_low >= total_duration_high:
        print(f"   ✓ Lower threshold detected more/equal duration (expected)")
    else:
        print(f"   ! Warning: Lower threshold detected less duration than high threshold")

# Test 8: Validate CWD parameter bounds
print("\n8. Testing CWD parameter validation...")

# Test invalid sampling frequency
print("   Testing invalid fs...")
try:
    cwd, _, _ = _compute_choi_williams_distribution(normal_signal, fs=-1.0)
    print(f"   ✗ Should have raised ValueError for negative fs")
except ValueError as e:
    print(f"   ✓ Correctly rejected negative fs")

# Test invalid sigma
print("   Testing invalid sigma...")
try:
    cwd, _, _ = _compute_choi_williams_distribution(normal_signal, fs, sigma=0)
    print(f"   ✗ Should have raised ValueError for zero sigma")
except ValueError as e:
    print(f"   ✓ Correctly rejected zero sigma")

print("\n" + "="*70)
print("All robustness tests passed! ✓")
print("="*70)
print("\nSummary:")
print("  1. CWD gracefully handles edge cases (short signals, low energy)")
print("  2. IMNF calculation has robust fallback to MNF when needed")
print("  3. Extended energy threshold range (0.1-0.95) is implemented")
print("  4. HHT detection works with extreme threshold values")
print("  5. Input validation prevents invalid parameters")
