# Quick Start Guide: Enhanced Features

This guide will help you quickly start using the three new enhanced features for sEMG analysis.

## Prerequisites

```bash
pip install numpy scipy pandas matplotlib scikit-learn PyWavelets ruptures
```

## 1. TKEO-Enhanced Detection (5 minutes)

The simplest way to benefit from TKEO preprocessing:

```python
from semg_preprocessing import (
    load_csv_data,
    apply_bandpass_filter,
    detect_muscle_activity
)

# Load your data
signal, df = load_csv_data('your_emg_data.csv', value_column=1)
fs = 1000  # Your sampling frequency

# Preprocess
filtered = apply_bandpass_filter(signal, fs, lowcut=20, highcut=450)

# Detect with TKEO (enabled by default)
segments = detect_muscle_activity(
    filtered,
    fs=fs,
    min_duration=0.5,  # Minimum action duration
    use_tkeo=True      # This is the default
)

print(f"Detected {len(segments)} muscle activity events")
for i, (start, end) in enumerate(segments):
    print(f"  Event {i+1}: {start/fs:.2f}s - {end/fs:.2f}s")
```

**That's it!** TKEO preprocessing is now working behind the scenes to improve changepoint detection.

## 2. Energy-Aware Merging for Dumbbell Exercises (5 minutes)

No code changes needed! The energy-aware merging is automatically applied when you use the detection function:

```python
# Detect dumbbell curl actions
# The algorithm will now correctly identify complete actions
# (lift + lower) as single events
segments = detect_muscle_activity(
    filtered,
    fs=1000,
    min_duration=0.5,  # Typical curl duration
    sensitivity=1.5    # Adjust if needed
)

# Each segment now represents a complete dumbbell action
print(f"Detected {len(segments)} complete dumbbell curl actions")
```

**How it works:**
- Adjacent segments with HIGH boundary energy → Merged (lift-to-lower transition)
- Adjacent segments with LOW boundary energy → Kept separate (different actions)
- Non-adjacent segments → Always kept separate

## 3. Batch HHT Export (10 minutes)

Export Hilbert spectra for all detected activity segments:

```python
from semg_preprocessing import detect_muscle_activity
from semg_preprocessing.hht import export_activity_segments_hht

# 1. Detect activity segments
segments = detect_muscle_activity(filtered, fs=1000, min_duration=0.5)

# 2. Export HHT for all segments in one line
export_info = export_activity_segments_hht(
    filtered,              # Your preprocessed signal
    segments,              # Detected segments
    fs=1000,
    output_dir='./hht_results',
    base_filename='bicep_curl'
)

# Done! Check ./hht_results/ for:
# - bicep_curl_001.npz (matrix data)
# - bicep_curl_001.png (visualization)
# - bicep_curl_002.npz
# - bicep_curl_002.png
# ... and so on

print(f"Exported {len(export_info)} Hilbert spectra")
```

**Loading the exported data:**

```python
import numpy as np

# Load a specific segment's Hilbert spectrum
data = np.load('./hht_results/bicep_curl_001.npz')
spectrum = data['spectrum']     # Frequency-time matrix
time = data['time']             # Time axis (normalized)
frequency = data['frequency']   # Frequency axis (Hz)
fs = data['sampling_rate']      # Original sampling rate

print(f"Spectrum shape: {spectrum.shape}")
print(f"Time range: {time[0]:.2f} to {time[-1]:.2f}")
print(f"Frequency range: {frequency[0]:.1f} to {frequency[-1]:.1f} Hz")
```

## Complete Example: From Raw Data to HHT Analysis

Here's a complete workflow in under 20 lines:

```python
from semg_preprocessing import (
    load_csv_data,
    apply_bandpass_filter,
    apply_notch_filter,
    detect_muscle_activity,
)
from semg_preprocessing.hht import export_activity_segments_hht

# Load and preprocess
signal, _ = load_csv_data('emg_data.csv', value_column=1)
fs = 1000
filtered = apply_bandpass_filter(signal, fs, 20, 450)
filtered = apply_notch_filter(filtered, fs, 50, harmonics=[1, 2])

# Detect with TKEO and energy-aware merging
segments = detect_muscle_activity(
    filtered, fs, min_duration=0.5, use_tkeo=True
)

# Export HHT for all segments
export_activity_segments_hht(
    filtered, segments, fs, './results', 'action'
)

print(f"✅ Processed {len(segments)} actions!")
```

## Customization Options

### Adjust TKEO Sensitivity

```python
# If detection is too aggressive
segments = detect_muscle_activity(
    filtered, fs,
    min_duration=0.5,
    sensitivity=2.0,  # Higher = less sensitive
    use_tkeo=True
)

# If missing events
segments = detect_muscle_activity(
    filtered, fs,
    min_duration=0.3,  # Lower minimum duration
    sensitivity=1.2,   # Lower = more sensitive
    use_tkeo=True
)
```

### Disable TKEO (for comparison)

```python
# Detect without TKEO
segments_without = detect_muscle_activity(
    filtered, fs, min_duration=0.5, use_tkeo=False
)

# Compare
print(f"With TKEO: {len(segments_with)} segments")
print(f"Without TKEO: {len(segments_without)} segments")
```

### Customize HHT Export

```python
export_activity_segments_hht(
    filtered, segments, fs,
    output_dir='./my_results',
    base_filename='my_exercise',
    n_freq_bins=256,        # Frequency resolution
    normalize_length=256,   # Time resolution
    use_ceemdan=True,       # High quality (slower)
    save_visualization=True,
    dpi=300                 # High-res images
)
```

### Fast HHT Export (for large datasets)

```python
export_activity_segments_hht(
    filtered, segments, fs,
    output_dir='./results',
    n_freq_bins=128,        # Lower resolution
    normalize_length=128,   # Lower resolution
    use_ceemdan=False,      # Faster EMD
    save_visualization=True,
    dpi=100                 # Lower DPI
)
```

## Troubleshooting

### "Detection splits my actions into multiple segments"

Try:
1. Lower `sensitivity` (1.0-1.5 instead of 1.5-2.0)
2. Ensure `use_tkeo=True` (default)
3. Check your `min_duration` setting

### "HHT export is too slow"

Try:
1. Reduce `n_freq_bins` to 128 or 64
2. Reduce `normalize_length` to 128 or 64
3. Set `use_ceemdan=False`
4. Lower `dpi` to 100 or 75

### "Spectrograms look mostly black"

This is normal - log-scale representation shows low amplitudes as dark. The high-energy regions (actual activity) will be bright.

### "TKEO creates noisy results"

The implementation includes automatic smoothing. If needed:
1. Apply more aggressive pre-filtering (lower highcut frequency)
2. Increase `min_duration` to filter out short bursts

## Next Steps

- Read the [full documentation](ENHANCED_FEATURES.md) for advanced usage
- Run the [demo script](examples/enhanced_detection_and_hht_demo.py) to see all features
- Check the [test suite](tests/test_enhanced_features.py) for more examples

## Need Help?

Open an issue on GitHub: https://github.com/PRIMOCOSMOS/sEMG-pre-processing/issues

## References

For the scientific background behind these features, see:
- Li et al. (2007) - TKEO for EMG onset detection
- Solnik et al. (2010) - TKEO signal conditioning
- Huang et al. (1998) - Hilbert-Huang Transform
