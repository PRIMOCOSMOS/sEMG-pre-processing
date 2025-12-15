# Enhanced sEMG Detection and Analysis Features

## Overview

This document describes the three major enhancements implemented in the sEMG preprocessing pipeline to improve muscle activity detection and analysis, specifically optimized for dumbbell exercise recognition.

## New Features

### 1. TKEO (Teager-Kaiser Energy Operator) Preprocessing

**Purpose**: Enhance changepoint detection performance by emphasizing rapid amplitude and frequency changes in the sEMG signal.

**Background**: 
The Teager-Kaiser Energy Operator (TKEO) is a nonlinear operator that is highly sensitive to instantaneous changes in both amplitude and frequency. Research has shown that applying TKEO as a preprocessing step before changepoint detection significantly improves the detection of muscle activity onsets and offsets.

**Formula**:
```
TKEO(x[n]) = x[n]² - x[n-1] × x[n+1]
```

**Key Features**:
- Emphasizes transitions and rapid changes in the signal
- Enhances detection of muscle activity boundaries
- Only used for changepoint detection; original signal is retained for display and analysis
- Can be enabled/disabled via `use_tkeo` parameter

**Usage**:
```python
from semg_preprocessing import detect_muscle_activity, apply_tkeo

# Apply TKEO directly to a signal
tkeo_signal = apply_tkeo(emg_signal)

# Use TKEO in detection (enabled by default)
segments = detect_muscle_activity(
    filtered_signal, 
    fs=1000,
    use_tkeo=True  # Default: True
)

# Disable TKEO if needed
segments = detect_muscle_activity(
    filtered_signal, 
    fs=1000,
    use_tkeo=False
)
```

**References**:
- Li et al. (2007) "Teager–Kaiser energy operation of surface EMG improves muscle activity onset detection" Ann Biomed Eng 35(9):1532–1538
- Solnik et al. (2010) "Teager-Kaiser energy operator signal conditioning improves EMG onset detection" Eur J Appl Physiol 110(3):489-498

---

### 2. Energy-Aware Segment Merging

**Purpose**: Intelligently merge or separate detected segments based on boundary energy states to correctly identify complete dumbbell exercise actions.

**Problem Statement**:
During dumbbell exercises, the transition points (e.g., lifting → lowering arm) can create strong changepoints that split a single action into multiple segments. Traditional time-based merging approaches don't account for whether a boundary represents a true rest period or just a transition within an action.

**Solution**:
The new energy-aware merging logic evaluates the energy state at segment boundaries:

1. **For Non-Adjacent Segments** (separated by inactive periods):
   - Always keep separate, even if both are activity events
   - Represents distinct actions with rest in between

2. **For Adjacent Segments** (directly touching or gap < 50ms):
   - Evaluate boundary energy state
   - **HIGH energy boundary** (≥70% of segment average) → **MERGE**
     - Indicates transition within same action (e.g., lift-to-lower)
   - **LOW energy boundary** (<70% of segment average) → **KEEP SEPARATE**
     - Indicates true rest or end of action

**Implementation Details**:
```python
# Energy evaluation at boundary
boundary_window = 50ms  # Evaluation window around changepoint
boundary_energy = mean(RMS[boundary_region])
segment_avg_energy = (energy_before + energy_after) / 2
energy_ratio = boundary_energy / segment_avg_energy

# Decision rule
if energy_ratio >= 0.7:  # HIGH energy state
    merge_segments()  # Part of same action
else:  # LOW energy state
    keep_separate()  # Different actions
```

**Benefits**:
- Correctly identifies complete dumbbell curl actions
- Handles lift-to-lower transitions without splitting the action
- Adaptive to signal characteristics
- Prevents inappropriate merging of distinct actions

**Usage**:
```python
from semg_preprocessing import detect_muscle_activity

# Energy-aware merging is automatically applied
segments = detect_muscle_activity(
    filtered_signal,
    fs=1000,
    min_duration=0.5  # Minimum action duration
)

# Each segment represents a complete dumbbell action
for i, (start, end) in enumerate(segments):
    duration = (end - start) / fs
    print(f"Action {i+1}: {start/fs:.2f}s - {end/fs:.2f}s ({duration:.2f}s)")
```

---

### 3. Batch HHT Hilbert Spectrum Export

**Purpose**: Export Hilbert-Huang Transform (HHT) analysis results for all detected activity segments in batch mode.

**Features**:
- **One file per segment**: Each activity segment gets its own NPZ matrix file and PNG visualization
- **Automatic numbering**: Files are numbered sequentially (e.g., `segment_001.npz`, `segment_002.npz`)
- **Complete data**: NPZ files contain spectrum matrix, time axis, frequency axis, and sampling rate
- **Publication-ready visualizations**: PNG images show time-frequency representation with proper labels and colorbars

**File Format**:

**NPZ Files** (compressed NumPy arrays):
```python
# Contents of each NPZ file
{
    'spectrum': np.ndarray,      # Hilbert spectrum matrix (freq_bins × time_samples)
    'time': np.ndarray,          # Normalized time axis [0, 1]
    'frequency': np.ndarray,     # Frequency axis [0, max_freq] Hz
    'sampling_rate': float,      # Original sampling rate
    'segment_index': int         # Zero-based segment index
}

# Loading NPZ files
data = np.load('segment_001.npz')
spectrum = data['spectrum']
time = data['time']
frequency = data['frequency']
```

**PNG Files**:
- High-resolution (150 DPI default, configurable)
- Jet colormap for clear visualization
- Proper axis labels and titles
- Colorbar showing amplitude scale

**Usage Example 1: Direct segment export**
```python
from semg_preprocessing.hht import export_hilbert_spectra_batch

# Assuming you have a list of activity segment arrays
segments = [segment1_array, segment2_array, segment3_array]

# Export all at once
export_info = export_hilbert_spectra_batch(
    segments,
    fs=1000,
    output_dir='./hht_results',
    base_filename='bicep_curl',
    n_freq_bins=256,
    normalize_length=256,
    save_visualization=True,
    dpi=150
)

# Output:
# ./hht_results/bicep_curl_001.npz
# ./hht_results/bicep_curl_001.png
# ./hht_results/bicep_curl_002.npz
# ./hht_results/bicep_curl_002.png
# ...
```

**Usage Example 2: Combined detection + export**
```python
from semg_preprocessing import detect_muscle_activity
from semg_preprocessing.hht import export_activity_segments_hht

# Detect activity segments
segments = detect_muscle_activity(filtered_signal, fs=1000, min_duration=0.5)

# Export HHT for all segments in one step
export_info = export_activity_segments_hht(
    filtered_signal,  # Full signal
    segments,         # List of (start, end) tuples
    fs=1000,
    output_dir='./hht_output',
    base_filename='activity_segment'
)

# Check export results
for info in export_info:
    print(f"Segment {info['segment_number']:03d}:")
    print(f"  NPZ: {info['npz_path']}")
    print(f"  PNG: {info['png_path']}")
```

**Parameters**:
- `segments`: List of sEMG signal arrays (1D numpy arrays)
- `fs`: Sampling frequency in Hz
- `output_dir`: Directory to save files
- `base_filename`: Base name for output files (default: "segment")
- `n_freq_bins`: Number of frequency bins (default: 256)
- `normalize_length`: Target time axis length (default: 256)
- `max_freq`: Maximum frequency in Hz (default: fs/2)
- `use_ceemdan`: Use CEEMDAN decomposition (default: True)
- `save_visualization`: Save PNG images (default: True)
- `dpi`: Image resolution (default: 150)

**Return Value**:
```python
[
    {
        'segment_index': 0,
        'segment_number': 1,
        'npz_path': '/path/to/segment_001.npz',
        'png_path': '/path/to/segment_001.png'
    },
    ...
]
```

---

## Complete Workflow Example

Here's a complete workflow demonstrating all three features:

```python
import numpy as np
from semg_preprocessing import (
    apply_bandpass_filter,
    apply_notch_filter,
    detect_muscle_activity,
    segment_signal,
)
from semg_preprocessing.hht import export_activity_segments_hht

# 1. Load and preprocess sEMG signal
from semg_preprocessing import load_csv_data
signal, df = load_csv_data('emg_data.csv', value_column=1)

# 2. Apply standard filtering
fs = 1000.0  # Sampling frequency
filtered = apply_bandpass_filter(signal, fs, lowcut=20, highcut=450)
filtered = apply_notch_filter(filtered, fs, freq=50, harmonics=[1, 2, 3])

# 3. Detect muscle activity with TKEO and energy-aware merging
segments = detect_muscle_activity(
    filtered,
    fs=fs,
    min_duration=0.5,      # Minimum action duration: 500ms
    sensitivity=1.5,       # Detection sensitivity
    use_tkeo=True,         # Enable TKEO preprocessing (default)
    use_multi_detector=True,
    n_detectors=3,
    fusion_method='confidence'
)

print(f"Detected {len(segments)} complete dumbbell actions")

# 4. Get segment details
segmented = segment_signal(filtered, segments, fs, include_metadata=True)
for i, seg in enumerate(segmented):
    print(f"Action {i+1}:")
    print(f"  Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
    print(f"  Duration: {seg['duration']:.2f}s")
    print(f"  Peak amplitude: {seg['peak_amplitude']:.3f}")

# 5. Export HHT Hilbert spectra for all actions
export_info = export_activity_segments_hht(
    filtered,
    segments,
    fs=fs,
    output_dir='./dumbbell_hht_analysis',
    base_filename='bicep_curl',
    n_freq_bins=256,
    normalize_length=256,
    use_ceemdan=True,
    save_visualization=True
)

print(f"\nExported {len(export_info)} Hilbert spectra to: ./dumbbell_hht_analysis/")
```

---

## Demonstration Script

A comprehensive demonstration script is provided: `examples/enhanced_detection_and_hht_demo.py`

Run it to see all three features in action:
```bash
cd examples
python enhanced_detection_and_hht_demo.py
```

**Outputs**:
1. `demo_tkeo_effect.png` - Visualization of TKEO preprocessing
2. `demo_energy_aware_merging.png` - Segment detection with energy-aware merging
3. `./hht_output_demo/` - Directory with Hilbert spectra for all segments

---

## Technical Details

### TKEO Implementation
- Discrete-time TKEO formula with boundary handling
- Smoothing applied to reduce noise (5ms window)
- Used for feature extraction in PELT algorithm
- Original signal preserved for all downstream analysis

### Energy-Aware Merging Algorithm
- Boundary window: 50ms around changepoint
- Context window: 100ms before/after for energy comparison
- Adaptive threshold: 70% of average segment energy
- Handles edge cases and signal boundaries robustly

### HHT Batch Export
- Uses CEEMDAN decomposition for robust IMF extraction
- Hilbert transform for instantaneous frequency and amplitude
- Log-scale amplitude representation for better visualization
- Compressed NPZ format for efficient storage
- Publication-quality PNG visualizations

---

## Performance Considerations

1. **TKEO Preprocessing**:
   - Computational cost: O(n) - very fast
   - Minimal overhead, can be left enabled by default

2. **Energy-Aware Merging**:
   - Computational cost: O(k) where k = number of segments
   - Negligible impact on overall detection time

3. **HHT Batch Export**:
   - Most computationally intensive operation
   - Time per segment: ~1-5 seconds (depends on segment length and parameters)
   - Parallelization possible for large batches
   - Consider reducing `n_freq_bins` or `normalize_length` for faster processing

---

## References and Further Reading

1. **TKEO**:
   - Kaiser, J. F. (1990). "On a simple algorithm to calculate the 'energy' of a signal"
   - Li et al. (2007). "Teager–Kaiser energy operation of surface EMG improves muscle activity onset detection"
   - Solnik et al. (2010). "Teager-Kaiser energy operator signal conditioning improves EMG onset detection"

2. **PELT Algorithm**:
   - Killick et al. (2012). "Optimal Detection of Changepoints With a Linear Computational Cost"
   - ruptures library: https://github.com/deepcharles/ruptures

3. **Hilbert-Huang Transform**:
   - Huang et al. (1998). "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis"
   - Torres et al. (2011). "A complete ensemble empirical mode decomposition with adaptive noise"

---

## Troubleshooting

**Q: Detection splits my dumbbell actions into multiple segments**
A: Try adjusting the `sensitivity` parameter. Lower values (1.0-1.5) are more sensitive and may merge transitions better. Also ensure `use_tkeo=True`.

**Q: HHT export is slow**
A: Reduce `n_freq_bins` and `normalize_length` to 128 or less. Disable CEEMDAN by setting `use_ceemdan=False` for faster processing (at the cost of some accuracy).

**Q: Exported spectrograms look mostly black**
A: This is normal - the function uses log-scale representation. Very low amplitudes appear black. Adjust `min_amplitude_percentile` if needed.

**Q: TKEO creates noisy output**
A: The implementation includes automatic smoothing. If needed, adjust the TKEO smoothing window in the source code or apply additional smoothing to your input signal.

---

## Citation

If you use these features in your research, please cite:

```bibtex
@software{semg_preprocessing_enhanced,
  title = {Enhanced sEMG Detection and Analysis Features},
  author = {PRIMOCOSMOS},
  year = {2024},
  url = {https://github.com/PRIMOCOSMOS/sEMG-pre-processing},
  note = {TKEO preprocessing, energy-aware merging, and batch HHT export}
}
```

---

## Contact and Support

For questions, issues, or feature requests, please open an issue on GitHub:
https://github.com/PRIMOCOSMOS/sEMG-pre-processing/issues
