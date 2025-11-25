# sEMG Preprocessing Toolkit - Project Structure

## Directory Structure

```
sEMG-pre-processing/
├── semg_preprocessing/          # Main package
│   ├── __init__.py             # Package initialization
│   ├── filters.py              # Filtering functions
│   ├── detection.py            # Muscle activity detection
│   └── utils.py                # Utility functions
├── examples/                    # Example scripts
│   ├── complete_pipeline.py    # Full preprocessing workflow
│   ├── compare_filters.py      # Filter comparison
│   └── detect_activity.py      # Detection comparison
├── tests/                       # Test suite
│   └── test_basic.py           # Basic functionality tests
├── data/                        # Sample data and results
│   ├── sample_emg.csv          # Synthetic EMG data
│   ├── processed_emg.csv       # Processed output
│   └── *.png                   # Visualization outputs
├── README.md                    # English documentation
├── 使用指南.md                  # Chinese usage guide
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── LICENSE                      # MIT License
└── .gitignore                  # Git ignore rules
```

## Module Overview

### semg_preprocessing/filters.py
Signal filtering functions:
- `apply_highpass_filter()` - Remove low-frequency artifacts
- `apply_lowpass_filter()` - Remove high-frequency noise
- `apply_bandpass_filter()` - Combined high+low pass
- `apply_notch_filter()` - Remove power line interference
- `remove_powerline_dft()` - DFT-based interference removal

### semg_preprocessing/detection.py
Muscle activity detection:
- `detect_muscle_activity()` - Main detection function
- `segment_signal()` - Segment based on detected activity
- `_detect_ruptures()` - Change point detection
- `_detect_amplitude()` - Amplitude threshold detection
- `_detect_combined()` - Combined method (recommended)

### semg_preprocessing/utils.py
Utility functions:
- `load_csv_data()` - Load sEMG from CSV
- `save_processed_data()` - Save processed signals
- `calculate_sampling_frequency()` - Estimate sampling rate
- `resample_signal()` - Change sampling frequency
- `normalize_signal()` - Signal normalization
- `split_into_windows()` - Windowed analysis

## Quick Reference

### Import Everything
```python
from semg_preprocessing import *
```

### Basic Workflow
```python
# 1. Load
signal, _ = load_csv_data('data.csv', value_column=1)

# 2. Filter
filtered = apply_bandpass_filter(signal, fs=1000, lowcut=20, highcut=450)
filtered = apply_notch_filter(filtered, fs=1000, freq=50, harmonics=[1,2,3])

# 3. Detect
segments = detect_muscle_activity(filtered, fs=1000, method='combined')

# 4. Segment
results = segment_signal(filtered, segments, fs=1000)

# 5. Save
save_processed_data('output.csv', filtered, fs=1000)
```

### Running Examples
```bash
cd examples
python complete_pipeline.py      # Full pipeline demo
python compare_filters.py         # Filter comparison
python detect_activity.py         # Detection comparison
```

### Running Tests
```bash
python tests/test_basic.py        # Run basic tests
```

## Key Features

### 1. Preprocessing Pipeline
✅ High-pass filtering (10-20Hz) - Remove motion artifacts, baseline drift, ECG
✅ Low-pass filtering (450-500Hz) - Remove high-frequency noise  
✅ Notch filtering (50/60Hz) - Remove power line interference
✅ Support for Butterworth & Chebyshev filters
✅ Zero-phase filtering using filtfilt

### 2. Detection Methods
✅ Ruptures-based change point detection
✅ Amplitude threshold with RMS envelope
✅ Combined method for robust detection
✅ Automatic segmentation with metadata

### 3. Data Handling
✅ CSV input/output (2nd column for signal)
✅ Automatic sampling frequency estimation
✅ Signal resampling and normalization
✅ Windowed analysis support

## Parameter Guidelines

### Filtering
- **High-pass cutoff**: 10-20Hz (use 20-30Hz for strong ECG rejection)
- **Low-pass cutoff**: 450-500Hz
- **Filter order**: 2-4 (higher orders may cause distortion)
- **Filter type**: 'butterworth' (smoother) or 'chebyshev' (sharper)

### Detection
- **Method**: 'combined' (recommended), 'ruptures', or 'amplitude'
- **Amplitude threshold**: Auto (2×RMS) or manual value
- **Min duration**: 0.1s (100ms) typical minimum
- **Ruptures penalty**: 3 (default), lower = more segments

### Powerline Removal
- **Frequency**: 50Hz (Europe/Asia), 60Hz (Americas)
- **Harmonics**: [1, 2, 3] typical (50, 100, 150 Hz)
- **Quality factor**: 30 (default), higher = narrower notch

## Common Issues & Solutions

### No segments detected
- Lower `amplitude_threshold`
- Reduce `pen` parameter (ruptures)
- Check if signal is properly filtered

### Too many false positives
- Increase `amplitude_threshold`  
- Increase `pen` parameter
- Increase `min_duration`
- Use 'combined' method

### Filter cutoff too high
- Ensure cutoff < Nyquist frequency (fs/2)
- Reduce sampling rate or increase cutoff

## Citation

If you use this toolkit in research, please cite:

```bibtex
@software{semg_preprocessing,
  title = {sEMG Signal Preprocessing Toolkit},
  author = {PRIMOCOSMOS},
  year = {2024},
  url = {https://github.com/PRIMOCOSMOS/sEMG-pre-processing}
}
```

## Support

- **Issues**: https://github.com/PRIMOCOSMOS/sEMG-pre-processing/issues
- **Documentation**: See README.md and 使用指南.md
- **Examples**: See examples/ directory

## License

MIT License - See LICENSE file for details
