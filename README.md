# sEMG Signal Preprocessing Toolkit

一个用于表面肌电图（sEMG）信号预处理的Python工具包，包括滤波、去噪、肌肉活动检测、特征提取和数据增强功能。

A comprehensive Python toolkit for surface electromyography (sEMG) signal preprocessing, including filtering, noise removal, muscle activity detection, feature extraction, and data augmentation.

## 📚 Documentation / 文档

- **[Enhanced Features Guide](ENHANCED_FEATURES.md)** - **NEW!** TKEO preprocessing, energy-aware merging, and batch HHT export
- **[Feature Algorithms](FEATURE_ALGORITHMS.md)** - Detailed mathematical formulas and physical meanings for all feature extraction algorithms
- **[GUI Guide](GUI_GUIDE.md)** - Graphical user interface usage guide
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[Project Structure](PROJECT_STRUCTURE.md)** - Code organization and architecture

## 🎯 Key Features / 核心功能

### ⭐ NEW: Enhanced Detection Features

**Latest improvements for dumbbell exercise recognition and HHT analysis:**

1. **TKEO (Teager-Kaiser Energy Operator) Preprocessing** 🆕
   - Enhances changepoint detection by emphasizing rapid amplitude/frequency changes
   - Significantly improves detection of muscle activity boundaries
   - Used internally by PELT algorithm (original signal preserved for analysis)
   - Can be enabled/disabled via `use_tkeo` parameter (default: True)
   - Research-backed approach: Li et al. (2007), Solnik et al. (2010)

2. **Energy-Aware Segment Merging** 🆕
   - Intelligent merging based on boundary energy states
   - Correctly identifies complete dumbbell actions (lift + lower as one event)
   - HIGH energy boundary → MERGE (transition within action)
   - LOW energy boundary → KEEP SEPARATE (different actions)
   - Non-adjacent segments always kept separate

3. **Batch HHT Hilbert Spectrum Export** 🆕
   - Export Hilbert spectra for ALL activity segments at once
   - One NPZ matrix file + one PNG visualization per segment
   - Automatic sequential numbering (segment_001, segment_002, ...)
   - Publication-ready visualizations with proper labels
   - Simple API: `export_activity_segments_hht(signal, segments, fs, output_dir)`

4. **HHT Algorithm Optimization** 🆕 **(December 2024)**
   - ✅ **No interpolation artifacts**: Replaced scipy.signal.resample with average pooling
   - ✅ **Valid sEMG frequency range**: 20-450Hz mapped to 256 frequency bins (not 0-Nyquist)
   - ✅ **Energy preservation**: HHT computed on original signal duration, then pooled to uniform size
   - ✅ **Better accuracy**: Avoids high-frequency artifacts introduced by interpolation
   - ✅ **Improved visualizations**: Hilbert spectrum images show meaningful sEMG frequency range

**📖 See [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) for detailed documentation and examples.**

### 1. EMG Data Preprocessing / EMG数据预处理

- **High-pass/Low-pass Filtering / 高通/低通滤波**
  - High-pass filter (10-20Hz): Remove motion artifacts, baseline drift, ECG interference
  - Low-pass filter (450-500Hz): Remove high-frequency noise
  - Supports Butterworth and Chebyshev filters
  - Adjustable filter order (recommended: 2-4)

- **Power Line Interference Removal / 工频干扰去除**
  - Method 1: Notch filter with harmonic cascading (50/60Hz and harmonics)
  - Method 2: DFT-based frequency domain removal with signal reconstruction

- **Batch Processing / 批量处理**
  - Process multiple files simultaneously
  - Unified parameters across all signals
  - Batch export capabilities

### 2. Muscle Activity Detection & Segmentation / 肌肉活动检测与分段

**Advanced PELT-Based Detection with Multi-Detector Ensemble**

The toolkit now implements a state-of-the-art muscle activity detection system using an enhanced PELT (Pruned Exact Linear Time) algorithm with multi-dimensional feature analysis and ensemble detection mechanisms.

#### Detection Algorithm: Advanced PELT

The new detection system uses only the **Combined** method, which is now powered by an advanced PELT algorithm with the following innovations:

**1. Energy-Based Adaptive Penalty Zones**
- Signal is divided into energy zones (low, medium, high) using K-means clustering
- Low energy zones: Lower penalty (more sensitive detection)
- High energy zones: Higher penalty (prevents over-segmentation)
- Formula: `penalty = base_penalty × zone_multiplier`
  - Low energy zone: 0.5× base penalty
  - Medium energy zone: 1.0× base penalty  
  - High energy zone: 2.0× base penalty

**2. Multi-Dimensional Feature Vectors**

The algorithm extracts 8 features across three domains:

**Time-Domain Features:**
- RMS (Root Mean Square) - signal energy
- MAV (Mean Absolute Value) - amplitude level
- VAR (Variance) - signal variability
- WL (Waveform Length) - signal complexity

**Frequency-Domain Features:**
- MNF (Mean Frequency) - spectral centroid
- MDF (Median Frequency) - spectral median

**Complexity Features:**
- ZCR (Zero Crossing Rate) - frequency indicator
- Sample Entropy (proxy) - signal regularity

All features are normalized and fed to PELT for robust change point detection.

**3. Multi-Detector Ensemble**

Runs multiple PELT detectors in parallel with different sensitivity levels:
- Number of detectors: 1-5 (default: 3)
- Sensitivity range: automatically distributed around base sensitivity
- Each detector independently identifies events

**Fusion Methods:**
- **Confidence** (recommended): Weighted by confidence scores
  - Each segment scored based on amplitude contrast, consistency, and duration
  - Confidence map created across all detectors
  - Threshold at 50th percentile of positive confidences
- **Voting**: Majority vote across detectors
  - Requires ≥50% of detectors to agree on a region
  - More conservative, reduces false positives
- **Union**: Combines all detections
  - Most sensitive, may include more false positives
  - Overlaps are merged

**4. Intelligent Dense Event Merging**

Automatically merges events with gaps < 50ms:
- Prevents over-segmentation in rhythmic/rapid activity
- Common in repetitive muscle contractions
- Merged segments must still satisfy min_duration constraint

**5. Strict Duration Enforcement**

**🔒 min_duration (HARD CONSTRAINT)**:
- Absolutely enforced at ALL stages
- Applied to: initial detection, merging, final output
- No segment can ever be shorter than this value
- Typical range: 0.01 - 10.0 seconds

**📏 max_duration (Optional Split Trigger)**:
- Long events exceeding this are split intelligently
- Uses PELT change points and RMS minima for natural breaks
- Each split segment must satisfy min_duration
- Typical range: 3.0 - 30.0 seconds

#### Algorithm Flow

1. **Feature Extraction**: Extract 8-dimensional feature vectors from preprocessed signal
2. **Energy Zone Computation**: Cluster signal into energy zones for adaptive penalties
3. **Multi-Detector Ensemble** (if enabled):
   - Run N detectors with sensitivity range [0.7×base, 1.3×base]
   - Each detector uses zone-specific adaptive penalties
   - Calculate confidence for each detected segment
4. **Fusion**: Combine detections using selected method (voting/confidence/union)
5. **Dense Event Merging**: Merge events with gaps < 50ms
6. **Duration Enforcement**: Final filter ensures all constraints satisfied

#### Parameter Tuning

**sensitivity** parameter (default: 1.5):
- Lower values (0.1 - 1.5): More sensitive, detects more segments
  - Lower confidence threshold
  - Lower amplitude threshold
  - More events detected (may include weaker activations)
- Medium values (1.5 - 2.5): Balanced, recommended for most cases
  - Good trade-off between sensitivity and specificity
- Higher values (2.5 - 4.0): Stricter, only strong activations
  - Higher confidence threshold
  - Higher amplitude threshold
  - Fewer events detected (only clear, strong activities)

**min_duration** parameter:
- Shorter (0.01 - 0.5s): Captures rapid contractions
- Medium (0.5 - 2.0s): Typical muscle contractions
- Longer (2.0 - 10.0s): Sustained activities only

**Example Usage:**
```python
from semg_preprocessing import detect_muscle_activity

# Two-stage amplitude-first combined detection (recommended)
segments = detect_muscle_activity(
    filtered_signal, 
    fs=1000,
    method='combined',           # Only supported method (PELT-based)
    min_duration=0.5,            # HARD: NO segment < 500ms
    max_duration=5.0,            # Soft: split events > 5s
    sensitivity=1.5,             # Controls PELT penalty (lower = more sensitive)
    n_detectors=3,               # Multi-detector ensemble
    fusion_method='confidence',  # How to combine detectors
    use_multi_detector=True      # Enable ensemble
)

# Each segment is a tuple: (start_index, end_index)
print(f"Detected {len(segments)} muscle activity events")

# Verify: ALL segments meet min_duration
durations = [(e-s)/1000 for s, e in segments]
assert all(d >= 0.5 for d in durations), "Duration constraint violated!"
```

**Key Advantages:**
- ✅ Multi-dimensional feature analysis (time, frequency, complexity)
- ✅ Energy-based adaptive penalty (context-aware detection)
- ✅ Multi-detector ensemble for robust detection
- ✅ Automatic dense event merging (gaps < 50ms)
- ✅ Strict enforcement of minimum duration (hard constraint)
- ✅ Works well across different signal characteristics and noise levels
- ✅ Direct interpretability: sensitivity → PELT penalty

### 3. Feature Extraction / 特征提取

**Time Domain Features:**
- WL (Waveform Length), ZC (Zero Crossings), SSC (Slope Sign Changes)
- RMS (Root Mean Square), MAV (Mean Absolute Value), VAR (Variance)

**Frequency Domain Features (Welch PSD-based):**
- MDF (Median Frequency), MNF (Mean Frequency)
- PKF (Peak Frequency), TTP (Total Power)
- IMNF (Instantaneous Mean Frequency using Choi-Williams Distribution)

**Fatigue Indicators:**
- WIRE51 (Wavelet Index - sym5 DWT-based)
- DI (Dimitrov Index - spectral moment ratio)

**See [FEATURE_ALGORITHMS.md](FEATURE_ALGORITHMS.md) for detailed formulas and interpretations.**

### 4. Hilbert-Huang Transform (HHT) / 希尔伯特-黄变换

**IMPROVED (2024):** HHT algorithm optimized to avoid interpolation artifacts and focus on valid sEMG frequency range.

- **CEEMDAN decomposition** for robust IMF extraction
- **Average pooling-based normalization** (no interpolation, no high-frequency artifacts)
- **Frequency mapping to 20-450Hz** (valid sEMG range, not 0-Nyquist)
- Production-ready HHT features:
  - Fixed IMF count (8) with zero-padding
  - Compute HHT on original signal duration, then pool to uniform size (256×256)
  - Unified time-frequency axes for CNN input
  - Energy conservation validation (<5% error typically)
  - Signal normalization and amplitude thresholding
  - Noise reduction and muscle activity representation
  - Batch export of Hilbert spectra for all activity segments

**Key Improvements:**
1. ✅ **No interpolation artifacts**: Uses average pooling instead of scipy.signal.resample
2. ✅ **Meaningful frequency range**: 20-450Hz maps to the 256 frequency bins (sEMG valid range)
3. ✅ **Energy preserved**: HHT computed on original signal, then downsampled
4. ✅ **Better visualization**: Hilbert spectrum PNG visualizations show 20-450Hz range

### 5. Data Augmentation / 数据增强

- CEEMDAN-based IMF recombination
- Batch file augmentation
- Generate artificial sEMG signals from multiple source signals
- Maintains physiological characteristics

### 6. File Format Support / 文件格式支持

- **CSV**: Standard comma-separated values (with header row options)
- **MAT**: MATLAB .mat files (n×1 or 1×n double arrays)

## System Architecture / 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    GUI Application (Tkinter)                │
│  - File Loading  - Filtering  - Detection  - Export         │
│  - Feature Analysis  - HHT Analysis  - Augmentation        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                Core Processing Modules                      │
├─────────────────────────────────────────────────────────────┤
│  1. utils.py                                                │
│     - File I/O (CSV, MAT)                                   │
│     - Batch loading with skip_rows support                  │
│                                                              │
│  2. filters.py                                              │
│     - Bandpass/Highpass/Lowpass filters                     │
│     - Notch filters (power line interference)               │
│     - DFT-based frequency removal                           │
│                                                              │
│  3. detection.py                                            │
│     - Ruptures-based change point detection                 │
│     - Amplitude threshold detection                         │
│     - Hybrid detection methods                              │
│     - Automatic segmentation                                │
│                                                              │
│  4. hht.py (Feature Extraction & HHT)                       │
│     ┌─────────────────────────────────────────────┐        │
│     │ Feature Extraction                          │        │
│     │ - Time domain: WL, ZC, SSC, RMS, MAV, VAR  │        │
│     │ - Frequency: MDF, MNF, PKF, TTP (Welch PSD)│        │
│     │ - Advanced: IMNF (CWD-based)               │        │
│     │ - Fatigue: WIRE51 (sym5 DWT), DI           │        │
│     └─────────────────────────────────────────────┘        │
│     ┌─────────────────────────────────────────────┐        │
│     │ HHT Analysis                                │        │
│     │ - EMD/CEEMDAN decomposition                │        │
│     │ - Hilbert transform & instantaneous freq    │        │
│     │ - Production HHT with validation           │        │
│     │ - Energy conservation checking              │        │
│     └─────────────────────────────────────────────┘        │
│                                                              │
│  5. augmentation.py                                         │
│     - CEEMDAN-based signal generation                       │
│     - IMF recombination (m=8)                               │
│     - Batch augmentation                                    │
└─────────────────────────────────────────────────────────────┘
```

### Processing Pipeline / 处理流程

```
Input Signal (CSV/MAT)
    ↓
[1] Preprocessing
    ├─ Bandpass Filter (20-450 Hz)
    ├─ Notch Filter (50/60 Hz + harmonics)
    └─ Normalization (optional)
    ↓
[2] Activity Detection
    ├─ Ruptures change point detection
    ├─ Amplitude threshold detection
    └─ Combined hybrid method
    ↓
[3] Segmentation
    └─ Extract activity segments with metadata
    ↓
[4] Feature Extraction (Per Segment)
    ├─ Time Domain Features
    ├─ Frequency Features (Welch PSD)
    ├─ IMNF (Choi-Williams)
    └─ Fatigue Indicators (WIRE51, DI)
    ↓
[5] Advanced Analysis (Optional)
    ├─ HHT Analysis (Time-Frequency)
    └─ Data Augmentation (CEEMDAN IMF)
    ↓
Output (CSV/NPZ/Visualization)
```

## Installation / 安装

```bash
# Clone the repository
git clone https://github.com/PRIMOCOSMOS/sEMG-pre-processing.git
cd sEMG-pre-processing

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Dependencies / 依赖项

- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- ruptures >= 1.1.7
- matplotlib >= 3.4.0

## Quick Start / 快速开始

### Basic Usage / 基本用法

```python
from semg_preprocessing import (
    load_csv_data,
    apply_bandpass_filter,
    apply_notch_filter,
    detect_muscle_activity,
    segment_signal,
)

# 1. Load data (CSV file, 2nd column contains EMG signal)
signal, df = load_csv_data('your_emg_data.csv', value_column=1)

# 2. Apply bandpass filter (20-450 Hz)
filtered = apply_bandpass_filter(signal, fs=1000, lowcut=20, highcut=450)

# 3. Remove power line interference (50 Hz)
filtered = apply_notch_filter(filtered, fs=1000, freq=50, harmonics=[1, 2, 3])

# 4. Detect muscle activity
segments = detect_muscle_activity(filtered, fs=1000, method='combined')

# 5. Segment the signal
segmented = segment_signal(filtered, segments, fs=1000)
```

### Complete Pipeline Example / 完整流程示例

```python
from semg_preprocessing import *

# Configuration
fs = 1000.0  # Sampling frequency in Hz

# Load data
signal, _ = load_csv_data('emg_data.csv', value_column=1)

# Preprocessing pipeline
filtered = apply_bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4)
filtered = apply_notch_filter(filtered, fs, freq=50, harmonics=[1, 2, 3])

# Detect and segment muscle activity
activity_segments = detect_muscle_activity(
    filtered, fs, 
    method='combined',
    min_duration=0.1
)

segments = segment_signal(filtered, activity_segments, fs, include_metadata=True)

# Print segment information
for i, seg in enumerate(segments):
    print(f"Segment {i+1}: {seg['start_time']:.3f}s - {seg['end_time']:.3f}s")
    print(f"  Duration: {seg['duration']:.3f}s")
    print(f"  Peak amplitude: {seg['peak_amplitude']:.3f}")
    print(f"  RMS: {seg['rms']:.3f}")
```

## API Reference / API参考

### Filtering Functions / 滤波函数

#### `apply_highpass_filter(data, fs, cutoff=20.0, order=4, filter_type='butterworth')`
高通滤波，去除运动伪影和基线漂移
- **cutoff**: 截止频率（推荐10-20Hz）
- **order**: 滤波器阶数（推荐2-4）
- **filter_type**: 'butterworth' 或 'chebyshev'

#### `apply_lowpass_filter(data, fs, cutoff=450.0, order=4, filter_type='butterworth')`
低通滤波，去除高频噪声
- **cutoff**: 截止频率（推荐450-500Hz）

#### `apply_bandpass_filter(data, fs, lowcut=20.0, highcut=450.0, order=4)`
带通滤波（高通+低通组合）

#### `apply_notch_filter(data, fs, freq=50.0, quality_factor=30.0, harmonics=None)`
陷波滤波器，去除工频干扰
- **freq**: 工频频率（欧洲/亚洲：50Hz，美洲：60Hz）
- **harmonics**: 谐波列表，如 [1, 2, 3] 表示50Hz、100Hz、150Hz

#### `remove_powerline_dft(data, fs, freq=50.0, harmonics=None, bandwidth=1.0)`
基于DFT的工频干扰去除

### Detection Functions / 检测函数

#### `detect_muscle_activity(data, fs, method='combined', ...)`
检测肌肉活动事件
- **method**: 'ruptures', 'amplitude', 或 'combined'（推荐）
- **amplitude_threshold**: 幅值阈值（默认自动计算）
- **min_duration**: 最小活动持续时间（秒）

#### `segment_signal(data, segments, fs, include_metadata=True)`
基于检测结果分段信号
- 返回包含信号片段和元数据的列表

### Utility Functions / 工具函数

#### `load_csv_data(filepath, value_column=1, has_header=True)`
从CSV文件加载sEMG数据
- **value_column**: 信号值所在列（默认为1，即第2列）

#### `save_processed_data(filepath, data, fs, include_time=True)`
保存处理后的数据到CSV

## Examples / 示例

The `examples/` directory contains several demonstration scripts:

1. **complete_pipeline.py** - 完整的预处理流程示例
2. **compare_filters.py** - 比较不同滤波方法
3. **detect_activity.py** - 肌肉活动检测示例

Run examples:
```bash
cd examples
python complete_pipeline.py
python compare_filters.py
python detect_activity.py
```

## Data Format / 数据格式

Input CSV file format (输入CSV格式):
```csv
Time,EMG_Signal
0.000,0.001
0.001,0.002
0.002,-0.001
...
```

- 第2列（索引1）包含sEMG信号值
- The 2nd column (index 1) contains the sEMG signal values

## Technical Details / 技术细节

### Filter Specifications / 滤波器规格

- **High-pass**: 10-20Hz, removes motion artifacts, baseline drift, ECG interference
- **Low-pass**: 450-500Hz, removes high-frequency noise (EMG signals typically <500Hz)
- **Notch**: 50Hz (or 60Hz) with harmonics, removes power line interference
- **Filter order**: 2-4 (higher orders may cause distortion)

### Detection Algorithm / 检测算法

The combined detection method:
1. Uses ruptures (Pelt algorithm) for change point detection
2. Applies amplitude threshold to identify true muscle activity
3. Merges overlapping segments
4. Filters by minimum duration

## Performance Considerations / 性能考虑

- Use `apply_bandpass_filter()` instead of separate high-pass and low-pass for better efficiency
- For large datasets, consider processing in chunks
- The 'combined' detection method provides best results but is slower than individual methods

## Contributing / 贡献

Contributions are welcome! Please feel free to submit a Pull Request.

## License / 许可证

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation / 引用

If you use this toolkit in your research, please cite:

```bibtex
@software{semg_preprocessing,
  title = {sEMG Signal Preprocessing Toolkit},
  author = {PRIMOCOSMOS},
  year = {2024},
  url = {https://github.com/PRIMOCOSMOS/sEMG-pre-processing}
}
```

## Contact / 联系方式

For questions and support, please open an issue on GitHub.

## Acknowledgments / 致谢

This toolkit uses the following open-source libraries:
- [ruptures](https://github.com/deepcharles/ruptures) for change point detection
- [scipy](https://scipy.org/) for signal processing
- [numpy](https://numpy.org/) for numerical computations
