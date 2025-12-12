# sEMG Signal Preprocessing Toolkit

一个用于表面肌电图（sEMG）信号预处理的Python工具包，包括滤波、去噪、肌肉活动检测、特征提取和数据增强功能。

A comprehensive Python toolkit for surface electromyography (sEMG) signal preprocessing, including filtering, noise removal, muscle activity detection, feature extraction, and data augmentation.

## 📚 Documentation / 文档

- **[Feature Algorithms](FEATURE_ALGORITHMS.md)** - Detailed mathematical formulas and physical meanings for all feature extraction algorithms
- **[GUI Guide](GUI_GUIDE.md)** - Graphical user interface usage guide
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[Project Structure](PROJECT_STRUCTURE.md)** - Code organization and architecture

## 🎯 Key Features / 核心功能

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

**Intelligent Event-Based Detection Algorithm**

The toolkit implements an advanced muscle activity detection system designed to identify meaningful physiological events (e.g., individual muscle contractions like bicep curls) with strict enforcement of duration constraints.

#### Detection Methods

1. **Ruptures**: Change point detection for structural signal changes
2. **Amplitude**: Threshold-based detection for sustained activity
3. **Rhythmic Patterns**: Local RMS variance for periodic movements
4. **Amplitude Trends**: Gradual activation pattern detection
5. **Combined** (⭐ Recommended): Intelligent holistic optimization with confidence scoring

#### Combined Method: Intelligent Event Detection

The combined method uses a four-stage approach to find optimal segmentation:

**Stage 1: Multi-Strategy Candidate Generation**
- Generates 6 different segmentation schemes:
  - Ruptures-based (structural changes)
  - Amplitude-based (sustained activity)
  - Rhythmic patterns (periodic movements)
  - Amplitude trends (gradual activation)
  - Hybrid 1: Ruptures refined by amplitude
  - Hybrid 2: Amplitude refined by ruptures

**Stage 2: Event Quality Scoring**

Each segmentation scheme is scored based on event quality metrics:

1. **RMS Consistency** (30% weight):
   - Measures coefficient of variation within each event
   - Lower CV = more coherent single event
   - Score: 0-10 points per event

2. **Duration Reasonableness** (25% weight):
   - Ideal range: 0.3 - 5.0 seconds for typical muscle contractions
   - Penalizes extremes (too short or too long)
   - Score: 0-10 points per event

3. **Boundary Quality** (25% weight):
   - Evaluates amplitude drops before/after events
   - Clear boundaries = better event separation
   - Score: 0-10 points per event

4. **Transition Sharpness** (20% weight):
   - Measures amplitude gradient at event boundaries
   - Sharp transitions = distinct events
   - Score: 0-10 points per event

**Scoring Formula:**
```
event_score = 0.30 × consistency + 0.25 × duration + 0.25 × boundary + 0.20 × transition
scheme_score = mean(event_scores) - |num_events - expected_events| × 0.5
```

**⚠️ CRITICAL**: Any scheme containing segments below `min_duration` receives a score of -∞ and is completely rejected.

**Stage 3: Confidence-Based Filtering**

Each potential event is assigned a confidence score (0-1) based on:

1. **Amplitude Elevation** (35% weight):
   - How much RMS exceeds surrounding baseline
   - Higher elevation = more confident it's real activity

2. **Signal Consistency** (30% weight):
   - Coefficient of variation within the event
   - Low CV = coherent single contraction

3. **Boundary Sharpness** (20% weight):
   - Rapid amplitude changes at start/end
   - Sharp transitions = clear event boundaries

4. **Duration Reasonableness** (15% weight):
   - Proximity to typical contraction durations (0.3-5s)
   - Extreme durations reduce confidence

**Confidence Formula:**
```
confidence = 0.35 × amplitude_elevation + 0.30 × consistency + 
             0.20 × boundary_sharpness + 0.15 × duration_reasonableness
```

**Confidence Threshold:**
- Adapts based on sensitivity: `threshold = 0.3 + (sensitivity - 1.0) × 0.1`
- Lower sensitivity → lower threshold → accepts more events
- Higher sensitivity → higher threshold → only high-confidence events

**Stage 4: Intelligent Refinement**

The best-scoring scheme with confidence filtering is post-processed:
- **Boundary Refinement**: Align event boundaries to local RMS minima
- **Similar Event Merging**: Merge adjacent events that are likely part of the same activity
  - Criteria: Small gap (<200ms), similar amplitudes, significant gap RMS
- **Final Hard Filter**: Absolutely ensure NO segment violates `min_duration`

#### Duration Constraints: Hard vs Soft

**🔒 min_duration (HARD CONSTRAINT)**:
- **Strictly enforced at ALL stages** - no segment can be shorter than this value
- Defines the valid solution space
- Candidate generation filters violations
- Scoring completely rejects schemes with violations (-∞ score)
- Post-processing never creates segments below this threshold
- Typical range: 0.01 - 2.0 seconds

**📏 max_duration (Soft Optimization Guide)**:
- Optional upper bound for event duration
- Long events exceeding this trigger intelligent splitting
- Uses multiple criteria: ruptures, RMS minima, amplitude drops
- Typical range: 3.0 - 10.0 seconds

#### Algorithm Philosophy

**Duration Constraints = Solution Space Boundaries**
- min_duration and max_duration define the valid solution space
- Within this space, the algorithm finds the optimal segmentation
- Not all candidate boundaries are activated
- Boundaries only created when:
  - Event confidence exceeds threshold
  - Duration constraints are satisfied
  - Overall segmentation quality improves

**Intelligent Boundary Decisions**
- Algorithm evaluates confidence difference between adjacent regions
- Boundaries activated only when confidence gap is significant
- Prevents over-segmentation while respecting constraints
- Ensures detected events are physiologically meaningful

#### Parameter Tuning

**sensitivity** parameter (default: 1.5):
- Lower values (0.5 - 1.5): More sensitive, detects subtle activities
  - Lower confidence threshold
  - More candidate boundaries considered
- Medium values (1.5 - 2.5): Balanced, recommended for most cases
- Higher values (2.5 - 4.0): Stricter, only strong activations
  - Higher confidence threshold
  - Fewer boundaries activated

**Example Usage:**
```python
from semg_preprocessing import detect_muscle_activity

# Intelligent combined detection (recommended)
segments = detect_muscle_activity(
    filtered_signal, 
    fs=1000,
    method='combined',
    min_duration=0.5,      # HARD: NO segment < 500ms
    max_duration=5.0,      # Soft: split events > 5s
    sensitivity=1.5        # Balanced sensitivity
)

# Each segment is a tuple: (start_index, end_index)
print(f"Detected {len(segments)} muscle activity events")

# Verify: ALL segments meet min_duration
durations = [(e-s)/1000 for s, e in segments]
assert all(d >= 0.5 for d in durations), "Duration constraint violated!"
```

**Key Advantages:**
- ✅ Finds meaningful physiological events, not arbitrary segments
- ✅ Strict enforcement of minimum duration (hard constraint)
- ✅ Confidence-based filtering removes low-quality detections
- ✅ Holistic optimization considers overall segmentation quality
- ✅ Intelligent boundary activation prevents over-segmentation
- ✅ Works well across different signal characteristics and noise levels
- ✅ Adaptive thresholds for diverse sEMG amplitude ranges

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

- CEEMDAN decomposition for robust IMF extraction
- Production-ready HHT with:
  - Fixed IMF count (8) with zero-padding
  - Unified time-frequency axes
  - Energy conservation validation (<5% error)
  - Signal normalization and amplitude thresholding
  - Noise reduction and muscle activity representation

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
