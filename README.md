# sEMG Signal Preprocessing Toolkit

一个用于表面肌电图（sEMG）信号预处理的Python工具包，包括滤波、去噪和肌肉活动检测功能。

A comprehensive Python toolkit for surface electromyography (sEMG) signal preprocessing, including filtering, noise removal, and muscle activity detection.

## Features / 功能特性

### 1. EMG数据预处理 / EMG Data Preprocessing

- **高通/低通滤波 / High-pass/Low-pass Filtering**
  - 高通滤波器（10-20Hz）：消除运动伪影、基线漂移和心电干扰
  - 低通滤波器（450-500Hz）：去除高频噪声
  - 支持巴特沃斯(Butterworth)和切比雪夫(Chebyshev)滤波器
  - 滤波器阶数可调（推荐2-4阶）

- **工频干扰去除 / Power Line Interference Removal**
  - 方案一：陷波器(Notch Filter)，可级联处理50Hz及其谐波
  - 方案二：DFT方法，去除目标频域成分后重建信号

- **其他生物信号干扰处理 / Other Biological Signal Interference**
  - 增强的高通滤波器设计，有效抑制ECG干扰（≤30Hz）

### 2. 肌肉活动检测与分段 / Muscle Activity Detection and Segmentation

- 基于ruptures库的变化点检测
- 基于幅值的活动检测
- 结合ruptures和幅值的混合方法（推荐）
- 自动信号分段和元数据提取

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
