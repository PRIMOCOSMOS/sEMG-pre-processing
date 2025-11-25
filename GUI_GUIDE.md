# GUI Application Guide

## 🎨 Graphical User Interface

This toolkit includes a user-friendly web-based GUI for easy-to-use signal preprocessing without coding.

### Quick Start

1. **Install dependencies** (if not already done):
```bash
pip install -r requirements.txt
```

2. **Launch the GUI**:
```bash
python gui_app.py
```

3. **Open in browser**:
The application will start on `http://localhost:7860`

### GUI Features

#### 📁 **Tab 1: Load Data**
- Upload CSV files containing sEMG signals
- Configure sampling frequency
- Specify signal column index
- Preview loaded signal

#### 🔧 **Tab 2: Apply Filters**
- **Bandpass Filter**: Remove motion artifacts and high-frequency noise
  - Adjustable high-pass cutoff (5-50 Hz)
  - Adjustable low-pass cutoff (200-500 Hz)
  - Filter order selection (2-6)
  
- **Notch Filter**: Remove power line interference
  - 50 Hz (Europe/Asia) or 60 Hz (Americas)
  - Harmonic filtering (e.g., 50, 100, 150 Hz)
  
- **Before/After Comparison**: Visual comparison of filtering results

#### 🎯 **Tab 3: Detect Activity**
- **Detection Methods**:
  - `multi_feature`: Most robust (recommended) - uses RMS, envelope, variance, energy
  - `combined`: Balance of speed and accuracy
  - `amplitude`: Fast threshold-based detection
  - `ruptures`: Change-point detection only

- **Advanced Options**:
  - Minimum segment duration
  - Clustering for activity/rest classification
  - Adaptive penalty parameter selection
  
- **Results Visualization**: See detected segments highlighted on the signal

#### 💾 **Tab 4: Export Results**
- Export full processed signal as CSV
- Export individual segments as separate CSV files
- Each segment includes metadata (start/end time, duration, peak, RMS)

#### ℹ️ **Tab 5: Help**
- Quick start guide
- Parameter explanations
- Usage tips

### Screenshots

The GUI provides an intuitive interface with:
- Real-time signal visualization
- Interactive parameter controls
- Progress feedback
- Export options

### Command Line Options

You can customize the GUI launch:

```python
# In gui_app.py, modify these parameters:
app.launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,        # Port number
    share=False,             # Set to True for public sharing link
    show_error=True          # Show detailed errors
)
```

### Remote Access

To access the GUI from another machine:
```bash
python gui_app.py
# Then navigate to http://YOUR_IP:7860
```

### Troubleshooting

**Port already in use?**
```python
# Change the port in gui_app.py:
app.launch(server_port=7861)  # Use different port
```

**Can't access from another machine?**
```python
# Make sure server_name is set to "0.0.0.0"
app.launch(server_name="0.0.0.0")
```

**Dependencies missing?**
```bash
pip install --upgrade -r requirements.txt
```

### Tips

1. **Always filter before detection** for best results
2. **Use multi_feature method** for most robust detection
3. **Adjust minimum duration** to filter out noise segments
4. **Check visualizations** to verify detection quality
5. **Export segments** for detailed analysis in other tools

### Example Workflow

1. **Load** your CSV file (column 1 = signal)
2. **Filter** with bandpass (20-450 Hz) and notch (50 Hz)
3. **Detect** with multi_feature method
4. **Review** detected segments in visualization
5. **Export** segments as individual CSV files
6. **Analyze** exported segments in your preferred tool

### Integration with Python Code

You can also use the toolkit programmatically alongside the GUI:

```python
from semg_preprocessing import *

# Load your data
signal, _ = load_csv_data('your_file.csv')

# Apply same filtering as GUI
filtered = apply_bandpass_filter(signal, fs=1000, lowcut=20, highcut=450)
filtered = apply_notch_filter(filtered, fs=1000, freq=50, harmonics=[1,2,3])

# Detect with multi-feature
segments = detect_muscle_activity(filtered, fs=1000, method='multi_feature')

# Export segments
export_segments_to_csv(filtered, segments, fs=1000, output_dir='./segments')
```
