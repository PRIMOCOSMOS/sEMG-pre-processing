# HHT-based Event Detection and Batch HHT Analysis - Implementation Summary

## Overview

This implementation adds two major features to the sEMG preprocessing toolkit:

1. **HHT-based Event Detection Algorithm** - A new parallel detection method alongside PELT
2. **Batch HHT Analysis in Segment Feature Analysis** - Process uploaded segments with HHT

## 1. HHT-based Event Detection Algorithm

### Implementation Details

**Location**: `semg_preprocessing/detection.py` - `detect_activity_hht()` function

**Key Features**:
- Computes full-signal Hilbert-Huang Transform with dynamic resolution
- Detects high-energy stripes in Hilbert spectrum (characteristic of muscle activity)
- Maps detected time-frequency patterns back to original signal
- Outputs segments compatible with existing pipeline

**Algorithm Steps**:
1. Compute HHT of entire signal with dynamic resolution (scales with signal duration)
2. Identify high-energy regions in spectrum using percentile threshold
3. Compute time-integrated energy profile
4. Detect temporally compact energy patterns (muscle events)
5. Map detected patterns back to time domain
6. Apply duration constraints (min/max duration)
7. Merge nearby segments (configurable gap)

**Parameters**:
- `energy_threshold` (0.5-0.9): Percentile threshold for high-energy detection
- `temporal_compactness` (0.1-0.7): Minimum energy density in time
- `resolution_per_second` (64-256): Time bins per second
- `adaptive_threshold_factor` (configurable): Sensitivity of adaptive threshold
- `merge_gap_ms` (configurable): Gap for merging nearby segments

**Visualization**:
- Three-panel display showing:
  1. Full Hilbert spectrum with detected regions
  2. Time-integrated energy profile with detection overlay
  3. Original signal with detected segments

### GUI Integration

**Location**: `gui_app.py` - "Detect Activity" tab

**Features**:
- Radio button to select between PELT and HHT methods
- HHT-specific parameter controls
- PELT parameters remain for backward compatibility
- Multi-panel visualization for HHT results
- Available in both single file and batch detection modes

## 2. Batch HHT Analysis in Segment Feature Analysis

### Implementation Details

**Location**: `gui_app.py` - `analyze_segment_features_hht()` and `export_segment_features_hht_csv()`

**Key Features**:
- Processes multiple uploaded segment files independently
- Each segment gets its own HHT decomposition (decoupled from detection HHT)
- Computes Hilbert spectrum with uniform resolution
- Extracts full suite of sEMG features from HHT results
- Visualizes first 6 spectra in grid layout

**Extracted Features**:
- Time domain: WL, ZC, SSC, RMS, MAV, VAR
- Frequency domain: MDF, MNF, IMNF, PKF, TTP
- Fatigue indicators: DI, WIRE51

**Export Functionality**:
- CSV export with all features
- Supports directory path input (auto filename: `segment_hht_features.csv`)
- Supports full file path input for custom naming

### GUI Integration

**Location**: `gui_app.py` - "Feature Analysis" tab, "HHT Batch Analysis" sub-tab

**Features**:
- Upload multiple segment files (CSV/MAT)
- Configure HHT parameters (frequency bins, normalized length, CEEMDAN)
- Visualize Hilbert spectra for first 6 segments
- Export features to CSV with flexible path handling

## Technical Highlights

### Code Quality Improvements

1. **Module-level imports**: HHT module imported at top of detection.py for performance
2. **Named constants**: Magic numbers extracted to named constants:
   - `HHT_MIN_TIME_BINS = 128`
   - `HHT_MAX_TIME_BINS = 2048`
   - `HHT_ADAPTIVE_THRESHOLD_FACTOR = 0.5`
   - `HHT_MERGE_GAP_MS = 50`
3. **Configurable parameters**: Previously hardcoded values now configurable
4. **Proper decoupling**: Detection HHT and segment analysis HHT are independent

### Security

- CodeQL scan completed: **0 security alerts**
- No vulnerabilities introduced

### Testing

1. **HHT Detection Test**: Synthetic signal with 3 activity bursts
   - Successfully detected 2/3 bursts with default parameters
   - Tunable parameters allow sensitivity adjustment
   - Visualization confirms correct spectrum analysis

2. **Batch HHT Analysis Test**: 3 synthetic segments with different frequency content
   - Successfully processed all segments
   - Correct feature extraction (MDF values match input frequencies)
   - Output format validated

## Files Modified

1. `semg_preprocessing/detection.py`: Added `detect_activity_hht()` function
2. `semg_preprocessing/__init__.py`: Exported new detection function
3. `gui_app.py`: 
   - Updated `detect_activity()` to support both PELT and HHT
   - Updated `detect_batch_activity()` to support both methods
   - Added `analyze_segment_features_hht()` for batch HHT
   - Added `export_segment_features_hht_csv()` for export
   - Updated UI with new controls and sub-tabs

## Usage Examples

### HHT Detection (Single File)

1. Load and filter signal
2. Go to "Detect Activity" tab
3. Select "HHT" detection method
4. Adjust parameters:
   - Energy threshold: 0.65 (higher = more selective)
   - Temporal compactness: 0.3 (higher = more strict)
   - Resolution: 128 bins/second
5. Click "Detect Activity"
6. View multi-panel visualization and detected segments

### Batch HHT on Segments

1. Go to "Feature Analysis" tab
2. Select "HHT Batch Analysis" sub-tab
3. Upload segment files (CSV or MAT)
4. Configure HHT parameters (freq bins, length, CEEMDAN)
5. Click "Compute HHT for All Segments"
6. View spectra and features
7. Export to CSV with custom path

## Benefits

1. **Alternative detection method**: HHT provides frequency-domain sensitivity
2. **Parallel to PELT**: Users can choose based on signal characteristics
3. **Complete workflow**: From detection to feature extraction
4. **Flexibility**: Multiple tunable parameters for different use cases
5. **Visualization**: Clear understanding of detection process
6. **Batch processing**: Efficient handling of multiple segments
7. **Export functionality**: Easy integration with analysis pipelines

## References

- Example NPZ file analyzed: `Test1_1_001.npz` (256×256 spectrum, 20-450 Hz)
- HHT frequency range: 20-450 Hz (sEMG effective range)
- Resolution mapping: 2-4s events → 256×256 matrix (reference from requirements)
