# Code Review Fixes Summary

## Issues Addressed

This document summarizes the fixes applied in response to the code review feedback on commit 1a8eef0.

## Issue 1: Non-Adaptive Merge Threshold ✅

### Problem
The `_merge_dense_events()` function calculated `signal_rms_mean` and `signal_rms_std` but never used them. The threshold (0.7) was hardcoded, making it not truly adaptive.

### Solution
1. **Added Coefficient of Variation (CV) calculation**:
   ```python
   energy_cv = signal_rms_std / signal_rms_mean
   ```

2. **Implemented adaptive threshold adjustment**:
   - High variability (CV > 0.5): `threshold = base_threshold × 0.9` (more aggressive)
   - Low variability (CV < 0.3): `threshold = base_threshold × 1.0` (conservative)
   - Medium variability: Linear interpolation between 0.9× and 1.0×

3. **Added `merge_threshold` parameter**:
   - Range: 0.4 to 0.9
   - Default: 0.7
   - Exposed in GUI as adjustable slider

### Benefits
- Truly adaptive to signal characteristics
- Helps keep sEMG envelope peaks within detected events
- User-configurable for different exercise types

---

## Issue 2: TKEO Not Exposed in UI ✅

### Problem
TKEO (Teager-Kaiser Energy Operator) preprocessing was implemented but not exposed in the GUI. Users couldn't toggle it on/off.

### Solution
1. **Added UI checkbox in both detection tabs**:
   - Label: "Enable TKEO Preprocessing"
   - Info: "Teager-Kaiser Energy Operator enhances changepoint detection"
   - Default: enabled (checked)

2. **Updated function signatures**:
   - Added `use_tkeo` parameter to `detect_activity()`
   - Added `use_tkeo` parameter to `detect_batch_activity()`
   - Parameter passed through to `detect_muscle_activity()`

3. **Added to both single and batch detection tabs**

### Benefits
- Users can now enable/disable TKEO as needed
- Visible control with helpful tooltip
- Maintains backward compatibility (default: enabled)

---

## Issue 3: HHT Export Not Working Properly ✅

### Problem
1. HHT export only exported a single spectrum (from `self.hht_results`)
2. Not one spectrum per detected segment
3. Batch export functionality didn't work
4. Files not organized by segment index
5. No separate folders for matrices and images

### Solution
1. **Replaced single export with batch export**:
   ```python
   from semg_preprocessing.hht import export_activity_segments_hht
   
   export_info = export_activity_segments_hht(
       self.filtered_signal,
       segment_tuples,
       fs=self.fs,
       output_dir=hht_matrices_dir,
       base_filename=prefix if prefix else 'segment',
       ...
   )
   ```

2. **Created organized folder structure**:
   - `hht_matrices/`: NPZ files with spectrum data
   - `hht_images/`: PNG visualization images

3. **Implemented segment-indexed naming**:
   - Format: `{prefix}_{index:03d}.npz` and `{prefix}_{index:03d}.png`
   - Example: `bicep_curl_001.npz`, `bicep_curl_002.npz`, etc.

4. **Added custom prefix support**:
   - Uses the prefix from "Custom filename prefix" field in export tab
   - Falls back to "segment" if no prefix provided

5. **Process flow**:
   - Export NPZ matrices for all segments
   - Load each saved spectrum and create PNG visualization
   - Save visualizations to separate images folder
   - Report success with file counts

### Benefits
- One matrix + one image per detected activity segment
- Organized folder structure for easy access
- Sequential numbering for easy identification
- Respects user-defined filename prefix
- Works when "Export HHT results" checkbox is selected

---

## Testing Results

All fixes have been tested and validated:

### Test 1: Adaptive Threshold
```python
# Test with different CV signals
segments = detect_muscle_activity(
    filtered, fs, min_duration=0.3, merge_threshold=0.7
)
# ✓ Threshold adjusts based on signal variability
```

### Test 2: TKEO Toggle
```python
# With TKEO (default)
segments_with = detect_muscle_activity(
    filtered, fs, use_tkeo=True
)

# Without TKEO
segments_without = detect_muscle_activity(
    filtered, fs, use_tkeo=False
)
# ✓ Both work correctly
```

### Test 3: Batch HHT Export
```python
export_info = export_activity_segments_hht(
    signal, segments, fs, output_dir, base_filename='test'
)
# ✓ Creates hht_matrices/ with NPZ files
# ✓ Creates hht_images/ with PNG files
# ✓ Files named by segment index (001, 002, ...)
```

---

## UI Changes

### Detection Tab - New Controls

**Advanced Detection Settings:**
- ☑ Enable TKEO Preprocessing
  - Default: enabled
  - Tooltip: "Teager-Kaiser Energy Operator enhances changepoint detection"
  
- Segment Merge Threshold: [slider 0.4 - 0.9]
  - Default: 0.7
  - Tooltip: "Energy ratio for merging adjacent segments (lower = more aggressive merging)"

### Export Tab - HHT Export Behavior

When "Export HHT results" is checked:
- Creates `hht_matrices/` directory with NPZ files
- Creates `hht_images/` directory with PNG files
- Files named: `{prefix}_{001}.npz`, `{prefix}_{001}.png`, etc.
- Uses custom prefix from "Custom filename prefix" field

---

## Backward Compatibility

All changes maintain backward compatibility:
- ✅ Default parameters preserve existing behavior
- ✅ TKEO enabled by default (as before)
- ✅ Merge threshold defaults to 0.7 (as before)
- ✅ HHT export enhanced but doesn't break existing code
- ✅ API additions are optional parameters

---

## Files Modified

1. **semg_preprocessing/detection.py**
   - Added adaptive threshold calculation using CV
   - Added `merge_threshold` parameter
   - Updated `_merge_dense_events()` signature
   - Updated `_detect_pelt_advanced()` signature
   - Maintained backward compatibility

2. **gui_app.py**
   - Added TKEO checkbox (single + batch tabs)
   - Added merge threshold slider (single + batch tabs)
   - Updated `detect_activity()` signature
   - Updated `detect_batch_activity()` signature
   - Replaced HHT export with batch export logic
   - Added hht_matrices/ and hht_images/ folders

---

## Commit Information

**Commit**: 1a8eef0
**Message**: Fix adaptive merging, add TKEO/merge UI controls, implement batch HHT export
**Date**: 2025-12-15

---

## References

- Original implementation: commit 6134fea
- Code review comment: #3655493425
- Enhanced features documentation: ENHANCED_FEATURES.md
