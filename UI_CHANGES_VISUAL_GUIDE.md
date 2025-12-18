# UI Changes Visual Guide / UI改动可视化指南

## Feature Export UI Changes / 特征导出UI改动

### Before (之前):
```
┌─────────────────────────────────────────────────────────────┐
│  Feature Export / 特征导出                                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Features CSV Path:                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ./output/segment_features.csv                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ❌ User must type full filename                             │
│  ❌ Same filename for different signals                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### After (之后):
```
┌─────────────────────────────────────────────────────────────┐
│  Feature Export / 特征导出                                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Output Directory (输出目录):                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ./output                                             │   │
│  └─────────────────────────────────────────────────────┘   │
│  ℹ️ Enter directory path (e.g., ./output).                  │
│     Filename will be auto-generated from signal file.       │
│                                                               │
│  ✅ Simple directory path only                               │
│  ✅ Auto-generates: {signal_name}_features.csv               │
│                                                               │
│  Examples:                                                    │
│  • Signal: bicep_curl.csv  → bicep_curl_features.csv        │
│  • Signal: test_data.mat   → test_data_features.csv         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## DI Display Changes / DI显示改动

### Text Display (文本显示)

#### Before:
```
Feature Statistics:
  RMS: 0.0234 ± 0.0012
  MDF: 85.23 ± 12.45 Hz
  DI: 0.00 ± 0.00          ❌ Not visible!
  WIRE51: 0.15 ± 0.03
```

#### After:
```
Feature Statistics:
  RMS: 0.0234 ± 0.0012
  MDF: 85.23 ± 12.45 Hz
  DI: 1.234568e-14 ± 5.678901e-15    ✅ Clear value!
  WIRE51: 0.15 ± 0.03
```

### Plot Display (图表显示)

#### Before:
```
Fatigue Indicators Trend
     │
 DI  │  ┌─────────────────────────────┐
     │  │ ∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙    │  ❌ All points near zero
0.00 │  │ (can't see variation)       │     Not informative
     │  └─────────────────────────────┘
     └──────────────────────────────────
        Segment Index
```

#### After:
```
Fatigue Indicators Trend (DI scaled ×10⁻¹⁴)
     │
 DI  │  ┌─────────────────────────────┐
     │  │          ∙     ∙             │  ✅ Clear variation
2.5  │  │      ∙       ∙               │     Values visible
     │  │   ∙                ∙         │     Trend obvious
1.0  │  │ ∙                     ∙      │
     │  └─────────────────────────────┘
     └──────────────────────────────────
        Segment Index
        
Y-axis: "DI Value (×10⁻¹⁴)"
Legend: "DI (×10⁻¹⁴)"
Values: Original DI × 10¹⁴ for display
```

---

## Code Implementation / 代码实现

### 1. Filename Generation / 文件名生成

```python
# New method in EMGProcessorGUI class
def _get_default_feature_filename(self):
    """Generate default filename based on signal filename."""
    if hasattr(self, 'current_filename') and self.current_filename:
        base_name = os.path.splitext(self.current_filename)[0]
        return f"{base_name}_features.csv"
    else:
        return "segment_features.csv"
```

**Usage Example:**
```python
# User loads: "bicep_curl.csv"
processor.current_filename = "bicep_curl.csv"
default = processor._get_default_feature_filename()
# Returns: "bicep_curl_features.csv"
```

### 2. Path Handling / 路径处理

```python
# In export_features_csv() method
if os.path.isdir(output_path) or \
   (not output_path.endswith('.csv') and not os.path.exists(output_path)):
    # It's a directory - add default filename
    default_filename = self._get_default_feature_filename()
    output_path = os.path.join(output_path, default_filename)
```

**Behavior:**
- Input: `./output` → Output: `./output/bicep_curl_features.csv`
- Input: `./output/custom.csv` → Output: `./output/custom.csv` (unchanged)

### 3. DI Formatting / DI格式化

#### Scientific Notation (科学计数法):
```python
# Before:
info += f"DI: {np.mean(di_values):.2f} ± {np.std(di_values):.2f}"
# Result: "DI: 0.00 ± 0.00"  ❌

# After:
info += f"DI: {np.mean(di_values):.6e} ± {np.std(di_values):.6e}"
# Result: "DI: 1.234568e-14 ± 5.678901e-15"  ✅
```

#### Scaled Plot Values (缩放图表值):
```python
# Before:
di_values = [f['DI'] for f in features_list]
# Values: [1.2e-14, 2.3e-14, ...]  ❌ Too small to see

# After:
di_values = [f['DI'] * 1e14 for f in features_list]
# Values: [1.2, 2.3, ...]  ✅ Visible range
ax.set_ylabel('DI Value (×10⁻¹⁴)', fontsize=11)
```

---

## Affected Locations / 影响位置

### DI Display Updates (7 locations):

1. **`extract_batch_features()`** - Batch summary
   - Line ~671: `.2f` → `.6e`
   - Line ~680: `.2f` → `.6e`

2. **`perform_batch_hht()`** - HHT results  
   - Line ~1186: `.4f` → `.6e`

3. **`_create_fatigue_indicator_plot()`** - Trend plot
   - Line ~1352-1354: Scale values by 1e14, update labels

4. **`_create_feature_summary()`** - Statistics
   - Line ~1410: Already `.6e` (kept)
   - Line ~1419: Already `.6e` (kept)

### UI Label Updates:

1. **Feature Export Section** (Line ~2250-2253):
   - Label: "Features CSV Path" → "Output Directory (输出目录)"
   - Default: `./output/segment_features.csv` → `./output`
   - Added placeholder with auto-generation hint

---

## Testing / 测试

### Test Script Results:
```
======================================================================
Testing UI Improvements
======================================================================

1. Testing default filename generation...
   Without filename: segment_features.csv
   With 'my_signal.csv': my_signal_features.csv
   With 'test_data.mat': test_data_features.csv
   ✓ Default filename generation works correctly

2. Testing DI value formatting...
   Scientific notation: 1.234568e-14
   Scaled (×10⁻¹⁴): 1.2346
   ✓ DI formatting displays small values correctly

3. Testing export path handling...
   Directory './output' → './output/signal_data_features.csv'
   Full path './output/custom_name.csv' → remains unchanged
   ✓ Path handling logic correct

======================================================================
All tests passed! ✓
======================================================================
```

---

## User Benefits / 用户收益

### Feature Export:
✅ **Simpler workflow**: Directory only, no typing filenames
✅ **Auto-organization**: Each signal gets unique filename  
✅ **No conflicts**: Different signals won't overwrite each other
✅ **Clear naming**: File origin always visible from filename
✅ **Backward compatible**: Full paths still work

### DI Display:
✅ **Visible values**: No more `0.0000`, shows actual magnitude
✅ **Scientific notation**: Standard format for small values
✅ **Clear trends**: Plot scaling makes variations obvious
✅ **Consistent**: Same format across all displays
✅ **No calculation changes**: Values computed identically

---

## Summary / 总结

**Changes Made:**
1. Feature export: Auto-generate filename from signal file
2. DI display: Scientific notation for text, scaled for plots

**No Changes:**
- ❌ DI calculation algorithm (unchanged)
- ❌ Feature extraction logic (unchanged)  
- ❌ Any data processing (unchanged)

**Result:**
- ✅ Better user experience
- ✅ Clearer information display
- ✅ Same accurate calculations
- ✅ Fully backward compatible
