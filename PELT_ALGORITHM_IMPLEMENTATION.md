# Advanced PELT Algorithm Implementation for sEMG Event Detection

## Overview

This document describes the new PELT (Pruned Exact Linear Time) based detection algorithm implemented for sEMG event recognition and segmentation. The algorithm addresses previous limitations by using a unified approach with advanced features.

## Key Improvements

### 1. Unified Single-Stage PELT Detection

**Previous approach**: Two-stage detection (event presence → boundary refinement)

**New approach**: Single-stage PELT with multi-dimensional features

**Benefits**:
- Simpler, more maintainable code
- Better performance
- More rigorous mathematical foundation
- Directly interpretable parameters

### 2. Energy-Based Adaptive Penalty Strategy

The signal is divided into energy zones using K-means clustering:

```
Energy Zones:
- Low energy zone (0):  penalty = base_penalty × 0.5  (more sensitive)
- Medium energy zone (1): penalty = base_penalty × 1.0  (normal)
- High energy zone (2): penalty = base_penalty × 2.0  (less sensitive)
```

**Base penalty formula**:
```python
base_penalty = 3.0 × sensitivity
```

Where `sensitivity` ranges from 0.1 (very sensitive) to 5.0 (very strict).

**Rationale**: 
- Low energy regions (baseline/rest) need lower penalties to detect subtle activity onsets
- High energy regions (active muscle) need higher penalties to avoid over-segmentation
- This context-aware approach improves detection quality

### 3. Multi-Dimensional Feature Vectors

The algorithm extracts 8 features across three domains:

#### Time-Domain Features (4)
1. **RMS (Root Mean Square)**: `sqrt(mean(signal²))`
   - Measures signal energy/power
2. **MAV (Mean Absolute Value)**: `mean(|signal|)`
   - Measures average amplitude
3. **VAR (Variance)**: `E[X²] - E[X]²`
   - Measures signal variability
4. **WL (Waveform Length)**: `sum(|gradient|)`
   - Measures signal complexity/irregularity

#### Frequency-Domain Features (2)
5. **MNF (Mean Frequency)**: `Σ(f·PSD) / Σ(PSD)`
   - Spectral centroid, indicates dominant frequency
6. **MDF (Median Frequency)**: Frequency at 50% of cumulative PSD
   - Robust frequency indicator

#### Complexity Features (2)
7. **ZCR (Zero Crossing Rate)**: Rate of sign changes
   - Simple frequency/complexity indicator
8. **Sample Entropy Proxy**: `log(VAR) / log(RMS)`
   - Approximates signal regularity

**Feature Normalization**: All features are standardized (zero mean, unit variance) before PELT.

### 4. Multi-Detector Ensemble System

#### Architecture

The system can run 1-5 parallel PELT detectors with different sensitivities:

```python
sensitivity_range = np.linspace(sensitivity × 0.7, sensitivity × 1.3, n_detectors)
```

For example, with `sensitivity=1.5` and `n_detectors=3`:
- Detector 1: sensitivity = 1.05 (more sensitive)
- Detector 2: sensitivity = 1.50 (base)
- Detector 3: sensitivity = 1.95 (less sensitive)

Each detector independently:
1. Computes adaptive penalties based on energy zones
2. Runs PELT on multi-dimensional features
3. Calculates confidence scores for each detected segment

#### Fusion Methods

**1. Confidence-Weighted Fusion (Recommended)**

Creates a confidence map by summing confidence scores:

```python
confidence_map[start:end] += confidence_score
```

Then thresholds at 50th percentile of positive confidences.

**Advantages**:
- Robust to detector disagreements
- Weights by quality of detections
- Good balance of sensitivity and precision

**2. Voting Fusion**

Counts how many detectors agree on each position:

```python
vote_map[start:end] += 1
threshold = n_detectors // 2 + 1  # Majority vote
```

**Advantages**:
- Conservative, reduces false positives
- Simple, interpretable
- Good when high precision is needed

**3. Union Fusion**

Simply combines all detections:

```python
all_segments = detector1_segs + detector2_segs + detector3_segs
merged = remove_overlaps(all_segments)
```

**Advantages**:
- Most sensitive, catches all possibilities
- Good when missing events is more costly than false positives

### 5. Intelligent Dense Event Merging

Events with gaps < 50ms are automatically merged:

```python
if gap < int(0.05 × fs):
    merged_segment = (current_start, next_end)
```

**Rationale**:
- Prevents over-segmentation in rhythmic activity
- 50ms is below typical muscle contraction onset time
- Physiologically meaningful threshold

### 6. Strict Duration Enforcement

**Hard Constraint**: `min_duration` is STRICTLY enforced at ALL stages:
1. Initial PELT detection: `min_size = int(min_duration × fs)`
2. After merging: Filter segments < min_duration
3. Final output: Double-check all segments ≥ min_duration

**Soft Constraint**: `max_duration` triggers intelligent splitting:
- Uses PELT change points within long segments
- Finds RMS envelope minima as natural break points
- Each split must satisfy min_duration

## Algorithm Flow

```
Input: preprocessed sEMG signal, fs, parameters
│
├─► 1. Extract Multi-Dimensional Features (8D)
│      - Time: RMS, MAV, VAR, WL
│      - Frequency: MNF, MDF
│      - Complexity: ZCR, Entropy
│      - Normalize features
│
├─► 2. Compute Energy Zones
│      - Calculate local energy
│      - K-means clustering (3 zones)
│      - Map: low→0, medium→1, high→2
│
├─► 3. Multi-Detector Ensemble (if enabled)
│   │
│   ├─► For each detector (different sensitivity):
│   │   ├─► Compute adaptive penalties
│   │   │      penalty[i] = base_penalty × zone_multiplier[zone[i]]
│   │   │
│   │   ├─► Run PELT with adaptive penalties
│   │   │      algo = Pelt(model='l2', min_size=min_samples)
│   │   │      change_points = algo.predict(pen=median_penalty)
│   │   │
│   │   ├─► Refine boundaries using local penalties
│   │   │      Move boundaries to positions with minimal penalty
│   │   │
│   │   └─► Calculate confidence scores
│   │          confidence = f(contrast, consistency, duration)
│   │
│   └─► Fuse detections
│          - Confidence: weighted by scores
│          - Voting: majority agreement
│          - Union: combine all
│
├─► 4. Merge Dense Events (gaps < 50ms)
│      - Prevents over-segmentation
│      - Respects min_duration
│
├─► 5. Enforce Duration Constraints
│      - Filter: keep only segments ≥ min_duration
│      - Split: segments > max_duration (if specified)
│
└─► Output: List of (start_index, end_index) tuples
```

## Confidence Score Calculation

For each detected segment:

```python
confidence = 0.5 × contrast_score + 
             0.3 × consistency_score + 
             0.2 × duration_score
```

Where:

**Contrast Score**: Amplitude difference from surroundings
```python
contrast = (segment_rms - surrounding_rms) / surrounding_rms
contrast_score = min(1.0, contrast / 2.0)
```

**Consistency Score**: Internal signal stability
```python
cv = std(segment) / mean(segment)  # Coefficient of variation
consistency_score = 1.0 / (1.0 + cv)
```

**Duration Score**: Physiological reasonableness
```python
if 0.1 ≤ duration ≤ 5.0:
    duration_score = 1.0
elif duration < 0.1:
    duration_score = duration / 0.1
else:
    duration_score = 5.0 / duration
```

## Usage Examples

### Basic Usage (Single Detector)

```python
from semg_preprocessing import detect_muscle_activity

segments = detect_muscle_activity(
    filtered_signal,
    fs=1000,
    method='combined',      # Only supported method
    min_duration=0.1,       # 100ms minimum
    max_duration=5.0,       # 5s maximum (optional)
    sensitivity=1.5,        # Balanced sensitivity
    use_multi_detector=False
)
```

### Advanced Usage (Multi-Detector Ensemble)

```python
segments = detect_muscle_activity(
    filtered_signal,
    fs=1000,
    method='combined',
    min_duration=0.1,
    max_duration=5.0,
    sensitivity=1.5,
    n_detectors=3,               # 3 parallel detectors
    fusion_method='confidence',  # Confidence-weighted fusion
    use_multi_detector=True
)
```

### High Sensitivity Detection

```python
segments = detect_muscle_activity(
    filtered_signal,
    fs=1000,
    method='combined',
    min_duration=0.05,      # Shorter minimum
    sensitivity=0.8,        # Lower = more sensitive
    n_detectors=5,          # More detectors
    fusion_method='union',  # Most sensitive fusion
    use_multi_detector=True
)
```

## Parameter Tuning Guidelines

### sensitivity (0.1 - 5.0)

- **0.1 - 0.8**: Very sensitive, detects subtle activity
  - Use for: Low-amplitude signals, subtle contractions
  - Risk: May detect noise as events
  
- **0.8 - 2.0**: Balanced (default: 1.5)
  - Use for: Most applications
  - Good balance of sensitivity and specificity
  
- **2.0 - 5.0**: Very strict, only strong activity
  - Use for: High SNR signals, obvious events only
  - Risk: May miss legitimate low-amplitude events

### n_detectors (1 - 5)

- **1**: Single detector (fastest)
- **3**: Recommended (good balance)
- **5**: Maximum robustness (slower)

### fusion_method

- **'confidence'**: Recommended for most cases
- **'voting'**: When precision > recall
- **'union'**: When recall > precision

### min_duration (0.01 - 10.0 seconds)

Set based on your application:
- **0.05 - 0.1s**: Very brief contractions
- **0.1 - 0.5s**: Typical muscle activity
- **0.5 - 2.0s**: Sustained contractions
- **> 2.0s**: Long-duration activity only

## Implementation References

The algorithm is based on rigorous signal processing principles and PELT theory:

1. **PELT Algorithm**: Killick et al. (2012) - Optimal detection of changepoints
2. **Energy-based Penalties**: Adaptive to local signal characteristics
3. **Multi-dimensional Features**: Standard sEMG feature extraction practices
4. **Ensemble Methods**: Combining multiple detectors for robustness
5. **Duration Constraints**: Physiologically-motivated hard and soft constraints

## Performance Characteristics

### Computational Complexity

- **Feature Extraction**: O(n) per feature, O(8n) total
- **Energy Zones**: O(n log n) for K-means
- **PELT per detector**: O(n) average case
- **Multi-detector**: O(k×n) where k = n_detectors
- **Fusion**: O(n) for all methods

**Overall**: O(k×n) where k is typically 1-5, so effectively linear in signal length.

### Memory Usage

- Feature matrix: 8 × n_samples × 8 bytes ≈ 64n bytes
- Energy zones: n_samples × 4 bytes ≈ 4n bytes
- Detector results: k × avg_segments × 16 bytes (minimal)

**Total**: ~70n bytes + minimal segment storage

## Critical Implementation Note

As specified in the requirements:

> **无论我在预处理后的信号中作事件检测分段时进行了任何处理，之后的相关特征参数计算都必须基于预处理后的原始信号对应的分段结果**

**Translation**: Regardless of any processing done during event detection on preprocessed signals, subsequent feature parameter calculations MUST be based on the segment results from the preprocessed original signal.

**Implementation**: 
- The detection algorithm works on the filtered/preprocessed signal
- The returned segment indices `(start, end)` should be used to extract segments from the SAME preprocessed signal for feature extraction
- Do NOT apply additional transformations between detection and feature extraction
- This prevents unknown issues from processing-induced feature changes

## Validation and Testing

All functionality has been validated:

✅ Import and basic detection test
✅ Realistic signal test with ground truth
✅ Multi-detector ensemble test
✅ Fusion method comparison
✅ Duration constraint enforcement
✅ All unit tests pass

## Future Enhancements

Possible improvements for future versions:

1. **Adaptive feature selection**: Automatically choose most discriminative features
2. **Online detection**: Real-time streaming mode
3. **GPU acceleration**: For large-scale batch processing
4. **Custom cost functions**: Allow user-defined PELT costs
5. **Learned penalties**: Machine learning-based penalty estimation

## Conclusion

The new PELT-based detection algorithm provides:

- ✅ Rigorous mathematical foundation
- ✅ Multi-dimensional feature analysis
- ✅ Context-aware adaptive penalties
- ✅ Ensemble robustness
- ✅ Intelligent event merging
- ✅ Strict constraint enforcement
- ✅ Direct parameter interpretability

This implementation addresses all requirements from the problem statement and provides a solid foundation for sEMG event detection and segmentation.
