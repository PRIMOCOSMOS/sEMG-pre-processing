"""
Muscle activity detection and signal segmentation.

This module implements muscle activity event detection using:
- Ruptures library for change point detection
- Amplitude-based analysis
- Combined ruptures and amplitude approach for robust detection
- Multi-feature fusion detection with adaptive parameters
"""

import numpy as np
import ruptures as rpt
from typing import List, Tuple, Optional, Dict, Union
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def detect_muscle_activity(
    data: np.ndarray,
    fs: float,
    method: str = "multi_feature",
    amplitude_threshold: Optional[float] = None,
    window_size: Optional[int] = None,
    min_duration: float = 0.1,
    max_duration: Optional[float] = None,
    **kwargs
) -> List[Tuple[int, int]]:
    """
    Detect muscle activity events in sEMG signal.
    
    This function provides multiple detection methods including advanced
    multi-feature fusion with adaptive parameters.
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data (1D array), should be preprocessed (filtered)
    fs : float
        Sampling frequency in Hz
    method : str, optional
        Detection method: 'ruptures', 'amplitude', 'combined', or 'multi_feature' (default: 'multi_feature')
    amplitude_threshold : float, optional
        Threshold for amplitude-based detection (default: auto-calculated as 2*RMS)
    window_size : int, optional
        Window size for feature calculation (default: fs/10)
    min_duration : float, optional
        Minimum duration of muscle activity in seconds (default: 0.1)
    max_duration : float, optional
        Maximum duration of muscle activity in seconds (default: None = no limit)
        Long segments exceeding this duration will be split using internal change detection
    **kwargs : dict
        Additional arguments:
        - For ruptures: model, pen, min_size
        - For multi_feature: use_clustering, adaptive_pen, n_clusters
    
    Returns:
    --------
    List[Tuple[int, int]]
        List of (start_index, end_index) tuples for each detected activity period
        
    Notes:
    ------
    - 'multi_feature' method is recommended for most robust detection
    - 'combined' method provides good balance of speed and accuracy
    - Data should be preprocessed (filtered) before detection
    - Adjust amplitude_threshold based on signal characteristics
    - max_duration helps split overly long segments into more meaningful motion periods
    """
    # Define valid parameters for each method to prevent TypeError
    ruptures_params = {'model', 'pen', 'min_size'}
    multi_feature_params = {'use_clustering', 'adaptive_pen', 'n_clusters', 'sensitivity', 
                            'model', 'pen', 'min_size'}
    
    if method == "ruptures":
        # Filter kwargs to only valid ruptures parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ruptures_params}
        segments = _detect_ruptures(data, fs, **filtered_kwargs)
    elif method == "amplitude":
        # Amplitude method uses sensitivity from kwargs if provided
        sensitivity = kwargs.get('sensitivity', 2.0)
        segments = _detect_amplitude(data, fs, amplitude_threshold, window_size, min_duration, sensitivity)
    elif method == "combined":
        # Filter kwargs to only valid ruptures parameters for combined
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ruptures_params}
        sensitivity = kwargs.get('sensitivity', 2.0)
        segments = _detect_combined(data, fs, amplitude_threshold, window_size, min_duration, sensitivity, **filtered_kwargs)
    elif method == "multi_feature":
        # Filter kwargs to only valid multi_feature parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in multi_feature_params}
        segments = _detect_multi_feature(data, fs, window_size, min_duration, **filtered_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ruptures', 'amplitude', 'combined', or 'multi_feature'")
    
    # Apply max_duration splitting if specified
    if max_duration is not None and max_duration > min_duration:
        segments = _split_long_segments(data, segments, fs, min_duration, max_duration)
    
    return segments


def _detect_ruptures(
    data: np.ndarray,
    fs: float,
    model: str = "l2",
    pen: float = 3,
    min_size: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Detect change points using ruptures library.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    model : str
        Ruptures model type ('l1', 'l2', 'rbf', 'normal')
    pen : float
        Penalty value (higher = fewer change points)
    min_size : int, optional
        Minimum segment size
    
    Returns:
    --------
    List[Tuple[int, int]]
        Detected segments
    """
    if min_size is None:
        min_size = int(fs / 10)  # Default: 0.1 seconds
    
    # Use Pelt algorithm for change point detection
    # Note: ruptures works better with 2D data
    data_2d = data.reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=min_size).fit(data_2d)
    change_points = algo.predict(pen=pen)
    
    # Remove the last point if it's the end of the signal
    if change_points and change_points[-1] == len(data):
        change_points = change_points[:-1]
    
    # If no change points detected, return empty list
    if not change_points:
        return []
    
    # Convert change points to segments
    segments = []
    start = 0
    for cp in change_points:
        if cp - start > min_size:
            segments.append((start, cp))
        start = cp
    
    # Add final segment if there's remaining data
    if start < len(data) and len(data) - start > min_size:
        segments.append((start, len(data)))
    
    return segments


def _calculate_adaptive_threshold(
    data: np.ndarray,
    envelope: np.ndarray,
    sensitivity: float = 1.5
) -> float:
    """
    Calculate adaptive threshold based on signal characteristics.
    
    This function analyzes the signal to determine appropriate threshold
    that works for both high and low amplitude sEMG recordings.
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal
    envelope : np.ndarray
        RMS envelope of signal
    sensitivity : float
        Sensitivity multiplier (lower = more sensitive, default: 1.5)
    
    Returns:
    --------
    float
        Adaptive threshold value
    """
    # Calculate signal statistics
    signal_rms = np.sqrt(np.mean(data ** 2))
    envelope_mean = np.mean(envelope)
    envelope_std = np.std(envelope)
    envelope_percentiles = np.percentile(envelope, [25, 50, 75, 90])
    
    # For very low amplitude signals, use percentile-based threshold
    if signal_rms < 0.01 * np.max(np.abs(data)):
        # Use 50th percentile + sensitivity factor
        threshold = envelope_percentiles[1] + sensitivity * 0.5 * envelope_std
    else:
        # For normal amplitude signals, use mean + sensitivity * std
        # Lower base multiplier from 2.0 to 1.5 for easier event recognition
        base_multiplier = 1.0 + (sensitivity - 1.0) * 0.5
        threshold = envelope_mean + base_multiplier * envelope_std
    
    # Ensure threshold is not too low (prevents noise detection)
    min_threshold = 0.1 * envelope_percentiles[2]  # At least 10% of 75th percentile
    threshold = max(threshold, min_threshold)
    
    return threshold


def _detect_amplitude(
    data: np.ndarray,
    fs: float,
    threshold: Optional[float] = None,
    window_size: Optional[int] = None,
    min_duration: float = 0.1,
    sensitivity: float = 1.5
) -> List[Tuple[int, int]]:
    """
    Detect muscle activity based on amplitude threshold with adaptive mechanisms.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    threshold : float, optional
        Amplitude threshold (default: calculated adaptively from signal characteristics)
    window_size : int, optional
        Window size for envelope calculation
    min_duration : float
        Minimum activity duration in seconds
    sensitivity : float
        Sensitivity multiplier for threshold (lower = more sensitive, default: 1.5)
        Range: 0.5 (very sensitive) to 5.0 (very strict)
    
    Returns:
    --------
    List[Tuple[int, int]]
        Detected activity periods
        
    Notes:
    ------
    Default sensitivity lowered from 2.0 to 1.5 to make event recognition easier.
    Adaptive threshold mechanism adjusts automatically for different signal amplitudes.
    """
    if window_size is None:
        window_size = int(fs / 10)  # Default: 0.1 seconds
    
    # Calculate envelope using RMS (Root Mean Square)
    envelope = _calculate_rms_envelope(data, window_size)
    
    # Auto-calculate threshold if not provided using adaptive mechanism
    if threshold is None:
        threshold = _calculate_adaptive_threshold(data, envelope, sensitivity)
    
    # Find regions above threshold
    above_threshold = envelope > threshold
    
    # Find transitions
    diff = np.diff(above_threshold.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if above_threshold[0]:
        starts = np.concatenate([[0], starts])
    if above_threshold[-1]:
        ends = np.concatenate([ends, [len(above_threshold)]])
    
    # Filter by minimum duration
    min_samples = int(min_duration * fs)
    segments = []
    for start, end in zip(starts, ends):
        if end - start >= min_samples:
            segments.append((int(start), int(end)))
    
    return segments


def _detect_combined(
    data: np.ndarray,
    fs: float,
    amplitude_threshold: Optional[float] = None,
    window_size: Optional[int] = None,
    min_duration: float = 0.1,
    sensitivity: float = 2.0,
    **kwargs
) -> List[Tuple[int, int]]:
    """
    Detect muscle activity events using intelligent holistic optimization.
    
    This method focuses on detecting meaningful muscle activity events (e.g., individual
    bicep curls) rather than mechanically satisfying parameter constraints. It uses:
    
    1. **Multi-Strategy Candidate Generation**:
       - Ruptures for structural change points
       - Amplitude-based detection for sustained activity
       - Rhythmic pattern detection for periodic movements
       - Amplitude trend analysis for gradual activation
    
    2. **Event Quality Scoring**:
       - RMS consistency within events (high = coherent single event)
       - Boundary quality (clear amplitude drops between events)
       - Duration reasonableness (penalize extremes)
       - Transition sharpness (rapid changes at boundaries)
    
    3. **Holistic Optimization**:
       - Generate multiple segmentation candidates
       - Score each holistically based on event characteristics
       - Select scheme with best overall quality score
       - Refine boundaries and merge similar adjacent events
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    amplitude_threshold : float, optional
        Amplitude threshold
    window_size : int, optional
        Window size for envelope calculation
    min_duration : float
        Minimum activity duration in seconds (optimization constraint)
    sensitivity : float
        Detection sensitivity (lower = more sensitive, default: 1.5)
        Affects candidate generation and scoring
    **kwargs : dict
        Additional arguments for ruptures (model, pen, min_size)
    
    Returns:
    --------
    List[Tuple[int, int]]
        Detected activity events with optimal segmentation
        
    Notes:
    ------
    This algorithm prioritizes finding meaningful physiological events over
    mechanical parameter satisfaction. Duration constraints are treated as
    optimization guides rather than hard cutoffs.
    """
    if window_size is None:
        window_size = int(fs / 10)
    
    min_samples = int(min_duration * fs)
    
    # Calculate RMS envelope for quality assessment
    rms_envelope = _calculate_rms_envelope(data, window_size)
    
    # Use adaptive threshold for amplitude-based detection
    if amplitude_threshold is None:
        amplitude_threshold = _calculate_adaptive_threshold(data, rms_envelope, sensitivity)
    
    # Filter kwargs to only pass valid ruptures parameters
    valid_ruptures_params = {'model', 'pen', 'min_size'}
    ruptures_kwargs = {k: v for k, v in kwargs.items() if k in valid_ruptures_params}
    
    # Adjust rupture penalty based on sensitivity
    if 'pen' not in ruptures_kwargs:
        ruptures_kwargs['pen'] = 3.0 * sensitivity / 2.0
    
    # === STEP 1: Generate candidate segmentation schemes ===
    
    # Candidate 1: Ruptures-based (structural changes)
    ruptures_segments = _detect_ruptures(data, fs, **ruptures_kwargs)
    
    # Candidate 2: Amplitude-based (sustained activity)
    amplitude_segments = _detect_amplitude(data, fs, amplitude_threshold, window_size, min_duration, sensitivity)
    
    # Candidate 3: Rhythmic patterns (periodic movements)
    rhythmic_segments = _detect_rhythmic_patterns(data, fs, window_size, min_duration, sensitivity)
    
    # Candidate 4: Amplitude trends (gradual activation)
    trend_segments = _detect_amplitude_trends(data, fs, window_size, min_duration, sensitivity)
    
    # Candidate 5: Hybrid 1 - Ruptures refined by amplitude
    hybrid1_segments = _refine_segments_by_amplitude(ruptures_segments, rms_envelope, min_samples, sensitivity)
    
    # Candidate 6: Hybrid 2 - Amplitude refined by ruptures
    hybrid2_segments = _refine_segments_by_ruptures(amplitude_segments, data, fs, min_samples, ruptures_kwargs)
    
    # Collect all candidate schemes
    candidates = [
        ruptures_segments,
        amplitude_segments,
        rhythmic_segments,
        trend_segments,
        hybrid1_segments,
        hybrid2_segments
    ]
    
    # === STEP 2: Score each candidate scheme holistically ===
    
    best_score = -np.inf
    best_segments = []
    
    for candidate_segments in candidates:
        if not candidate_segments:
            continue
        
        # Score this segmentation scheme
        score = _score_segmentation_scheme(
            data, candidate_segments, rms_envelope, fs, 
            min_duration, sensitivity
        )
        
        if score > best_score:
            best_score = score
            best_segments = candidate_segments
    
    # === STEP 3: Intelligent post-processing and refinement ===
    
    if best_segments:
        # Refine boundaries for better event separation
        refined_segments = _refine_event_boundaries(
            data, best_segments, rms_envelope, min_samples
        )
        
        # Merge events that are likely part of the same activity
        final_segments = _merge_similar_events(
            data, refined_segments, rms_envelope, fs, min_duration
        )
    else:
        final_segments = []
    
    return final_segments


def _calculate_rms_envelope(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate RMS (Root Mean Square) envelope of the signal.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    window_size : int
        Window size for RMS calculation
    
    Returns:
    --------
    np.ndarray
        RMS envelope
    """
    # Square the signal
    squared = data ** 2
    
    # Apply moving average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(squared, kernel, mode='same')
    
    # Take square root to get RMS
    rms = np.sqrt(smoothed)
    
    return rms


def _merge_overlapping_segments(segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping segments.
    
    Parameters:
    -----------
    segments : List[Tuple[int, int]]
        List of (start, end) tuples
    
    Returns:
    --------
    List[Tuple[int, int]]
        Merged segments
    """
    if not segments:
        return []
    
    # Sort segments by start index
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    merged = [sorted_segments[0]]
    
    for current_start, current_end in sorted_segments[1:]:
        last_start, last_end = merged[-1]
        
        # Check for overlap or adjacency
        if current_start <= last_end:
            # Merge
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap, add as new segment
            merged.append((current_start, current_end))
    
    return merged


def segment_signal(
    data: np.ndarray,
    segments: List[Tuple[int, int]],
    fs: float,
    include_metadata: bool = True
) -> List[Dict]:
    """
    Segment the signal based on detected activity periods.
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data
    segments : List[Tuple[int, int]]
        List of (start_index, end_index) tuples from detect_muscle_activity
    fs : float
        Sampling frequency in Hz
    include_metadata : bool, optional
        Include metadata (duration, peak amplitude, etc.) for each segment
    
    Returns:
    --------
    List[Dict]
        List of dictionaries containing:
        - 'data': np.ndarray, the signal segment
        - 'start_index': int, start index in original signal
        - 'end_index': int, end index in original signal
        - 'start_time': float, start time in seconds (if include_metadata)
        - 'end_time': float, end time in seconds (if include_metadata)
        - 'duration': float, duration in seconds (if include_metadata)
        - 'peak_amplitude': float, maximum amplitude (if include_metadata)
        - 'mean_amplitude': float, mean amplitude (if include_metadata)
        - 'rms': float, RMS value (if include_metadata)
    """
    segmented_data = []
    
    for start_idx, end_idx in segments:
        segment_dict = {
            'data': data[start_idx:end_idx],
            'start_index': start_idx,
            'end_index': end_idx,
        }
        
        if include_metadata:
            segment = data[start_idx:end_idx]
            segment_dict.update({
                'start_time': start_idx / fs,
                'end_time': end_idx / fs,
                'duration': (end_idx - start_idx) / fs,
                'peak_amplitude': np.max(np.abs(segment)),
                'mean_amplitude': np.mean(np.abs(segment)),
                'rms': np.sqrt(np.mean(segment ** 2)),
            })
        
        segmented_data.append(segment_dict)
    
    return segmented_data


def _detect_multi_feature(
    data: np.ndarray,
    fs: float,
    window_size: Optional[int] = None,
    min_duration: float = 0.1,
    use_clustering: bool = True,
    adaptive_pen: bool = True,
    n_clusters: int = 2,
    sensitivity: float = 1.0,
    **ruptures_kwargs
) -> List[Tuple[int, int]]:
    """
    Detect muscle activity using multi-feature fusion approach.
    
    This advanced method extracts multiple features (RMS, envelope, variance)
    and uses them together with ruptures for robust change point detection.
    Optionally uses clustering to identify activity vs rest states.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    window_size : int, optional
        Window size for feature extraction
    min_duration : float
        Minimum activity duration in seconds
    use_clustering : bool
        Use K-means clustering for activity/rest classification
    adaptive_pen : bool
        Use adaptive penalty parameter selection
    n_clusters : int
        Number of clusters for K-means (default: 2 for activity/rest)
    sensitivity : float
        Detection sensitivity (default: 1.0)
        Lower = more sensitive (more segments), Higher = stricter (fewer segments)
        Range: 0.1 (very sensitive) to 3.0 (very strict)
    **ruptures_kwargs : dict
        Additional arguments for ruptures
    
    Returns:
    --------
    List[Tuple[int, int]]
        Detected activity periods
    """
    if window_size is None:
        window_size = int(fs / 10)  # Default: 0.1 seconds
    
    # Extract multiple features
    features = _extract_multi_features(data, window_size)
    
    # Normalize features for ruptures
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Determine penalty parameter (adjusted by sensitivity)
    if adaptive_pen:
        base_pen = _calculate_adaptive_penalty(features_normalized, fs)
        pen = base_pen * sensitivity  # Apply sensitivity modifier
    else:
        pen = ruptures_kwargs.get('pen', 3) * sensitivity
    
    # Use ruptures on multi-feature data
    model = ruptures_kwargs.get('model', 'l2')
    min_size = ruptures_kwargs.get('min_size', int(fs / 10))
    
    algo = rpt.Pelt(model=model, min_size=min_size).fit(features_normalized)
    change_points = algo.predict(pen=pen)
    
    # Remove the last point if it's the end of the signal
    if change_points and change_points[-1] == len(data):
        change_points = change_points[:-1]
    
    if not change_points:
        return []
    
    # Convert change points to segments
    segments = []
    start = 0
    for cp in change_points:
        if cp - start > min_size:
            segments.append((start, cp))
        start = cp
    
    # Add final segment if there's remaining data
    if start < len(data) and len(data) - start > min_size:
        segments.append((start, len(data)))
    
    # Use clustering to classify segments as activity or rest
    if use_clustering and segments:
        segments = _filter_segments_by_clustering(
            data, segments, features, n_clusters
        )
    
    # Filter by minimum duration
    min_samples = int(min_duration * fs)
    segments = [(s, e) for s, e in segments if e - s >= min_samples]
    
    return segments


def _extract_multi_features(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Extract multiple features from the signal for robust detection.
    
    Features extracted:
    1. RMS (Root Mean Square) envelope
    2. Absolute amplitude envelope
    3. Sliding window variance
    4. Signal energy
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    window_size : int
        Window size for feature calculation
    
    Returns:
    --------
    np.ndarray
        Feature matrix of shape (n_samples, n_features)
    """
    n_samples = len(data)
    
    # Feature 1: RMS envelope
    rms = _calculate_rms_envelope(data, window_size)
    
    # Feature 2: Absolute amplitude envelope (smoothed)
    abs_env = np.abs(data)
    kernel = np.ones(window_size) / window_size
    abs_env_smooth = np.convolve(abs_env, kernel, mode='same')
    
    # Feature 3: Sliding window variance
    variance = _calculate_sliding_variance(data, window_size)
    
    # Feature 4: Signal energy (smoothed squared signal)
    energy = data ** 2
    energy_smooth = np.convolve(energy, kernel, mode='same')
    
    # Stack features
    features = np.column_stack([rms, abs_env_smooth, variance, energy_smooth])
    
    return features


def _calculate_sliding_variance(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate sliding window variance of the signal using efficient convolution.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    window_size : int
        Window size
    
    Returns:
    --------
    np.ndarray
        Sliding variance
    """
    # Use convolution for efficient calculation
    # Var(X) = E[X^2] - E[X]^2
    kernel = np.ones(window_size) / window_size
    
    # Calculate E[X] - mean
    mean = np.convolve(data, kernel, mode='same')
    
    # Calculate E[X^2] - mean of squares
    mean_sq = np.convolve(data ** 2, kernel, mode='same')
    
    # Variance = E[X^2] - E[X]^2
    variance = mean_sq - mean ** 2
    
    # Handle numerical errors (variance should be >= 0)
    variance = np.maximum(variance, 0)
    
    return variance


def _calculate_adaptive_penalty(features: np.ndarray, fs: float) -> float:
    """
    Calculate adaptive penalty parameter based on signal characteristics.
    
    Uses the median absolute deviation (MAD) of feature changes to
    estimate an appropriate penalty value.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    fs : float
        Sampling frequency
    
    Returns:
    --------
    float
        Adaptive penalty value
    """
    # Calculate feature gradients (rate of change)
    feature_diffs = np.abs(np.diff(features, axis=0))
    
    # Use median absolute deviation as a robust estimate
    mad = np.median(np.abs(feature_diffs - np.median(feature_diffs, axis=0)), axis=0)
    
    # Penalty is proportional to the inverse of variability
    # More variability -> lower penalty (more change points)
    # Less variability -> higher penalty (fewer change points)
    median_mad = np.median(mad)
    
    if median_mad < 1e-10:
        return 3.0  # Default fallback
    
    # Scale penalty based on MAD
    # This is a heuristic that works well in practice
    penalty = np.log10(1 / median_mad + 1) * 2
    
    # Clip to reasonable range
    penalty = np.clip(penalty, 0.5, 10.0)
    
    return penalty


def _filter_segments_by_clustering(
    data: np.ndarray,
    segments: List[Tuple[int, int]],
    features: np.ndarray,
    n_clusters: int = 2
) -> List[Tuple[int, int]]:
    """
    Filter segments using K-means clustering to identify activity vs rest.
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal
    segments : List[Tuple[int, int]]
        Detected segments
    features : np.ndarray
        Feature matrix
    n_clusters : int
        Number of clusters (default: 2 for activity/rest)
    
    Returns:
    --------
    List[Tuple[int, int]]
        Filtered segments (activity segments only)
    """
    if not segments:
        return []
    
    # Calculate mean features for each segment
    segment_features = []
    for start, end in segments:
        seg_features = features[start:end].mean(axis=0)
        segment_features.append(seg_features)
    
    segment_features = np.array(segment_features)
    
    # Apply K-means clustering
    try:
        # Try using n_init='auto' for sklearn >= 1.4, fallback to 10 for older versions
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    except TypeError:
        # Older sklearn versions don't support n_init='auto'
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    labels = kmeans.fit_predict(segment_features)
    
    # Identify the activity cluster (higher feature values)
    cluster_means = np.array([segment_features[labels == i].mean() 
                              for i in range(n_clusters)])
    activity_cluster = np.argmax(cluster_means)
    
    # Filter segments to keep only activity cluster
    activity_segments = [seg for seg, label in zip(segments, labels) 
                        if label == activity_cluster]
    
    return activity_segments


def _detect_rhythmic_patterns(
    data: np.ndarray,
    fs: float,
    window_size: int,
    min_duration: float,
    sensitivity: float = 2.0
) -> List[Tuple[int, int]]:
    """
    Detect rhythmic muscle activity patterns using local RMS variance analysis.
    
    This method is particularly sensitive to rhythmic outbursts with varying amplitude,
    which are common in sEMG during repetitive movements.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    window_size : int
        Window size for local analysis
    min_duration : float
        Minimum duration in seconds
    sensitivity : float
        Detection sensitivity (lower = more sensitive)
    
    Returns:
    --------
    List[Tuple[int, int]]
        Detected rhythmic segments
    """
    # Calculate RMS envelope
    rms_envelope = _calculate_rms_envelope(data, window_size)
    
    # Calculate local variance of RMS (captures rhythmicity)
    rms_variance = _calculate_sliding_variance(rms_envelope, window_size)
    
    # Normalize variance
    rms_variance_normalized = rms_variance / (np.mean(rms_variance) + 1e-10)
    
    # Adaptive threshold based on sensitivity
    # Lower sensitivity -> lower threshold -> more segments
    threshold = sensitivity * 0.5
    
    # Find regions with high RMS variance (rhythmic activity)
    rhythmic_regions = rms_variance_normalized > threshold
    
    # Find transitions
    diff = np.diff(rhythmic_regions.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if rhythmic_regions[0]:
        starts = np.concatenate([[0], starts])
    if rhythmic_regions[-1]:
        ends = np.concatenate([ends, [len(rhythmic_regions)]])
    
    # Filter by minimum duration
    min_samples = int(min_duration * fs)
    segments = []
    for start, end in zip(starts, ends):
        if end - start >= min_samples:
            segments.append((int(start), int(end)))
    
    return segments


def _detect_amplitude_trends(
    data: np.ndarray,
    fs: float,
    window_size: int,
    min_duration: float,
    sensitivity: float = 2.0
) -> List[Tuple[int, int]]:
    """
    Detect low-amplitude outbursts using amplitude trend analysis.
    
    This method is designed to catch gradual amplitude increases that might
    be missed by simple threshold-based detection, especially in rhythmic movements
    where amplitude varies over time.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    window_size : int
        Window size for trend calculation
    min_duration : float
        Minimum duration in seconds
    sensitivity : float
        Detection sensitivity (lower = more sensitive)
    
    Returns:
    --------
    List[Tuple[int, int]]
        Detected trend-based segments
    """
    # Calculate absolute amplitude envelope
    abs_envelope = np.abs(data)
    kernel = np.ones(window_size) / window_size
    smoothed_envelope = np.convolve(abs_envelope, kernel, mode='same')
    
    # Calculate the gradient (rate of change) of the envelope
    gradient = np.gradient(smoothed_envelope)
    gradient_smoothed = np.convolve(gradient, kernel, mode='same')
    
    # Calculate local mean and std of the envelope for adaptive threshold
    local_mean = np.convolve(smoothed_envelope, kernel, mode='same')
    local_std = np.sqrt(np.convolve((smoothed_envelope - local_mean)**2, kernel, mode='same'))
    
    # Adaptive threshold: regions where envelope exceeds local baseline
    # Lower sensitivity = more sensitive to small changes
    # Adjusted multiplier for better low-amplitude detection
    threshold_multiplier = 0.4 + (sensitivity - 1.0) * 0.25  # Range: 0.4 to 1.0
    adaptive_threshold = local_mean + threshold_multiplier * local_std
    
    # Detect regions above adaptive threshold
    above_threshold = smoothed_envelope > adaptive_threshold
    
    # Also consider regions with significant positive gradient (rising activity)
    gradient_threshold = np.percentile(np.abs(gradient_smoothed), 60) / sensitivity
    rising_activity = gradient_smoothed > gradient_threshold
    
    # Combine both criteria
    activity_regions = above_threshold | rising_activity
    
    # Find transitions
    diff = np.diff(activity_regions.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if activity_regions[0]:
        starts = np.concatenate([[0], starts])
    if activity_regions[-1]:
        ends = np.concatenate([ends, [len(activity_regions)]])
    
    # Filter by minimum duration
    min_samples = int(min_duration * fs)
    segments = []
    for start, end in zip(starts, ends):
        if end - start >= min_samples:
            segments.append((int(start), int(end)))
    
    return segments


def _split_long_segments(
    data: np.ndarray,
    segments: List[Tuple[int, int]],
    fs: float,
    min_duration: float,
    max_duration: float
) -> List[Tuple[int, int]]:
    """
    Split segments that exceed maximum duration using internal change detection.
    
    This function uses multiple criteria to split long segments:
    1. Rupture-based change points within the segment
    2. Local minima in RMS envelope (natural breaks)
    3. Amplitude drops below local threshold
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal
    segments : List[Tuple[int, int]]
        Input segments
    fs : float
        Sampling frequency
    min_duration : float
        Minimum duration in seconds
    max_duration : float
        Maximum duration in seconds
    
    Returns:
    --------
    List[Tuple[int, int]]
        Split segments (all within max_duration)
    """
    max_samples = int(max_duration * fs)
    min_samples = int(min_duration * fs)
    split_segments = []
    
    for start, end in segments:
        segment_length = end - start
        
        # If segment is within limit, keep as is
        if segment_length <= max_samples:
            split_segments.append((start, end))
            continue
        
        # Segment is too long, need to split it
        segment_data = data[start:end]
        
        # Method 1: Use ruptures to find internal change points
        try:
            data_2d = segment_data.reshape(-1, 1)
            # Use lower penalty to detect more change points
            algo = rpt.Pelt(model='l2', min_size=min_samples).fit(data_2d)
            internal_cps = algo.predict(pen=1.5)
            
            # Remove last point if it's the end
            if internal_cps and internal_cps[-1] == len(segment_data):
                internal_cps = internal_cps[:-1]
        except:
            internal_cps = []
        
        # Method 2: Find local minima in RMS envelope as natural break points
        window_size = int(fs / 10)
        rms_envelope = _calculate_rms_envelope(segment_data, window_size)
        
        # Find local minima (potential breaks)
        # Invert to find minima
        peaks, _ = find_peaks(-rms_envelope, distance=min_samples, prominence=np.std(rms_envelope) * 0.3)
        local_minima = peaks.tolist()
        
        # Method 3: Find amplitude drops
        mean_rms = np.mean(rms_envelope)
        std_rms = np.std(rms_envelope)
        low_amplitude_threshold = mean_rms - 0.5 * std_rms
        below_threshold = rms_envelope < low_amplitude_threshold
        
        # Find midpoints of low-amplitude regions as potential splits
        diff = np.diff(below_threshold.astype(int))
        low_amp_starts = np.where(diff == 1)[0] + 1
        low_amp_ends = np.where(diff == -1)[0] + 1
        
        # Get midpoints of low-amplitude regions
        amplitude_drops = []
        for la_start, la_end in zip(low_amp_starts, low_amp_ends):
            if la_end - la_start > min_samples // 2:  # Only consider significant drops
                midpoint = (la_start + la_end) // 2
                amplitude_drops.append(midpoint)
        
        # Combine all potential split points
        all_split_points = sorted(set(internal_cps + local_minima + amplitude_drops))
        
        # Filter split points: must be at least min_samples apart
        filtered_splits = []
        last_split = 0
        for sp in all_split_points:
            if sp - last_split >= min_samples:
                filtered_splits.append(sp)
                last_split = sp
        
        # Create sub-segments
        if filtered_splits:
            sub_start = 0
            for split_point in filtered_splits:
                if split_point - sub_start >= min_samples:
                    split_segments.append((start + sub_start, start + split_point))
                    sub_start = split_point
            
            # Add final sub-segment
            if len(segment_data) - sub_start >= min_samples:
                split_segments.append((start + sub_start, end))
        else:
            # No good split points found, force split at max_duration intervals
            num_splits = int(np.ceil(segment_length / max_samples))
            split_size = segment_length // num_splits
            
            for i in range(num_splits):
                sub_start = start + i * split_size
                sub_end = start + (i + 1) * split_size if i < num_splits - 1 else end
                if sub_end - sub_start >= min_samples:
                    split_segments.append((sub_start, sub_end))
    
    return split_segments


def _score_segmentation_scheme(
    data: np.ndarray,
    segments: List[Tuple[int, int]],
    rms_envelope: np.ndarray,
    fs: float,
    min_duration: float,
    sensitivity: float
) -> float:
    """
    Score a segmentation scheme based on event quality metrics.
    
    Scoring criteria:
    1. RMS consistency within events (higher is better)
    2. Boundary quality (clear amplitude drops between events)
    3. Duration reasonableness (penalize too short/long events)
    4. Transition sharpness (rapid changes at boundaries)
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal
    segments : List[Tuple[int, int]]
        Segmentation scheme to score
    rms_envelope : np.ndarray
        RMS envelope of signal
    fs : float
        Sampling frequency
    min_duration : float
        Minimum duration constraint
    sensitivity : float
        Sensitivity parameter
    
    Returns:
    --------
    float
        Quality score (higher is better)
    """
    if not segments:
        return -np.inf
    
    min_samples = int(min_duration * fs)
    total_score = 0.0
    
    for i, (start, end) in enumerate(segments):
        segment_length = end - start
        
        # Skip segments that violate minimum duration
        if segment_length < min_samples:
            total_score -= 100.0  # Heavy penalty
            continue
        
        segment_rms = rms_envelope[start:end]
        
        # Metric 1: RMS consistency within event (coefficient of variation)
        # Lower CV = more consistent = better single event
        mean_rms = np.mean(segment_rms)
        std_rms = np.std(segment_rms)
        if mean_rms > 1e-10:
            cv = std_rms / mean_rms
            consistency_score = 10.0 * (1.0 / (1.0 + cv))  # 0 to 10
        else:
            consistency_score = 0.0
        
        # Metric 2: Duration reasonableness
        # Ideal duration range: 0.3 to 5 seconds for typical muscle contractions
        # Lowered minimum from 0.5 to 0.3 for more flexible event recognition
        duration_seconds = segment_length / fs
        ideal_duration_range = (0.3, 5.0)
        
        if ideal_duration_range[0] <= duration_seconds <= ideal_duration_range[1]:
            duration_score = 10.0
        elif duration_seconds < ideal_duration_range[0]:
            # Too short
            ratio = duration_seconds / ideal_duration_range[0]
            duration_score = 10.0 * ratio
        else:
            # Too long
            ratio = ideal_duration_range[1] / duration_seconds
            duration_score = 10.0 * ratio
        
        # Metric 3: Boundary quality (amplitude drop before/after)
        boundary_score = 0.0
        boundary_window = min(int(fs * 0.1), segment_length // 4)  # 100ms window
        
        # Check start boundary
        if start > boundary_window:
            before_rms = np.mean(rms_envelope[start - boundary_window:start])
            event_start_rms = np.mean(segment_rms[:boundary_window])
            if event_start_rms > before_rms:
                boundary_score += 5.0 * (event_start_rms / (before_rms + 1e-10))
        
        # Check end boundary
        if end + boundary_window < len(rms_envelope):
            after_rms = np.mean(rms_envelope[end:end + boundary_window])
            event_end_rms = np.mean(segment_rms[-boundary_window:])
            if event_end_rms > after_rms:
                boundary_score += 5.0 * (event_end_rms / (after_rms + 1e-10))
        
        boundary_score = min(boundary_score, 10.0)  # Cap at 10
        
        # Metric 4: Transition sharpness (rapid amplitude change at boundaries)
        transition_score = 0.0
        
        # Start transition
        if start + boundary_window < end:
            start_gradient = np.abs(segment_rms[boundary_window] - segment_rms[0])
            transition_score += min(start_gradient / (mean_rms + 1e-10) * 5.0, 5.0)
        
        # End transition
        if end - boundary_window > start:
            end_gradient = np.abs(segment_rms[-1] - segment_rms[-boundary_window])
            transition_score += min(end_gradient / (mean_rms + 1e-10) * 5.0, 5.0)
        
        # Combine scores with weights
        event_score = (
            0.30 * consistency_score +
            0.25 * duration_score +
            0.25 * boundary_score +
            0.20 * transition_score
        )
        
        total_score += event_score
    
    # Normalize by number of segments to favor schemes with reasonable event counts
    num_segments = len(segments)
    if num_segments > 0:
        avg_score = total_score / num_segments
        
        # Bonus for reasonable number of events (not too many, not too few)
        signal_duration = len(data) / fs
        expected_events = max(1, int(signal_duration / 2.0))  # Rough estimate
        event_count_penalty = abs(num_segments - expected_events) * 0.5
        
        final_score = avg_score - event_count_penalty
    else:
        final_score = -np.inf
    
    return final_score


def _refine_segments_by_amplitude(
    segments: List[Tuple[int, int]],
    rms_envelope: np.ndarray,
    min_samples: int,
    sensitivity: float
) -> List[Tuple[int, int]]:
    """
    Refine ruptures-based segments using amplitude information.
    
    Splits segments at points where amplitude drops significantly,
    indicating potential event boundaries.
    """
    refined = []
    threshold_factor = 0.7  # 70% of mean RMS
    
    for start, end in segments:
        segment_rms = rms_envelope[start:end]
        mean_rms = np.mean(segment_rms)
        threshold = mean_rms * threshold_factor
        
        # Find points below threshold
        below = segment_rms < threshold
        
        # Find transitions
        diff = np.diff(below.astype(int))
        drop_starts = np.where(diff == 1)[0] + 1
        drop_ends = np.where(diff == -1)[0] + 1
        
        # Use significant drops as split points
        split_points = []
        for ds, de in zip(drop_starts, drop_ends):
            if de - ds > min_samples // 4:  # Significant drop
                midpoint = (ds + de) // 2
                split_points.append(start + midpoint)
        
        # Create sub-segments
        if split_points:
            sub_start = start
            for sp in split_points:
                if sp - sub_start >= min_samples:
                    refined.append((sub_start, sp))
                    sub_start = sp
            
            if end - sub_start >= min_samples:
                refined.append((sub_start, end))
        else:
            refined.append((start, end))
    
    return refined


def _refine_segments_by_ruptures(
    segments: List[Tuple[int, int]],
    data: np.ndarray,
    fs: float,
    min_samples: int,
    ruptures_kwargs: dict
) -> List[Tuple[int, int]]:
    """
    Refine amplitude-based segments using rupture detection.
    
    Applies ruptures within each amplitude segment to find internal
    change points that may indicate separate events.
    """
    refined = []
    
    for start, end in segments:
        segment_data = data[start:end]
        
        if len(segment_data) < 2 * min_samples:
            refined.append((start, end))
            continue
        
        # Apply ruptures within segment
        try:
            data_2d = segment_data.reshape(-1, 1)
            algo = rpt.Pelt(model='l2', min_size=min_samples // 2).fit(data_2d)
            internal_cps = algo.predict(pen=2.0)
            
            if internal_cps and internal_cps[-1] == len(segment_data):
                internal_cps = internal_cps[:-1]
            
            if internal_cps:
                sub_start = 0
                for cp in internal_cps:
                    if cp - sub_start >= min_samples:
                        refined.append((start + sub_start, start + cp))
                        sub_start = cp
                
                if len(segment_data) - sub_start >= min_samples:
                    refined.append((start + sub_start, end))
            else:
                refined.append((start, end))
        except:
            refined.append((start, end))
    
    return refined


def _refine_event_boundaries(
    data: np.ndarray,
    segments: List[Tuple[int, int]],
    rms_envelope: np.ndarray,
    min_samples: int
) -> List[Tuple[int, int]]:
    """
    Refine event boundaries to better align with amplitude changes.
    
    Adjusts segment start/end points to local minima in RMS envelope
    for cleaner event separation.
    """
    refined = []
    search_window = min(min_samples // 2, 50)  # Search within this range
    
    for start, end in segments:
        new_start = start
        new_end = end
        
        # Refine start boundary
        if start > search_window:
            search_region = rms_envelope[start - search_window:start + search_window]
            local_min_idx = np.argmin(search_region)
            new_start = start - search_window + local_min_idx
        
        # Refine end boundary
        if end + search_window < len(rms_envelope):
            search_region = rms_envelope[end - search_window:end + search_window]
            local_min_idx = np.argmin(search_region)
            new_end = end - search_window + local_min_idx
        
        # Ensure minimum duration is maintained
        if new_end - new_start >= min_samples:
            refined.append((new_start, new_end))
        else:
            refined.append((start, end))  # Keep original if refinement violates constraint
    
    return refined


def _merge_similar_events(
    data: np.ndarray,
    segments: List[Tuple[int, int]],
    rms_envelope: np.ndarray,
    fs: float,
    min_duration: float
) -> List[Tuple[int, int]]:
    """
    Merge adjacent events that are likely part of the same activity.
    
    Criteria for merging:
    1. Short gap between events (< 200ms)
    2. Similar amplitude levels
    3. Gap amplitude is significant (not complete rest)
    """
    if len(segments) <= 1:
        return segments
    
    merged = []
    i = 0
    max_gap_samples = int(fs * 0.2)  # 200ms max gap for merging
    
    while i < len(segments):
        current_start, current_end = segments[i]
        
        # Try to merge with next segment
        while i + 1 < len(segments):
            next_start, next_end = segments[i + 1]
            gap = next_start - current_end
            
            # Check if should merge
            should_merge = False
            
            if gap <= max_gap_samples:
                # Gap is small
                current_rms = np.mean(rms_envelope[current_start:current_end])
                next_rms = np.mean(rms_envelope[next_start:next_end])
                gap_rms = np.mean(rms_envelope[current_end:next_start])
                
                # Merge if amplitudes are similar and gap is not complete rest
                amplitude_similar = abs(current_rms - next_rms) / (current_rms + 1e-10) < 0.5
                gap_significant = gap_rms > 0.3 * min(current_rms, next_rms)
                
                if amplitude_similar and gap_significant:
                    should_merge = True
            
            if should_merge:
                # Merge by extending current segment
                current_end = next_end
                i += 1
            else:
                break
        
        merged.append((current_start, current_end))
        i += 1
    
    return merged
