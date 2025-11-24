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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def detect_muscle_activity(
    data: np.ndarray,
    fs: float,
    method: str = "multi_feature",
    amplitude_threshold: Optional[float] = None,
    window_size: Optional[int] = None,
    min_duration: float = 0.1,
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
    """
    if method == "ruptures":
        return _detect_ruptures(data, fs, **kwargs)
    elif method == "amplitude":
        return _detect_amplitude(data, fs, amplitude_threshold, window_size, min_duration)
    elif method == "combined":
        return _detect_combined(data, fs, amplitude_threshold, window_size, min_duration, **kwargs)
    elif method == "multi_feature":
        return _detect_multi_feature(data, fs, window_size, min_duration, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ruptures', 'amplitude', 'combined', or 'multi_feature'")


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


def _detect_amplitude(
    data: np.ndarray,
    fs: float,
    threshold: Optional[float] = None,
    window_size: Optional[int] = None,
    min_duration: float = 0.1
) -> List[Tuple[int, int]]:
    """
    Detect muscle activity based on amplitude threshold.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    threshold : float, optional
        Amplitude threshold (default: 2 * RMS of signal)
    window_size : int, optional
        Window size for envelope calculation
    min_duration : float
        Minimum activity duration in seconds
    
    Returns:
    --------
    List[Tuple[int, int]]
        Detected activity periods
    """
    if window_size is None:
        window_size = int(fs / 10)  # Default: 0.1 seconds
    
    # Calculate envelope using RMS (Root Mean Square)
    envelope = _calculate_rms_envelope(data, window_size)
    
    # Auto-calculate threshold if not provided
    if threshold is None:
        # Use 2 times the RMS value of the signal as threshold
        # This is more robust than using standard deviation
        threshold = 2.0 * np.sqrt(np.mean(data ** 2))
    
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
    **ruptures_kwargs
) -> List[Tuple[int, int]]:
    """
    Detect muscle activity using combined ruptures and amplitude approach.
    
    This method first uses ruptures to detect potential change points,
    then uses amplitude analysis to refine the detection.
    
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
        Minimum activity duration in seconds
    **ruptures_kwargs : dict
        Additional arguments for ruptures
    
    Returns:
    --------
    List[Tuple[int, int]]
        Detected activity periods
    """
    # Get segments from both methods
    ruptures_segments = _detect_ruptures(data, fs, **ruptures_kwargs)
    amplitude_segments = _detect_amplitude(data, fs, amplitude_threshold, window_size, min_duration)
    
    # Combine segments by finding overlaps
    combined_segments = []
    
    for r_start, r_end in ruptures_segments:
        for a_start, a_end in amplitude_segments:
            # Check for overlap
            overlap_start = max(r_start, a_start)
            overlap_end = min(r_end, a_end)
            
            if overlap_start < overlap_end:
                # There is overlap, merge the segments
                merged_start = min(r_start, a_start)
                merged_end = max(r_end, a_end)
                combined_segments.append((merged_start, merged_end))
    
    # Merge overlapping segments in combined list
    if combined_segments:
        combined_segments = _merge_overlapping_segments(combined_segments)
    
    # If no combined segments found, use amplitude detection as fallback
    if not combined_segments:
        combined_segments = amplitude_segments
    
    return combined_segments


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
    
    # Determine penalty parameter
    if adaptive_pen:
        pen = _calculate_adaptive_penalty(features_normalized, fs)
    else:
        pen = ruptures_kwargs.get('pen', 3)
    
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
    Calculate sliding window variance of the signal.
    
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
    variance = np.zeros(len(data))
    half_window = window_size // 2
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window_data = data[start:end]
        variance[i] = np.var(window_data)
    
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
