"""
Muscle activity detection and signal segmentation.

This module implements muscle activity event detection using:
- Ruptures library for change point detection
- Amplitude-based analysis
- Combined ruptures and amplitude approach for robust detection
"""

import numpy as np
import ruptures as rpt
from typing import List, Tuple, Optional, Dict


def detect_muscle_activity(
    data: np.ndarray,
    fs: float,
    method: str = "combined",
    amplitude_threshold: Optional[float] = None,
    window_size: Optional[int] = None,
    min_duration: float = 0.1,
    **kwargs
) -> List[Tuple[int, int]]:
    """
    Detect muscle activity events in sEMG signal.
    
    This function combines ruptures change point detection with amplitude analysis
    to identify periods of muscle activity.
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data (1D array), should be preprocessed (filtered)
    fs : float
        Sampling frequency in Hz
    method : str, optional
        Detection method: 'ruptures', 'amplitude', or 'combined' (default: 'combined')
    amplitude_threshold : float, optional
        Threshold for amplitude-based detection (default: auto-calculated as 2*RMS)
    window_size : int, optional
        Window size for amplitude envelope calculation (default: fs/10)
    min_duration : float, optional
        Minimum duration of muscle activity in seconds (default: 0.1)
    **kwargs : dict
        Additional arguments for ruptures detection:
        - model: str, model for ruptures ('l1', 'l2', 'rbf', 'normal') default: 'l2'
        - pen: float, penalty value for ruptures (default: 3)
        - min_size: int, minimum segment size for ruptures (default: fs/10)
    
    Returns:
    --------
    List[Tuple[int, int]]
        List of (start_index, end_index) tuples for each detected activity period
        
    Notes:
    ------
    - 'combined' method is recommended for robust detection
    - Data should be preprocessed (filtered) before detection
    - Adjust amplitude_threshold based on signal characteristics
    """
    if method == "ruptures":
        return _detect_ruptures(data, fs, **kwargs)
    elif method == "amplitude":
        return _detect_amplitude(data, fs, amplitude_threshold, window_size, min_duration)
    elif method == "combined":
        return _detect_combined(data, fs, amplitude_threshold, window_size, min_duration, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ruptures', 'amplitude', or 'combined'")


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
