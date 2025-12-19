"""
Muscle activity detection and signal segmentation.

This module implements muscle activity event detection using:
- Advanced PELT (Pruned Exact Linear Time) algorithm with adaptive penalty
- Multi-dimensional feature vectors (time-domain, frequency-domain, complexity)
- Energy-based penalty zones for improved detection
- Multi-detector ensemble with voting/fusion mechanisms
- HHT-based detection using Hilbert spectrum analysis
"""

import numpy as np
import ruptures as rpt
from typing import List, Tuple, Optional, Dict, Union
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import entropy
from . import hht as hht_module

# HHT detection constants
HHT_MIN_TIME_BINS = 128  # Minimum time bins for HHT resolution
HHT_MAX_TIME_BINS = 2048  # Maximum time bins for HHT resolution
HHT_ADAPTIVE_THRESHOLD_FACTOR = 0.3  # Factor for adaptive energy threshold - lower values increase detection sensitivity
HHT_MERGE_GAP_MS = 50  # Gap in milliseconds for merging nearby segments

# HHT algorithm parameters - thresholds and bounds
HHT_MIN_ENERGY_THRESHOLD = 0.3  # Minimum allowed adjusted energy threshold (percentile)
HHT_MAX_ENERGY_THRESHOLD = 0.95  # Maximum allowed adjusted energy threshold (percentile)
HHT_NOISE_FLOOR_PERCENTILE = 5  # Percentile for minimum adaptive threshold (noise floor)
HHT_MAX_THRESHOLD_PERCENTILE = 60  # Percentile for maximum adaptive threshold
HHT_MIN_COMPACTNESS = 0.1  # Minimum allowed temporal compactness
HHT_MAX_COMPACTNESS = 0.8  # Maximum allowed temporal compactness
HHT_LOCAL_WINDOW_MIN_SIZE = 5  # Minimum size for local contrast window
HHT_LOCAL_WINDOW_FRACTION = 20  # Fraction of signal for local context (1/20 = 5%)
HHT_RMS_WINDOW_DIVISOR = 10  # Divisor for RMS window calculation (fs/10 = 100ms window)


def apply_tkeo(signal: np.ndarray) -> np.ndarray:
    """
    Apply Teager-Kaiser Energy Operator (TKEO) to enhance signal for changepoint detection.
    
    The TKEO is a nonlinear operator that emphasizes high-frequency, high-amplitude components
    of the signal, making it particularly effective for detecting muscle activity transitions.
    
    Formula: TKEO(x[n]) = x[n]² - x[n-1] × x[n+1]
    
    This operator is highly sensitive to instantaneous changes in both amplitude and frequency,
    making it superior to simple amplitude-based methods for detecting muscle activity onsets.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input sEMG signal (1D array)
    
    Returns:
    --------
    np.ndarray
        TKEO-transformed signal with enhanced transitions
    
    References:
    -----------
    - Li et al. (2007) "Teager–Kaiser energy operation of surface EMG improves 
      muscle activity onset detection" Ann Biomed Eng 35(9):1532–1538
    - Solnik et al. (2010) "Teager-Kaiser energy operator signal conditioning 
      improves EMG onset detection" Eur J Appl Physiol 110(3):489-498
    """
    n = len(signal)
    tkeo_signal = np.zeros(n)
    
    # Apply TKEO formula: x[n]² - x[n-1] × x[n+1]
    # Handle boundaries by using edge values
    for i in range(1, n - 1):
        tkeo_signal[i] = signal[i] ** 2 - signal[i - 1] * signal[i + 1]
    
    # Handle boundary conditions
    # For first sample, use forward difference approximation
    tkeo_signal[0] = signal[0] ** 2 - signal[0] * signal[1]
    # For last sample, use backward difference approximation
    tkeo_signal[-1] = signal[-1] ** 2 - signal[-2] * signal[-1]
    
    # Take absolute value to ensure positive energy values
    tkeo_signal = np.abs(tkeo_signal)
    
    return tkeo_signal


def detect_muscle_activity(
    data: np.ndarray,
    fs: float,
    method: str = "combined",
    amplitude_threshold: Optional[float] = None,
    window_size: Optional[int] = None,
    min_duration: float = 0.1,
    max_duration: Optional[float] = None,
    **kwargs
) -> Union[List[Tuple[int, int]], Dict[str, Union[List[Tuple[int, int]], List[int]]]]:
    """
    Detect muscle activity events in sEMG signal using advanced PELT algorithm.
    
    This function uses an enhanced PELT (Pruned Exact Linear Time) algorithm with:
    - Energy-based adaptive penalty strategy
    - Multi-dimensional feature vectors (time, frequency, complexity domains)
    - Multi-detector ensemble with voting/fusion mechanisms
    - Activity vs non-activity classification after segmentation
    - Intelligent event merging for dense events (gaps < 50ms)
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data (1D array), should be preprocessed (filtered)
    fs : float
        Sampling frequency in Hz
    method : str, optional
        Detection method: only 'combined' is supported (default: 'combined')
        Note: Other methods have been deprecated
    amplitude_threshold : float, optional
        Not used in new PELT implementation (kept for API compatibility)
    window_size : int, optional
        Window size for feature calculation (default: fs/10)
    min_duration : float, optional
        Minimum duration of muscle activity in seconds (default: 0.1)
        This is STRICTLY enforced - no segment shorter than this will be produced
    max_duration : float, optional
        Maximum duration of muscle activity in seconds (default: None = no limit)
        Long segments exceeding this duration will be split
    **kwargs : dict
        Additional arguments:
        - sensitivity: float or list - Detection sensitivity, directly affects penalty
          Single value (0.1-5.0) or list of values for each detector
        - n_detectors: int (1-5) - Number of parallel PELT detectors (default: 3)
        - detector_sensitivities: list - Individual sensitivities for each detector (overrides sensitivity)
        - fusion_method: str ('voting', 'confidence', 'union') - How to combine detector outputs
        - use_multi_detector: bool - Enable multi-detector ensemble (default: True)
        - classify_segments: bool - Enable activity/non-activity classification (default: True)
        - use_clustering: bool - Use clustering for classification instead of threshold (default: False)
        - classification_threshold: float - Controls classification strictness (default: 0.5)
          Negative values = very lenient (includes low-intensity events)
          0.0 = median threshold (50% of segments)
          Positive values = more strict (fewer segments)
          Range: -2.0 to 2.0, where 0.5 is balanced
          Can be negative to handle cases where high-amplitude bursts skew the mean
        - use_tkeo: bool - Apply TKEO preprocessing for improved changepoint detection (default: True)
        - merge_threshold: float - Energy ratio threshold for segment merging (default: 0.7)
          Range: 0.3-0.9, where lower = more aggressive merging (extended range for robustness)
        - max_merge_count: int - Maximum number of PELT segments to merge into one event (default: 3)
          Prevents merging of truly independent actions
        - return_changepoints: bool - If True, return dict with segments and changepoints (default: False)
    
    Returns:
    --------
    List[Tuple[int, int]] or Dict
        If return_changepoints=False (default):
            List of (start_index, end_index) tuples for each detected ACTIVE muscle period
        If return_changepoints=True:
            Dict with keys:
                'segments': List of (start_index, end_index) tuples for active segments
                'changepoints': List of all PELT-detected change point indices
        ALL segments are guaranteed to satisfy min_duration and max_duration constraints
        Only segments classified as "activity" are returned (if classify_segments=True)
        
    Notes:
    ------
    - Only 'combined' method is now supported (uses advanced PELT algorithm)
    - Data should be preprocessed (filtered) before detection
    - Sensitivity parameter directly controls PELT penalty for better interpretability
    - Multi-detector ensemble provides more robust detection
    - After PELT segmentation, segments are classified as active/inactive
    - classification_threshold allows control over classification strictness
    - Dense events with gaps < 50ms are automatically merged
    - TKEO (Teager-Kaiser Energy Operator) is applied by default for better changepoint detection
    """
    if method != "combined":
        raise ValueError(f"Only 'combined' method is supported. Other methods have been deprecated.")
    
    # Extract parameters
    sensitivity = kwargs.get('sensitivity', 1.5)
    n_detectors = kwargs.get('n_detectors', 3)
    detector_sensitivities = kwargs.get('detector_sensitivities', None)
    fusion_method = kwargs.get('fusion_method', 'confidence')
    use_multi_detector = kwargs.get('use_multi_detector', True)
    classify_segments = kwargs.get('classify_segments', True)
    use_clustering = kwargs.get('use_clustering', False)
    classification_threshold = kwargs.get('classification_threshold', 0.5)
    return_changepoints = kwargs.get('return_changepoints', False)
    use_tkeo = kwargs.get('use_tkeo', True)
    merge_threshold = kwargs.get('merge_threshold', 0.7)
    max_merge_count = kwargs.get('max_merge_count', 3)
    
    # Use new advanced PELT detection
    segments, changepoints = _detect_pelt_advanced(
        data, fs, window_size, min_duration, 
        sensitivity, n_detectors, detector_sensitivities, fusion_method, use_multi_detector, use_tkeo, merge_threshold, max_merge_count
    )
    
    # Apply max_duration splitting if specified
    if max_duration is not None and max_duration > min_duration:
        segments = _split_long_segments(data, segments, fs, min_duration, max_duration)
    
    # Classify segments as activity vs non-activity
    if classify_segments and segments:
        segments = _classify_activity_segments(
            data, segments, fs, window_size, use_clustering, classification_threshold
        )
    
    # Return results based on return_changepoints flag
    if return_changepoints:
        return {
            'segments': segments,
            'changepoints': changepoints
        }
    else:
        return segments


def _detect_pelt_advanced(
    data: np.ndarray,
    fs: float,
    window_size: Optional[int],
    min_duration: float,
    sensitivity: Union[float, List[float]],
    n_detectors: int,
    detector_sensitivities: Optional[List[float]],
    fusion_method: str,
    use_multi_detector: bool,
    use_tkeo: bool = True,
    merge_threshold: float = 0.7,
    max_merge_count: int = 3
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Advanced PELT-based muscle activity detection with multi-detector ensemble.
    
    This implements the new detection algorithm with:
    1. Optional TKEO preprocessing for enhanced changepoint detection
    2. Energy-based adaptive penalty zones
    3. Multi-dimensional feature vectors
    4. Multiple PELT detectors with individual sensitivities
    5. Voting/confidence-weighted fusion
    6. Intelligent event merging with energy-aware boundary evaluation
    7. Limit on number of merged segments to prevent merging independent actions
    8. Strict min/max duration enforcement
    
    Parameters:
    -----------
    data : np.ndarray
        Input preprocessed sEMG signal (original signal for display)
    fs : float
        Sampling frequency
    window_size : int, optional
        Window size for feature extraction
    min_duration : float
        Minimum event duration (seconds) - STRICTLY enforced
    sensitivity : float or list
        Detection sensitivity (0.1-5.0) - directly affects penalty
        Can be single value or list for backward compatibility
    n_detectors : int
        Number of parallel detectors
    detector_sensitivities : list, optional
        Individual sensitivities for each detector (overrides automatic range)
    fusion_method : str
        Method to combine detectors: 'voting', 'confidence', 'union'
    use_multi_detector : bool
        Whether to use multi-detector ensemble
    use_tkeo : bool
        Whether to apply TKEO preprocessing for changepoint detection
    merge_threshold : float
        Energy ratio threshold for segment merging (default: 0.7)
    max_merge_count : int
        Maximum number of PELT segments to merge into one event (default: 3)
    
    Returns:
    --------
    Tuple[List[Tuple[int, int]], List[int]]
        - Detected event segments (before activity classification)
        - All PELT-detected change point indices
    """
    if window_size is None:
        window_size = int(fs / 10)  # Default: 0.1 seconds
    
    min_samples = int(min_duration * fs)
    
    # Apply TKEO preprocessing if enabled (for changepoint detection only)
    if use_tkeo:
        # Apply TKEO to enhance transitions for better changepoint detection
        tkeo_data = apply_tkeo(data)
        # Smooth TKEO output to reduce noise
        kernel_size = max(3, int(fs / 200))  # 5ms smoothing window
        kernel = np.ones(kernel_size) / kernel_size
        tkeo_data = np.convolve(tkeo_data, kernel, mode='same')
        # Use TKEO-enhanced signal for feature extraction
        detection_signal = tkeo_data
    else:
        # Use original signal for detection
        detection_signal = data
    
    # Step 1: Extract multi-dimensional features (optimized)
    # Note: We use TKEO-enhanced signal for feature extraction if enabled
    features = _extract_multidimensional_features_fast(detection_signal, fs, window_size)
    
    # Step 2: Compute energy zones for adaptive penalty
    # Use original signal for energy zones to maintain energy-based context
    energy_zones = _compute_energy_zones(data, window_size)
    
    all_changepoints = []
    
    if use_multi_detector and n_detectors > 1:
        # Step 3: Run multiple PELT detectors with individual sensitivities
        all_detections = []
        detector_confidences = []
        
        # Determine sensitivity for each detector
        if detector_sensitivities is not None and len(detector_sensitivities) == n_detectors:
            # Use user-provided individual sensitivities
            sensitivity_range = detector_sensitivities
        elif isinstance(sensitivity, list) and len(sensitivity) == n_detectors:
            # sensitivity is already a list
            sensitivity_range = sensitivity
        else:
            # Create sensitivity range around base sensitivity (backward compatibility)
            base_sens = sensitivity if isinstance(sensitivity, (int, float)) else sensitivity[0]
            sensitivity_range = np.linspace(base_sens * 0.7, base_sens * 1.3, n_detectors)
        
        for i, det_sensitivity in enumerate(sensitivity_range):
            # Compute adaptive penalties for this detector
            penalties = _compute_adaptive_penalties(
                features, energy_zones, det_sensitivity
            )
            
            # Run PELT with zone-specific penalties
            segments, changepoints = _run_pelt_with_adaptive_penalty(
                features, penalties, min_samples
            )
            
            # Collect all changepoints
            all_changepoints.extend(changepoints)
            
            # Calculate confidence scores for each segment
            confidences = [_calculate_segment_confidence(data, seg, fs) for seg in segments]
            
            all_detections.append(segments)
            detector_confidences.append(confidences)
        
        # Step 4: Fuse detections using selected method
        segments = _fuse_detections(
            all_detections, detector_confidences, fusion_method, min_samples
        )
    else:
        # Single detector mode
        single_sens = sensitivity if isinstance(sensitivity, (int, float)) else sensitivity[0]
        penalties = _compute_adaptive_penalties(features, energy_zones, single_sens)
        segments, changepoints = _run_pelt_with_adaptive_penalty(features, penalties, min_samples)
        all_changepoints.extend(changepoints)
    
    # Remove duplicate changepoints and sort
    all_changepoints = sorted(list(set(all_changepoints)))
    
    # Step 5: Merge dense events (gaps < 50ms) with adaptive threshold and merge limit
    segments = _merge_dense_events(data, segments, fs, min_samples, merge_threshold, max_merge_count)
    
    # Step 6: Final strict enforcement of duration constraints
    segments = [(s, e) for s, e in segments if (e - s) >= min_samples]
    
    return segments, all_changepoints


def _extract_multidimensional_features(
    data: np.ndarray,
    fs: float,
    window_size: int
) -> np.ndarray:
    """
    Extract multi-dimensional feature vectors including time-domain,
    frequency-domain, and complexity features for PELT algorithm.
    
    Features extracted:
    1. Time-domain: RMS, MAV, VAR, WL (Waveform Length)
    2. Frequency-domain: MNF (Mean Frequency), MDF (Median Frequency)
    3. Complexity: Sample Entropy, Zero Crossing Rate
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    window_size : int
        Window size for feature extraction
    
    Returns:
    --------
    np.ndarray
        Feature matrix of shape (n_samples, n_features)
    """
    n_samples = len(data)
    kernel = np.ones(window_size) / window_size
    
    # Time-domain features
    # 1. RMS (Root Mean Square)
    rms = np.sqrt(np.convolve(data ** 2, kernel, mode='same'))
    
    # 2. MAV (Mean Absolute Value)
    mav = np.convolve(np.abs(data), kernel, mode='same')
    
    # 3. VAR (Variance)
    mean = np.convolve(data, kernel, mode='same')
    var = np.convolve(data ** 2, kernel, mode='same') - mean ** 2
    var = np.maximum(var, 0)  # Handle numerical errors
    
    # 4. WL (Waveform Length) - approximated by gradient magnitude
    wl = np.abs(np.gradient(data))
    wl = np.convolve(wl, kernel, mode='same')
    
    # Frequency-domain features (calculated over sliding windows)
    # 5. MNF (Mean Frequency) and 6. MDF (Median Frequency)
    mnf, mdf = _calculate_frequency_features(data, fs, window_size)
    
    # Complexity features
    # 7. Zero Crossing Rate
    zcr = _calculate_zero_crossing_rate(data, window_size)
    
    # 8. Sample Entropy (approximation using local variance ratio)
    # True sample entropy is computationally expensive, so we use a proxy
    sampen_proxy = np.log1p(var) / (np.log1p(rms) + 1e-10)
    
    # Stack all features
    features = np.column_stack([rms, mav, var, wl, mnf, mdf, zcr, sampen_proxy])
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    return features_normalized


def _calculate_frequency_features(
    data: np.ndarray,
    fs: float,
    window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate frequency-domain features: MNF (Mean Frequency) and MDF (Median Frequency).
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    window_size : int
        Window size
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (MNF array, MDF array)
    """
    n_samples = len(data)
    step = window_size // 4  # 75% overlap
    
    mnf = np.zeros(n_samples)
    mdf = np.zeros(n_samples)
    
    # Calculate for each window
    for i in range(0, n_samples, step):
        start = max(0, i - window_size // 2)
        end = min(n_samples, i + window_size // 2)
        
        if end - start < window_size // 2:
            continue
        
        window_data = data[start:end]
        
        # Compute power spectral density
        freqs, psd = scipy_signal.welch(window_data, fs=fs, nperseg=min(256, len(window_data)))
        
        # Mean frequency
        mnf_val = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        
        # Median frequency
        cumsum = np.cumsum(psd)
        cumsum_norm = cumsum / (cumsum[-1] + 1e-10)
        mdf_idx = np.argmin(np.abs(cumsum_norm - 0.5))
        mdf_val = freqs[mdf_idx]
        
        # Assign to output arrays
        mnf[start:end] = mnf_val
        mdf[start:end] = mdf_val
    
    return mnf, mdf


def _extract_multidimensional_features_fast(
    data: np.ndarray,
    fs: float,
    window_size: int
) -> np.ndarray:
    """
    Optimized version of multi-dimensional feature extraction.
    
    Optimizations:
    - Reduced frequency feature calculation (less overlap, coarser resolution)
    - Simplified complexity features
    - Cached convolution operations
    
    Features extracted:
    1. Time-domain: RMS, MAV, VAR
    2. Frequency-domain: MNF (simplified)
    3. Complexity: ZCR
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    window_size : int
        Window size for feature extraction
    
    Returns:
    --------
    np.ndarray
        Feature matrix of shape (n_samples, 5 features)
    """
    n_samples = len(data)
    kernel = np.ones(window_size) / window_size
    
    # Time-domain features (fast)
    # 1. RMS (Root Mean Square)
    rms = np.sqrt(np.convolve(data ** 2, kernel, mode='same'))
    
    # 2. MAV (Mean Absolute Value)
    mav = np.convolve(np.abs(data), kernel, mode='same')
    
    # 3. VAR (Variance)
    mean = np.convolve(data, kernel, mode='same')
    var = np.convolve(data ** 2, kernel, mode='same') - mean ** 2
    var = np.maximum(var, 0)  # Handle numerical errors
    
    # Frequency-domain features (optimized - less frequent calculation)
    # 4. MNF (Mean Frequency) - simplified
    mnf = _calculate_frequency_features_fast(data, fs, window_size)
    
    # Complexity features (simplified)
    # 5. Zero Crossing Rate
    zcr = _calculate_zero_crossing_rate(data, window_size)
    
    # Stack all features
    features = np.column_stack([rms, mav, var, mnf, zcr])
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    return features_normalized


def _calculate_frequency_features_fast(
    data: np.ndarray,
    fs: float,
    window_size: int
) -> np.ndarray:
    """
    Fast calculation of mean frequency feature with reduced temporal resolution.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    window_size : int
        Window size
    
    Returns:
    --------
    np.ndarray
        MNF array
    """
    n_samples = len(data)
    step = window_size  # No overlap for speed
    
    mnf = np.zeros(n_samples)
    
    # Calculate for each window with larger steps
    for i in range(0, n_samples, step):
        end = min(n_samples, i + window_size)
        
        if end - i < window_size // 2:
            continue
        
        window_data = data[i:end]
        
        # Compute power spectral density (simplified)
        freqs, psd = scipy_signal.welch(window_data, fs=fs, nperseg=min(128, len(window_data)))
        
        # Mean frequency
        mnf_val = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        
        # Assign to output array
        mnf[i:end] = mnf_val
    
    return mnf


def _calculate_zero_crossing_rate(
    data: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Calculate zero crossing rate over sliding window.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    window_size : int
        Window size
    
    Returns:
    --------
    np.ndarray
        Zero crossing rate
    """
    # Find zero crossings
    zero_crossings = np.abs(np.diff(np.sign(data))) > 0
    
    # Count in sliding window
    kernel = np.ones(window_size)
    zcr = np.convolve(zero_crossings.astype(float), kernel, mode='same')
    
    # Normalize by window size
    zcr = zcr / window_size
    
    # Pad to match input length
    zcr = np.concatenate([[zcr[0]], zcr])
    
    return zcr


def _compute_energy_zones(
    data: np.ndarray,
    window_size: int,
    n_zones: int = 3
) -> np.ndarray:
    """
    Compute energy-based zones for adaptive penalty strategy.
    
    Divides signal into zones based on local energy:
    - Low energy zones: use low penalty (more sensitive)
    - Medium energy zones: use medium penalty
    - High energy zones: use high penalty (less sensitive)
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    window_size : int
        Window size for energy calculation
    n_zones : int
        Number of energy zones (default: 3)
    
    Returns:
    --------
    np.ndarray
        Zone labels (0 to n_zones-1) for each sample
    """
    # Calculate local energy
    energy = data ** 2
    kernel = np.ones(window_size) / window_size
    local_energy = np.convolve(energy, kernel, mode='same')
    
    # Use K-means clustering to identify energy zones
    energy_2d = local_energy.reshape(-1, 1)
    
    try:
        kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init='auto')
    except TypeError:
        kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
    
    zones = kmeans.fit_predict(energy_2d)
    
    # Reorder zones so that 0 = low energy, n_zones-1 = high energy
    cluster_means = np.array([local_energy[zones == i].mean() for i in range(n_zones)])
    zone_order = np.argsort(cluster_means)
    
    # Remap zones
    zone_map = {old_label: new_label for new_label, old_label in enumerate(zone_order)}
    zones_reordered = np.array([zone_map[z] for z in zones])
    
    return zones_reordered


def _compute_adaptive_penalties(
    features: np.ndarray,
    energy_zones: np.ndarray,
    sensitivity: float
) -> np.ndarray:
    """
    Compute adaptive penalty values based on energy zones and sensitivity.
    
    Low energy zones get lower penalties (more sensitive detection).
    High energy zones get higher penalties (less sensitive, avoid over-segmentation).
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    energy_zones : np.ndarray
        Energy zone labels for each sample
    sensitivity : float
        Global sensitivity parameter
    
    Returns:
    --------
    np.ndarray
        Penalty value for each sample
    """
    n_samples = len(energy_zones)
    n_zones = len(np.unique(energy_zones))
    
    # Base penalty from sensitivity (inversely proportional)
    # sensitivity: 0.1 (very sensitive) -> high base penalty modifier
    # sensitivity: 5.0 (very strict) -> low base penalty modifier
    base_penalty = 3.0 * sensitivity
    
    # Zone-specific multipliers
    # Low energy (zone 0): 0.5x base (more sensitive)
    # Medium energy (zone 1): 1.0x base
    # High energy (zone 2): 2.0x base (less sensitive)
    zone_multipliers = np.linspace(0.5, 2.0, n_zones)
    
    # Assign penalties based on zones
    penalties = np.zeros(n_samples)
    for i in range(n_zones):
        mask = energy_zones == i
        penalties[mask] = base_penalty * zone_multipliers[i]
    
    return penalties


def _run_pelt_with_adaptive_penalty(
    features: np.ndarray,
    penalties: np.ndarray,
    min_samples: int
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Run PELT algorithm with position-dependent adaptive penalties.
    
    Since ruptures doesn't directly support position-dependent penalties,
    we use a strategy of running PELT with an average penalty and then
    refining boundaries based on local penalty values.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    penalties : np.ndarray
        Penalty values for each position
    min_samples : int
        Minimum segment size
    
    Returns:
    --------
    Tuple[List[Tuple[int, int]], List[int]]
        - Detected segments
        - Change point indices detected by PELT
    """
    # Use median penalty as the base penalty for PELT
    median_penalty = np.median(penalties)
    
    # Run PELT
    algo = rpt.Pelt(model='l2', min_size=min_samples).fit(features)
    change_points = algo.predict(pen=median_penalty)
    
    # Store original changepoints for return
    original_changepoints = change_points.copy()
    
    # Remove the last point if it's the end of the signal
    if change_points and change_points[-1] == len(features):
        change_points = change_points[:-1]
    
    if not change_points:
        return [], original_changepoints
    
    # Convert change points to segments
    segments = []
    start = 0
    for cp in change_points:
        if cp - start >= min_samples:
            segments.append((start, cp))
        start = cp
    
    # Add final segment if there's remaining data
    if start < len(features) and len(features) - start >= min_samples:
        segments.append((start, len(features)))
    
    # Refine boundaries based on local penalties
    # Move boundaries to positions with locally minimal penalty (easier to split)
    refined_segments = []
    for i, (start, end) in enumerate(segments):
        # Check boundaries within a small window
        search_window = min(min_samples // 2, 50)
        
        # Refine start boundary (except for first segment)
        if i > 0 and start > search_window:
            window_start = max(0, start - search_window)
            window_end = min(len(penalties), start + search_window)
            local_penalties = penalties[window_start:window_end]
            min_idx = np.argmin(local_penalties)
            refined_start = window_start + min_idx
        else:
            refined_start = start
        
        # Refine end boundary (except for last segment)
        if i < len(segments) - 1 and end < len(penalties) - search_window:
            window_start = max(0, end - search_window)
            window_end = min(len(penalties), end + search_window)
            local_penalties = penalties[window_start:window_end]
            min_idx = np.argmin(local_penalties)
            refined_end = window_start + min_idx
        else:
            refined_end = end
        
        # Ensure minimum duration
        if refined_end - refined_start >= min_samples:
            refined_segments.append((refined_start, refined_end))
    
    return refined_segments, original_changepoints


def _calculate_segment_confidence(
    data: np.ndarray,
    segment: Tuple[int, int],
    fs: float
) -> float:
    """
    Calculate confidence score for a detected segment.
    
    Confidence is based on:
    1. Amplitude contrast with surrounding regions
    2. Internal consistency (low variance)
    3. Duration reasonableness
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal
    segment : Tuple[int, int]
        Segment boundaries
    fs : float
        Sampling frequency
    
    Returns:
    --------
    float
        Confidence score (0 to 1)
    """
    start, end = segment
    segment_length = end - start
    
    if segment_length == 0:
        return 0.0
    
    segment_data = data[start:end]
    segment_rms = np.sqrt(np.mean(segment_data ** 2))
    
    # 1. Amplitude contrast
    window = min(segment_length, int(fs * 0.5))
    
    before_rms = 0.0
    after_rms = 0.0
    count = 0
    
    if start >= window:
        before_rms = np.sqrt(np.mean(data[start - window:start] ** 2))
        count += 1
    if end + window < len(data):
        after_rms = np.sqrt(np.mean(data[end:end + window] ** 2))
        count += 1
    
    if count > 0:
        surrounding_rms = (before_rms + after_rms) / count
        contrast = (segment_rms - surrounding_rms) / (surrounding_rms + 1e-10)
        contrast_score = min(1.0, contrast / 2.0)
    else:
        contrast_score = 0.5
    
    # 2. Internal consistency
    cv = np.std(segment_data) / (segment_rms + 1e-10)
    consistency_score = 1.0 / (1.0 + cv)
    
    # 3. Duration reasonableness
    duration_seconds = segment_length / fs
    if 0.1 <= duration_seconds <= 5.0:
        duration_score = 1.0
    elif duration_seconds < 0.1:
        duration_score = duration_seconds / 0.1
    else:
        duration_score = 5.0 / duration_seconds
    
    # Weighted combination
    confidence = 0.5 * contrast_score + 0.3 * consistency_score + 0.2 * duration_score
    
    return confidence


def _fuse_detections(
    all_detections: List[List[Tuple[int, int]]],
    detector_confidences: List[List[float]],
    fusion_method: str,
    min_samples: int
) -> List[Tuple[int, int]]:
    """
    Fuse detections from multiple PELT detectors using voting or confidence weighting.
    
    Parameters:
    -----------
    all_detections : List[List[Tuple[int, int]]]
        Detections from each detector
    detector_confidences : List[List[float]]
        Confidence scores for each detection
    fusion_method : str
        'voting': majority vote, 'confidence': confidence-weighted, 'union': combine all
    min_samples : int
        Minimum segment size
    
    Returns:
    --------
    List[Tuple[int, int]]
        Fused segments
    """
    if fusion_method == 'union':
        # Simply combine all detections and remove overlaps
        all_segments = []
        for detections in all_detections:
            all_segments.extend(detections)
        return _merge_overlapping_segments(all_segments)
    
    elif fusion_method == 'voting':
        # Use majority voting: keep segments detected by at least n/2 detectors
        n_detectors = len(all_detections)
        threshold = n_detectors // 2 + 1
        
        # Create a voting map
        if not all_detections or not all_detections[0]:
            return []
        
        signal_length = max(end for detections in all_detections for _, end in detections) if any(all_detections) else 0
        if signal_length == 0:
            return []
        
        vote_map = np.zeros(signal_length, dtype=int)
        
        for detections in all_detections:
            for start, end in detections:
                vote_map[start:end] += 1
        
        # Find regions with enough votes
        above_threshold = vote_map >= threshold
        
        # Find transitions
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(above_threshold)]])
        
        segments = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_samples]
        return segments
    
    elif fusion_method == 'confidence':
        # Confidence-weighted fusion: weight each detector's contribution by confidence
        if not all_detections or not all_detections[0]:
            return []
        
        signal_length = max(end for detections in all_detections for _, end in detections) if any(all_detections) else 0
        if signal_length == 0:
            return []
        
        confidence_map = np.zeros(signal_length, dtype=float)
        
        for detections, confidences in zip(all_detections, detector_confidences):
            for (start, end), conf in zip(detections, confidences):
                confidence_map[start:end] += conf
        
        # Normalize by number of detectors
        confidence_map /= len(all_detections)
        
        # Use adaptive threshold based on confidence distribution
        threshold = np.percentile(confidence_map[confidence_map > 0], 50) if np.any(confidence_map > 0) else 0.5
        
        # Find regions above threshold
        above_threshold = confidence_map >= threshold
        
        # Find transitions
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(above_threshold)]])
        
        segments = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_samples]
        return segments
    
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")


def _classify_activity_segments(
    data: np.ndarray,
    segments: List[Tuple[int, int]],
    fs: float,
    window_size: Optional[int],
    use_clustering: bool = False,
    classification_threshold: float = 0.5
) -> List[Tuple[int, int]]:
    """
    Classify segments as activity or non-activity based on signal features.
    
    After PELT detection creates segments at change points, this function
    determines which segments represent actual muscle activity vs rest periods.
    Only segments classified as "activity" are returned.
    
    Classification methods:
    1. Adaptive threshold (default): Uses time and frequency domain features
    2. K-means clustering: Automatically separates activity from rest
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal
    segments : List[Tuple[int, int]]
        All detected segments from PELT
    fs : float
        Sampling frequency
    window_size : int, optional
        Window size for feature extraction
    use_clustering : bool
        If True, use K-means clustering; otherwise use adaptive threshold
    classification_threshold : float
        Controls classification strictness (default: 0.5)
        For adaptive threshold: 
          - Can be negative (e.g., -1.0) for very lenient classification
          - 0.0 uses median as threshold (50% of segments)
          - Positive values are more strict
          - Range: -2.0 to 2.0, allows handling high-amplitude bursts that skew mean
        For clustering: percentile threshold to separate clusters (0-1)
        Range: -2.0 to 2.0, where lower values are less strict
    
    Returns:
    --------
    List[Tuple[int, int]]
        Only segments classified as muscle activity
    """
    if not segments:
        return []
    
    if window_size is None:
        window_size = int(fs / 10)
    
    # Extract features for each segment
    segment_features = []
    for start, end in segments:
        segment_data = data[start:end]
        
        # Time-domain features
        rms = np.sqrt(np.mean(segment_data ** 2))
        mav = np.mean(np.abs(segment_data))
        var = np.var(segment_data)
        
        # Frequency-domain feature (simplified)
        if len(segment_data) >= 64:
            freqs, psd = scipy_signal.welch(segment_data, fs=fs, nperseg=min(128, len(segment_data)))
            mnf = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
            power = np.sum(psd)
        else:
            mnf = 0
            power = 0
        
        # Combine features
        segment_features.append([rms, mav, var, mnf, power])
    
    segment_features = np.array(segment_features)
    
    if use_clustering:
        # Method 1: K-means clustering (2 clusters: activity vs rest)
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        except TypeError:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        
        labels = kmeans.fit_predict(segment_features)
        
        # Calculate normalized feature sum for each segment
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(segment_features)
        activity_scores = features_normalized.mean(axis=1)
        
        # Identify activity cluster (higher feature values)
        cluster_means = np.array([activity_scores[labels == i].mean() 
                                  for i in range(2)])
        activity_cluster = np.argmax(cluster_means)
        
        # Apply threshold within activity cluster based on classification_threshold
        # Lower threshold = less strict (keep more segments)
        activity_cluster_scores = activity_scores[labels == activity_cluster]
        if len(activity_cluster_scores) > 0:
            # Use percentile based on threshold: 0.5 -> 50th percentile (median)
            # Lower values (e.g., 0.2) -> 20th percentile (less strict)
            # Higher values (e.g., 0.8) -> 80th percentile (more strict)
            percentile = classification_threshold * 100
            percentile = np.clip(percentile, 0, 100)
            score_threshold = np.percentile(activity_cluster_scores, percentile)
        else:
            score_threshold = cluster_means[activity_cluster]
        
        # Keep segments in activity cluster with scores above threshold
        activity_segments = [seg for seg, label, score in zip(segments, labels, activity_scores) 
                            if label == activity_cluster and score >= score_threshold]
    else:
        # Method 2: Adaptive threshold (faster, more interpretable)
        # Normalize each feature individually to ensure equal weight
        # This prevents features with larger scales from dominating
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(segment_features)
        
        # Calculate combined activity score for each segment
        # Use only intensity-related features: RMS, MAV, VAR, and Power (skip MNF at index 3)
        # Each feature is already normalized (mean=0, std=1), giving equal weight
        activity_scores = (
            features_normalized[:, 0] +  # RMS (normalized)
            features_normalized[:, 1] +  # MAV (normalized)
            features_normalized[:, 2] +  # VAR (normalized)
            features_normalized[:, 4]    # Power (normalized)
        ) / 4.0
        
        # Adaptive threshold based on score distribution
        # Use median + (std * classification_threshold) to separate activity from rest
        # classification_threshold can be negative to be less strict
        # Negative values allow segments below median to be classified as activity
        # Examples:
        #   -1.0: median - 1.0*std (very lenient, includes low-intensity events)
        #   -0.5: median - 0.5*std (lenient)
        #    0.0: median (balanced, 50% of segments)
        #    0.5: median + 0.5*std (default, moderately strict)
        #    1.0: median + 1.0*std (strict)
        threshold = np.median(activity_scores) + np.std(activity_scores) * classification_threshold
        
        # Keep segments above threshold
        activity_segments = [seg for seg, score in zip(segments, activity_scores) 
                            if score >= threshold]
    
    return activity_segments


def _merge_dense_events(
    data: np.ndarray,
    segments: List[Tuple[int, int]],
    fs: float,
    min_samples: int,
    adaptive_threshold: float = 0.7,
    max_merge_count: int = 3
) -> List[Tuple[int, int]]:
    """
    Intelligently merge adjacent events based on boundary energy state.
    
    New merging strategy for dumbbell exercise recognition:
    1. For non-adjacent segments (separated by inactive periods): Keep separate
    2. For adjacent segments (directly touching): Evaluate boundary energy state
       - If boundary is in HIGH energy state → MERGE (part of same action)
       - If boundary is in LOW energy state → KEEP SEPARATE (different actions)
    3. Limit merging to max_merge_count segments to prevent merging independent actions
    
    This addresses the issue where arm lift/lower transitions can create strong
    changepoints that split a single dumbbell action into multiple segments.
    
    The threshold is adaptive based on signal characteristics:
    - Base threshold can be adjusted via adaptive_threshold parameter
    - Automatically adjusts for signals with high variance (multiple energy levels)
    - For uniform signals, uses the base threshold directly
    - More aggressive merging in high-amplitude regions to keep peaks within events
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal (not TKEO)
    segments : List[Tuple[int, int]]
        Detected segments
    fs : float
        Sampling frequency
    min_samples : int
        Minimum segment size
    adaptive_threshold : float, optional
        Base energy ratio threshold for merging (default: 0.7)
        Lower values = more aggressive merging
        Higher values = more conservative merging
        Range: 0.3 - 0.9 recommended (extended for more aggressive merging)
    max_merge_count : int, optional
        Maximum number of original PELT segments to merge into one event (default: 3)
        Prevents merging of truly independent actions
    
    Returns:
    --------
    List[Tuple[int, int]]
        Merged segments with energy-aware logic
    """
    if len(segments) <= 1:
        return segments
    
    # Calculate RMS envelope for energy evaluation
    window_size = int(fs / 10)  # 100ms window
    kernel = np.ones(window_size) / window_size
    rms_envelope = np.sqrt(np.convolve(data ** 2, kernel, mode='same'))
    
    # Compute global energy statistics for adaptive thresholding
    signal_rms_mean = np.mean(rms_envelope)
    signal_rms_std = np.std(rms_envelope)
    
    # Calculate coefficient of variation (CV) to assess signal variability
    # CV = std / mean, measures relative variability
    if signal_rms_mean > 1e-10:
        energy_cv = signal_rms_std / signal_rms_mean
    else:
        energy_cv = 0.0
    
    # Adaptive threshold adjustment based on signal characteristics
    # More aggressive strategy for high-amplitude regions:
    # - For signals with high variability (CV > 0.5), be MORE aggressive (lower threshold)
    # - This helps keep envelope peaks within detected events
    # - For signals with low variability (CV < 0.3), use base threshold
    if energy_cv > 0.5:
        # High variability signal - be significantly more aggressive in merging
        # Use 0.8× threshold to capture more transitions within same action
        adjusted_threshold = adaptive_threshold * 0.8
    elif energy_cv < 0.3:
        # Low variability signal - use base threshold
        adjusted_threshold = adaptive_threshold
    else:
        # Medium variability - interpolate
        # Linear interpolation between 0.8× and 1.0× based on CV in [0.3, 0.5]
        interpolation_factor = 0.8 + 0.2 * (0.5 - energy_cv) / 0.2
        adjusted_threshold = adaptive_threshold * interpolation_factor
    
    merged = []
    i = 0
    
    while i < len(segments):
        current_start, current_end = segments[i]
        merge_count = 1  # Track how many segments have been merged
        
        # Try to merge with next segments (only if adjacent or nearly adjacent)
        while i + 1 < len(segments) and merge_count < max_merge_count:
            next_start, next_end = segments[i + 1]
            gap = next_start - current_end
            
            # Define adjacency threshold (segments separated by < 50ms gap)
            adjacency_threshold = int(0.05 * fs)
            
            should_merge = False
            
            if gap <= adjacency_threshold:
                # Segments are adjacent or nearly adjacent
                # Evaluate energy state at boundary
                
                # Define boundary evaluation window (around the changepoint)
                boundary_window = min(int(fs * 0.05), min_samples // 4)  # 50ms or quarter of min duration
                
                # Get boundary region energy (from both sides of the gap)
                boundary_start = max(0, current_end - boundary_window)
                boundary_end = min(len(rms_envelope), next_start + boundary_window)
                
                # If gap exists, include gap region
                if gap > 0:
                    boundary_region_energy = rms_envelope[boundary_start:boundary_end]
                else:
                    # Directly adjacent, evaluate at the junction
                    boundary_region_energy = rms_envelope[boundary_start:boundary_end]
                
                # Calculate local energy metrics
                boundary_mean_energy = np.mean(boundary_region_energy)
                
                # Get surrounding energy context for comparison
                # Before boundary
                before_window_start = max(0, boundary_start - boundary_window * 2)
                before_region = rms_envelope[before_window_start:boundary_start]
                before_energy = np.mean(before_region) if len(before_region) > 0 else 0
                
                # After boundary
                after_window_end = min(len(rms_envelope), boundary_end + boundary_window * 2)
                after_region = rms_envelope[boundary_end:after_window_end]
                after_energy = np.mean(after_region) if len(after_region) > 0 else 0
                
                # Average energy in the two segments
                segment_avg_energy = (before_energy + after_energy) / 2
                
                # Determine if boundary is HIGH or LOW energy state
                # HIGH energy: boundary energy is close to or higher than surrounding segment energy
                # LOW energy: boundary energy is significantly lower than surrounding energy
                
                # Calculate energy ratio: boundary_energy / segment_avg_energy
                if segment_avg_energy > 1e-10:
                    energy_ratio = boundary_mean_energy / segment_avg_energy
                else:
                    energy_ratio = 1.0
                
                # Adaptive threshold based on signal characteristics
                # Use the adjusted threshold calculated from global signal statistics
                high_energy_threshold = adjusted_threshold
                
                if energy_ratio >= high_energy_threshold:
                    # Boundary is in HIGH energy state
                    # This suggests it's a transition within the same activity (e.g., lift-to-lower transition)
                    # MERGE these segments
                    should_merge = True
                else:
                    # Boundary is in LOW energy state
                    # This suggests it's a true rest period or end of activity
                    # DON'T MERGE - keep as separate events
                    should_merge = False
            else:
                # Segments are NOT adjacent (gap > 50ms)
                # According to requirement: keep them separate regardless of their activity status
                should_merge = False
            
            if should_merge:
                # Merge: extend current segment to include next
                current_end = next_end
                merge_count += 1  # Increment merge counter
                i += 1
            else:
                # Don't merge: stop trying to merge with subsequent segments
                break
        
        # Add the (potentially merged) segment if it meets minimum duration
        if current_end - current_start >= min_samples:
            merged.append((current_start, current_end))
        
        i += 1
    
    return merged


# ============================================================================
# Legacy functions (kept for compatibility and helper functions)
# ============================================================================


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
    Detect muscle activity events using two-stage amplitude-first approach.
    
    **TWO-STAGE DETECTION ALGORITHM:**
    
    **Stage 1 - Event Presence Detection (Amplitude-Weighted):**
    Primary reliance on amplitude to determine IF events exist. This reduces missed
    events by using a liberal amplitude-based initial detection.
    
    **Stage 2 - Boundary Refinement (Multi-Factor):**
    Once event presence is confirmed, use ruptures and other quality metrics to
    determine precise WHERE events start and end.
    
    **Key Improvements:**
    - Amplitude gets 70% weight in presence detection (vs 35% in confidence scoring)
    - Significantly lowered detection thresholds to catch more events
    - Separate presence detection from boundary determination
    - Ensures no event overlaps
    - Strict enforcement of min/max duration constraints
    
    **Algorithm Flow:**
    1. Calculate adaptive amplitude threshold based on signal statistics
    2. Identify ALL regions exceeding amplitude threshold (candidate events)
    3. For each candidate: calculate confidence score (amplitude-weighted 70%)
    4. Accept candidates meeting lowered confidence threshold (0.25-0.40 range)
    5. Refine boundaries using ruptures and local minima for precise start/end
    6. Merge similar nearby events while respecting min_duration
    7. Final filtering ensures no overlaps and duration constraints met
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    amplitude_threshold : float, optional
        Amplitude threshold (auto-calculated if None)
    window_size : int, optional
        Window size for envelope calculation
    min_duration : float
        Minimum activity duration in seconds (HARD CONSTRAINT - strictly enforced)
        No segment shorter than this will ever be produced
    sensitivity : float
        Detection sensitivity (lower = more sensitive, default: 1.5)
        Affects amplitude threshold and confidence threshold
    **kwargs : dict
        Additional arguments for ruptures (model, pen, min_size)
    
    Returns:
    --------
    List[Tuple[int, int]]
        Detected activity events with non-overlapping segments
        ALL segments guaranteed to be >= min_duration
        
    Notes:
    ------
    This two-stage approach significantly reduces missed events by prioritizing
    amplitude for presence detection, then using multi-factor analysis for precise
    boundary determination. Duration constraints are absolute hard limits.
    """
    if window_size is None:
        window_size = int(fs / 10)
    
    min_samples = int(min_duration * fs)
    
    # Calculate RMS envelope for all stages
    rms_envelope = _calculate_rms_envelope(data, window_size)
    
    # Use adaptive threshold for amplitude-based detection (liberal for Stage 1)
    if amplitude_threshold is None:
        amplitude_threshold = _calculate_adaptive_threshold(data, rms_envelope, sensitivity)
    
    # Lower the threshold further to catch more events in Stage 1
    # This reduces missed events by being more aggressive initially
    amplitude_threshold *= 0.8  # 20% lower threshold for presence detection
    
    # ===========================================
    # === STAGE 1: EVENT PRESENCE DETECTION ===
    # ===========================================
    #
    # Goal: Identify ALL regions where muscle activity MIGHT exist
    # Strategy: Liberal amplitude-based detection with lowered thresholds
    
    # Find all regions where RMS exceeds the lowered amplitude threshold
    above_threshold = rms_envelope > amplitude_threshold
    
    # Find transitions
    diff = np.diff(above_threshold.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if above_threshold[0]:
        starts = np.concatenate([[0], starts])
    if above_threshold[-1]:
        ends = np.concatenate([ends, [len(above_threshold)]])
    
    # Generate candidate event regions (liberal, may include false positives)
    candidate_regions = []
    for start, end in zip(starts, ends):
        if end - start >= min_samples:  # Respect minimum duration
            candidate_regions.append((int(start), int(end)))
    
    if not candidate_regions:
        return []  # No activity detected
    
    # For each candidate region, calculate confidence score (amplitude-weighted 70%)
    # Lowered confidence threshold: 0.25 to 0.40 (vs previous 0.30 to 0.50)
    confidence_threshold = 0.25 + (sensitivity - 1.0) * 0.075  # Range: 0.25 to 0.40
    
    confirmed_events = []
    for region in candidate_regions:
        # Calculate confidence with HIGH weight on amplitude
        confidence = _calculate_event_presence_confidence(
            data, region, rms_envelope, fs, amplitude_threshold
        )
        
        if confidence >= confidence_threshold:
            confirmed_events.append(region)
    
    if not confirmed_events:
        return []  # No confident events found
    
    # ===============================================
    # === STAGE 2: BOUNDARY REFINEMENT ===
    # ===============================================
    #
    # Goal: Determine precise start/end points for confirmed events
    # Strategy: Use ruptures and quality metrics to refine boundaries
    
    # Prepare ruptures parameters
    valid_ruptures_params = {'model', 'pen', 'min_size'}
    ruptures_kwargs = {k: v for k, v in kwargs.items() if k in valid_ruptures_params}
    
    if 'pen' not in ruptures_kwargs:
        ruptures_kwargs['pen'] = 3.0 * sensitivity / 2.0
    
    refined_events = []
    
    for event in confirmed_events:
        start, end = event
        event_data = data[start:end]
        event_rms = rms_envelope[start:end]
        
        # Use ruptures to find internal structure and refine boundaries
        try:
            # Apply ruptures within this event region
            algo = rpt.Pelt(**ruptures_kwargs)
            algo.fit(event_data)
            internal_breaks = algo.predict(pen=ruptures_kwargs.get('pen', 3.0))
            
            # Use breaks to find the most coherent sub-segment
            # Or refine boundaries to local minima
            refined_start, refined_end = _refine_event_boundaries_with_ruptures(
                event_data, event_rms, internal_breaks, start, end, min_samples
            )
            
            refined_events.append((refined_start, refined_end))
        except:
            # If ruptures fails, use local minima to refine boundaries
            refined_start = start + np.argmin(event_rms[:len(event_rms)//4])  # First quarter
            refined_end = start + len(event_rms)//4*3 + np.argmin(event_rms[len(event_rms)//4*3:])  # Last quarter
            
            if refined_end - refined_start >= min_samples:
                refined_events.append((refined_start, refined_end))
            else:
                refined_events.append((start, end))  # Keep original if refinement too short
    
    # Merge nearby similar events while respecting min_duration
    merged_events = _merge_similar_events(
        data, refined_events, rms_envelope, fs, min_duration
    )
    
    # Ensure no overlaps (critical requirement)
    non_overlapping_events = _remove_overlaps(merged_events, min_samples)
    
    # FINAL HARD FILTER: Absolutely ensure no segment violates min_duration
    final_events = [(s, e) for s, e in non_overlapping_events if (e - s) >= min_samples]
    
    return final_events


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
        Minimum duration constraint (HARD LIMIT)
    sensitivity : float
        Sensitivity parameter
    
    Returns:
    --------
    float
        Quality score (higher is better)
        Returns -inf if ANY segment violates minimum duration
    """
    if not segments:
        return -np.inf
    
    min_samples = int(min_duration * fs)
    
    # HARD CONSTRAINT: Reject ANY scheme with segments below min_duration
    for start, end in segments:
        if (end - start) < min_samples:
            return -np.inf  # Completely reject this scheme
    
    total_score = 0.0
    
    for i, (start, end) in enumerate(segments):
        segment_length = end - start
        
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
    
    # HARD FILTER: Ensure no merged segment violates min_duration
    min_samples = int(min_duration * fs)
    merged = [(s, e) for s, e in merged if (e - s) >= min_samples]
    
    return merged


def _calculate_event_confidence(
    data: np.ndarray,
    segment: Tuple[int, int],
    rms_envelope: np.ndarray,
    fs: float,
    sensitivity: float
) -> float:
    """
    Calculate confidence score for a potential muscle activity event.
    
    Confidence is based on multiple criteria:
    1. Amplitude elevation above baseline
    2. Signal consistency within event (low variance = good)
    3. Boundary sharpness (clear transitions)
    4. Duration reasonableness for physiological events
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal
    segment : Tuple[int, int]
        Event segment (start, end)
    rms_envelope : np.ndarray
        RMS envelope
    fs : float
        Sampling frequency
    sensitivity : float
        Sensitivity parameter (affects confidence threshold)
    
    Returns:
    --------
    float
        Confidence score (0 to 1, higher = more confident this is a real event)
    """
    start, end = segment
    segment_length = end - start
    
    if segment_length == 0:
        return 0.0
    
    segment_rms = rms_envelope[start:end]
    mean_rms = np.mean(segment_rms)
    
    # Criterion 1: Amplitude elevation (how much above baseline)
    # Compare to surrounding regions
    window = min(segment_length, int(fs * 0.5))  # 500ms context
    
    baseline_rms = 0.0
    baseline_count = 0
    
    if start >= window:
        baseline_rms += np.mean(rms_envelope[start - window:start])
        baseline_count += 1
    if end + window < len(rms_envelope):
        baseline_rms += np.mean(rms_envelope[end:end + window])
        baseline_count += 1
    
    if baseline_count > 0:
        baseline_rms /= baseline_count
        amplitude_elevation = (mean_rms - baseline_rms) / (baseline_rms + 1e-10)
        amplitude_score = min(1.0, amplitude_elevation / 2.0)  # Normalize to 0-1
    else:
        amplitude_score = 0.5  # No baseline, moderate confidence
    
    # Criterion 2: Consistency (low variance within event)
    std_rms = np.std(segment_rms)
    cv = std_rms / (mean_rms + 1e-10)
    consistency_score = 1.0 / (1.0 + cv)  # Lower CV = higher score
    
    # Criterion 3: Boundary sharpness
    # Check for sharp transitions at start and end
    boundary_window = min(int(fs * 0.05), segment_length // 4)  # 50ms
    
    start_sharpness = 0.5
    end_sharpness = 0.5
    
    if start >= boundary_window:
        before_start = np.mean(rms_envelope[start - boundary_window:start])
        after_start = np.mean(segment_rms[:boundary_window])
        start_sharpness = min(1.0, (after_start - before_start) / (before_start + 1e-10) / 2.0)
    
    if end + boundary_window < len(rms_envelope):
        before_end = np.mean(segment_rms[-boundary_window:])
        after_end = np.mean(rms_envelope[end:end + boundary_window])
        end_sharpness = min(1.0, (before_end - after_end) / (after_end + 1e-10) / 2.0)
    
    boundary_score = (start_sharpness + end_sharpness) / 2.0
    
    # Criterion 4: Duration reasonableness
    # Typical muscle contraction: 0.3-5 seconds
    duration_seconds = segment_length / fs
    if 0.3 <= duration_seconds <= 5.0:
        duration_score = 1.0
    elif duration_seconds < 0.3:
        duration_score = duration_seconds / 0.3
    else:
        duration_score = 5.0 / duration_seconds
    
    # Weighted combination
    confidence = (
        0.35 * amplitude_score +
        0.30 * consistency_score +
        0.20 * boundary_score +
        0.15 * duration_score
    )
    
    return confidence


def _calculate_event_presence_confidence(
    data: np.ndarray,
    region: Tuple[int, int],
    rms_envelope: np.ndarray,
    fs: float,
    amplitude_threshold: float
) -> float:
    """
    Calculate confidence score for event PRESENCE (Stage 1).
    
    This function emphasizes amplitude (70% weight) to determine if an event
    likely exists in the given region. Other factors are secondary.
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal
    region : Tuple[int, int]
        Candidate event region (start, end)
    rms_envelope : np.ndarray
        RMS envelope
    fs : float
        Sampling frequency
    amplitude_threshold : float
        Amplitude threshold used for detection
    
    Returns:
    --------
    float
        Presence confidence score (0 to 1, higher = more likely a real event)
    """
    start, end = region
    region_length = end - start
    
    if region_length == 0:
        return 0.0
    
    region_rms = rms_envelope[start:end]
    mean_rms = np.mean(region_rms)
    
    # PRIMARY CRITERION (70%): Amplitude elevation
    # How much above the detection threshold
    amplitude_ratio = mean_rms / (amplitude_threshold + 1e-10)
    amplitude_score = min(1.0, amplitude_ratio / 2.0)  # Normalize: 2x threshold = perfect
    
    # SECONDARY CRITERION (20%): Signal consistency
    std_rms = np.std(region_rms)
    cv = std_rms / (mean_rms + 1e-10)
    consistency_score = 1.0 / (1.0 + cv)
    
    # TERTIARY CRITERION (10%): Duration reasonableness
    duration_seconds = region_length / fs
    if 0.3 <= duration_seconds <= 5.0:
        duration_score = 1.0
    elif duration_seconds < 0.3:
        duration_score = duration_seconds / 0.3
    else:
        duration_score = 5.0 / duration_seconds
    
    # Weighted combination - HEAVILY weighted toward amplitude
    confidence = (
        0.70 * amplitude_score +      # PRIMARY: amplitude determines presence
        0.20 * consistency_score +     # SECONDARY: consistency support
        0.10 * duration_score          # TERTIARY: duration check
    )
    
    return confidence


def _refine_event_boundaries_with_ruptures(
    event_data: np.ndarray,
    event_rms: np.ndarray,
    internal_breaks: List[int],
    global_start: int,
    global_end: int,
    min_samples: int
) -> Tuple[int, int]:
    """
    Refine event boundaries using ruptures internal structure.
    
    Parameters:
    -----------
    event_data : np.ndarray
        Signal data for this event
    event_rms : np.ndarray
        RMS envelope for this event
    internal_breaks : List[int]
        Break points from ruptures within event
    global_start : int
        Global start index of event
    global_end : int
        Global end index of event
    min_samples : int
        Minimum samples for valid segment
    
    Returns:
    --------
    Tuple[int, int]
        Refined (start, end) indices in global coordinates
    """
    # Find local minima near start and end for better boundaries
    quarter_len = len(event_rms) // 4
    
    if quarter_len > 0:
        # Refine start: find minimum in first quarter
        start_region = event_rms[:quarter_len]
        local_start = np.argmin(start_region)
        
        # Refine end: find minimum in last quarter
        end_region = event_rms[-quarter_len:]
        local_end = len(event_rms) - quarter_len + np.argmin(end_region)
        
        refined_start = global_start + local_start
        refined_end = global_start + local_end
        
        # Ensure minimum duration
        if refined_end - refined_start >= min_samples:
            return (refined_start, refined_end)
    
    # If refinement fails, return original
    return (global_start, global_end)


def _remove_overlaps(
    segments: List[Tuple[int, int]],
    min_samples: int
) -> List[Tuple[int, int]]:
    """
    Remove overlapping segments while respecting minimum duration.
    
    When segments overlap, keep the one with larger extent or merge them
    if they're close enough.
    
    Parameters:
    -----------
    segments : List[Tuple[int, int]]
        List of (start, end) tuples
    min_samples : int
        Minimum samples for valid segment
    
    Returns:
    --------
    List[Tuple[int, int]]
        Non-overlapping segments
    """
    if not segments:
        return []
    
    # Sort by start index
    sorted_segs = sorted(segments, key=lambda x: x[0])
    
    non_overlapping = [sorted_segs[0]]
    
    for current_start, current_end in sorted_segs[1:]:
        last_start, last_end = non_overlapping[-1]
        
        # Check for overlap
        if current_start < last_end:
            # Overlap detected - merge or keep larger
            merged_start = min(last_start, current_start)
            merged_end = max(last_end, current_end)
            
            # Replace last segment with merged one
            if merged_end - merged_start >= min_samples:
                non_overlapping[-1] = (merged_start, merged_end)
            # else keep the previous one
        else:
            # No overlap - add as new segment
            if current_end - current_start >= min_samples:
                non_overlapping.append((current_start, current_end))
    
    return non_overlapping


def detect_activity_hht(
    data: np.ndarray,
    fs: float,
    min_duration: float = 0.1,
    max_duration: Optional[float] = None,
    energy_threshold: float = 0.4,
    temporal_compactness: float = 0.15,
    min_freq: float = 20.0,
    max_freq: float = 450.0,
    resolution_per_second: int = 128,
    adaptive_threshold_factor: float = HHT_ADAPTIVE_THRESHOLD_FACTOR,
    merge_gap_ms: float = HHT_MERGE_GAP_MS,
    return_spectrum: bool = False,
    sensitivity: float = 1.0,
    local_contrast_weight: float = 0.3,
    **kwargs
) -> Union[List[Tuple[int, int]], Dict[str, Union[List[Tuple[int, int]], np.ndarray]]]:
    """
    Detect muscle activity events using Hilbert-Huang Transform (HHT) analysis.
    
    This method computes the full-signal HHT, analyzes the Hilbert spectrum for
    high-energy stripes (characteristic of muscle activity), and maps detected
    time segments back to the original signal.
    
    Algorithm (Enhanced with Baseline Resting State):
    1. Compute HHT of the entire signal with dynamic resolution using CEEMDAN
    2. Compute time-integrated energy profile from Hilbert spectrum
    3. Estimate baseline (resting state) energy from lower percentiles
    4. Set threshold relative to baseline rather than global statistics
    5. Detect temporally compact energy patterns (muscle events)
    6. Map detected patterns back to time domain with user-adjustable sensitivity
    7. Apply duration constraints and segment merging
    
    Key Innovation - Baseline Resting State Approach:
    The HHT spectrum shows clear baseline (resting) energy with minimal variation,
    while muscle activities appear as distinct elevations above baseline. The 
    detection threshold is set relative to this baseline, ensuring that any 
    sustained elevation above the "calm" resting state is detected. This approach:
    - Detects weak activities that would be missed by global statistics
    - Is not skewed by high-intensity peaks
    - Adapts to the signal's inherent baseline level
    - Captures complete muscle action segments (rising, peak, falling phases)
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data (1D array), should be preprocessed (filtered)
    fs : float
        Sampling frequency in Hz
    min_duration : float, optional
        Minimum duration of muscle activity in seconds (default: 0.1)
        Muscle actions like dumbbell lifts typically require at least 0.5-1.0 seconds
    max_duration : float, optional
        Maximum duration of muscle activity in seconds (default: None = no limit)
        Typical dumbbell lift actions are usually 1-10 seconds
    energy_threshold : float, optional
        Percentile threshold for high-energy detection (0-1, default: 0.4)
        Used for spectrum high-energy mask in compactness filtering
    temporal_compactness : float, optional
        Minimum ratio of time bins that must have high energy (0-1, default: 0.15)
        Lower values include broader segments containing muscle activity, not just instantaneous peaks
    min_freq : float, optional
        Minimum frequency for HHT analysis (default: 20.0 Hz for sEMG)
    max_freq : float, optional
        Maximum frequency for HHT analysis (default: 450.0 Hz for sEMG)
    resolution_per_second : int, optional
        Time resolution per second (default: 128, i.e., 128 time bins per second)
        Total resolution scales with signal duration
    adaptive_threshold_factor : float, optional
        **DEPRECATED - Not used in baseline approach (kept for API compatibility)**
        This parameter is not used in the new baseline resting state detection.
        Will be removed in a future version.
        For controlling sensitivity, use the 'sensitivity' parameter instead.
    merge_gap_ms : float, optional
        Gap in milliseconds for merging nearby segments (default: 50ms)
    return_spectrum : bool, optional
        If True, return dict with segments and spectrum data for visualization
    sensitivity : float, optional
        User-adjustable detection sensitivity (default: 1.0)
        Range: 0.1 (very sensitive, detects weak events) to 3.0 (very strict)
        - Lower values reduce baseline margin, detecting more events
        - Higher values increase baseline margin, only detecting stronger events
        Controls the margin above baseline: threshold = baseline_mean + (2.0/sensitivity) * baseline_std
    local_contrast_weight : float, optional
        Weight for local contrast in energy evaluation (default: 0.3)
        Range: 0.0 (pure global) to 1.0 (pure local)
        Higher values emphasize local energy contrast over global threshold
    **kwargs : dict
        Additional arguments (for compatibility)
    
    Returns:
    --------
    List[Tuple[int, int]] or Dict
        If return_spectrum=False (default):
            List of (start_index, end_index) tuples for detected muscle activity segments
        If return_spectrum=True:
            Dict with keys:
                'segments': List of (start_index, end_index) tuples
                'spectrum': Full Hilbert spectrum matrix (freq_bins x time_bins)
                'spectrum_log': Log-scaled spectrum for visualization
                'time': Time axis array
                'frequency': Frequency axis array
                'detection_mask': Boolean mask of detected regions on spectrum
                'time_energy': Time-integrated energy profile
                'combined_energy': Combined global+local energy
                'active_time_bins': Boolean array of active time bins
                'threshold_info': Dict with threshold calculation details including:
                    - baseline_mean, baseline_std: Resting state statistics
                    - baseline_threshold: Threshold relative to baseline
                    - adaptive_threshold: Final threshold after safety bounds
    
    Notes:
    ------
    - Signal should be preprocessed (filtered) before detection
    - Uses CEEMDAN decomposition for robust IMF extraction
    - Resolution dynamically scales: for 2-4s signal → 256-512 time bins
    - Uses average pooling logic, no interpolation (consistent with existing HHT)
    - Frequency range limited to sEMG effective range (20-450 Hz)
    - High-energy stripes in spectrum indicate muscle activity
    - Baseline approach detects all activities above resting state, regardless of intensity
    - Local contrast weight enables detection of events that stand out locally
      even if they are not the highest energy globally
    
    Examples:
    ---------
    >>> # Basic usage
    >>> segments = detect_activity_hht(signal, fs=1000, min_duration=0.5)
    
    >>> # More sensitive (detects weaker activities)
    >>> segments = detect_activity_hht(signal, fs=1000, sensitivity=0.7)
    
    >>> # Get full results with spectrum for visualization
    >>> result = detect_activity_hht(signal, fs=1000, return_spectrum=True)
    >>> segments = result['segments']
    >>> spectrum = result['spectrum_log']
    """
    
    signal_length = len(data)
    signal_duration = signal_length / fs
    
    # Dynamic resolution: scale based on signal duration
    # For 2-4s signals → 256-512 time bins (as per requirement)
    target_time_bins = int(signal_duration * resolution_per_second)
    target_time_bins = max(HHT_MIN_TIME_BINS, min(target_time_bins, HHT_MAX_TIME_BINS))  # Reasonable bounds
    
    # Frequency bins: use same number as time bins for square matrix (common for CNN input)
    n_freq_bins = target_time_bins
    
    # Compute full-signal HHT
    # Use compute_hilbert_spectrum which handles the decomposition and spectrum generation
    spectrum, time_axis, freq_axis = hht_module.compute_hilbert_spectrum(
        data, 
        fs=fs,
        n_freq_bins=n_freq_bins,
        min_freq=min_freq,
        max_freq=max_freq,
        normalize_length=target_time_bins
    )
    
    # Create log-scaled spectrum for visualization (improved display)
    epsilon = 1e-10
    spectrum_log = np.log1p(spectrum / (np.percentile(spectrum[spectrum > 0], 5) + epsilon)) if np.any(spectrum > 0) else spectrum.copy()
    
    # ===== IMPROVED ENERGY-BASED DETECTION =====
    # Strategy: Combine global percentile threshold with local contrast analysis
    # This allows detection of events that are locally prominent even if not globally highest
    
    # 1. Threshold spectrum at percentile (adjusted by sensitivity)
    # Lower sensitivity = lower percentile threshold = more sensitive detection
    adjusted_energy_threshold = energy_threshold * sensitivity
    adjusted_energy_threshold = np.clip(adjusted_energy_threshold, HHT_MIN_ENERGY_THRESHOLD, HHT_MAX_ENERGY_THRESHOLD)
    threshold = np.percentile(spectrum, adjusted_energy_threshold * 100)
    high_energy_mask = spectrum > threshold
    
    # 2. Compute time-integrated energy (sum over frequency at each time bin)
    time_energy = np.sum(spectrum, axis=0)
    
    # 3. Compute local contrast energy for improved detection
    # This helps detect events that stand out locally even if not highest globally
    local_window_size = max(HHT_LOCAL_WINDOW_MIN_SIZE, target_time_bins // HHT_LOCAL_WINDOW_FRACTION)
    local_contrast = _compute_local_energy_contrast(time_energy, local_window_size)
    
    # 4. Detect active time regions using combined global and local criteria
    # Normalize both energy metrics
    if np.max(time_energy) > 0:
        time_energy_norm = time_energy / np.max(time_energy)
    else:
        time_energy_norm = time_energy
    
    if np.max(local_contrast) > 0:
        local_contrast_norm = local_contrast / np.max(local_contrast)
    else:
        local_contrast_norm = local_contrast
    
    # Combine global and local energy metrics
    # Weight can be adjusted by user (local_contrast_weight)
    combined_energy = (1 - local_contrast_weight) * time_energy_norm + local_contrast_weight * local_contrast_norm
    
    # ===== BASELINE RESTING STATE APPROACH =====
    # Key insight: HHT spectrum shows clear baseline (resting state) vs activity
    # In resting state, energy remains near baseline with minimal variation
    # Any sustained elevation above baseline indicates muscle activity
    # This approach is more robust than global statistics which can be skewed by strong peaks
    
    # 1. Estimate baseline (resting state) energy level
    # Use lower percentiles to capture the typical resting energy
    # Sort energy values and take lower portion as baseline candidates
    sorted_energy = np.sort(combined_energy)
    baseline_percentile = 30  # Use lower 30% of energy values to estimate baseline
    baseline_sample_size = max(10, int(len(sorted_energy) * baseline_percentile / 100))
    baseline_energy = sorted_energy[:baseline_sample_size]
    
    # Calculate baseline statistics
    baseline_mean = np.mean(baseline_energy)
    baseline_std = np.std(baseline_energy)
    
    # 2. Set threshold relative to baseline rather than global mean
    # The threshold is: baseline + margin above typical baseline variation
    # This ensures we detect any activity that rises above the "calm" resting state
    # Sensitivity parameter controls the margin
    baseline_margin_factor = 2.0 / sensitivity  # Lower sensitivity = smaller margin = more sensitive
    baseline_threshold = baseline_mean + baseline_margin_factor * baseline_std
    
    # 3. Add safety bounds to prevent extreme thresholds
    # Still use percentiles as safety bounds, but they're now less restrictive
    min_safe_threshold = np.percentile(combined_energy, HHT_NOISE_FLOOR_PERCENTILE)
    max_safe_threshold = np.percentile(combined_energy, HHT_MAX_THRESHOLD_PERCENTILE)
    adaptive_threshold = np.clip(baseline_threshold, min_safe_threshold, max_safe_threshold)
    
    # 4. Apply threshold to detect active regions
    active_time_bins = combined_energy > adaptive_threshold
    
    # 5. Find contiguous active regions using connected component analysis
    # Convert boolean mask to segments
    segments_in_bins = []
    in_segment = False
    start_bin = 0
    
    for i, is_active in enumerate(active_time_bins):
        if is_active and not in_segment:
            # Start of new segment
            start_bin = i
            in_segment = True
        elif not is_active and in_segment:
            # End of segment
            segments_in_bins.append((start_bin, i))
            in_segment = False
    
    # Handle case where segment extends to end
    if in_segment:
        segments_in_bins.append((start_bin, len(active_time_bins)))
    
    # 6. Filter segments by temporal compactness
    # Check if high-energy patterns are compact within detected regions
    # Adjust compactness threshold based on sensitivity
    adjusted_compactness = temporal_compactness / sensitivity
    adjusted_compactness = np.clip(adjusted_compactness, HHT_MIN_COMPACTNESS, HHT_MAX_COMPACTNESS)
    
    filtered_segments = []
    for start_bin, end_bin in segments_in_bins:
        segment_length = end_bin - start_bin
        if segment_length == 0:
            continue
            
        # Check compactness: ratio of high-energy bins within segment
        segment_mask = high_energy_mask[:, start_bin:end_bin]
        has_high_energy = np.any(segment_mask, axis=0)  # Time bins with any high energy
        compactness = np.sum(has_high_energy) / segment_length
        
        if compactness >= adjusted_compactness:
            filtered_segments.append((start_bin, end_bin))
    
    # 7. Map time bins back to original signal indices
    # Time bins correspond to normalized time axis
    bin_to_sample_ratio = signal_length / target_time_bins
    
    segments_in_samples = []
    for start_bin, end_bin in filtered_segments:
        start_sample = int(start_bin * bin_to_sample_ratio)
        end_sample = int(end_bin * bin_to_sample_ratio)
        
        # Ensure within bounds
        start_sample = max(0, start_sample)
        end_sample = min(signal_length, end_sample)
        
        segments_in_samples.append((start_sample, end_sample))
    
    # 8. Apply duration constraints
    min_samples = int(min_duration * fs)
    segments_filtered = []
    
    for start, end in segments_in_samples:
        duration_samples = end - start
        
        # Check minimum duration
        if duration_samples < min_samples:
            continue
        
        # Apply max_duration splitting if specified
        if max_duration is not None and max_duration > min_duration:
            max_samples = int(max_duration * fs)
            if duration_samples > max_samples:
                # Split into multiple segments at energy minima
                segment_energy = time_energy_norm[int(start / bin_to_sample_ratio):int(end / bin_to_sample_ratio)]
                if len(segment_energy) > 1:
                    # Find local minima for natural split points
                    split_points = _find_energy_split_points(
                        data[start:end], fs, max_samples, min_samples
                    )
                    
                    if split_points:
                        prev_split = 0
                        for sp in split_points:
                            if sp - prev_split >= min_samples:
                                segments_filtered.append((start + prev_split, start + sp))
                            prev_split = sp
                        # Add final segment
                        if end - start - prev_split >= min_samples:
                            segments_filtered.append((start + prev_split, end))
                    else:
                        # Fallback: simple uniform split
                        for split_start in range(start, end, max_samples):
                            split_end = min(split_start + max_samples, end)
                            if split_end - split_start >= min_samples:
                                segments_filtered.append((split_start, split_end))
                else:
                    segments_filtered.append((start, end))
            else:
                segments_filtered.append((start, end))
        else:
            segments_filtered.append((start, end))
    
    # 9. Merge nearby segments with energy-aware logic
    # Merge segments with configurable gap
    merge_gap_samples = int(merge_gap_ms / 1000.0 * fs)
    if len(segments_filtered) > 1:
        merged_segments = [segments_filtered[0]]
        for start, end in segments_filtered[1:]:
            last_start, last_end = merged_segments[-1]
            gap = start - last_end
            
            if gap <= merge_gap_samples:
                # Check if the merged segment would exceed max_duration
                merged_duration = end - last_start
                if max_duration is None or merged_duration <= int(max_duration * fs):
                    # Merge with previous
                    merged_segments[-1] = (last_start, end)
                else:
                    # Don't merge if it would exceed max duration
                    merged_segments.append((start, end))
            else:
                merged_segments.append((start, end))
        segments_filtered = merged_segments
    
    # Return results
    if return_spectrum:
        # Create detection mask for visualization
        detection_mask = np.zeros(spectrum.shape, dtype=bool)
        for start_bin, end_bin in filtered_segments:
            detection_mask[:, start_bin:end_bin] = True
        
        # Threshold information for debugging/display
        threshold_info = {
            'spectrum_percentile_threshold': threshold,  # Percentile threshold used for spectrum high-energy mask
            'adaptive_threshold': adaptive_threshold,
            'sensitivity': sensitivity,
            'adjusted_energy_threshold': adjusted_energy_threshold,
            'adjusted_compactness': adjusted_compactness,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'baseline_threshold': baseline_threshold,
            'baseline_margin_factor': baseline_margin_factor
        }
        
        return {
            'segments': segments_filtered,
            'spectrum': spectrum,
            'spectrum_log': spectrum_log,
            'time': time_axis,
            'frequency': freq_axis,
            'detection_mask': detection_mask,
            'time_energy': time_energy,
            'combined_energy': combined_energy,
            'active_time_bins': active_time_bins,
            'threshold_info': threshold_info
        }
    else:
        return segments_filtered


def _compute_local_energy_contrast(energy_profile: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute local energy contrast for each time bin.
    
    This measures how much the energy at each time point stands out compared
    to its local neighborhood. High values indicate locally prominent events.
    
    Parameters:
    -----------
    energy_profile : np.ndarray
        Time-integrated energy at each time bin
    window_size : int
        Size of local window for computing neighborhood statistics
    
    Returns:
    --------
    np.ndarray
        Local contrast values (same length as energy_profile)
    """
    n = len(energy_profile)
    contrast = np.zeros(n)
    
    for i in range(n):
        # Define local window
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        
        # Local neighborhood statistics (excluding center)
        neighborhood = np.concatenate([energy_profile[start:i], energy_profile[i+1:end]])
        
        if len(neighborhood) > 0:
            local_mean = np.mean(neighborhood)
            local_std = np.std(neighborhood) + 1e-10
            
            # Contrast: how many std above local mean
            contrast[i] = max(0, (energy_profile[i] - local_mean) / local_std)
    
    return contrast


def _find_energy_split_points(
    segment_data: np.ndarray,
    fs: float,
    max_samples: int,
    min_samples: int
) -> List[int]:
    """
    Find optimal split points for long segments based on energy minima.
    
    Parameters:
    -----------
    segment_data : np.ndarray
        Signal data for the segment
    fs : float
        Sampling frequency
    max_samples : int
        Maximum segment length
    min_samples : int
        Minimum segment length
    
    Returns:
    --------
    List[int]
        List of split point indices within the segment
    """
    # Compute RMS envelope using a 100ms window
    window_size = int(fs / HHT_RMS_WINDOW_DIVISOR)
    if window_size < 1:
        window_size = 1
    kernel = np.ones(window_size) / window_size
    rms_envelope = np.sqrt(np.convolve(segment_data ** 2, kernel, mode='same'))
    
    split_points = []
    segment_length = len(segment_data)
    
    if segment_length <= max_samples:
        return []
    
    # Find local minima as potential split points
    current_pos = 0
    while current_pos + max_samples < segment_length:
        # Look for minimum in the region around max_samples
        search_start = max(current_pos + min_samples, current_pos + max_samples // 2)
        search_end = min(current_pos + max_samples + max_samples // 2, segment_length - min_samples)
        
        if search_end <= search_start:
            # Fallback: just split at max_samples
            split_points.append(current_pos + max_samples)
            current_pos = current_pos + max_samples
        else:
            # Find minimum energy point
            search_region = rms_envelope[search_start:search_end]
            min_idx = np.argmin(search_region) + search_start
            split_points.append(min_idx)
            current_pos = min_idx
    
    return split_points
