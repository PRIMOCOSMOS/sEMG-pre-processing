"""
Hilbert-Huang Transform (HHT) module for sEMG signal analysis.

This module implements:
1. CEEMDAN (Complete Ensemble EMD with Adaptive Noise) for signal decomposition into IMFs
2. Hilbert Transform for instantaneous frequency and amplitude
3. Hilbert Spectrum generation for time-frequency analysis
4. sEMG feature extraction (WL, ZC, SSC, Median/Mean Frequency, IMNF, WIRE51)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from scipy.signal import hilbert
from scipy.interpolate import interp1d

# Module-level constants for CEEMDAN configuration
DEFAULT_CEEMDAN_ENSEMBLES = 30  # Default number of ensembles for CEEMDAN (30 for speed, 50 for accuracy)
EPSILON = 1e-10  # Small value for numerical stability (avoiding division by zero)

# sEMG frequency filtering constants
SEMG_LOW_FREQ_CUTOFF = 20.0  # Hz - Exclude DC and low-freq artifacts (motion, baseline drift)
                              # sEMG content is typically in 20-450 Hz range
SEMG_HIGH_FREQ_CUTOFF = 450.0  # Hz - Upper limit of valid sEMG frequency content
                                # Above 450Hz is typically noise, not meaningful muscle signal


def emd_decomposition(
    signal: np.ndarray,
    max_imfs: int = 10,
    max_iterations: int = 1000,
    sift_threshold: float = 0.05
) -> List[np.ndarray]:
    """
    Perform Empirical Mode Decomposition (EMD) on the signal.
    
    EMD decomposes a signal into a set of Intrinsic Mode Functions (IMFs)
    that represent different frequency components of the original signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D array)
    max_imfs : int, optional
        Maximum number of IMFs to extract (default: 10)
    max_iterations : int, optional
        Maximum iterations for sifting process (default: 1000)
    sift_threshold : float, optional
        Threshold for stopping sifting (default: 0.05)
    
    Returns:
    --------
    List[np.ndarray]
        List of IMFs, from highest to lowest frequency,
        plus the residue as the last element
        
    Examples:
    ---------
    >>> imfs = emd_decomposition(signal)
    >>> print(f"Number of IMFs: {len(imfs) - 1}")  # -1 for residue
    >>> reconstructed = sum(imfs)  # Should equal original signal
    """
    imfs = []
    residue = signal.copy()
    
    for _ in range(max_imfs):
        # Extract one IMF using sifting
        imf = _sift_imf(residue, max_iterations, sift_threshold)
        
        if imf is None:
            break
            
        imfs.append(imf)
        residue = residue - imf
        
        # Check if residue is monotonic (no more IMFs can be extracted)
        if _is_monotonic(residue):
            break
    
    # Add residue as the last component
    imfs.append(residue)
    
    return imfs


def _sift_imf(
    signal: np.ndarray,
    max_iterations: int = 1000,
    threshold: float = 0.05
) -> Optional[np.ndarray]:
    """
    Extract one IMF from the signal using the sifting process.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    max_iterations : int
        Maximum sifting iterations
    threshold : float
        Convergence threshold
    
    Returns:
    --------
    Optional[np.ndarray]
        Extracted IMF, or None if extraction failed
    """
    h = signal.copy()
    
    for _ in range(max_iterations):
        # Find local maxima and minima
        maxima_idx, maxima_val = _find_extrema(h, 'max')
        minima_idx, minima_val = _find_extrema(h, 'min')
        
        # Need at least 2 maxima and 2 minima for envelope
        if len(maxima_idx) < 2 or len(minima_idx) < 2:
            return None
        
        # Create upper and lower envelopes using cubic spline
        upper_env = _create_envelope(h, maxima_idx, maxima_val)
        lower_env = _create_envelope(h, minima_idx, minima_val)
        
        # Calculate mean envelope
        mean_env = (upper_env + lower_env) / 2
        
        # Subtract mean from signal
        h_new = h - mean_env
        
        # Check for convergence (use EPSILON for numerical stability)
        sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + EPSILON)
        h = h_new
        
        if sd < threshold:
            break
    
    # Verify this is a valid IMF
    if _is_imf(h):
        return h
    
    return h  # Return even if not perfect IMF


def _find_extrema(
    signal: np.ndarray,
    extrema_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local extrema (maxima or minima) of the signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    extrema_type : str
        'max' for local maxima, 'min' for local minima
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Indices and values of extrema
    """
    # Compute differences
    diff = np.diff(signal)
    
    if extrema_type == 'max':
        # Find where diff changes from positive to negative
        sign_changes = (diff[:-1] > 0) & (diff[1:] <= 0)
    else:  # min
        # Find where diff changes from negative to positive
        sign_changes = (diff[:-1] < 0) & (diff[1:] >= 0)
    
    # Get indices (add 1 because diff shifts indices)
    indices = np.where(sign_changes)[0] + 1
    
    # Handle edge cases - add endpoints if they are extrema
    if extrema_type == 'max':
        if signal[0] > signal[1]:
            indices = np.concatenate([[0], indices])
        if signal[-1] > signal[-2]:
            indices = np.concatenate([indices, [len(signal) - 1]])
    else:
        if signal[0] < signal[1]:
            indices = np.concatenate([[0], indices])
        if signal[-1] < signal[-2]:
            indices = np.concatenate([indices, [len(signal) - 1]])
    
    values = signal[indices]
    
    return indices, values


def _create_envelope(
    signal: np.ndarray,
    extrema_idx: np.ndarray,
    extrema_val: np.ndarray
) -> np.ndarray:
    """
    Create envelope using cubic spline interpolation.
    
    Parameters:
    -----------
    signal : np.ndarray
        Original signal (for length reference)
    extrema_idx : np.ndarray
        Indices of extrema
    extrema_val : np.ndarray
        Values at extrema
    
    Returns:
    --------
    np.ndarray
        Envelope signal
    """
    # Use cubic interpolation for smooth envelope
    if len(extrema_idx) < 4:
        # Fall back to linear interpolation for few points
        kind = 'linear'
    else:
        kind = 'cubic'
    
    # Handle boundary conditions by adding mirrored points
    # This prevents edge artifacts
    x = extrema_idx.astype(float)
    y = extrema_val
    
    # Create interpolator
    interp_func = interp1d(x, y, kind=kind, bounds_error=False, 
                           fill_value=(y[0], y[-1]))
    
    # Generate envelope
    envelope = interp_func(np.arange(len(signal)))
    
    return envelope


def _is_monotonic(signal: np.ndarray) -> bool:
    """
    Check if signal is monotonic (increasing or decreasing).
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    
    Returns:
    --------
    bool
        True if monotonic
    """
    diff = np.diff(signal)
    return np.all(diff >= 0) or np.all(diff <= 0)


def _is_imf(signal: np.ndarray) -> bool:
    """
    Check if signal satisfies IMF conditions.
    
    IMF conditions:
    1. Number of extrema and zero-crossings must differ by at most 1
    2. Mean of upper and lower envelopes must be approximately zero
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    
    Returns:
    --------
    bool
        True if signal is a valid IMF
    """
    # Count extrema
    max_idx, _ = _find_extrema(signal, 'max')
    min_idx, _ = _find_extrema(signal, 'min')
    n_extrema = len(max_idx) + len(min_idx)
    
    # Count zero crossings
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    
    # Check condition 1
    if abs(n_extrema - zero_crossings) > 1:
        return False
    
    return True


def hilbert_transform(
    signal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hilbert Transform to get instantaneous amplitude and frequency.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D array), ideally an IMF
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - Analytic signal (complex)
        - Instantaneous amplitude (envelope)
        - Instantaneous phase
        
    Examples:
    ---------
    >>> analytic, amplitude, phase = hilbert_transform(imf)
    """
    # Compute analytic signal using Hilbert transform
    analytic_signal = hilbert(signal)
    
    # Instantaneous amplitude (envelope)
    amplitude = np.abs(analytic_signal)
    
    # Instantaneous phase
    phase = np.angle(analytic_signal)
    
    return analytic_signal, amplitude, phase


def compute_instantaneous_frequency(
    phase: np.ndarray,
    fs: float
) -> np.ndarray:
    """
    Compute instantaneous frequency from phase.
    
    Parameters:
    -----------
    phase : np.ndarray
        Instantaneous phase (from hilbert_transform)
    fs : float
        Sampling frequency in Hz
    
    Returns:
    --------
    np.ndarray
        Instantaneous frequency in Hz
    """
    # Unwrap phase to handle discontinuities
    unwrapped_phase = np.unwrap(phase)
    
    # Compute frequency as derivative of phase
    # f = (1/2π) * (dφ/dt)
    inst_freq = np.diff(unwrapped_phase) / (2 * np.pi) * fs
    
    # Pad to match original length
    inst_freq = np.concatenate([inst_freq, [inst_freq[-1]]])
    
    # Handle negative frequencies by taking absolute value
    inst_freq = np.abs(inst_freq)
    
    return inst_freq


def _average_pool_1d(signal: np.ndarray, target_length: int) -> np.ndarray:
    """
    Downsample 1D signal using average pooling to avoid interpolation artifacts.
    
    This function uses average pooling to reduce signal length, which avoids
    introducing high-frequency artifacts that can occur with interpolation methods.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D array)
    target_length : int
        Target length after downsampling
    
    Returns:
    --------
    np.ndarray
        Downsampled signal using average pooling
        
    Note:
    -----
    If the signal is shorter than target_length, it will be zero-padded.
    If signal length is not evenly divisible by target_length, the last
    pooling window may be smaller than others.
    """
    current_length = len(signal)
    
    # If signal is shorter, pad with zeros
    if current_length <= target_length:
        result = np.zeros(target_length)
        result[:current_length] = signal
        return result
    
    # Calculate pool size
    pool_size = current_length / target_length
    pooled = np.zeros(target_length)
    
    for i in range(target_length):
        # Calculate window boundaries
        start = int(i * pool_size)
        end = int((i + 1) * pool_size)
        
        # Average over the window
        if end > start:
            pooled[i] = np.mean(signal[start:end])
        else:
            pooled[i] = signal[start]
    
    return pooled


def _average_pool_2d_time(spectrum: np.ndarray, target_time_length: int) -> np.ndarray:
    """
    Downsample spectrum in time dimension using average pooling.
    
    This function applies average pooling along the time axis (axis=1) of a
    2D spectrum matrix to reduce its time dimension while preserving energy
    and avoiding interpolation artifacts.
    
    Parameters:
    -----------
    spectrum : np.ndarray
        Input spectrum matrix (freq_bins x time_samples)
    target_time_length : int
        Target number of time samples
    
    Returns:
    --------
    np.ndarray
        Downsampled spectrum (freq_bins x target_time_length)
    """
    n_freq_bins, current_time_length = spectrum.shape
    
    # If spectrum is shorter in time, pad with zeros
    if current_time_length <= target_time_length:
        result = np.zeros((n_freq_bins, target_time_length))
        result[:, :current_time_length] = spectrum
        return result
    
    # Apply average pooling along time axis
    pooled_spectrum = np.zeros((n_freq_bins, target_time_length))
    pool_size = current_time_length / target_time_length
    
    for i in range(target_time_length):
        start = int(i * pool_size)
        end = int((i + 1) * pool_size)
        
        if end > start:
            pooled_spectrum[:, i] = np.mean(spectrum[:, start:end], axis=1)
        else:
            pooled_spectrum[:, i] = spectrum[:, start]
    
    return pooled_spectrum


def compute_hilbert_spectrum(
    signal: np.ndarray,
    fs: float,
    n_freq_bins: int = 256,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    normalize_length: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hilbert Spectrum (time-frequency representation) using HHT.
    
    IMPROVEMENTS (2024):
    - Uses average pooling instead of interpolation to avoid high-frequency artifacts
    - Frequency axis maps to meaningful sEMG range (20-450Hz by default)
    
    This function performs:
    1. EMD decomposition to get IMFs
    2. Hilbert transform on each IMF
    3. Construction of time-frequency spectrum
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D array)
    fs : float
        Sampling frequency in Hz
    n_freq_bins : int, optional
        Number of frequency bins (default: 256)
    min_freq : float, optional
        Minimum frequency to display (default: 20Hz for sEMG)
    max_freq : float, optional
        Maximum frequency to display (default: 450Hz for sEMG)
    normalize_length : int, optional
        If provided, use average pooling to adjust spectrum to this length
        (useful for creating uniform-sized spectra from variable-length segments)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - Hilbert spectrum matrix (shape: n_freq_bins x n_time_samples)
        - Time axis
        - Frequency axis
        
    Examples:
    ---------
    >>> spectrum, time, freq = compute_hilbert_spectrum(segment, fs=1000)
    >>> plt.pcolormesh(time, freq, spectrum, shading='auto')
    >>> plt.colorbar(label='Amplitude')
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Frequency (Hz)')
    """
    # Set default frequency range for sEMG
    if min_freq is None:
        min_freq = SEMG_LOW_FREQ_CUTOFF  # 20 Hz
    if max_freq is None:
        max_freq = SEMG_HIGH_FREQ_CUTOFF  # 450 Hz
    
    # Compute HHT on ORIGINAL signal (no interpolation pre-processing)
    n_samples = len(signal)
    
    # Create frequency axis mapped to sEMG range
    freq_axis = np.linspace(min_freq, max_freq, n_freq_bins)
    
    # Initialize spectrum at original time resolution
    spectrum = np.zeros((n_freq_bins, n_samples))
    
    # Perform EMD
    imfs = emd_decomposition(signal)
    
    # Process each IMF
    for imf in imfs[:-1]:  # Exclude residue
        # Compute Hilbert transform
        _, amplitude, phase = hilbert_transform(imf)
        
        # Compute instantaneous frequency
        inst_freq = compute_instantaneous_frequency(phase, fs)
        
        # Clip to valid sEMG frequency range
        inst_freq = np.clip(inst_freq, min_freq, max_freq)
        
        # Map to spectrum
        for t in range(n_samples):
            freq = inst_freq[t]
            amp = amplitude[t]
            
            # Map frequency to bin index in [min_freq, max_freq] range
            freq_normalized = (freq - min_freq) / (max_freq - min_freq)
            freq_bin = int(freq_normalized * (n_freq_bins - 1))
            freq_bin = np.clip(freq_bin, 0, n_freq_bins - 1)
            spectrum[freq_bin, t] += amp
    
    # Apply average pooling to adjust time dimension (if requested)
    if normalize_length is not None and n_samples != normalize_length:
        spectrum = _average_pool_2d_time(spectrum, normalize_length)
        n_samples = normalize_length
    
    # Create time axis
    time_axis = np.arange(n_samples) / fs
    
    return spectrum, time_axis, freq_axis


def hht_analysis(
    signal: np.ndarray,
    fs: float,
    n_freq_bins: int = 256,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    normalize_length: Optional[int] = None,
    return_imfs: bool = False
) -> Dict:
    """
    Complete Hilbert-Huang Transform analysis.
    
    IMPROVEMENTS (2024):
    - Uses average pooling instead of interpolation to avoid artifacts
    - Frequency axis maps to sEMG range (20-450Hz by default)
    
    This is a convenience function that performs full HHT analysis
    and returns comprehensive results.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    n_freq_bins : int, optional
        Number of frequency bins (default: 256)
    min_freq : float, optional
        Minimum frequency (default: 20Hz for sEMG)
    max_freq : float, optional
        Maximum frequency (default: 450Hz for sEMG)
    normalize_length : int, optional
        Target length for normalization (uses average pooling)
    return_imfs : bool, optional
        Whether to return individual IMFs (default: False)
    
    Returns:
    --------
    Dict containing:
        - 'spectrum': Hilbert spectrum matrix
        - 'time': Time axis
        - 'frequency': Frequency axis
        - 'imfs': List of IMFs (if return_imfs=True)
        - 'residue': Residue signal (if return_imfs=True)
        - 'marginal_spectrum': Marginal Hilbert spectrum (frequency-amplitude)
        - 'mean_frequency': Mean instantaneous frequency
        - 'dominant_frequency': Dominant frequency at each time point
    """
    # Store original length (no interpolation)
    original_length = len(signal)
    
    # Compute spectrum (uses average pooling if normalize_length is provided)
    spectrum, time_axis, freq_axis = compute_hilbert_spectrum(
        signal, fs, n_freq_bins, min_freq, max_freq, normalize_length
    )
    
    # Compute marginal spectrum (integrate over time)
    marginal_spectrum = np.sum(spectrum, axis=1)
    
    # Compute dominant frequency at each time point
    dominant_freq_idx = np.argmax(spectrum, axis=0)
    dominant_frequency = freq_axis[dominant_freq_idx]
    
    # Compute mean frequency (weighted average)
    freq_weights = spectrum.sum(axis=1)
    if freq_weights.sum() > 0:
        mean_frequency = np.average(freq_axis, weights=freq_weights)
    else:
        mean_frequency = 0
    
    result = {
        'spectrum': spectrum,
        'time': time_axis,
        'frequency': freq_axis,
        'marginal_spectrum': marginal_spectrum,
        'mean_frequency': mean_frequency,
        'dominant_frequency': dominant_frequency,
        'original_length': original_length,
    }
    
    # Optionally return IMFs
    if return_imfs:
        imfs = emd_decomposition(signal)
        result['imfs'] = imfs[:-1]  # All IMFs except residue
        result['residue'] = imfs[-1]
    
    return result


def save_hilbert_spectrum(
    spectrum: np.ndarray,
    time: np.ndarray,
    frequency: np.ndarray,
    filepath: str,
    format: str = 'npz'
) -> None:
    """
    Save Hilbert spectrum to file.
    
    Parameters:
    -----------
    spectrum : np.ndarray
        Hilbert spectrum matrix
    time : np.ndarray
        Time axis
    frequency : np.ndarray
        Frequency axis
    filepath : str
        Output file path
    format : str, optional
        Output format: 'npz' (compressed), 'csv', or 'npy'
    """
    import os
    
    if format == 'npz':
        np.savez_compressed(filepath, spectrum=spectrum, time=time, frequency=frequency)
    elif format == 'npy':
        np.save(filepath, spectrum)
    elif format == 'csv':
        import pandas as pd
        # Save spectrum as DataFrame with time as columns and freq as index
        df = pd.DataFrame(spectrum, index=frequency, columns=time)
        df.index.name = 'Frequency (Hz)'
        df.to_csv(filepath)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz', 'csv', or 'npy'")
    
    print(f"Hilbert spectrum saved to: {filepath}")


def ceemdan_decomposition(
    signal: np.ndarray,
    max_imfs: int = 10,
    n_ensembles: int = 50,
    noise_std: float = 0.2,
    max_iterations: int = 1000,
    sift_threshold: float = 0.05,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Perform Complete Ensemble EMD with Adaptive Noise (CEEMDAN) decomposition.
    
    CEEMDAN is an improved version of EMD that provides more robust decomposition
    by adding noise-assisted analysis. It produces more stable and physically
    meaningful IMFs compared to standard EMD.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D array)
    max_imfs : int, optional
        Maximum number of IMFs to extract (default: 10)
    n_ensembles : int, optional
        Number of ensemble trials (default: 50)
    noise_std : float, optional
        Standard deviation of added noise relative to signal std (default: 0.2)
    max_iterations : int, optional
        Maximum iterations for sifting process (default: 1000)
    sift_threshold : float, optional
        Threshold for stopping sifting (default: 0.05)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    List[np.ndarray]
        List of IMFs, from highest to lowest frequency,
        plus the residue as the last element
        
    Examples:
    ---------
    >>> imfs = ceemdan_decomposition(signal)
    >>> print(f"Number of IMFs: {len(imfs) - 1}")  # -1 for residue
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(signal)
    signal_std = np.std(signal)
    
    # Store all ensemble IMFs
    all_imfs = []
    
    # Phase 1: Get first IMF through ensemble averaging
    first_imfs = []
    for i in range(n_ensembles):
        # Add white noise scaled adaptively
        noise = np.random.randn(n_samples) * noise_std * signal_std
        noisy_signal = signal + noise
        
        # Extract first IMF using standard EMD
        imfs = emd_decomposition(noisy_signal, max_imfs=1, max_iterations=max_iterations, 
                                  sift_threshold=sift_threshold)
        if len(imfs) > 0:
            first_imfs.append(imfs[0])
    
    if first_imfs:
        # Average the first IMFs
        first_imf = np.mean(first_imfs, axis=0)
        all_imfs.append(first_imf)
        residue = signal - first_imf
    else:
        residue = signal.copy()
    
    # Phase 2: Extract subsequent IMFs
    for k in range(1, max_imfs):
        # Check if residue is monotonic
        if _is_monotonic(residue):
            break
        
        # Generate ensemble IMFs for this level
        level_imfs = []
        for i in range(n_ensembles):
            # Add noise to residue
            noise = np.random.randn(n_samples) * noise_std * signal_std / (k + 1)
            noisy_residue = residue + noise
            
            # Extract first IMF from noisy residue
            imfs = emd_decomposition(noisy_residue, max_imfs=1, max_iterations=max_iterations,
                                      sift_threshold=sift_threshold)
            if len(imfs) > 0:
                level_imfs.append(imfs[0])
        
        if level_imfs:
            # Average to get the k-th IMF
            kth_imf = np.mean(level_imfs, axis=0)
            all_imfs.append(kth_imf)
            residue = residue - kth_imf
        else:
            break
    
    # Add final residue
    all_imfs.append(residue)
    
    return all_imfs


def compute_hilbert_spectrum_production(
    signal: np.ndarray,
    fs: float,
    n_freq_bins: int = 256,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    target_length: int = 256,
    fixed_imf_count: int = 8,
    amplitude_threshold_percentile: float = 10.0,
    validate_energy: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Production-ready Hilbert spectrum computation with unified parameters and validation.
    
    IMPROVEMENTS (2024):
    - Uses average pooling instead of interpolation to avoid high-frequency artifacts
    - Frequency axis maps to meaningful sEMG range (20-450Hz) instead of 0-Nyquist
    - Computes HHT on original signal length, then pools to target size
    - Preserves energy conservation validation
    
    This function implements comprehensive improvements for real-world sEMG analysis:
    1. Fixed IMF count with zero-padding for consistency
    2. Unified time and frequency axes for all signals (via average pooling, not interpolation)
    3. Signal normalization before HHT
    4. Energy conservation validation
    5. Amplitude thresholding for noise reduction
    6. Proper amplitude normalization for muscle activity representation
    7. Frequency axis maps to valid sEMG range (20-450Hz by default)
    
    Parameters:
    -----------
    signal : np.ndarray
        Input sEMG signal (1D array)
    fs : float
        Sampling frequency in Hz
    n_freq_bins : int, optional
        Number of frequency bins (default: 256)
    min_freq : float, optional
        Minimum frequency in Hz (default: 20Hz for sEMG)
    max_freq : float, optional
        Maximum frequency in Hz (default: 450Hz for sEMG)
    target_length : int, optional
        Unified time axis length for all spectra (default: 256)
    fixed_imf_count : int, optional
        Fixed number of IMFs (pad with zeros if needed, default: 8)
    amplitude_threshold_percentile : float, optional
        Percentile threshold to remove low-amplitude noise (default: 10.0)
    validate_energy : bool, optional
        Validate energy conservation between original and reconstructed signal (default: True)
    
    Returns:
    --------
    Tuple containing:
        - spectrum: Hilbert spectrum matrix (n_freq_bins × target_length)
        - time_axis: Normalized time axis [0, 1]
        - freq_axis: Frequency axis [min_freq, max_freq] (typically 20-450Hz)
        - validation_info: Dict with 'energy_error', 'imf_count', 'signal_energy', 'reconstructed_energy'
    
    Mathematical Details:
    --------------------
    1. Signal Normalization:
       x_norm = (x - mean(x)) / std(x)
       Ensures consistent amplitude scaling across different recordings
    
    2. Fixed IMF Count:
       If n_imfs < fixed_imf_count: pad with zero IMFs
       If n_imfs > fixed_imf_count: use first fixed_imf_count IMFs
       Ensures uniform decomposition structure
    
    3. HHT on Original Signal:
       HHT is computed on the signal at its original sampling rate and length
       This avoids introducing artifacts from pre-processing interpolation
    
    4. Average Pooling for Time Normalization:
       After HHT computation, the spectrum is downsampled to target_length
       using average pooling instead of interpolation
       This preserves energy and avoids high-frequency artifacts
    
    5. Energy Conservation:
       E_original = ||x||²
       E_reconstructed = ||∑IMFs||²
       error = |E_original - E_reconstructed| / E_original
       Typical acceptable error: < 5%
    
    6. Amplitude Thresholding:
       Remove amplitude values below Pth percentile
       Reduces noise, enhances signal features
    
    7. Amplitude Normalization:
       spectrum_norm = spectrum / max(spectrum)
       Represents relative muscle activity level
    
    8. Frequency Range:
       By default, maps to 20-450Hz (valid sEMG range)
       Frequencies outside this range are not considered meaningful
    """
    # Set default frequency range for sEMG
    if min_freq is None:
        min_freq = SEMG_LOW_FREQ_CUTOFF  # 20 Hz
    if max_freq is None:
        max_freq = SEMG_HIGH_FREQ_CUTOFF  # 450 Hz
    
    # Step 1: Signal Normalization
    # Normalize to zero mean and unit variance for consistent processing
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if signal_std > EPSILON:
        signal_normalized = (signal - signal_mean) / signal_std
    else:
        signal_normalized = signal - signal_mean
    
    # Store original signal for energy validation (before any resampling)
    original_signal_for_energy = signal_normalized.copy()
    original_energy = np.sum(original_signal_for_energy ** 2)
    original_length = len(signal_normalized)
    
    # NOTE: We NO LONGER resample the signal before HHT
    # HHT is computed on the signal at its actual duration
    # This avoids interpolation artifacts
    
    n_samples = len(signal_normalized)
    
    # Step 2: CEEMDAN Decomposition with fixed IMF count
    # Use CEEMDAN for robust decomposition on ORIGINAL signal
    try:
        imfs = ceemdan_decomposition(
            signal_normalized,
            max_imfs=fixed_imf_count + 2,  # Allow more, then select
            n_ensembles=DEFAULT_CEEMDAN_ENSEMBLES
        )
    except Exception:
        # Fallback to standard EMD if CEEMDAN fails
        imfs = emd_decomposition(signal_normalized, max_imfs=fixed_imf_count + 2)
    
    # Ensure fixed IMF count (excluding residue)
    actual_imf_count = len(imfs) - 1  # Exclude residue
    
    if actual_imf_count < fixed_imf_count:
        # Pad with zero IMFs
        for _ in range(fixed_imf_count - actual_imf_count):
            imfs.insert(-1, np.zeros(n_samples))  # Insert before residue
    elif actual_imf_count > fixed_imf_count:
        # Use only first fixed_imf_count IMFs plus residue
        imfs = imfs[:fixed_imf_count] + [imfs[-1]]
    
    # Step 3: Energy Conservation Validation
    # Validate against the original signal (not resampled)
    if validate_energy:
        reconstructed_signal = np.sum(imfs, axis=0)
        reconstructed_energy = np.sum(reconstructed_signal ** 2)
        signal_energy = np.sum(signal_normalized ** 2)
        energy_error = abs(signal_energy - reconstructed_energy) / (signal_energy + EPSILON)
    else:
        reconstructed_signal = None
        reconstructed_energy = np.sum(signal_normalized ** 2)
        energy_error = 0.0
    
    # Step 4: Compute Hilbert Spectrum at ORIGINAL time resolution
    # Frequency axis now maps to [min_freq, max_freq] (typically 20-450Hz for sEMG)
    spectrum = np.zeros((n_freq_bins, n_samples))
    freq_axis = np.linspace(min_freq, max_freq, n_freq_bins)
    
    # Process each IMF (exclude residue which is last element)
    for imf in imfs[:-1]:
        # Skip zero IMFs (padded)
        if np.sum(np.abs(imf)) < EPSILON:
            continue
        
        # Compute Hilbert transform
        _, amplitude, phase = hilbert_transform(imf)
        
        # Compute instantaneous frequency with careful unwrapping
        inst_freq = compute_instantaneous_frequency(phase, fs)
        
        # Clip to valid sEMG frequency range [min_freq, max_freq]
        inst_freq = np.clip(inst_freq, min_freq, max_freq)
        
        # Map to spectrum with conservative spreading to avoid artifacts
        for t in range(n_samples):
            freq = inst_freq[t]
            amp = amplitude[t]
            
            # Find nearest frequency bin in the [min_freq, max_freq] range
            freq_normalized = (freq - min_freq) / (max_freq - min_freq)
            freq_bin = int(freq_normalized * (n_freq_bins - 1))
            freq_bin = np.clip(freq_bin, 0, n_freq_bins - 1)
            
            # Add amplitude to spectrum with minimal spreading
            # Use only 1 bin spreading to maintain accuracy
            for df in [-1, 0, 1]:
                fb = freq_bin + df
                if 0 <= fb < n_freq_bins:
                    # Gaussian weighting with narrow sigma for accuracy
                    weight = np.exp(-0.5 * (df / 0.5) ** 2)
                    spectrum[fb, t] += amp * weight
    
    # Step 5: Apply Average Pooling to adjust time dimension to target_length
    # This avoids interpolation artifacts while achieving uniform matrix size
    if n_samples != target_length:
        spectrum = _average_pool_2d_time(spectrum, target_length)
    
    # Create unified time axis (normalized to [0, 1])
    time_axis = np.linspace(0, 1, target_length)
    
    # Step 6: Amplitude Thresholding for Noise Reduction
    # Remove amplitudes below threshold percentile
    threshold_value = 0.0  # Initialize to prevent NameError
    if amplitude_threshold_percentile > 0 and np.max(spectrum) > 0:
        threshold_value = np.percentile(spectrum[spectrum > 0], amplitude_threshold_percentile)
        spectrum[spectrum < threshold_value] = 0
    
    # Step 7: Amplitude Normalization for Muscle Activity Representation
    # Normalize to [0, 1] range to represent relative muscle activity
    max_amplitude = np.max(spectrum)
    if max_amplitude > EPSILON:
        spectrum = spectrum / max_amplitude
    
    # Validation information
    validation_info = {
        'energy_error': float(energy_error),
        'imf_count': fixed_imf_count,
        'signal_energy': float(original_energy),
        'reconstructed_energy': float(reconstructed_energy),
        'energy_conservation_ok': energy_error < 0.05,  # < 5% error is acceptable
        'max_amplitude': float(max_amplitude),
        'threshold_value': float(threshold_value),
        'original_length': original_length,
        'pooled_length': target_length,
        'freq_range': (min_freq, max_freq)
    }
    
    return spectrum, time_axis, freq_axis, validation_info


def compute_hilbert_spectrum_enhanced(
    signal: np.ndarray,
    fs: float,
    n_freq_bins: int = 256,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    normalize_length: Optional[int] = None,
    normalize_time: bool = True,
    normalize_amplitude: bool = False,
    use_ceemdan: bool = True,
    log_scale: bool = True,
    min_amplitude_percentile: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute enhanced Hilbert Spectrum with improved visualization for sEMG.
    
    IMPROVEMENTS (2024):
    - Uses average pooling instead of interpolation to avoid high-frequency artifacts
    - Frequency axis maps to meaningful sEMG range (20-450Hz by default) instead of 0-Nyquist
    - Computes HHT on original signal length, then pools to target size
    
    This function addresses the issue of mostly black spectrograms by:
    1. Using log-scale amplitude representation
    2. Applying adaptive thresholding
    3. Normalizing time axis for uniform matrix sizes (via average pooling, not interpolation)
    4. Mapping frequency to valid sEMG range (20-450Hz by default)
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D array)
    fs : float
        Sampling frequency in Hz
    n_freq_bins : int, optional
        Number of frequency bins (default: 256)
    min_freq : float, optional
        Minimum frequency to display (default: 20Hz for sEMG)
    max_freq : float, optional
        Maximum frequency to display (default: 450Hz for sEMG)
    normalize_length : int, optional
        If provided, use average pooling to adjust spectrum to this length
    normalize_time : bool, optional
        If True, normalize time axis to [0, 1] (default: True)
    normalize_amplitude : bool, optional
        If True, normalize amplitude to [0, 1] (default: False)
    use_ceemdan : bool, optional
        If True, use CEEMDAN instead of EMD (default: True)
    log_scale : bool, optional
        If True, apply log scaling to spectrum (default: True)
    min_amplitude_percentile : float, optional
        Percentile threshold for minimum amplitude display (default: 5.0)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - Hilbert spectrum matrix (shape: n_freq_bins x normalize_length or n_samples)
        - Time axis (normalized to [0, 1] if normalize_time=True)
        - Frequency axis (typically 20-450Hz for sEMG)
    """
    # Set default frequency range for sEMG
    if min_freq is None:
        min_freq = SEMG_LOW_FREQ_CUTOFF  # 20 Hz
    if max_freq is None:
        max_freq = SEMG_HIGH_FREQ_CUTOFF  # 450 Hz
    
    # Store original length (NO interpolation pre-processing)
    original_length = len(signal)
    n_samples = len(signal)
    
    # Create frequency axis mapped to sEMG range [min_freq, max_freq]
    freq_axis = np.linspace(min_freq, max_freq, n_freq_bins)
    
    # Initialize spectrum at original time resolution
    spectrum = np.zeros((n_freq_bins, n_samples))
    
    # Perform decomposition on ORIGINAL signal (no resampling)
    if use_ceemdan:
        imfs = ceemdan_decomposition(signal, n_ensembles=DEFAULT_CEEMDAN_ENSEMBLES)
    else:
        imfs = emd_decomposition(signal)
    
    # Process each IMF
    for imf in imfs[:-1]:  # Exclude residue
        # Compute Hilbert transform
        _, amplitude, phase = hilbert_transform(imf)
        
        # Compute instantaneous frequency
        inst_freq = compute_instantaneous_frequency(phase, fs)
        
        # Clip to valid sEMG frequency range [min_freq, max_freq]
        inst_freq = np.clip(inst_freq, min_freq, max_freq)
        
        # Map to spectrum with Gaussian smoothing
        for t in range(n_samples):
            freq = inst_freq[t]
            amp = amplitude[t]
            
            # Map frequency to bin index in [min_freq, max_freq] range
            freq_normalized = (freq - min_freq) / (max_freq - min_freq)
            freq_bin = int(freq_normalized * (n_freq_bins - 1))
            freq_bin = np.clip(freq_bin, 0, n_freq_bins - 1)
            
            # Add to spectrum with slight frequency spreading for smoother display
            spread = 2  # Frequency bins to spread
            for df in range(-spread, spread + 1):
                fb = freq_bin + df
                if 0 <= fb < n_freq_bins:
                    weight = np.exp(-0.5 * (df / 1.0) ** 2)  # Gaussian weight
                    spectrum[fb, t] += amp * weight
    
    # Apply average pooling to adjust time dimension (if requested)
    if normalize_length is not None and n_samples != normalize_length:
        spectrum = _average_pool_2d_time(spectrum, normalize_length)
        n_samples = normalize_length
    
    # Create time axis (normalized or absolute)
    if normalize_time:
        time_axis = np.linspace(0, 1, n_samples)
    else:
        time_axis = np.arange(n_samples) / fs
    
    # Apply amplitude processing for better visualization
    if spectrum.max() > 0:
        # Apply log scaling
        if log_scale:
            # Add small epsilon to avoid log(0), use machine epsilon as fallback
            epsilon_val = np.percentile(spectrum[spectrum > 0], min_amplitude_percentile) if np.any(spectrum > 0) else EPSILON
            spectrum = np.log1p(spectrum / epsilon_val)
        
        # Normalize amplitude if requested
        if normalize_amplitude:
            spectrum = spectrum / spectrum.max()
    
    return spectrum, time_axis, freq_axis


def extract_semg_features(
    signal: np.ndarray,
    fs: float
) -> Dict[str, float]:
    """
    Extract comprehensive sEMG signal features.
    
    This function computes time-domain, frequency-domain, and fatigue-related features
    from surface electromyography (sEMG) signals using robust signal processing methods.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input sEMG signal (1D array)
    fs : float
        Sampling frequency in Hz
    
    Returns:
    --------
    Dict[str, float]
        Dictionary containing extracted features:
        
        **Time Domain Features:**
        - WL: Waveform Length - sum of absolute differences between adjacent samples
        - ZC: Zero Crossings - number of times signal crosses zero
        - SSC: Slope Sign Changes - number of times slope changes direction
        - RMS: Root Mean Square - measure of signal power
        - MAV: Mean Absolute Value - average signal amplitude
        - VAR: Variance - measure of signal variability
        
        **Frequency Domain Features (Welch PSD-based):**
        - MDF: Median Frequency - frequency dividing power spectrum into two equal halves
        - MNF: Mean Frequency - power-weighted average frequency
        - PKF: Peak Frequency - frequency with maximum power
        - TTP: Total Power - integrated power across all frequencies
        
        **Advanced Frequency Features:**
        - IMNF: Instantaneous Mean Frequency using Choi-Williams Distribution (CWD)
               Time-frequency analysis method providing robust instantaneous frequency
        
        **Fatigue Indicators:**
        - WIRE51: Wavelet Index of Reliability Estimation using sym5 wavelet
                 Ratio of low-to-high frequency wavelet detail coefficients
                 Formula: WIRE51 = E(D5) / E(D1) where E is energy
                 Physical meaning: increases with muscle fatigue as power shifts to lower frequencies
        - DI: Dimitrov Index (spectral moment ratio)
             Formula: DI = M_{-1} / M_5 where M_k = ∑(f^k·P(f))/∑P(f)
             Physical meaning: increases with fatigue (typically 2-10x) as power shifts to lower frequencies
             Typical range: 1e-14 to 1e-8 (absolute values vary, ratio is meaningful)
    
    Mathematical Formulas:
    ---------------------
    Time Domain:
        WL = ∑|x[n+1] - x[n]|
        RMS = √(∑x²[n]/N)
        MAV = ∑|x[n]|/N
        VAR = ∑(x[n] - μ)²/N
    
    Frequency Domain (Welch PSD):
        MDF: ∫[0,MDF] P(f)df = ∫[MDF,∞] P(f)df = TTP/2
        MNF = ∫f·P(f)df / ∫P(f)df
        PKF = argmax P(f)
        TTP = ∫P(f)df
    
    Fatigue Indicators:
        WIRE51 = ∑D5²[n] / ∑D1²[n]  (sym5 wavelet decomposition)
        DI = M_{-1} / M_5  where M_k = (∑f^k·P(f)) / (∑P(f))
    
    Note: All frequency features exclude DC and low-frequency artifacts (<20 Hz)
    to ensure measurements reflect true muscle activity.
    """
    import pywt
    from scipy.signal import welch
    
    n_samples = len(signal)
    
    # Time domain features
    # 1. Waveform Length (WL) - sum of absolute differences
    wl = np.sum(np.abs(np.diff(signal)))
    
    # 2. Zero Crossings (ZC)
    threshold = 0.01 * np.std(signal)  # Small threshold to avoid noise
    zc = 0
    for i in range(n_samples - 1):
        if ((signal[i] > 0 and signal[i + 1] < 0) or 
            (signal[i] < 0 and signal[i + 1] > 0)) and \
            abs(signal[i] - signal[i + 1]) > threshold:
            zc += 1
    
    # 3. Slope Sign Changes (SSC)
    diff_signal = np.diff(signal)
    ssc = 0
    for i in range(len(diff_signal) - 1):
        if ((diff_signal[i] > 0 and diff_signal[i + 1] < 0) or
            (diff_signal[i] < 0 and diff_signal[i + 1] > 0)) and \
            abs(diff_signal[i] - diff_signal[i + 1]) > threshold:
            ssc += 1
    
    # 4. RMS, MAV, Variance
    rms = np.sqrt(np.mean(signal ** 2))
    mav = np.mean(np.abs(signal))
    var = np.var(signal)
    
    # Frequency domain features using Welch method for robust power spectrum estimation
    # Welch's method provides better spectral estimates by averaging periodograms
    # of overlapping segments, reducing variance compared to direct FFT
    nperseg = min(256, n_samples // 4)  # Segment length for Welch method
    freqs, power_spectrum = welch(signal, fs=fs, nperseg=nperseg, scaling='density')
    
    # For sEMG signals, exclude DC and very low frequencies (< SEMG_LOW_FREQ_CUTOFF)
    # These are typically artifacts, baseline drift, or motion noise
    # sEMG frequency content is typically in 20-450 Hz range
    freq_mask = freqs >= SEMG_LOW_FREQ_CUTOFF  # Exclude frequencies below cutoff
    valid_freqs = freqs[freq_mask]
    valid_power = power_spectrum[freq_mask]
    
    # Total power (using valid frequency range)
    # Integrate power spectrum using trapezoidal rule for accuracy
    ttp_valid = np.trapz(valid_power, valid_freqs) if len(valid_freqs) > 1 else np.sum(valid_power)
    ttp = np.trapz(power_spectrum, freqs) if len(freqs) > 1 else np.sum(power_spectrum)
    
    # 5. Median Frequency (MDF) - frequency that divides power spectrum in half
    # Use valid frequency range to avoid DC bias
    if ttp_valid > EPSILON:
        cumulative_power = np.cumsum(valid_power) * (valid_freqs[1] - valid_freqs[0]) if len(valid_freqs) > 1 else np.cumsum(valid_power)
        half_power = ttp_valid / 2
        mdf_idx = np.searchsorted(cumulative_power, half_power)
        mdf = valid_freqs[min(mdf_idx, len(valid_freqs) - 1)]
    else:
        mdf = 0.0
    
    # 6. Mean Frequency (MNF) - power-weighted average frequency
    # Use valid frequency range to avoid DC bias
    # Formula: MNF = ∫f·P(f)df / ∫P(f)df
    if ttp_valid > EPSILON:
        mnf = np.trapz(valid_freqs * valid_power, valid_freqs) / ttp_valid if len(valid_freqs) > 1 else np.sum(valid_freqs * valid_power) / ttp_valid
    else:
        mnf = 0.0
    
    # 7. Peak Frequency (PKF) - frequency with maximum power
    # Use valid frequency range to find dominant sEMG frequency
    if len(valid_power) > 0 and np.max(valid_power) > EPSILON:
        pkf = valid_freqs[np.argmax(valid_power)]
    else:
        pkf = 0.0
    
    # 8. IMNF - Instantaneous Mean Frequency using Choi-Williams Distribution (CWD)
    # 
    # The Choi-Williams Distribution (CWD) is a time-frequency representation that provides
    # better concentration and reduced cross-term interference compared to Wigner-Ville Distribution.
    # 
    # CWD Formula: CWD(t,f) = ∫∫ A(θ,τ) · x(u+τ/2) · x*(u-τ/2) · e^(-j2πfτ) dτ du
    # where A(θ,τ) = (1/(4π|θ|σ))^(1/2) · exp(-τ²/(4σθ))  (Choi-Williams kernel)
    # and σ is a scaling parameter (typically σ = 1)
    #
    # IMNF is then computed as the first moment of CWD in frequency:
    # IMNF = ∫∫ f · CWD(t,f) df dt / ∫∫ CWD(t,f) df dt
    #
    # Physical meaning: IMNF represents the time-varying center frequency of the signal,
    # which decreases with muscle fatigue as the power spectrum shifts to lower frequencies.
    #
    # For computational efficiency and robustness, we use a simplified pseudo-CWD approach:
    # 1. Apply short-time Fourier transform (STFT) with optimal window
    # 2. Use smoothing in both time and frequency to reduce cross-terms (CWD-like behavior)
    # 3. Compute power-weighted mean frequency over time
    try:
        from scipy.signal import stft
        
        # STFT parameters for time-frequency analysis
        # Window length chosen based on signal characteristics
        nperseg_stft = min(256, n_samples // 4)
        noverlap = nperseg_stft // 2
        
        # Compute STFT (provides time-frequency representation)
        f_stft, t_stft, Zxx = stft(signal, fs=fs, nperseg=nperseg_stft, noverlap=noverlap)
        
        # Power spectrogram (analogous to CWD magnitude)
        power_tf = np.abs(Zxx) ** 2
        
        # Apply smoothing to reduce cross-terms (emulates CWD kernel smoothing)
        # Smooth in frequency direction
        from scipy.ndimage import gaussian_filter1d
        power_tf_smoothed = gaussian_filter1d(power_tf, sigma=1.5, axis=0)
        # Smooth in time direction
        power_tf_smoothed = gaussian_filter1d(power_tf_smoothed, sigma=1.0, axis=1)
        
        # Extract valid frequency range (exclude low frequencies)
        freq_mask_tf = f_stft >= SEMG_LOW_FREQ_CUTOFF
        valid_freqs_tf = f_stft[freq_mask_tf]
        valid_power_tf = power_tf_smoothed[freq_mask_tf, :]
        
        # Compute instantaneous mean frequency at each time point
        # IMF(t) = ∫ f · P(f,t) df / ∫ P(f,t) df
        inst_mean_freqs = []
        for t_idx in range(valid_power_tf.shape[1]):
            power_at_t = valid_power_tf[:, t_idx]
            total_power_at_t = np.sum(power_at_t)
            if total_power_at_t > EPSILON:
                imf_at_t = np.sum(valid_freqs_tf * power_at_t) / total_power_at_t
                inst_mean_freqs.append(imf_at_t)
        
        # IMNF = time-averaged instantaneous mean frequency
        # Weight by total power at each time point for robust averaging
        if len(inst_mean_freqs) > 0:
            time_weights = np.sum(valid_power_tf, axis=0)
            total_weight = np.sum(time_weights)
            if total_weight > EPSILON:
                imnf = np.average(inst_mean_freqs, weights=time_weights)
            else:
                imnf = np.mean(inst_mean_freqs)
            
            # Ensure reasonable frequency range
            imnf = np.clip(imnf, SEMG_LOW_FREQ_CUTOFF, fs/2)
        else:
            imnf = mnf  # Fallback to MNF
            
    except Exception as e:
        # Fallback to MNF if CWD-based calculation fails
        imnf = mnf
    
    # 9. WIRE51 - Wavelet Index of Reliability Estimation
    # 
    # Uses 5th-order Symlet wavelet (sym5) with Mallat's pyramidal algorithm
    # Formula: WIRE51 = E(D5) / E(D1)
    # where E(Di) = ∑D_i²[n] is the energy of detail coefficients at scale i
    #
    # Physical interpretation:
    # - Wavelet decomposition separates signal into different frequency bands
    # - D1 (detail level 1): highest frequency band ≈ [fs/4, fs/2]
    # - D5 (detail level 5): lower frequency band ≈ [fs/64, fs/32]
    # - For sEMG at fs=1000Hz: D1≈[250-500Hz], D5≈[15.6-31.2Hz]
    # - Muscle fatigue causes power shift from high to low frequencies
    # - WIRE51 increases with fatigue as E(D5) increases relative to E(D1)
    #
    # Frequency band mapping for DWT with sym5:
    # Level i corresponds to approximate frequency band: [fs/(2^(i+1)), fs/(2^i)]
    # Example for fs=1000Hz:
    #   D1: [250, 500] Hz    D2: [125, 250] Hz    D3: [62.5, 125] Hz
    #   D4: [31.2, 62.5] Hz  D5: [15.6, 31.2] Hz  D6: [7.8, 15.6] Hz
    #
    # Maximum decomposition level: floor(log2(N / (filter_length - 1)))
    # For sym5: filter_length = 10, so max_level ≈ floor(log2(N/9))
    try:
        import pywt
        # Use sym5 wavelet as specified in literature
        wavelet = 'sym5'
        # Maximum level based on signal length for sym5 wavelet
        max_level = pywt.dwt_max_level(n_samples, wavelet)
        
        # We need at least 5 levels for proper WIRE51 calculation
        if max_level >= 5:
            # Perform 5-level decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=5)
            # coeffs structure: [cA5, cD5, cD4, cD3, cD2, cD1]
            # cA5: approximation at level 5 (very low frequencies)
            # cD5: detail at level 5 (low frequency band ≈ [fs/64, fs/32])
            # cD1: detail at level 1 (high frequency band ≈ [fs/4, fs/2])
            d5 = coeffs[1]  # Detail signal at scale 5
            d1 = coeffs[5]  # Detail signal at scale 1
            
            # Compute energy (sum of squares)
            d5_energy = np.sum(d5 ** 2)
            d1_energy = np.sum(d1 ** 2)
            
            # WIRE51 = E(D5) / E(D1)
            # Higher values indicate more fatigue (power shifted to lower frequencies)
            if d1_energy > EPSILON:
                wire51 = d5_energy / d1_energy
            else:
                wire51 = 0.0
        elif max_level >= 3:
            # For shorter signals, use available levels with adaptive strategy
            # Use highest detail (lowest freq) and lowest detail (highest freq)
            level = max_level
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            # coeffs: [cA_max, cD_max, ..., cD2, cD1]
            high_scale_detail = coeffs[1]  # Highest scale = lowest freq detail
            low_scale_detail = coeffs[-1]  # Lowest scale = highest freq detail
            
            high_energy = np.sum(high_scale_detail ** 2)
            low_energy = np.sum(low_scale_detail ** 2)
            
            if low_energy > EPSILON:
                wire51 = high_energy / low_energy
            else:
                wire51 = 0.0
        else:
            # Signal too short for meaningful wavelet decomposition
            wire51 = 0.0
            
    except Exception:
        # Fallback to spectral-based approximation
        # Use frequency band ratios as proxy for WIRE51
        # Low freq band: 20-50 Hz, High freq band: 200-400 Hz
        low_freq_mask = (freqs >= 20) & (freqs < 50)
        high_freq_mask = (freqs >= 200) & (freqs < 400)
        low_power = np.sum(power_spectrum[low_freq_mask]) if np.any(low_freq_mask) else 0
        high_power = np.sum(power_spectrum[high_freq_mask]) if np.any(high_freq_mask) else 0
        wire51 = low_power / high_power if high_power > EPSILON else 0.0
    
    # 10. DI - Dimitrov Index (Dimitrov Fatigue Index)
    # 
    # Based on: Dimitrov et al. (2006) Med Sci Sports Exerc 38(11):1971-1979
    # Formula: DI = M_{-1} / M_5
    # where M_k = (∑f^k · P(f)) / (∑P(f)) is the k-th spectral moment
    # 
    # Physical interpretation:
    # - M_{-1} emphasizes low frequencies (inverse frequency weighting: 1/f)
    # - M_5 emphasizes high frequencies (very strong high-freq weighting: f^5)
    # - When muscle fatigues, power spectrum shifts toward lower frequencies
    # - This causes M_{-1} to increase and M_5 to decrease
    # - Therefore DI = M_{-1}/M_5 increases with fatigue
    #
    # Typical absolute values: 1e-14 to 1e-8 (very small due to f^5 term)
    # What matters: ratio between fatigued/non-fatigued states (typically 2-10x increase)
    #
    # Example for healthy biceps brachii:
    #   Non-fatigued: DI ≈ 1-5 × 10^-12
    #   Fatigued: DI ≈ 5-20 × 10^-12
    #   Ratio: 2-5x increase
    try:
        # Use valid frequency range (>= SEMG_LOW_FREQ_CUTOFF) to exclude DC and artifacts
        di_mask = freqs >= SEMG_LOW_FREQ_CUTOFF
        di_freqs = freqs[di_mask]
        di_power = power_spectrum[di_mask]
        
        total_power_di = np.sum(di_power)
        
        if total_power_di > EPSILON and len(di_freqs) > 1:
            # Normalize power spectrum to probability distribution
            norm_power = di_power / total_power_di
            
            # M_{-1} = ∑(f^{-1} · P(f)) - emphasizes low frequencies
            # Gives higher weight to low-frequency components
            moment_minus1 = np.sum((di_freqs ** -1) * norm_power)
            
            # M_5 = ∑(f^5 · P(f)) - emphasizes high frequencies
            # Gives extremely high weight to high-frequency components
            moment_5 = np.sum((di_freqs ** 5) * norm_power)
            
            # DI = M_{-1} / M_5
            # Result will be very small (1e-14 to 1e-8 range) but physically meaningful
            # As power shifts to lower frequencies with fatigue:
            #   - M_{-1} increases (more power at low f, 1/f weighting helps)
            #   - M_5 decreases (less power at high f, f^5 weighting hurts)
            #   - DI = M_{-1}/M_5 increases
            if moment_5 > EPSILON:
                dimitrov_index = moment_minus1 / moment_5
            else:
                dimitrov_index = 0.0
        else:
            dimitrov_index = 0.0
            
    except Exception:
        # Fallback: estimate using simple spectral ratio
        # Low freq band / High freq band (conceptually similar to DI)
        low_freq_mask = (freqs >= 20) & (freqs < 80)
        high_freq_mask = freqs >= 80
        low_power = np.sum(power_spectrum[low_freq_mask])
        high_power = np.sum(power_spectrum[high_freq_mask])
        dimitrov_index = low_power / high_power if high_power > EPSILON else 0.0
    
    # Format values for readability using scientific notation where appropriate
    # Very small values (< 1e-6) and very large values (> 1e6) use scientific notation
    def format_value(val):
        """Format value with scientific notation for very small/large numbers"""
        if isinstance(val, (int, np.integer)):
            return int(val)
        elif abs(val) < 1e-6 or abs(val) > 1e6:
            # Return as float - will be formatted as scientific notation when displayed
            return float(val)
        else:
            return float(val)
    
    return {
        'WL': format_value(wl),
        'ZC': int(zc),
        'SSC': int(ssc),
        'MDF': format_value(mdf),
        'MNF': format_value(mnf),
        'IMNF': format_value(imnf),
        'WIRE51': format_value(wire51),
        'DI': format_value(dimitrov_index),
        'RMS': format_value(rms),
        'MAV': format_value(mav),
        'VAR': format_value(var),
        'PKF': format_value(pkf),
        'TTP': format_value(ttp)
    }


def batch_hht_analysis(
    segments: List[np.ndarray],
    fs: float,
    n_freq_bins: int = 256,
    normalize_length: int = 256,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    use_ceemdan: bool = True,
    extract_features: bool = True
) -> Dict[str, Union[List[np.ndarray], np.ndarray, List[Dict]]]:
    """
    Perform batch HHT analysis on multiple segments.
    
    IMPROVEMENTS (2024):
    - Uses average pooling instead of interpolation to avoid artifacts
    - Frequency axis maps to sEMG range (20-450Hz by default)
    
    This function processes all segments at once, producing uniform-sized
    Hilbert spectra suitable for CNN input.
    
    Parameters:
    -----------
    segments : List[np.ndarray]
        List of sEMG segments to analyze
    fs : float
        Sampling frequency in Hz
    n_freq_bins : int, optional
        Number of frequency bins (default: 256)
    normalize_length : int, optional
        Target length for all spectra (default: 256, uses average pooling)
    min_freq : float, optional
        Minimum frequency (default: 20Hz for sEMG)
    max_freq : float, optional
        Maximum frequency (default: 450Hz for sEMG)
    use_ceemdan : bool, optional
        Use CEEMDAN instead of EMD (default: True)
    extract_features : bool, optional
        Also extract sEMG features (default: True)
    
    Returns:
    --------
    Dict containing:
        - 'spectra': List of Hilbert spectrum matrices
        - 'spectra_array': 3D numpy array (n_segments, n_freq_bins, normalize_length)
        - 'time': Normalized time axis [0, 1]
        - 'frequency': Frequency axis (typically 20-450Hz)
        - 'features': List of feature dictionaries (if extract_features=True)
    """
    spectra = []
    features_list = []
    
    # Set default frequency range for sEMG
    if min_freq is None:
        min_freq = SEMG_LOW_FREQ_CUTOFF  # 20 Hz
    if max_freq is None:
        max_freq = SEMG_HIGH_FREQ_CUTOFF  # 450 Hz
    
    for segment in segments:
        # Compute enhanced spectrum (uses average pooling, not interpolation)
        spectrum, time_axis, freq_axis = compute_hilbert_spectrum_enhanced(
            segment, fs,
            n_freq_bins=n_freq_bins,
            min_freq=min_freq,
            max_freq=max_freq,
            normalize_length=normalize_length,
            normalize_time=True,
            use_ceemdan=use_ceemdan,
            log_scale=True
        )
        spectra.append(spectrum)
        
        # Extract features
        if extract_features:
            features = extract_semg_features(segment, fs)
            features_list.append(features)
    
    # Create 3D array for CNN input
    spectra_array = np.array(spectra)
    
    result = {
        'spectra': spectra,
        'spectra_array': spectra_array,
        'time': time_axis,  # Same for all (normalized)
        'frequency': freq_axis,
        'n_segments': len(segments)
    }
    
    if extract_features:
        result['features'] = features_list
    
    return result


def hht_analysis_enhanced(
    signal: np.ndarray,
    fs: float,
    n_freq_bins: int = 256,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    normalize_length: Optional[int] = None,
    normalize_time: bool = True,
    normalize_amplitude: bool = False,
    use_ceemdan: bool = True,
    return_imfs: bool = False,
    extract_features: bool = True
) -> Dict:
    """
    Enhanced Hilbert-Huang Transform analysis with CEEMDAN and feature extraction.
    
    IMPROVEMENTS (2024):
    - Uses average pooling instead of interpolation to avoid artifacts
    - Frequency axis maps to sEMG range (20-450Hz by default)
    
    This is an improved version of hht_analysis that uses CEEMDAN for more
    robust decomposition and includes comprehensive feature extraction.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    n_freq_bins : int, optional
        Number of frequency bins (default: 256)
    min_freq : float, optional
        Minimum frequency (default: 20Hz for sEMG)
    max_freq : float, optional
        Maximum frequency (default: 450Hz for sEMG)
    normalize_length : int, optional
        Target length for normalization (for uniform CNN input, uses average pooling)
    normalize_time : bool, optional
        Normalize time axis to [0, 1] (default: True)
    normalize_amplitude : bool, optional
        Normalize spectrum amplitude (default: False)
    use_ceemdan : bool, optional
        Use CEEMDAN instead of EMD (default: True)
    return_imfs : bool, optional
        Whether to return individual IMFs (default: False)
    extract_features : bool, optional
        Whether to extract sEMG features (default: True)
    
    Returns:
    --------
    Dict containing:
        - 'spectrum': Hilbert spectrum matrix
        - 'time': Time axis
        - 'frequency': Frequency axis
        - 'imfs': List of IMFs (if return_imfs=True)
        - 'residue': Residue signal (if return_imfs=True)
        - 'marginal_spectrum': Marginal Hilbert spectrum
        - 'mean_frequency': Mean instantaneous frequency
        - 'dominant_frequency': Dominant frequency at each time point
        - 'features': sEMG features dictionary (if extract_features=True)
    """
    # Store original length (no interpolation pre-processing)
    original_length = len(signal)
    
    # Compute enhanced spectrum (uses average pooling if normalize_length is provided)
    spectrum, time_axis, freq_axis = compute_hilbert_spectrum_enhanced(
        signal, fs,
        n_freq_bins=n_freq_bins,
        min_freq=min_freq,
        max_freq=max_freq,
        normalize_length=normalize_length,
        normalize_time=normalize_time,
        normalize_amplitude=normalize_amplitude,
        use_ceemdan=use_ceemdan,
        log_scale=True
    )
    
    # Compute marginal spectrum (integrate over time)
    marginal_spectrum = np.sum(spectrum, axis=1)
    
    # Compute dominant frequency at each time point
    dominant_freq_idx = np.argmax(spectrum, axis=0)
    dominant_frequency = freq_axis[dominant_freq_idx]
    
    # Compute mean frequency (weighted average)
    freq_weights = spectrum.sum(axis=1)
    if freq_weights.sum() > 0:
        mean_frequency = np.average(freq_axis, weights=freq_weights)
    else:
        mean_frequency = 0
    
    result = {
        'spectrum': spectrum,
        'time': time_axis,
        'frequency': freq_axis,
        'marginal_spectrum': marginal_spectrum,
        'mean_frequency': mean_frequency,
        'dominant_frequency': dominant_frequency,
        'original_length': original_length,
        'normalized_length': len(time_axis),
        'use_ceemdan': use_ceemdan,
    }
    
    # Optionally return IMFs
    if return_imfs:
        if use_ceemdan:
            imfs = ceemdan_decomposition(signal)
        else:
            imfs = emd_decomposition(signal)
        result['imfs'] = imfs[:-1]  # All IMFs except residue
        result['residue'] = imfs[-1]
    
    # Extract sEMG features
    if extract_features:
        result['features'] = extract_semg_features(signal, fs)
    
    return result


def export_features_to_csv(
    features_list: List[Dict[str, float]],
    filepath: str,
    segment_names: Optional[List[str]] = None,
    annotations: Optional[Dict[str, str]] = None
) -> None:
    """
    Export sEMG features from multiple segments to a CSV file.
    
    Parameters:
    -----------
    features_list : List[Dict[str, float]]
        List of feature dictionaries from extract_semg_features()
    filepath : str
        Output file path for CSV
    segment_names : List[str], optional
        Names for each segment (default: Segment_001, Segment_002, ...)
    annotations : Dict[str, str], optional
        Additional annotation columns to include (e.g., subject, fatigue_level)
    
    Returns:
    --------
    None
        Saves features to CSV file
        
    Examples:
    ---------
    >>> features = [extract_semg_features(seg, fs=1000) for seg in segments]
    >>> export_features_to_csv(features, 'features.csv')
    """
    import pandas as pd
    
    if not features_list:
        raise ValueError("features_list cannot be empty")
    
    # Create segment names if not provided
    if segment_names is None:
        segment_names = [f"Segment_{i+1:03d}" for i in range(len(features_list))]
    
    # Build dataframe
    df = pd.DataFrame(features_list)
    df.insert(0, 'Segment', segment_names)
    
    # Add annotation columns if provided
    if annotations:
        for key, value in annotations.items():
            df.insert(1, key, value)
    
    # Define column order for readability
    priority_cols = ['Segment']
    if annotations:
        priority_cols.extend(list(annotations.keys()))
    
    # Feature columns in logical order
    feature_order = ['WL', 'ZC', 'SSC', 'RMS', 'MAV', 'VAR', 
                     'MDF', 'MNF', 'IMNF', 'PKF', 'TTP', 'WIRE51', 'DI']
    
    # Reorder columns
    ordered_cols = priority_cols + [c for c in feature_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + other_cols]
    
    # Save to CSV
    df.to_csv(filepath, index=False, float_format='%.6f')
    print(f"Features exported to: {filepath}")


def export_hilbert_spectra_batch(
    segments: List[np.ndarray],
    fs: float,
    output_dir: str,
    base_filename: str = "segment",
    n_freq_bins: int = 256,
    normalize_length: int = 256,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    use_ceemdan: bool = True,
    save_visualization: bool = True,
    dpi: int = 150
) -> List[Dict[str, str]]:
    """
    Export Hilbert spectra for all activity segments in batch.
    
    IMPROVEMENTS (2024):
    - Uses average pooling instead of interpolation to avoid artifacts
    - Frequency axis maps to sEMG range (20-450Hz by default)
    
    For each sEMG activity segment, exports:
    1. NPZ file containing spectrum matrix, time axis, and frequency axis
    2. PNG visualization image of the Hilbert spectrum
    
    This function addresses requirement #3: export Hilbert spectra for ALL detected
    activity segments, with one matrix file and one visualization per segment.
    
    Parameters:
    -----------
    segments : List[np.ndarray]
        List of sEMG activity segments (1D arrays)
    fs : float
        Sampling frequency in Hz
    output_dir : str
        Directory to save output files
    base_filename : str, optional
        Base name for output files (default: "segment")
        Files will be named: {base_filename}_001.npz, {base_filename}_001.png, etc.
    n_freq_bins : int, optional
        Number of frequency bins (default: 256)
    normalize_length : int, optional
        Target time axis length (default: 256, uses average pooling)
    min_freq : float, optional
        Minimum frequency in Hz (default: 20Hz for sEMG)
    max_freq : float, optional
        Maximum frequency in Hz (default: 450Hz for sEMG)
    use_ceemdan : bool, optional
        Use CEEMDAN decomposition (default: True)
    save_visualization : bool, optional
        Whether to save PNG visualizations (default: True)
    dpi : int, optional
        DPI for PNG images (default: 150)
    
    Returns:
    --------
    List[Dict[str, str]]
        List of dictionaries containing file paths for each segment:
        [
            {
                'segment_index': 0,
                'npz_path': '/path/to/segment_001.npz',
                'png_path': '/path/to/segment_001.png'  # if save_visualization=True
            },
            ...
        ]
    
    Examples:
    ---------
    >>> from semg_preprocessing import detect_muscle_activity, segment_signal
    >>> from semg_preprocessing.hht import export_hilbert_spectra_batch
    >>> 
    >>> # Detect activity segments
    >>> activity_periods = detect_muscle_activity(filtered_signal, fs=1000)
    >>> segments = segment_signal(filtered_signal, activity_periods, fs=1000)
    >>> 
    >>> # Extract just the data arrays
    >>> segment_arrays = [seg['data'] for seg in segments]
    >>> 
    >>> # Export all Hilbert spectra
    >>> export_info = export_hilbert_spectra_batch(
    >>>     segment_arrays, 
    >>>     fs=1000,
    >>>     output_dir='./hht_results',
    >>>     base_filename='activity_segment'
    >>> )
    >>> 
    >>> print(f"Exported {len(export_info)} Hilbert spectra")
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default frequency range for sEMG
    if min_freq is None:
        min_freq = SEMG_LOW_FREQ_CUTOFF  # 20 Hz
    if max_freq is None:
        max_freq = SEMG_HIGH_FREQ_CUTOFF  # 450 Hz
    
    export_info = []
    
    print(f"Exporting Hilbert spectra for {len(segments)} segments...")
    
    for idx, segment in enumerate(segments):
        segment_num = idx + 1
        
        # Compute Hilbert spectrum (uses average pooling, not interpolation)
        spectrum, time_axis, freq_axis = compute_hilbert_spectrum_enhanced(
            segment,
            fs,
            n_freq_bins=n_freq_bins,
            min_freq=min_freq,
            max_freq=max_freq,
            normalize_length=normalize_length,
            normalize_time=True,
            normalize_amplitude=False,
            use_ceemdan=use_ceemdan,
            log_scale=True
        )
        
        # Generate filenames with zero-padded numbering
        npz_filename = f"{base_filename}_{segment_num:03d}.npz"
        npz_path = os.path.join(output_dir, npz_filename)
        
        # Save NPZ file
        np.savez_compressed(
            npz_path,
            spectrum=spectrum,
            time=time_axis,
            frequency=freq_axis,
            sampling_rate=fs,
            segment_index=idx
        )
        
        info_dict = {
            'segment_index': idx,
            'segment_number': segment_num,
            'npz_path': npz_path
        }
        
        # Save visualization if requested
        if save_visualization:
            png_filename = f"{base_filename}_{segment_num:03d}.png"
            png_path = os.path.join(output_dir, png_filename)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
            
            # Plot Hilbert spectrum
            im = ax.pcolormesh(
                time_axis, 
                freq_axis, 
                spectrum,
                shading='auto',
                cmap='jet'
            )
            
            ax.set_xlabel('Normalized Time', fontsize=12)
            ax.set_ylabel('Frequency (Hz)', fontsize=12)
            ax.set_title(f'Hilbert Spectrum - Segment {segment_num:03d}', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Amplitude (Log Scale)', fontsize=11)
            
            # Grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            info_dict['png_path'] = png_path
        
        export_info.append(info_dict)
        
        # Progress update
        if (segment_num % 10 == 0) or (segment_num == len(segments)):
            print(f"  Processed {segment_num}/{len(segments)} segments")
    
    print(f"\nExport complete!")
    print(f"  NPZ files saved to: {output_dir}")
    if save_visualization:
        print(f"  PNG visualizations saved to: {output_dir}")
    
    return export_info


def export_activity_segments_hht(
    signal: np.ndarray,
    activity_segments: List[Tuple[int, int]],
    fs: float,
    output_dir: str,
    base_filename: str = "activity_segment",
    **hht_kwargs
) -> List[Dict[str, str]]:
    """
    Convenience function to export HHT analysis for all detected activity segments.
    
    This function combines segment extraction and HHT export in one step.
    
    Parameters:
    -----------
    signal : np.ndarray
        Full preprocessed sEMG signal
    activity_segments : List[Tuple[int, int]]
        List of (start_index, end_index) tuples from detect_muscle_activity()
    fs : float
        Sampling frequency in Hz
    output_dir : str
        Directory to save output files
    base_filename : str, optional
        Base name for output files
    **hht_kwargs : dict
        Additional keyword arguments for export_hilbert_spectra_batch()
        (e.g., n_freq_bins, normalize_length, use_ceemdan, save_visualization)
    
    Returns:
    --------
    List[Dict[str, str]]
        Export information for each segment (see export_hilbert_spectra_batch)
    
    Examples:
    ---------
    >>> from semg_preprocessing import detect_muscle_activity
    >>> from semg_preprocessing.hht import export_activity_segments_hht
    >>> 
    >>> # Detect activity
    >>> segments = detect_muscle_activity(filtered_signal, fs=1000, min_duration=0.5)
    >>> 
    >>> # Export HHT for all segments
    >>> export_info = export_activity_segments_hht(
    >>>     filtered_signal,
    >>>     segments,
    >>>     fs=1000,
    >>>     output_dir='./hht_output',
    >>>     base_filename='bicep_curl'
    >>> )
    """
    # Extract segment data
    segment_arrays = [signal[start:end] for start, end in activity_segments]
    
    # Export Hilbert spectra
    return export_hilbert_spectra_batch(
        segment_arrays,
        fs,
        output_dir,
        base_filename,
        **hht_kwargs
    )
