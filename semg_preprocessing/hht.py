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


def compute_hilbert_spectrum(
    signal: np.ndarray,
    fs: float,
    n_freq_bins: int = 256,
    max_freq: Optional[float] = None,
    normalize_length: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hilbert Spectrum (time-frequency representation) using HHT.
    
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
    max_freq : float, optional
        Maximum frequency to display (default: fs/2)
    normalize_length : int, optional
        If provided, resample signal to this length before HHT
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
    from scipy.signal import resample
    
    # Optionally normalize length
    if normalize_length is not None and len(signal) != normalize_length:
        signal = resample(signal, normalize_length)
    
    n_samples = len(signal)
    
    if max_freq is None:
        max_freq = fs / 2
    
    # Create time and frequency axes
    time_axis = np.arange(n_samples) / fs
    freq_axis = np.linspace(0, max_freq, n_freq_bins)
    
    # Initialize spectrum
    spectrum = np.zeros((n_freq_bins, n_samples))
    
    # Perform EMD
    imfs = emd_decomposition(signal)
    
    # Process each IMF
    for imf in imfs[:-1]:  # Exclude residue
        # Compute Hilbert transform
        _, amplitude, phase = hilbert_transform(imf)
        
        # Compute instantaneous frequency
        inst_freq = compute_instantaneous_frequency(phase, fs)
        
        # Map to spectrum
        for t in range(n_samples):
            freq = inst_freq[t]
            amp = amplitude[t]
            
            # Find nearest frequency bin
            if 0 <= freq <= max_freq:
                freq_bin = int(freq / max_freq * (n_freq_bins - 1))
                freq_bin = min(freq_bin, n_freq_bins - 1)
                spectrum[freq_bin, t] += amp
    
    return spectrum, time_axis, freq_axis


def hht_analysis(
    signal: np.ndarray,
    fs: float,
    n_freq_bins: int = 256,
    max_freq: Optional[float] = None,
    normalize_length: Optional[int] = None,
    return_imfs: bool = False
) -> Dict:
    """
    Complete Hilbert-Huang Transform analysis.
    
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
    max_freq : float, optional
        Maximum frequency (default: fs/2)
    normalize_length : int, optional
        Target length for normalization
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
    from scipy.signal import resample
    
    # Normalize length if requested
    original_length = len(signal)
    if normalize_length is not None and len(signal) != normalize_length:
        signal = resample(signal, normalize_length)
    
    # Compute spectrum
    spectrum, time_axis, freq_axis = compute_hilbert_spectrum(
        signal, fs, n_freq_bins, max_freq, normalize_length=None
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


def compute_hilbert_spectrum_enhanced(
    signal: np.ndarray,
    fs: float,
    n_freq_bins: int = 256,
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
    
    This function addresses the issue of mostly black spectrograms by:
    1. Using log-scale amplitude representation
    2. Applying adaptive thresholding
    3. Normalizing time axis for uniform matrix sizes
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D array)
    fs : float
        Sampling frequency in Hz
    n_freq_bins : int, optional
        Number of frequency bins (default: 256)
    max_freq : float, optional
        Maximum frequency to display (default: fs/2)
    normalize_length : int, optional
        If provided, resample signal to this length before HHT
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
        - Frequency axis
    """
    from scipy.signal import resample
    
    # Optionally normalize length
    original_length = len(signal)
    if normalize_length is not None and len(signal) != normalize_length:
        signal = resample(signal, normalize_length)
    
    n_samples = len(signal)
    
    if max_freq is None:
        max_freq = fs / 2
    
    # Create time axis (normalized or absolute)
    if normalize_time:
        time_axis = np.linspace(0, 1, n_samples)
    else:
        time_axis = np.arange(n_samples) / fs
    
    freq_axis = np.linspace(0, max_freq, n_freq_bins)
    
    # Initialize spectrum
    spectrum = np.zeros((n_freq_bins, n_samples))
    
    # Perform decomposition
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
        
        # Map to spectrum with Gaussian smoothing
        for t in range(n_samples):
            freq = inst_freq[t]
            amp = amplitude[t]
            
            if 0 <= freq <= max_freq:
                # Find nearest frequency bin
                freq_bin = int(freq / max_freq * (n_freq_bins - 1))
                freq_bin = min(freq_bin, n_freq_bins - 1)
                
                # Add to spectrum with slight frequency spreading for smoother display
                spread = 2  # Frequency bins to spread
                for df in range(-spread, spread + 1):
                    fb = freq_bin + df
                    if 0 <= fb < n_freq_bins:
                        weight = np.exp(-0.5 * (df / 1.0) ** 2)  # Gaussian weight
                        spectrum[fb, t] += amp * weight
    
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
    
    Parameters:
    -----------
    signal : np.ndarray
        Input sEMG signal
    fs : float
        Sampling frequency in Hz
    
    Returns:
    --------
    Dict[str, float]
        Dictionary containing:
        - WL: Waveform Length
        - ZC: Zero Crossings
        - SSC: Slope Sign Changes
        - MDF: Median Frequency
        - MNF: Mean Frequency
        - IMNF: Instantaneous Mean Frequency (from HHT)
        - WIRE51: Wavelet-based Index of Reliability Estimation (DWT-based)
        - DI: Dimitrov Index (fatigue indicator, DWT-based)
        - RMS: Root Mean Square
        - MAV: Mean Absolute Value
        - VAR: Variance
        - PKF: Peak Frequency
        - TTP: Total Power
    """
    import pywt
    
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
    
    # Frequency domain features
    # Compute power spectrum
    fft_signal = np.fft.rfft(signal)
    power_spectrum = np.abs(fft_signal) ** 2
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    
    # For sEMG signals, exclude DC and very low frequencies (< SEMG_LOW_FREQ_CUTOFF)
    # These are typically artifacts, baseline drift, or motion noise
    # sEMG frequency content is typically in 20-450 Hz range
    freq_mask = freqs >= SEMG_LOW_FREQ_CUTOFF  # Exclude frequencies below cutoff
    valid_freqs = freqs[freq_mask]
    valid_power = power_spectrum[freq_mask]
    
    # Total power (using valid frequency range)
    ttp_valid = np.sum(valid_power)
    ttp = np.sum(power_spectrum)  # Total including DC for compatibility
    
    # 5. Median Frequency (MDF) - frequency that divides power spectrum in half
    # Use valid frequency range to avoid DC bias
    if ttp_valid > 0:
        cumulative_power = np.cumsum(valid_power)
        half_power = ttp_valid / 2
        mdf_idx = np.searchsorted(cumulative_power, half_power)
        mdf = valid_freqs[min(mdf_idx, len(valid_freqs) - 1)]
    else:
        mdf = 0
    
    # 6. Mean Frequency (MNF) - weighted average of frequencies
    # Use valid frequency range to avoid DC bias
    if ttp_valid > 0:
        mnf = np.sum(valid_freqs * valid_power) / ttp_valid
    else:
        mnf = 0
    
    # 7. Peak Frequency (PKF)
    # Use valid frequency range to find dominant sEMG frequency
    if len(valid_power) > 0:
        pkf = valid_freqs[np.argmax(valid_power)]
    else:
        pkf = 0
    
    # 8. IMNF - Instantaneous Mean Frequency
    # Computed using Hilbert transform to obtain instantaneous frequency.
    # IMNF is the power-weighted average of instantaneous frequencies.
    # Formula: IMNF = sum(IF(t) * A^2(t)) / sum(A^2(t))
    # where IF(t) is instantaneous frequency and A(t) is instantaneous amplitude
    try:
        from scipy.signal import hilbert as scipy_hilbert, butter, filtfilt
        
        # Apply high-pass filter to remove DC and low-frequency drift
        # This prevents low-frequency artifacts from affecting instantaneous frequency
        # Use 4th order Butterworth high-pass filter at SEMG_LOW_FREQ_CUTOFF
        min_fs_for_filter = 2 * SEMG_LOW_FREQ_CUTOFF  # Minimum sampling rate for filtering
        if fs > min_fs_for_filter:  # Only filter if sampling rate is high enough
            nyquist = fs / 2
            cutoff = SEMG_LOW_FREQ_CUTOFF / nyquist
            max_normalized_cutoff = 1.0  # Maximum valid normalized cutoff frequency
            if cutoff < max_normalized_cutoff:  # Valid cutoff frequency
                b, a = butter(4, cutoff, btype='high')
                signal_filtered = filtfilt(b, a, signal)
            else:
                signal_filtered = signal
        else:
            signal_filtered = signal
        
        # Compute analytic signal from filtered signal
        analytic = scipy_hilbert(signal_filtered)
        inst_amplitude = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))
        
        # Compute instantaneous frequency from phase derivative
        inst_freq_signal = np.diff(inst_phase) / (2 * np.pi) * fs
        inst_freq_signal = np.concatenate([inst_freq_signal, [inst_freq_signal[-1]]])
        inst_freq_signal = np.abs(inst_freq_signal)
        
        # Clip to valid sEMG frequency range (SEMG_LOW_FREQ_CUTOFF to Nyquist)
        inst_freq_signal = np.clip(inst_freq_signal, SEMG_LOW_FREQ_CUTOFF, fs/2)
        
        # Power-weighted average using amplitude^2 as power
        power = inst_amplitude ** 2
        total_power_inst = np.sum(power)
        if total_power_inst > EPSILON:
            imnf = np.sum(inst_freq_signal * power) / total_power_inst
        else:
            imnf = mnf
        
        # Ensure reasonable frequency range
        imnf = np.clip(imnf, SEMG_LOW_FREQ_CUTOFF, fs/2)
    except Exception:
        imnf = mnf  # Fallback to MNF
    
    # 9. WIRE51 - Wavelet Index of Reliability Estimation (DWT-based)
    # Uses 5th-order Symlet wavelet (sym5) with Mallat algorithm
    # Formula: WIRE51 = sum(D5[n]^2) / sum(D1[n]^2)
    # where D5 and D1 are detail signals at scales 5 and 1 respectively
    # Maximum decomposition level depends on signal length: floor(log2(N))
    try:
        import pywt
        # Use sym5 wavelet as specified in literature
        wavelet = 'sym5'
        # Maximum level based on signal length for sym5 wavelet
        max_level = pywt.dwt_max_level(n_samples, wavelet)
        
        # We need at least 5 levels for proper WIRE51 calculation
        if max_level >= 5:
            coeffs = pywt.wavedec(signal, wavelet, level=5)
            # coeffs structure: [cA5, cD5, cD4, cD3, cD2, cD1]
            # D5 is coeffs[1] (detail at level 5)
            # D1 is coeffs[5] (detail at level 1)
            d5 = coeffs[1]  # Detail signal at scale 5 (low frequency details)
            d1 = coeffs[5]  # Detail signal at scale 1 (high frequency details)
            
            d5_power = np.sum(d5 ** 2)
            d1_power = np.sum(d1 ** 2)
            
            # WIRE51 = D5_power / D1_power, higher values indicate more fatigue
            wire51 = d5_power / (d1_power + EPSILON)
        else:
            # For short signals, use available levels
            # Fallback: use highest and lowest detail levels available
            level = min(max_level, 3)  # At least 3 levels if possible
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            if len(coeffs) >= 3:
                # Use highest detail (low freq) and lowest detail (high freq)
                high_scale_detail = coeffs[1]  # Highest scale detail available
                low_scale_detail = coeffs[-1]  # Lowest scale detail available
                wire51 = np.sum(high_scale_detail ** 2) / (np.sum(low_scale_detail ** 2) + EPSILON)
            else:
                wire51 = 0.0
    except Exception:
        # Fallback to spectral-based calculation
        # WIRE51 approximation: high_freq_power / low_freq_power
        low_freq_mask = freqs < 50
        high_freq_mask = freqs >= 50
        low_power = np.sum(power_spectrum[low_freq_mask])
        high_power = np.sum(power_spectrum[high_freq_mask])
        wire51 = high_power / (low_power + EPSILON) if low_power > EPSILON else 0
    
    # 10. DI - Dimitrov Index (Dimitrov Fatigue Index, FInsm5)
    # Based on Dimitrov et al. (2006): Med Sci Sports Exerc 38(11):1971-1979
    # Formula: DI = M_{-1} / M_5
    # where M_k = sum(f^k * P(f)) / sum(P(f))
    # 
    # Physical interpretation:
    # - M_{-1} gives more weight to low frequencies (1/f weighting)
    # - M_5 gives more weight to high frequencies (f^5 weighting)
    # - DI = M_{-1} / M_5 increases with fatigue as spectrum shifts to lower frequencies
    # 
    # Typical values: 1e-11 to 1e-9 (very small due to f^5 term)
    # The ratio between fatigued/non-fatigued conditions is what matters (typically 2-10x increase)
    try:
        # Calculate spectral moments with original formula
        # Use valid frequency range (>= SEMG_LOW_FREQ_CUTOFF) to exclude DC and low-frequency artifacts
        di_mask = freqs >= SEMG_LOW_FREQ_CUTOFF
        di_freqs = freqs[di_mask]
        di_power = power_spectrum[di_mask]
        
        total_power_di = np.sum(di_power)
        
        if total_power_di > EPSILON and len(di_freqs) > 0:
            # Normalize power spectrum to probability distribution
            norm_power = di_power / total_power_di
            
            # M_{-1} = sum(f^{-1} * P(f)) - emphasizes low frequencies
            moment_minus1 = np.sum((di_freqs ** -1) * norm_power)
            
            # M_5 = sum(f^5 * P(f)) - emphasizes high frequencies
            moment_5 = np.sum((di_freqs ** 5) * norm_power)
            
            # DI = M_{-1} / M_5
            # This will be a very small number (1e-11 to 1e-9 range)
            # but increases with fatigue as power shifts to lower frequencies
            if moment_5 > EPSILON:
                dimitrov_index = moment_minus1 / moment_5
            else:
                dimitrov_index = 0.0
        else:
            dimitrov_index = 0.0
            
    except Exception:
        # Fallback: estimate using spectral ratio (higher = more fatigue)
        # This approximates the DI concept: low freq / high freq power
        low_freq_mask = freqs < 80  # Low frequency band
        high_freq_mask = freqs >= 80  # High frequency band
        low_power = np.sum(power_spectrum[low_freq_mask])
        high_power = np.sum(power_spectrum[high_freq_mask])
        dimitrov_index = low_power / (high_power + EPSILON) if high_power > EPSILON else 0
    
    return {
        'WL': float(wl),
        'ZC': int(zc),
        'SSC': int(ssc),
        'MDF': float(mdf),
        'MNF': float(mnf),
        'IMNF': float(imnf),
        'WIRE51': float(wire51),
        'DI': float(dimitrov_index),
        'RMS': float(rms),
        'MAV': float(mav),
        'VAR': float(var),
        'PKF': float(pkf),
        'TTP': float(ttp)
    }


def batch_hht_analysis(
    segments: List[np.ndarray],
    fs: float,
    n_freq_bins: int = 256,
    normalize_length: int = 256,
    max_freq: Optional[float] = None,
    use_ceemdan: bool = True,
    extract_features: bool = True
) -> Dict[str, Union[List[np.ndarray], np.ndarray, List[Dict]]]:
    """
    Perform batch HHT analysis on multiple segments.
    
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
        Target length for all spectra (default: 256)
    max_freq : float, optional
        Maximum frequency (default: fs/2)
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
        - 'frequency': Frequency axis
        - 'features': List of feature dictionaries (if extract_features=True)
    """
    spectra = []
    features_list = []
    
    if max_freq is None:
        max_freq = fs / 2
    
    for segment in segments:
        # Compute enhanced spectrum
        spectrum, time_axis, freq_axis = compute_hilbert_spectrum_enhanced(
            segment, fs,
            n_freq_bins=n_freq_bins,
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
    max_freq : float, optional
        Maximum frequency (default: fs/2)
    normalize_length : int, optional
        Target length for normalization (for uniform CNN input)
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
    from scipy.signal import resample
    
    # Store original length
    original_length = len(signal)
    
    # Normalize length if requested
    if normalize_length is not None and len(signal) != normalize_length:
        signal = resample(signal, normalize_length)
    
    # Compute enhanced spectrum
    spectrum, time_axis, freq_axis = compute_hilbert_spectrum_enhanced(
        signal, fs,
        n_freq_bins=n_freq_bins,
        max_freq=max_freq,
        normalize_length=None,  # Already resampled above
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
        'normalized_length': len(signal),
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
