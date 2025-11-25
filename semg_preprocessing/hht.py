"""
Hilbert-Huang Transform (HHT) module for sEMG signal analysis.

This module implements:
1. Empirical Mode Decomposition (EMD) for signal decomposition into IMFs
2. Hilbert Transform for instantaneous frequency and amplitude
3. Hilbert Spectrum generation for time-frequency analysis
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.signal import hilbert
from scipy.interpolate import interp1d


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
        
        # Check for convergence
        sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-10)
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
