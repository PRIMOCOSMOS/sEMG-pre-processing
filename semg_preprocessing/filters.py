"""
Signal filtering functions for sEMG preprocessing.

This module implements various filtering techniques for sEMG signal preprocessing:
- High-pass filtering (10-20Hz) to remove motion artifacts and baseline drift
- Low-pass filtering (450-500Hz) to remove high-frequency noise
- Bandpass filtering (combination of high-pass and low-pass)
- Notch filtering for power line interference (50Hz and harmonics)
- DFT-based power line interference removal
"""

import numpy as np
from scipy import signal
from typing import Optional, Union, List


def apply_highpass_filter(
    data: np.ndarray,
    fs: float,
    cutoff: float = 20.0,
    order: int = 4,
    filter_type: str = "butterworth"
) -> np.ndarray:
    """
    Apply high-pass filter to remove motion artifacts, baseline drift, and ECG interference.
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data (1D array)
    fs : float
        Sampling frequency in Hz
    cutoff : float, optional
        Cutoff frequency in Hz (default: 20.0, recommended: 10-20Hz)
    order : int, optional
        Filter order (default: 4, recommended: 2-4)
    filter_type : str, optional
        Type of filter: 'butterworth' or 'chebyshev' (default: 'butterworth')
    
    Returns:
    --------
    np.ndarray
        Filtered signal
        
    Notes:
    ------
    - High-pass filtering removes low-frequency components including:
      * Motion artifacts
      * Baseline drift
      * ECG interference (mostly below 30Hz)
    - Higher cutoff frequencies (e.g., 20Hz) provide better ECG rejection
    - Be cautious with higher orders as they may cause signal distortion
    """
    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    
    if filter_type.lower() == "butterworth":
        b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
    elif filter_type.lower() == "chebyshev":
        # Chebyshev Type I filter with 0.5 dB ripple
        b, a = signal.cheby1(order, 0.5, normalized_cutoff, btype='high', analog=False)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Use 'butterworth' or 'chebyshev'")
    
    # Use filtfilt for zero-phase filtering
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data


def apply_lowpass_filter(
    data: np.ndarray,
    fs: float,
    cutoff: float = 450.0,
    order: int = 4,
    filter_type: str = "butterworth"
) -> np.ndarray:
    """
    Apply low-pass filter to remove high-frequency noise.
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data (1D array)
    fs : float
        Sampling frequency in Hz
    cutoff : float, optional
        Cutoff frequency in Hz (default: 450.0, recommended: 450-500Hz)
    order : int, optional
        Filter order (default: 4, recommended: 2-4)
    filter_type : str, optional
        Type of filter: 'butterworth' or 'chebyshev' (default: 'butterworth')
    
    Returns:
    --------
    np.ndarray
        Filtered signal
        
    Notes:
    ------
    - EMG signal highest frequency components are around 400-500Hz
    - Low-pass filtering removes noise above the useful EMG frequency range
    """
    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff >= 1.0:
        raise ValueError(f"Cutoff frequency ({cutoff}Hz) must be less than Nyquist frequency ({nyquist}Hz)")
    
    if filter_type.lower() == "butterworth":
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    elif filter_type.lower() == "chebyshev":
        # Chebyshev Type I filter with 0.5 dB ripple
        b, a = signal.cheby1(order, 0.5, normalized_cutoff, btype='low', analog=False)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Use 'butterworth' or 'chebyshev'")
    
    # Use filtfilt for zero-phase filtering
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data


def apply_bandpass_filter(
    data: np.ndarray,
    fs: float,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    order: int = 4,
    filter_type: str = "butterworth"
) -> np.ndarray:
    """
    Apply bandpass filter (combination of high-pass and low-pass).
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data (1D array)
    fs : float
        Sampling frequency in Hz
    lowcut : float, optional
        Low cutoff frequency in Hz (default: 20.0)
    highcut : float, optional
        High cutoff frequency in Hz (default: 450.0)
    order : int, optional
        Filter order (default: 4, recommended: 2-4)
    filter_type : str, optional
        Type of filter: 'butterworth' or 'chebyshev' (default: 'butterworth')
    
    Returns:
    --------
    np.ndarray
        Filtered signal
        
    Notes:
    ------
    - Combines high-pass and low-pass filtering in a single operation
    - More efficient than applying filters separately
    """
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if high >= 1.0:
        raise ValueError(f"High cutoff frequency ({highcut}Hz) must be less than Nyquist frequency ({nyquist}Hz)")
    
    if filter_type.lower() == "butterworth":
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
    elif filter_type.lower() == "chebyshev":
        # Chebyshev Type I filter with 0.5 dB ripple
        b, a = signal.cheby1(order, 0.5, [low, high], btype='band', analog=False)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Use 'butterworth' or 'chebyshev'")
    
    # Use filtfilt for zero-phase filtering
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data


def apply_notch_filter(
    data: np.ndarray,
    fs: float,
    freq: float = 50.0,
    quality_factor: float = 30.0,
    harmonics: Optional[List[int]] = None
) -> np.ndarray:
    """
    Apply notch filter to remove power line interference.
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data (1D array)
    fs : float
        Sampling frequency in Hz
    freq : float, optional
        Frequency to remove in Hz (default: 50.0 for European power line)
        Use 60.0 for US/North American power line
    quality_factor : float, optional
        Quality factor (default: 30.0, higher = narrower notch)
    harmonics : List[int], optional
        List of harmonic multipliers to filter (e.g., [1, 2, 3] for 50Hz, 100Hz, 150Hz)
        If None, only filters the fundamental frequency
    
    Returns:
    --------
    np.ndarray
        Filtered signal
        
    Notes:
    ------
    - Notch filters are designed to remove specific frequency components
    - Power line interference typically occurs at 50Hz (Europe/Asia) or 60Hz (Americas)
    - Harmonics (100Hz, 150Hz, etc.) may also need to be filtered
    - Cascading multiple notch filters for harmonics is common practice
    """
    filtered_data = data.copy()
    
    if harmonics is None:
        harmonics = [1]
    
    for harmonic in harmonics:
        target_freq = freq * harmonic
        
        # Check if target frequency is below Nyquist
        if target_freq >= fs / 2.0:
            print(f"Warning: Harmonic frequency {target_freq}Hz exceeds Nyquist frequency. Skipping.")
            continue
        
        # Design notch filter
        b, a = signal.iirnotch(target_freq, quality_factor, fs)
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, filtered_data)
    
    return filtered_data


def remove_powerline_dft(
    data: np.ndarray,
    fs: float,
    freq: float = 50.0,
    harmonics: Optional[List[int]] = None,
    bandwidth: float = 1.0
) -> np.ndarray:
    """
    Remove power line interference using DFT (Discrete Fourier Transform) method.
    
    This method transforms the signal to frequency domain, removes target frequency
    components, and reconstructs the signal.
    
    Parameters:
    -----------
    data : np.ndarray
        Input sEMG signal data (1D array)
    fs : float
        Sampling frequency in Hz
    freq : float, optional
        Power line frequency in Hz (default: 50.0)
        Use 60.0 for US/North American power line
    harmonics : List[int], optional
        List of harmonic multipliers to remove (e.g., [1, 2, 3])
        If None, only removes the fundamental frequency
    bandwidth : float, optional
        Bandwidth around target frequency to remove in Hz (default: 1.0)
    
    Returns:
    --------
    np.ndarray
        Filtered signal with power line interference removed
        
    Notes:
    ------
    - Alternative to notch filtering
    - Provides more precise control over frequency removal
    - May be more effective for certain types of interference
    """
    if harmonics is None:
        harmonics = [1]
    
    # Perform FFT
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1/fs)
    
    # Create a copy for modification
    fft_filtered = fft_data.copy()
    
    # Remove each harmonic
    for harmonic in harmonics:
        target_freq = freq * harmonic
        
        # Find indices within bandwidth of target frequency
        # Consider both positive and negative frequencies
        freq_mask = np.abs(np.abs(freqs) - target_freq) < bandwidth
        
        # Zero out the target frequency components
        fft_filtered[freq_mask] = 0
    
    # Inverse FFT to reconstruct signal
    filtered_data = np.fft.ifft(fft_filtered).real
    
    return filtered_data
