"""
Data Augmentation module for sEMG signals.

This module implements CEEMDAN-based data augmentation techniques:
1. IMF decomposition and recombination for synthetic signal generation
2. IMF mixing across different signals for data augmentation
3. Various augmentation strategies for building sEMG datasets
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from .hht import emd_decomposition, ceemdan_decomposition, DEFAULT_CEEMDAN_ENSEMBLES


def augment_by_imf_mixing(
    signal: np.ndarray,
    n_augmented: int = 5,
    imf_perturbation: float = 0.1,
    use_ceemdan: bool = True,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate augmented signals by perturbing IMF components.
    
    This method decomposes the signal using CEEMDAN/EMD, then creates variations
    by slightly modifying each IMF's amplitude and phase.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input sEMG signal
    n_augmented : int, optional
        Number of augmented signals to generate (default: 5)
    imf_perturbation : float, optional
        Maximum perturbation ratio for IMFs (default: 0.1 = 10%)
    use_ceemdan : bool, optional
        Use CEEMDAN instead of EMD (default: True)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    List[np.ndarray]
        List of augmented signals (including original)
        
    Examples:
    ---------
    >>> augmented_signals = augment_by_imf_mixing(segment, n_augmented=10)
    >>> print(f"Generated {len(augmented_signals)} signals")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Decompose original signal using CEEMDAN or EMD
    if use_ceemdan:
        imfs = ceemdan_decomposition(signal, n_ensembles=DEFAULT_CEEMDAN_ENSEMBLES)
    else:
        imfs = emd_decomposition(signal)
    
    augmented = [signal.copy()]  # Include original
    
    for _ in range(n_augmented):
        # Create new signal by perturbing IMFs
        new_signal = np.zeros_like(signal)
        
        for imf in imfs:
            # Random amplitude scaling
            scale = 1.0 + np.random.uniform(-imf_perturbation, imf_perturbation)
            # Random phase shift (circular shift)
            shift = int(np.random.uniform(-len(imf) * imf_perturbation, 
                                          len(imf) * imf_perturbation))
            
            perturbed_imf = scale * np.roll(imf, shift)
            new_signal += perturbed_imf
        
        augmented.append(new_signal)
    
    return augmented


def augment_by_imf_recombination(
    signals: List[np.ndarray],
    n_augmented: int = 5,
    n_imfs_to_swap: Optional[int] = None,
    use_ceemdan: bool = True,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate augmented signals by recombining IMFs from different signals.
    
    This method extracts IMFs from multiple signals and creates new synthetic
    signals by combining IMFs from different sources.
    
    Parameters:
    -----------
    signals : List[np.ndarray]
        List of input sEMG signals (should be similar in nature/length)
    n_augmented : int, optional
        Number of augmented signals to generate (default: 5)
    n_imfs_to_swap : int, optional
        Number of IMFs to swap (default: random)
    use_ceemdan : bool, optional
        Use CEEMDAN instead of EMD (default: True)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    List[np.ndarray]
        List of augmented signals
        
    Examples:
    ---------
    >>> signals = [segment1, segment2, segment3]
    >>> augmented = augment_by_imf_recombination(signals, n_augmented=10)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if len(signals) < 2:
        raise ValueError("Need at least 2 signals for recombination")
    
    # Normalize lengths by resampling to match the first signal
    from scipy.signal import resample
    target_length = len(signals[0])
    
    normalized_signals = []
    for sig in signals:
        if len(sig) != target_length:
            sig = resample(sig, target_length)
        normalized_signals.append(sig)
    
    # Decompose all signals using CEEMDAN or EMD
    if use_ceemdan:
        all_imfs = [ceemdan_decomposition(sig, n_ensembles=DEFAULT_CEEMDAN_ENSEMBLES) for sig in normalized_signals]
    else:
        all_imfs = [emd_decomposition(sig) for sig in normalized_signals]
    
    # Find minimum number of IMFs across all signals
    min_n_imfs = min(len(imfs) for imfs in all_imfs)
    
    augmented = []
    
    for _ in range(n_augmented):
        # Start with a random base signal
        base_idx = np.random.randint(len(all_imfs))
        base_imfs = all_imfs[base_idx]
        
        # Determine how many IMFs to swap
        if n_imfs_to_swap is None:
            n_swap = np.random.randint(1, min_n_imfs)
        else:
            n_swap = min(n_imfs_to_swap, min_n_imfs - 1)
        
        # Randomly select which IMFs to swap
        swap_indices = np.random.choice(min_n_imfs - 1, n_swap, replace=False)
        
        # Create new signal
        new_signal = np.zeros(target_length)
        
        for i, imf in enumerate(base_imfs):
            if i < min_n_imfs - 1:  # Exclude residue
                if i in swap_indices:
                    # Swap with IMF from another signal
                    donor_idx = np.random.randint(len(all_imfs))
                    while donor_idx == base_idx:
                        donor_idx = np.random.randint(len(all_imfs))
                    new_signal += all_imfs[donor_idx][i]
                else:
                    new_signal += imf
            else:
                # Keep residue from base
                new_signal += imf
        
        augmented.append(new_signal)
    
    return augmented


def augment_by_imf_scaling(
    signal: np.ndarray,
    n_augmented: int = 5,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    selective_imfs: Optional[List[int]] = None,
    use_ceemdan: bool = True,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate augmented signals by scaling specific IMF components.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input sEMG signal
    n_augmented : int, optional
        Number of augmented signals to generate (default: 5)
    scale_range : Tuple[float, float], optional
        Range of scaling factors (default: 0.8 to 1.2)
    selective_imfs : List[int], optional
        Indices of IMFs to scale (default: all except residue)
    use_ceemdan : bool, optional
        Use CEEMDAN instead of EMD (default: True)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    List[np.ndarray]
        List of augmented signals
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use CEEMDAN or EMD
    if use_ceemdan:
        imfs = ceemdan_decomposition(signal, n_ensembles=DEFAULT_CEEMDAN_ENSEMBLES)
    else:
        imfs = emd_decomposition(signal)
    n_imfs = len(imfs) - 1  # Exclude residue
    
    if selective_imfs is None:
        selective_imfs = list(range(n_imfs))
    
    augmented = [signal.copy()]  # Include original
    
    for _ in range(n_augmented):
        new_signal = np.zeros_like(signal)
        
        for i, imf in enumerate(imfs):
            if i < n_imfs and i in selective_imfs:
                # Apply random scaling
                scale = np.random.uniform(scale_range[0], scale_range[1])
                new_signal += scale * imf
            else:
                new_signal += imf
        
        augmented.append(new_signal)
    
    return augmented


def augment_by_noise_injection(
    signal: np.ndarray,
    n_augmented: int = 5,
    noise_level: float = 0.05,
    noise_type: str = 'gaussian',
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate augmented signals by adding controlled noise.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input sEMG signal
    n_augmented : int, optional
        Number of augmented signals to generate (default: 5)
    noise_level : float, optional
        Noise level relative to signal std (default: 0.05)
    noise_type : str, optional
        Type of noise: 'gaussian', 'uniform', or 'pink' (default: 'gaussian')
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    List[np.ndarray]
        List of augmented signals
    """
    if seed is not None:
        np.random.seed(seed)
    
    signal_std = np.std(signal)
    augmented = [signal.copy()]  # Include original
    
    for _ in range(n_augmented):
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level * signal_std, len(signal))
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level * signal_std, 
                                      noise_level * signal_std, len(signal))
        elif noise_type == 'pink':
            # Generate pink noise (1/f)
            noise = _generate_pink_noise(len(signal)) * noise_level * signal_std
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        augmented.append(signal + noise)
    
    return augmented


def _generate_pink_noise(n_samples: int) -> np.ndarray:
    """
    Generate pink (1/f) noise using spectral shaping.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    
    Returns:
    --------
    np.ndarray
        Pink noise signal (normalized)
    """
    # Generate white noise
    white = np.random.randn(n_samples)
    
    # FFT
    fft = np.fft.rfft(white)
    
    # Create 1/f filter
    frequencies = np.fft.rfftfreq(n_samples)
    frequencies[0] = 1  # Avoid division by zero
    pink_filter = 1 / np.sqrt(frequencies)
    
    # Apply filter
    pink_fft = fft * pink_filter
    
    # Inverse FFT
    pink = np.fft.irfft(pink_fft, n_samples)
    
    # Normalize
    pink = pink / np.std(pink)
    
    return pink


def augment_by_time_warping(
    signal: np.ndarray,
    n_augmented: int = 5,
    warp_factor: float = 0.1,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate augmented signals by time warping.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input sEMG signal
    n_augmented : int, optional
        Number of augmented signals to generate (default: 5)
    warp_factor : float, optional
        Maximum time warping factor (default: 0.1)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    List[np.ndarray]
        List of augmented signals (all same length as original)
    """
    from scipy.interpolate import interp1d
    
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(signal)
    augmented = [signal.copy()]  # Include original
    
    for _ in range(n_augmented):
        # Create random warping curve
        n_knots = 5
        knot_positions = np.linspace(0, n_samples - 1, n_knots)
        knot_values = knot_positions + np.random.uniform(
            -warp_factor * n_samples, warp_factor * n_samples, n_knots
        )
        
        # Ensure monotonicity
        knot_values = np.sort(knot_values)
        knot_values = np.clip(knot_values, 0, n_samples - 1)
        
        # Create warping function
        warp_func = interp1d(knot_positions, knot_values, kind='cubic',
                            fill_value='extrapolate')
        
        # Generate warped time indices
        warped_indices = warp_func(np.arange(n_samples))
        warped_indices = np.clip(warped_indices, 0, n_samples - 1)
        
        # Interpolate signal at warped indices
        signal_interp = interp1d(np.arange(n_samples), signal, 
                                 kind='linear', fill_value='extrapolate')
        warped_signal = signal_interp(warped_indices)
        
        augmented.append(warped_signal)
    
    return augmented


def comprehensive_augmentation(
    signal: np.ndarray,
    n_per_method: int = 2,
    methods: Optional[List[str]] = None,
    use_ceemdan: bool = True,
    seed: Optional[int] = None
) -> Dict[str, List[np.ndarray]]:
    """
    Apply multiple augmentation methods to generate a comprehensive set of variations.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input sEMG signal
    n_per_method : int, optional
        Number of augmented signals per method (default: 2)
    methods : List[str], optional
        List of methods to use (default: all)
        Options: 'imf_mixing', 'imf_scaling', 'noise', 'time_warp'
    use_ceemdan : bool, optional
        Use CEEMDAN instead of EMD (default: True)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    Dict[str, List[np.ndarray]]
        Dictionary with method names as keys and list of augmented signals as values
        
    Examples:
    ---------
    >>> augmented = comprehensive_augmentation(segment, n_per_method=3)
    >>> for method, signals in augmented.items():
    ...     print(f"{method}: {len(signals)} signals")
    """
    if methods is None:
        methods = ['imf_mixing', 'imf_scaling', 'noise', 'time_warp']
    
    results = {'original': [signal.copy()]}
    
    if 'imf_mixing' in methods:
        results['imf_mixing'] = augment_by_imf_mixing(
            signal, n_augmented=n_per_method, use_ceemdan=use_ceemdan, seed=seed
        )[1:]  # Exclude original
    
    if 'imf_scaling' in methods:
        results['imf_scaling'] = augment_by_imf_scaling(
            signal, n_augmented=n_per_method, use_ceemdan=use_ceemdan, seed=seed
        )[1:]  # Exclude original
    
    if 'noise' in methods:
        results['noise'] = augment_by_noise_injection(
            signal, n_augmented=n_per_method, seed=seed
        )[1:]  # Exclude original
    
    if 'time_warp' in methods:
        results['time_warp'] = augment_by_time_warping(
            signal, n_augmented=n_per_method, seed=seed
        )[1:]  # Exclude original
    
    return results


def batch_augmentation(
    segments: List[np.ndarray],
    augmentation_method: str = 'imf_mixing',
    n_augmented: int = 5,
    target_length: Optional[int] = None,
    use_ceemdan: bool = True,
    seed: Optional[int] = None,
    **kwargs
) -> List[np.ndarray]:
    """
    Apply augmentation to a batch of segments.
    
    Parameters:
    -----------
    segments : List[np.ndarray]
        List of sEMG segments to augment
    augmentation_method : str, optional
        Method to use: 'imf_mixing', 'imf_scaling', 'noise', 'time_warp', 
        'imf_recombination' (default: 'imf_mixing')
    n_augmented : int, optional
        Number of augmented signals per segment (default: 5)
    target_length : int, optional
        Resample all segments to this length (default: keep original)
    use_ceemdan : bool, optional
        Use CEEMDAN instead of EMD (default: True)
    seed : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional arguments for specific augmentation method
    
    Returns:
    --------
    List[np.ndarray]
        List of all augmented signals (originals + augmented)
    """
    from scipy.signal import resample
    
    all_augmented = []
    
    # Optionally normalize lengths
    if target_length is not None:
        segments = [resample(seg, target_length) for seg in segments]
    
    # Special case for imf_recombination which needs multiple signals
    if augmentation_method == 'imf_recombination' and len(segments) >= 2:
        augmented = augment_by_imf_recombination(
            segments, n_augmented=n_augmented * len(segments), 
            use_ceemdan=use_ceemdan, seed=seed
        )
        return list(segments) + augmented
    
    # Process each segment
    for i, segment in enumerate(segments):
        seg_seed = None if seed is None else seed + i
        
        if augmentation_method == 'imf_mixing':
            augmented = augment_by_imf_mixing(
                segment, n_augmented=n_augmented, use_ceemdan=use_ceemdan, 
                seed=seg_seed, **kwargs
            )
        elif augmentation_method == 'imf_scaling':
            augmented = augment_by_imf_scaling(
                segment, n_augmented=n_augmented, use_ceemdan=use_ceemdan,
                seed=seg_seed, **kwargs
            )
        elif augmentation_method == 'noise':
            augmented = augment_by_noise_injection(
                segment, n_augmented=n_augmented, seed=seg_seed, **kwargs
            )
        elif augmentation_method == 'time_warp':
            augmented = augment_by_time_warping(
                segment, n_augmented=n_augmented, seed=seg_seed, **kwargs
            )
        else:
            raise ValueError(f"Unknown augmentation method: {augmentation_method}")
        
        all_augmented.extend(augmented)
    
    return all_augmented
