"""
sEMG Preprocessing Toolkit

A comprehensive toolkit for surface electromyography (sEMG) signal preprocessing,
including filtering, noise removal, muscle activity detection, HHT analysis,
and data augmentation.
"""

__version__ = "0.2.0"

from .filters import (
    apply_highpass_filter,
    apply_lowpass_filter,
    apply_bandpass_filter,
    apply_notch_filter,
    remove_powerline_dft,
)

from .detection import (
    detect_muscle_activity,
    segment_signal,
)

from .utils import (
    load_csv_data,
    save_processed_data,
    export_segments_to_csv,
)

from .hht import (
    emd_decomposition,
    hilbert_transform,
    compute_instantaneous_frequency,
    compute_hilbert_spectrum,
    hht_analysis,
    save_hilbert_spectrum,
)

from .augmentation import (
    augment_by_imf_mixing,
    augment_by_imf_recombination,
    augment_by_imf_scaling,
    augment_by_noise_injection,
    augment_by_time_warping,
    comprehensive_augmentation,
    batch_augmentation,
)

__all__ = [
    # Filters
    "apply_highpass_filter",
    "apply_lowpass_filter",
    "apply_bandpass_filter",
    "apply_notch_filter",
    "remove_powerline_dft",
    # Detection
    "detect_muscle_activity",
    "segment_signal",
    # I/O
    "load_csv_data",
    "save_processed_data",
    "export_segments_to_csv",
    # HHT
    "emd_decomposition",
    "hilbert_transform",
    "compute_instantaneous_frequency",
    "compute_hilbert_spectrum",
    "hht_analysis",
    "save_hilbert_spectrum",
    # Augmentation
    "augment_by_imf_mixing",
    "augment_by_imf_recombination",
    "augment_by_imf_scaling",
    "augment_by_noise_injection",
    "augment_by_time_warping",
    "comprehensive_augmentation",
    "batch_augmentation",
]
