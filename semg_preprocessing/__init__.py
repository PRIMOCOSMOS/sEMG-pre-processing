"""
sEMG Preprocessing Toolkit

A comprehensive toolkit for surface electromyography (sEMG) signal preprocessing,
including filtering, noise removal, muscle activity detection, HHT analysis,
CEEMDAN decomposition, feature extraction, and data augmentation.
"""

__version__ = "0.5.0"

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
    apply_tkeo,
)

from .utils import (
    load_csv_data,
    load_mat_data,
    load_signal_file,
    save_processed_data,
    export_segments_to_csv,
)

from .hht import (
    emd_decomposition,
    ceemdan_decomposition,
    hilbert_transform,
    compute_instantaneous_frequency,
    compute_hilbert_spectrum,
    compute_hilbert_spectrum_enhanced,
    hht_analysis,
    hht_analysis_enhanced,
    batch_hht_analysis,
    extract_semg_features,
    export_features_to_csv,
    save_hilbert_spectrum,
    export_hilbert_spectra_batch,
    export_activity_segments_hht,
)

from .augmentation import (
    augment_by_imf_mixing,
    augment_by_imf_recombination,
    augment_ceemdan_random_imf,
    trim_signals_to_equal_length,
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
    "apply_tkeo",
    # I/O
    "load_csv_data",
    "load_mat_data",
    "load_signal_file",
    "save_processed_data",
    "export_segments_to_csv",
    # HHT
    "emd_decomposition",
    "ceemdan_decomposition",
    "hilbert_transform",
    "compute_instantaneous_frequency",
    "compute_hilbert_spectrum",
    "compute_hilbert_spectrum_enhanced",
    "hht_analysis",
    "hht_analysis_enhanced",
    "batch_hht_analysis",
    "extract_semg_features",
    "export_features_to_csv",
    "save_hilbert_spectrum",
    "export_hilbert_spectra_batch",
    "export_activity_segments_hht",
    # Augmentation
    "augment_by_imf_mixing",
    "augment_by_imf_recombination",
    "augment_ceemdan_random_imf",
    "trim_signals_to_equal_length",
    "augment_by_imf_scaling",
    "augment_by_noise_injection",
    "augment_by_time_warping",
    "comprehensive_augmentation",
    "batch_augmentation",
]
