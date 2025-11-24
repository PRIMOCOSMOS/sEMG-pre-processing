"""
sEMG Preprocessing Toolkit

A comprehensive toolkit for surface electromyography (sEMG) signal preprocessing,
including filtering, noise removal, and muscle activity detection.
"""

__version__ = "0.1.0"

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

__all__ = [
    "apply_highpass_filter",
    "apply_lowpass_filter",
    "apply_bandpass_filter",
    "apply_notch_filter",
    "remove_powerline_dft",
    "detect_muscle_activity",
    "segment_signal",
    "load_csv_data",
    "save_processed_data",
    "export_segments_to_csv",
]
