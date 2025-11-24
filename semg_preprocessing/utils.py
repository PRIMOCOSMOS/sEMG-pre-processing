"""
Utility functions for data I/O and signal processing helpers.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union


def load_csv_data(
    filepath: str,
    value_column: int = 1,
    has_header: bool = True,
    delimiter: str = ','
) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """
    Load sEMG data from CSV file.
    
    According to the specification, the CSV file has sEMG signal values
    in the 2nd column (index 1).
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    value_column : int, optional
        Column index containing the signal values (default: 1 for 2nd column)
    has_header : bool, optional
        Whether the CSV file has a header row (default: True)
    delimiter : str, optional
        CSV delimiter (default: ',')
    
    Returns:
    --------
    Tuple[np.ndarray, Optional[pd.DataFrame]]
        - Signal data as numpy array (1D)
        - Full dataframe (for accessing other columns if needed)
        
    Examples:
    ---------
    >>> signal, df = load_csv_data('emg_data.csv')
    >>> signal, df = load_csv_data('emg_data.csv', value_column=2, has_header=False)
    """
    # Read CSV file
    if has_header:
        df = pd.read_csv(filepath, delimiter=delimiter)
    else:
        df = pd.read_csv(filepath, delimiter=delimiter, header=None)
    
    # Extract signal data from specified column
    if value_column >= len(df.columns):
        raise ValueError(f"Column index {value_column} is out of range. CSV has {len(df.columns)} columns.")
    
    # Get the signal data
    signal_data = df.iloc[:, value_column].values
    
    # Convert to float and handle any non-numeric values
    try:
        signal_data = signal_data.astype(float)
    except ValueError as e:
        raise ValueError(f"Error converting column {value_column} to numeric values: {e}")
    
    return signal_data, df


def save_processed_data(
    filepath: str,
    data: np.ndarray,
    fs: Optional[float] = None,
    include_time: bool = True,
    original_df: Optional[pd.DataFrame] = None,
    additional_columns: Optional[dict] = None
) -> None:
    """
    Save processed sEMG data to CSV file.
    
    Parameters:
    -----------
    filepath : str
        Output file path
    data : np.ndarray
        Processed signal data
    fs : float, optional
        Sampling frequency in Hz (required if include_time=True)
    include_time : bool, optional
        Include time column in output (default: True)
    original_df : pd.DataFrame, optional
        Original dataframe to preserve other columns
    additional_columns : dict, optional
        Dictionary of additional columns to include {column_name: values}
    
    Examples:
    ---------
    >>> save_processed_data('filtered_emg.csv', filtered_signal, fs=1000)
    >>> save_processed_data('filtered_emg.csv', filtered_signal, fs=1000, 
    ...                     additional_columns={'envelope': envelope_data})
    """
    output_df = pd.DataFrame()
    
    # Add time column if requested
    if include_time:
        if fs is None:
            raise ValueError("Sampling frequency (fs) is required when include_time=True")
        time = np.arange(len(data)) / fs
        output_df['Time (s)'] = time
    
    # Add the processed signal data
    output_df['Signal'] = data
    
    # Add additional columns if provided
    if additional_columns:
        for col_name, col_data in additional_columns.items():
            if len(col_data) != len(data):
                raise ValueError(f"Length mismatch: {col_name} has {len(col_data)} values, expected {len(data)}")
            output_df[col_name] = col_data
    
    # Save to CSV
    output_df.to_csv(filepath, index=False)
    print(f"Processed data saved to: {filepath}")


def calculate_sampling_frequency(
    time_data: np.ndarray
) -> float:
    """
    Calculate sampling frequency from time data.
    
    Parameters:
    -----------
    time_data : np.ndarray
        Time values array
    
    Returns:
    --------
    float
        Estimated sampling frequency in Hz
    """
    # Calculate mean time difference
    dt = np.mean(np.diff(time_data))
    fs = 1.0 / dt
    return fs


def resample_signal(
    data: np.ndarray,
    original_fs: float,
    target_fs: float
) -> np.ndarray:
    """
    Resample signal to a different sampling frequency.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    original_fs : float
        Original sampling frequency in Hz
    target_fs : float
        Target sampling frequency in Hz
    
    Returns:
    --------
    np.ndarray
        Resampled signal
    """
    from scipy import signal as scipy_signal
    
    # Calculate resampling ratio
    num_samples = int(len(data) * target_fs / original_fs)
    
    # Resample using scipy
    resampled = scipy_signal.resample(data, num_samples)
    
    return resampled


def normalize_signal(
    data: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize signal data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    method : str, optional
        Normalization method: 'zscore', 'minmax', or 'maxabs' (default: 'zscore')
    
    Returns:
    --------
    np.ndarray
        Normalized signal
    """
    if method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        normalized = (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        # Min-max normalization (0 to 1)
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'maxabs':
        # Max absolute value normalization (-1 to 1)
        normalized = data / np.max(np.abs(data))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def split_into_windows(
    data: np.ndarray,
    window_size: int,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Split signal into overlapping windows.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    window_size : int
        Size of each window in samples
    overlap : float, optional
        Overlap ratio between windows (0 to 1, default: 0.5)
    
    Returns:
    --------
    np.ndarray
        2D array where each row is a window
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be between 0 and 1 (exclusive)")
    
    step_size = int(window_size * (1 - overlap))
    
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        windows.append(window)
    
    return np.array(windows)


def export_segments_to_csv(
    data: np.ndarray,
    segments: list,
    fs: float,
    output_dir: str,
    prefix: str = "segment",
    include_time: bool = True,
    original_df: Optional[pd.DataFrame] = None
) -> list:
    """
    Export detected muscle activity segments as individual CSV files.
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal data
    segments : list
        List of (start_index, end_index) tuples or list of segment dicts
    fs : float
        Sampling frequency in Hz
    output_dir : str
        Directory to save segment CSV files
    prefix : str, optional
        Prefix for output filenames (default: 'segment')
    include_time : bool, optional
        Include time column in output (default: True)
    original_df : pd.DataFrame, optional
        Original dataframe to preserve additional columns
    
    Returns:
    --------
    list
        List of saved file paths
        
    Examples:
    ---------
    >>> from semg_preprocessing import detect_muscle_activity, export_segments_to_csv
    >>> segments = detect_muscle_activity(filtered_signal, fs=1000)
    >>> files = export_segments_to_csv(filtered_signal, segments, fs=1000, 
    ...                                 output_dir='./segments')
    >>> print(f"Saved {len(files)} segment files")
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # Handle both list of tuples and list of dicts
    for i, seg in enumerate(segments, 1):
        if isinstance(seg, tuple):
            start_idx, end_idx = seg
            segment_data = data[start_idx:end_idx]
        elif isinstance(seg, dict):
            start_idx = seg['start_index']
            end_idx = seg['end_index']
            segment_data = seg['data']
        else:
            raise ValueError("Segments must be tuples (start, end) or dicts with 'start_index', 'end_index', 'data'")
        
        # Create filename
        filename = f"{prefix}_{i:03d}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data
        output_df = pd.DataFrame()
        
        if include_time:
            # Create time array relative to segment start
            time = np.arange(len(segment_data)) / fs
            output_df['Time (s)'] = time
        
        output_df['Signal'] = segment_data
        
        # Add metadata as header comments if segment is a dict
        if isinstance(seg, dict) and 'duration' in seg:
            # Write metadata as comments at the top
            with open(filepath, 'w') as f:
                f.write(f"# Segment {i}\n")
                f.write(f"# Start time: {seg.get('start_time', start_idx/fs):.3f} s\n")
                f.write(f"# End time: {seg.get('end_time', end_idx/fs):.3f} s\n")
                f.write(f"# Duration: {seg.get('duration', (end_idx-start_idx)/fs):.3f} s\n")
                if 'peak_amplitude' in seg:
                    f.write(f"# Peak amplitude: {seg['peak_amplitude']:.4f}\n")
                if 'rms' in seg:
                    f.write(f"# RMS: {seg['rms']:.4f}\n")
                f.write("#\n")
                # Write the dataframe
                output_df.to_csv(f, index=False)
        else:
            # Simple CSV without metadata
            output_df.to_csv(filepath, index=False)
        
        saved_files.append(filepath)
        print(f"Saved segment {i} to: {filepath}")
    
    return saved_files
