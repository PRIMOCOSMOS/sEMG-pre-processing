"""
Utility functions for data I/O and signal processing helpers.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Dict
import os
import glob
from scipy.io import loadmat


def load_csv_data(
    filepath: str,
    value_column: int = 1,
    has_header: bool = True,
    delimiter: str = ',',
    skip_rows: int = 0
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
    skip_rows : int, optional
        Number of rows to skip at the beginning of the file before reading data
        (default: 0). Use this for files with multiple header/metadata rows.
        For example, if the file has 2 header rows and data starts from row 3,
        set skip_rows=2 and has_header=False, or skip_rows=1 and has_header=True.
    
    Returns:
    --------
    Tuple[np.ndarray, Optional[pd.DataFrame]]
        - Signal data as numpy array (1D)
        - Full dataframe (for accessing other columns if needed)
        
    Examples:
    ---------
    >>> signal, df = load_csv_data('emg_data.csv')
    >>> signal, df = load_csv_data('emg_data.csv', value_column=2, has_header=False)
    >>> # For files with 2 header rows (metadata + column names):
    >>> signal, df = load_csv_data('emg_data.csv', skip_rows=1, has_header=True)
    >>> # For files with 2 header rows but no column names:
    >>> signal, df = load_csv_data('emg_data.csv', skip_rows=2, has_header=False)
    """
    # Read CSV file with optional row skipping
    if has_header:
        df = pd.read_csv(filepath, delimiter=delimiter, skiprows=skip_rows)
    else:
        df = pd.read_csv(filepath, delimiter=delimiter, header=None, skiprows=skip_rows)
    
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


def batch_load_csv(
    input_dir: str,
    pattern: str = "*.csv",
    value_column: int = 1,
    has_header: bool = True,
    delimiter: str = ',',
    skip_rows: int = 0
) -> List[Dict]:
    """
    Load multiple CSV files from a directory for batch processing.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CSV files
    pattern : str, optional
        File pattern to match (default: "*.csv")
    value_column : int, optional
        Column index containing signal values (default: 1)
    has_header : bool, optional
        Whether CSV files have header rows (default: True)
    delimiter : str, optional
        CSV delimiter (default: ',')
    skip_rows : int, optional
        Number of rows to skip at the beginning of each file (default: 0).
        Use this for files with multiple header/metadata rows.
    
    Returns:
    --------
    List[Dict]
        List of dictionaries containing:
        - 'filepath': Original file path
        - 'filename': File name without extension
        - 'signal': Signal data as numpy array
        - 'df': Full dataframe
        
    Examples:
    ---------
    >>> data_list = batch_load_csv('./raw_data/')
    >>> for data in data_list:
    ...     print(f"Loaded {data['filename']}: {len(data['signal'])} samples")
    >>> # For files with 2 header rows:
    >>> data_list = batch_load_csv('./raw_data/', skip_rows=1, has_header=True)
    """
    # Find all matching files
    file_pattern = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {input_dir}")
    
    loaded_data = []
    
    for filepath in files:
        try:
            signal, df = load_csv_data(filepath, value_column, has_header, delimiter, skip_rows)
            filename = os.path.splitext(os.path.basename(filepath))[0]
            
            loaded_data.append({
                'filepath': filepath,
                'filename': filename,
                'signal': signal,
                'df': df
            })
            print(f"Loaded: {filename} ({len(signal)} samples)")
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            continue
    
    print(f"\nSuccessfully loaded {len(loaded_data)} files")
    return loaded_data


def load_mat_data(
    filepath: str,
    variable_name: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Load sEMG data from MATLAB .mat file.
    
    Supports .mat files containing n×1 or 1×n double arrays where n is the number
    of samples. Automatically detects and reshapes the data to 1D array.
    
    Parameters:
    -----------
    filepath : str
        Path to the .mat file
    variable_name : str, optional
        Name of the variable in the .mat file to load. If None, will automatically
        detect and load the first numeric array found (excluding MATLAB metadata).
    
    Returns:
    --------
    Tuple[np.ndarray, Dict]
        - Signal data as numpy array (1D)
        - Dictionary containing all variables from the .mat file
        
    Examples:
    ---------
    >>> signal, mat_data = load_mat_data('emg_data.mat')
    >>> signal, mat_data = load_mat_data('emg_data.mat', variable_name='emg_signal')
    
    Raises:
    -------
    ValueError
        If no suitable numeric array is found in the .mat file
    FileNotFoundError
        If the file does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load .mat file
    try:
        mat_contents = loadmat(filepath)
    except Exception as e:
        raise ValueError(f"Error loading .mat file: {e}")
    
    # Filter out MATLAB metadata variables (start with __)
    data_vars = {k: v for k, v in mat_contents.items() if not k.startswith('__')}
    
    if len(data_vars) == 0:
        raise ValueError("No data variables found in .mat file (only MATLAB metadata)")
    
    # Find the signal data
    signal_data = None
    
    if variable_name is not None:
        # Use specified variable name
        if variable_name not in data_vars:
            available = ', '.join(data_vars.keys())
            raise ValueError(f"Variable '{variable_name}' not found. Available: {available}")
        signal_data = data_vars[variable_name]
    else:
        # Auto-detect: find first numeric array
        for var_name, var_data in data_vars.items():
            if isinstance(var_data, np.ndarray) and var_data.dtype.kind in 'biufc':  # numeric types
                signal_data = var_data
                print(f"Auto-detected signal variable: '{var_name}'")
                break
    
    if signal_data is None:
        raise ValueError("No suitable numeric array found in .mat file")
    
    # Reshape to 1D array
    # Handle n×1, 1×n, or 1D arrays
    if signal_data.ndim == 2:
        if signal_data.shape[0] == 1:
            # 1×n array
            signal_data = signal_data.flatten()
        elif signal_data.shape[1] == 1:
            # n×1 array
            signal_data = signal_data.flatten()
        else:
            raise ValueError(f"Expected n×1 or 1×n array, got shape {signal_data.shape}")
    elif signal_data.ndim == 1:
        # Already 1D
        pass
    else:
        raise ValueError(f"Expected 1D or 2D array, got {signal_data.ndim}D array")
    
    # Ensure float type
    signal_data = signal_data.astype(float)
    
    return signal_data, mat_contents


def load_signal_file(
    filepath: str,
    value_column: int = 1,
    has_header: bool = True,
    delimiter: str = ',',
    skip_rows: int = 0,
    mat_variable: Optional[str] = None
) -> Tuple[np.ndarray, Union[pd.DataFrame, Dict, None]]:
    """
    Load sEMG signal from either CSV or MAT file (auto-detects based on extension).
    
    Parameters:
    -----------
    filepath : str
        Path to the file (.csv or .mat)
    value_column : int, optional
        For CSV: Column index containing signal values (default: 1)
    has_header : bool, optional
        For CSV: Whether file has header row (default: True)
    delimiter : str, optional
        For CSV: Delimiter (default: ',')
    skip_rows : int, optional
        For CSV: Number of rows to skip (default: 0)
    mat_variable : str, optional
        For MAT: Variable name to load (default: auto-detect)
    
    Returns:
    --------
    Tuple[np.ndarray, Union[pd.DataFrame, Dict, None]]
        - Signal data as numpy array (1D)
        - DataFrame (for CSV) or Dict (for MAT) or None
        
    Examples:
    ---------
    >>> signal, data = load_signal_file('emg_data.csv')
    >>> signal, data = load_signal_file('emg_data.mat')
    >>> signal, data = load_signal_file('emg_data.mat', mat_variable='my_signal')
    """
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext == '.mat':
        return load_mat_data(filepath, mat_variable)
    elif ext == '.csv':
        return load_csv_data(filepath, value_column, has_header, delimiter, skip_rows)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .csv, .mat")


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
    annotations: Optional[Dict] = None
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
    annotations : Dict, optional
        Annotation data to include in each segment file:
        - subject: Subject identifier
        - fatigue_level: Fatigue condition (e.g., 'fresh', 'fatigued')
        - quality_rating: Motion quality (e.g., 1-5)
        - action_type: Type of muscle action
        - notes: Additional notes
    
    Returns:
    --------
    list
        List of saved file paths
        
    Examples:
    ---------
    >>> from semg_preprocessing import detect_muscle_activity, export_segments_to_csv
    >>> segments = detect_muscle_activity(filtered_signal, fs=1000)
    >>> annotations = {
    ...     'subject': 'S01',
    ...     'fatigue_level': 'fresh',
    ...     'quality_rating': 4,
    ...     'action_type': 'bicep_curl'
    ... }
    >>> files = export_segments_to_csv(filtered_signal, segments, fs=1000, 
    ...                                 output_dir='./segments', annotations=annotations)
    >>> print(f"Saved {len(files)} segment files")
    """
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
        
        # Write to file with metadata
        with open(filepath, 'w') as f:
            f.write(f"# Segment {i}\n")
            f.write(f"# Sampling frequency: {fs} Hz\n")
            f.write(f"# Start time: {start_idx/fs:.3f} s\n")
            f.write(f"# End time: {end_idx/fs:.3f} s\n")
            f.write(f"# Duration: {(end_idx-start_idx)/fs:.3f} s\n")
            f.write(f"# Samples: {len(segment_data)}\n")
            
            # Add segment statistics if available
            if isinstance(seg, dict):
                if 'peak_amplitude' in seg:
                    f.write(f"# Peak amplitude: {seg['peak_amplitude']:.4f}\n")
                if 'rms' in seg:
                    f.write(f"# RMS: {seg['rms']:.4f}\n")
            else:
                # Calculate basic stats
                f.write(f"# Peak amplitude: {np.max(np.abs(segment_data)):.4f}\n")
                f.write(f"# RMS: {np.sqrt(np.mean(segment_data**2)):.4f}\n")
            
            # Add annotations if provided
            if annotations:
                f.write("#\n")
                f.write("# === Annotations ===\n")
                if 'subject' in annotations:
                    f.write(f"# Subject: {annotations['subject']}\n")
                if 'fatigue_level' in annotations:
                    f.write(f"# Fatigue level: {annotations['fatigue_level']}\n")
                if 'quality_rating' in annotations:
                    f.write(f"# Quality rating: {annotations['quality_rating']}\n")
                if 'action_type' in annotations:
                    f.write(f"# Action type: {annotations['action_type']}\n")
                if 'notes' in annotations:
                    f.write(f"# Notes: {annotations['notes']}\n")
                # Add any custom annotations
                for key, value in annotations.items():
                    if key not in ['subject', 'fatigue_level', 'quality_rating', 'action_type', 'notes']:
                        f.write(f"# {key}: {value}\n")
            
            f.write("#\n")
            # Write the dataframe
            output_df.to_csv(f, index=False)
        
        saved_files.append(filepath)
        print(f"Saved segment {i} to: {filepath}")
    
    return saved_files


def export_segments_with_annotations(
    data: np.ndarray,
    segments: list,
    fs: float,
    output_dir: str,
    base_annotations: Dict,
    segment_annotations: Optional[List[Dict]] = None,
    prefix: str = "segment"
) -> List[str]:
    """
    Export segments with per-segment annotations.
    
    Parameters:
    -----------
    data : np.ndarray
        Original signal data
    segments : list
        List of segment tuples or dicts
    fs : float
        Sampling frequency in Hz
    output_dir : str
        Output directory
    base_annotations : Dict
        Base annotations applied to all segments (subject, session, etc.)
    segment_annotations : List[Dict], optional
        Per-segment annotations (one dict per segment)
        Each dict can contain: quality_rating, fatigue_level, action_type, notes
    prefix : str, optional
        Filename prefix
    
    Returns:
    --------
    List[str]
        List of saved file paths
        
    Examples:
    ---------
    >>> base_annot = {'subject': 'S01', 'session': 1}
    >>> seg_annot = [
    ...     {'quality_rating': 5, 'fatigue_level': 'fresh'},
    ...     {'quality_rating': 4, 'fatigue_level': 'mild'},
    ...     {'quality_rating': 3, 'fatigue_level': 'moderate'}
    ... ]
    >>> files = export_segments_with_annotations(
    ...     signal, segments, fs=1000, 
    ...     output_dir='./output',
    ...     base_annotations=base_annot,
    ...     segment_annotations=seg_annot
    ... )
    """
    saved_files = []
    
    for i, seg in enumerate(segments):
        # Combine base and per-segment annotations
        combined_annotations = base_annotations.copy()
        
        if segment_annotations and i < len(segment_annotations):
            combined_annotations.update(segment_annotations[i])
        
        # Add segment index
        combined_annotations['segment_index'] = i + 1
        
        # Export single segment
        files = export_segments_to_csv(
            data=data,
            segments=[seg],
            fs=fs,
            output_dir=output_dir,
            prefix=f"{prefix}_{i+1:03d}",
            annotations=combined_annotations
        )
        saved_files.extend(files)
    
    return saved_files


def create_annotation_summary(
    output_dir: str,
    segment_files: List[str],
    annotations_list: List[Dict],
    output_filename: str = "annotations_summary.csv"
) -> str:
    """
    Create a summary CSV file containing all segment annotations.
    
    Parameters:
    -----------
    output_dir : str
        Output directory
    segment_files : List[str]
        List of segment file paths
    annotations_list : List[Dict]
        List of annotation dictionaries (one per segment)
    output_filename : str, optional
        Output filename (default: 'annotations_summary.csv')
    
    Returns:
    --------
    str
        Path to the summary file
    """
    summary_data = []
    
    for filepath, annotations in zip(segment_files, annotations_list):
        row = {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
        }
        row.update(annotations)
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    print(f"Annotation summary saved to: {output_path}")
    return output_path


def resample_to_fixed_length(
    data: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Resample signal to a fixed number of samples.
    
    Useful for creating uniform-sized inputs for machine learning models.
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    target_length : int
        Target number of samples
    
    Returns:
    --------
    np.ndarray
        Resampled signal with exactly target_length samples
    """
    from scipy.signal import resample
    return resample(data, target_length)


def batch_resample_segments(
    segments: List[np.ndarray],
    target_length: int
) -> List[np.ndarray]:
    """
    Resample multiple segments to the same length.
    
    Parameters:
    -----------
    segments : List[np.ndarray]
        List of signal segments
    target_length : int
        Target length for all segments
    
    Returns:
    --------
    List[np.ndarray]
        List of resampled segments
    """
    return [resample_to_fixed_length(seg, target_length) for seg in segments]
