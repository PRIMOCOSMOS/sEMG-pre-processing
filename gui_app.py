"""
Gradio GUI Application for sEMG Signal Preprocessing

This provides a user-friendly web interface for:
- Loading CSV files with sEMG data (single or batch)
- Applying preprocessing filters
- Detecting muscle activity
- HHT (Hilbert-Huang Transform) analysis
- EMD-based data augmentation
- Visualizing results
- Exporting processed data and segments with annotations
"""

import os
import sys
import numpy as np
import pandas as pd

# Set matplotlib to use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread safety with Gradio
import matplotlib.pyplot as plt

import gradio as gr
from pathlib import Path

# Ensure semg_preprocessing is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semg_preprocessing import (
    load_csv_data,
    load_mat_data,
    load_signal_file,
    apply_bandpass_filter,
    apply_notch_filter,
    detect_muscle_activity,
    detect_activity_hht,
    segment_signal,
    save_processed_data,
    export_segments_to_csv,
    # HHT functions
    emd_decomposition,
    ceemdan_decomposition,
    compute_hilbert_spectrum,
    compute_hilbert_spectrum_enhanced,
    hht_analysis,
    hht_analysis_enhanced,
    batch_hht_analysis,
    extract_semg_features,
    export_features_to_csv,
    save_hilbert_spectrum,
    # Augmentation functions
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

from semg_preprocessing.utils import batch_load_csv, resample_to_fixed_length


class EMGProcessorGUI:
    """GUI application for sEMG signal preprocessing."""
    
    def __init__(self):
        self.signal = None
        self.filtered_signal = None
        self.fs = 1000.0
        self.segments = None
        self.segment_data = None
        self.df = None
        self.current_filename = None
        # Batch processing state
        self.batch_data = []
        self.batch_results = []
        self.batch_filtered = []  # Store filtered signals for batch
        self.batch_segments = []  # Store segments for batch
        self.batch_features = []  # Store features for batch
        # HHT results
        self.hht_results = None
        # Augmentation results
        self.augmented_signals = None
    
    def _get_default_feature_filename(self):
        """Generate default filename for feature export based on current signal filename."""
        if hasattr(self, 'current_filename') and self.current_filename:
            # Remove extension and add _features.csv
            base_name = os.path.splitext(self.current_filename)[0]
            return f"{base_name}_features.csv"
        else:
            return "segment_features.csv"
    
    def _create_batch_subplot_axes(self, n_items, max_show=8, figsize_per_row=2.5, n_cols=2):
        """
        Create subplot axes for batch visualization.
        
        Parameters:
        -----------
        n_items : int
            Total number of items to potentially show
        max_show : int
            Maximum number of subplots to create (default: 8)
        figsize_per_row : float
            Height per row in inches (default: 2.5)
        n_cols : int
            Number of columns (default: 2)
        
        Returns:
        --------
        Tuple[Figure, np.ndarray, int]
            - Matplotlib figure
            - Flattened array of axes
            - Number of items to show
        """
        n_show = min(n_items, max_show)
        n_rows = (n_show + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 if n_cols == 2 else 12, figsize_per_row * n_rows))
        
        if n_show == 1:
            axes = np.array([[axes]])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes = axes.flatten()
        return fig, axes, n_show
        
    def load_file(self, file_obj, fs, value_column, has_header, skip_rows):
        """Load signal file (CSV or MAT) and extract signal."""
        try:
            if file_obj is None:
                return "Please upload a file", None
            
            # Load the data using unified loader
            self.fs = float(fs)
            self.signal, self.df = load_signal_file(
                file_obj.name,
                value_column=int(value_column),
                has_header=has_header,
                skip_rows=int(skip_rows)
            )
            self.current_filename = os.path.basename(file_obj.name)
            file_ext = os.path.splitext(self.current_filename)[1].lower()
            
            # Create preview plot
            fig, ax = plt.subplots(figsize=(12, 4))
            time = np.arange(len(self.signal)) / self.fs
            ax.plot(time, self.signal, 'b-', linewidth=0.5, alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Loaded sEMG Signal: {self.current_filename}')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            file_type_str = "MAT file" if file_ext == '.mat' else "CSV file"
            info = f"""
✅ {file_type_str} loaded successfully!
- Filename: {self.current_filename}
- Samples: {len(self.signal)}
- Duration: {len(self.signal)/self.fs:.2f} seconds
- Sampling frequency: {self.fs} Hz
- Signal range: [{self.signal.min():.3f}, {self.signal.max():.3f}]
{f'- Skipped rows: {skip_rows}' if file_ext == '.csv' else ''}
            """
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error loading file: {str(e)}\n{traceback.format_exc()}", None
    
    def load_batch_files(self, file_objs, fs, value_column, has_header, skip_rows, progress=gr.Progress()):
        """Load multiple signal files (CSV or MAT) for batch processing."""
        try:
            if file_objs is None or len(file_objs) == 0:
                return "Please upload files", None
            
            self.fs = float(fs)
            self.batch_data = []
            self.batch_filtered = []
            self.batch_segments = []
            self.batch_features = []
            
            progress(0.1, desc="Loading files...")
            
            for i, file_obj in enumerate(file_objs):
                progress((i + 1) / len(file_objs) * 0.8, desc=f"Loading {os.path.basename(file_obj.name)}...")
                
                try:
                    signal, df = load_signal_file(
                        file_obj.name,
                        value_column=int(value_column),
                        has_header=has_header,
                        skip_rows=int(skip_rows)
                    )
                    
                    self.batch_data.append({
                        'filename': os.path.basename(file_obj.name),
                        'filepath': file_obj.name,
                        'signal': signal,
                        'df': df
                    })
                except Exception as e:
                    print(f"Warning: Failed to load {file_obj.name}: {e}")
            
            progress(1.0, desc="Complete!")
            
            # Create summary plot showing all files using helper method
            fig, axes, n_show = self._create_batch_subplot_axes(len(self.batch_data))
            
            for i, data in enumerate(self.batch_data[:n_show]):
                time = np.arange(len(data['signal'])) / self.fs
                axes[i].plot(time, data['signal'], 'b-', linewidth=0.5, alpha=0.7)
                axes[i].set_title(data['filename'], fontsize=9)
                axes[i].set_xlabel('Time (s)', fontsize=8)
                axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for j in range(n_show, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            
            info = f"""
✅ Batch loading complete!
- Files loaded: {len(self.batch_data)} / {len(file_objs)}
- Sampling frequency: {self.fs} Hz
- Skipped rows: {skip_rows}

**Loaded files:**
"""
            for data in self.batch_data:
                info += f"\n- {data['filename']}: {len(data['signal'])} samples ({len(data['signal'])/self.fs:.2f}s)"
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error loading files: {str(e)}\n{traceback.format_exc()}", None
    
    def apply_filters(self, lowcut, highcut, filter_order, notch_freq, harmonics_str):
        """Apply bandpass and notch filters."""
        try:
            if self.signal is None:
                return "Please load a file first", None
            
            # Apply bandpass filter
            self.filtered_signal = apply_bandpass_filter(
                self.signal,
                fs=self.fs,
                lowcut=float(lowcut),
                highcut=float(highcut),
                order=int(filter_order)
            )
            
            # Apply notch filter if enabled
            if notch_freq > 0:
                harmonics = [int(x.strip()) for x in harmonics_str.split(',') if x.strip()]
                self.filtered_signal = apply_notch_filter(
                    self.filtered_signal,
                    fs=self.fs,
                    freq=float(notch_freq),
                    harmonics=harmonics
                )
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            time = np.arange(len(self.signal)) / self.fs
            
            # Original signal
            ax1.plot(time, self.signal, 'b-', linewidth=0.5, alpha=0.7)
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Original Signal')
            ax1.grid(True, alpha=0.3)
            
            # Filtered signal
            ax2.plot(time, self.filtered_signal, 'g-', linewidth=0.5, alpha=0.7)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.set_title('Filtered Signal')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            info = f"""
✅ Filtering applied successfully!
- Bandpass: {lowcut}-{highcut} Hz (order {filter_order})
- Notch: {notch_freq} Hz (harmonics: {harmonics_str if notch_freq > 0 else 'N/A'})
            """
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error applying filters: {str(e)}\n{traceback.format_exc()}", None
    
    def apply_batch_filters(self, lowcut, highcut, filter_order, notch_freq, harmonics_str, progress=gr.Progress()):
        """Apply bandpass and notch filters to all batch-loaded files."""
        try:
            if not self.batch_data:
                return "❌ Please load batch files first", None
            
            self.batch_filtered = []
            harmonics = [int(x.strip()) for x in harmonics_str.split(',') if x.strip()] if notch_freq > 0 else []
            
            progress(0.1, desc="Starting batch filtering...")
            
            for i, data in enumerate(self.batch_data):
                progress((i + 1) / len(self.batch_data) * 0.8, desc=f"Filtering {data['filename']}...")
                
                # Apply bandpass filter
                filtered = apply_bandpass_filter(
                    data['signal'],
                    fs=self.fs,
                    lowcut=float(lowcut),
                    highcut=float(highcut),
                    order=int(filter_order)
                )
                
                # Apply notch filter if enabled
                if notch_freq > 0:
                    filtered = apply_notch_filter(
                        filtered,
                        fs=self.fs,
                        freq=float(notch_freq),
                        harmonics=harmonics
                    )
                
                self.batch_filtered.append({
                    'filename': data['filename'],
                    'original': data['signal'],
                    'filtered': filtered
                })
            
            progress(1.0, desc="Complete!")
            
            # Also set single signal for downstream processing if only one file
            if len(self.batch_filtered) == 1:
                self.signal = self.batch_data[0]['signal']
                self.filtered_signal = self.batch_filtered[0]['filtered']
            
            # Create summary visualization using helper method
            fig, axes, n_show = self._create_batch_subplot_axes(len(self.batch_filtered), max_show=6, figsize_per_row=3)
            
            for i, data in enumerate(self.batch_filtered[:n_show]):
                time = np.arange(len(data['filtered'])) / self.fs
                axes[i].plot(time, data['filtered'], 'g-', linewidth=0.5, alpha=0.7)
                axes[i].set_title(f"{data['filename']} (Filtered)", fontsize=9)
                axes[i].set_xlabel('Time (s)', fontsize=8)
                axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for j in range(n_show, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            
            info = f"""
✅ Batch filtering complete!
- Files processed: {len(self.batch_filtered)}
- Bandpass: {lowcut}-{highcut} Hz (order {filter_order})
- Notch: {notch_freq} Hz (harmonics: {harmonics_str if notch_freq > 0 else 'N/A'})

**Filtered files:**
"""
            for data in self.batch_filtered:
                info += f"\n- {data['filename']}: {len(data['filtered'])} samples"
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error applying batch filters: {str(e)}\n{traceback.format_exc()}", None
    
    def detect_activity(self, detection_method, min_duration, max_duration, sensitivity, n_detectors, fusion_method, 
                       use_multi_detector, use_clustering, detector_sens_str, classification_threshold,
                       use_tkeo, merge_threshold, max_merge_count,
                       hht_energy_threshold, hht_temporal_compactness, hht_resolution, 
                       hht_sensitivity, hht_local_contrast_weight, progress=gr.Progress()):
        """Detect muscle activity segments using PELT or HHT algorithm."""
        try:
            if self.filtered_signal is None:
                return "❌ Please apply filters first", None
            
            progress(0.1, desc="Initializing detection...")
            
            # Convert max_duration (None or float)
            max_dur = float(max_duration) if max_duration and max_duration > 0 else None
            
            if detection_method == "PELT":
                # Parse individual detector sensitivities if provided
                detector_sensitivities = None
                if use_multi_detector and detector_sens_str and detector_sens_str.strip():
                    try:
                        # Parse comma-separated values
                        detector_sensitivities = [float(x.strip()) for x in detector_sens_str.split(',')]
                        if len(detector_sensitivities) != n_detectors:
                            return f"❌ Number of detector sensitivities ({len(detector_sensitivities)}) must match number of detectors ({n_detectors})", None
                    except ValueError:
                        return "❌ Invalid detector sensitivities format. Use comma-separated numbers (e.g., 1.0, 1.5, 2.0)", None
                
                # Detect muscle activity using PELT method
                progress(0.3, desc="Detecting muscle activity with PELT...")
                result = detect_muscle_activity(
                    self.filtered_signal,
                    fs=self.fs,
                    method="combined",
                    min_duration=float(min_duration),
                    max_duration=max_dur,
                    sensitivity=float(sensitivity),
                    n_detectors=int(n_detectors),
                    detector_sensitivities=detector_sensitivities,
                    fusion_method=fusion_method,
                    use_multi_detector=use_multi_detector,
                    classify_segments=True,
                    use_clustering=use_clustering,
                    classification_threshold=float(classification_threshold),
                    use_tkeo=use_tkeo,
                    merge_threshold=float(merge_threshold),
                    max_merge_count=int(max_merge_count),
                    return_changepoints=True
                )
                
                # Extract segments and changepoints from result
                self.segments = result['segments']
                changepoints = result['changepoints']
                
                # Get detailed segment information
                progress(0.6, desc="Extracting segment information...")
                self.segment_data = segment_signal(
                    self.filtered_signal,
                    self.segments,
                    fs=self.fs,
                    include_metadata=True
                )
                
                # Create visualization
                progress(0.8, desc="Creating visualization...")
                fig, ax = plt.subplots(figsize=(12, 6))
                time = np.arange(len(self.filtered_signal)) / self.fs
                
                ax.plot(time, self.filtered_signal, 'k-', linewidth=0.5, alpha=0.5, label='Filtered Signal')
                
                # Mark all PELT changepoints with vertical lines
                for cp in changepoints:
                    ax.axvline(cp/self.fs, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                
                # Highlight detected segments
                colors = plt.cm.Set1(np.linspace(0, 1, max(len(self.segments), 1)))
                for i, (start, end) in enumerate(self.segments):
                    ax.axvspan(start/self.fs, end/self.fs, alpha=0.3, color=colors[i % len(colors)],
                              label=f'Segment {i+1}' if i < 5 else '')
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'PELT Detection: {len(self.segments)} active segments, {len(changepoints)} changepoints')
                ax.grid(True, alpha=0.3)
                if self.segments:
                    ax.legend(loc='upper right', ncol=2)
                
                plt.tight_layout()
                
                # Create info text
                progress(0.9, desc="Formatting results...")
                info = f"✅ **PELT Detection**: {len(self.segments)} muscle activity segments from {len(changepoints)} changepoints:\n\n"
                for i, seg in enumerate(self.segment_data[:10], 1):
                    info += f"**Segment {i}**: {seg['start_time']:.3f}s - {seg['end_time']:.3f}s "
                    info += f"(duration: {seg['duration']:.3f}s, peak: {seg['peak_amplitude']:.3f}, RMS: {seg['rms']:.3f})\n"
                
                if len(self.segment_data) > 10:
                    info += f"\n... and {len(self.segment_data) - 10} more segments"
                
            else:  # HHT method
                progress(0.3, desc="Computing full-signal HHT...")
                
                # Detect muscle activity using HHT method with enhanced parameters
                result = detect_activity_hht(
                    self.filtered_signal,
                    fs=self.fs,
                    min_duration=float(min_duration),
                    max_duration=max_dur,
                    energy_threshold=float(hht_energy_threshold),
                    temporal_compactness=float(hht_temporal_compactness),
                    resolution_per_second=int(hht_resolution),
                    sensitivity=float(hht_sensitivity),
                    local_contrast_weight=float(hht_local_contrast_weight),
                    return_spectrum=True
                )
                
                self.segments = result['segments']
                spectrum = result['spectrum']
                spectrum_log = result.get('spectrum_log', np.log1p(spectrum))  # Use log-scaled spectrum for display
                time_axis = result['time']
                freq_axis = result['frequency']
                detection_mask = result['detection_mask']
                time_energy = result['time_energy']
                combined_energy = result.get('combined_energy', time_energy)
                threshold_info = result.get('threshold_info', {})
                
                # Get detailed segment information
                progress(0.6, desc="Extracting segment information...")
                self.segment_data = segment_signal(
                    self.filtered_signal,
                    self.segments,
                    fs=self.fs,
                    include_metadata=True
                )
                
                # Create multi-panel visualization with improved colormap
                progress(0.8, desc="Creating visualization...")
                fig, axes = plt.subplots(3, 1, figsize=(12, 14))
                
                # Panel 1: Hilbert Spectrum with colored logarithmic display
                ax1 = axes[0]
                # Use log-scaled spectrum for better visualization
                # Show original HHT result as logarithmic colored spectrum only (no white bands)
                im = ax1.pcolormesh(time_axis, freq_axis, spectrum_log, shading='auto', cmap='jet')
                ax1.set_xlabel('Time (s)', fontsize=11)
                ax1.set_ylabel('Frequency (Hz)', fontsize=11)
                ax1.set_title('Hilbert Spectrum (Log Scale)', fontsize=12, fontweight='bold')
                cbar = plt.colorbar(im, ax=ax1)
                cbar.set_label('Log Amplitude', fontsize=10)
                
                # Panel 2: Time energy profile with detection and threshold visualization
                ax2 = axes[1]
                time_bins = np.arange(len(time_energy))
                # Normalize for display
                if np.max(time_energy) > 0:
                    time_energy_norm = time_energy / np.max(time_energy)
                else:
                    time_energy_norm = time_energy
                if np.max(combined_energy) > 0:
                    combined_norm = combined_energy / np.max(combined_energy)
                else:
                    combined_norm = combined_energy
                
                ax2.plot(time_axis, time_energy_norm, 'b-', linewidth=1, alpha=0.6, label='Global Energy')
                ax2.plot(time_axis, combined_norm, 'g-', linewidth=1.5, label='Combined (Global+Local)')
                
                # Show adaptive threshold line
                adaptive_thresh = threshold_info.get('adaptive_threshold', np.mean(combined_norm))
                ax2.axhline(y=adaptive_thresh, color='r', linestyle='--', linewidth=1.5, 
                           label=f'Threshold ({adaptive_thresh:.2f})')
                
                # Mark detected regions
                active_bins = result['active_time_bins']
                ax2.fill_between(time_axis, 0, combined_norm, where=active_bins, alpha=0.3, color='red', label='Detected')
                ax2.set_xlabel('Time (s)', fontsize=11)
                ax2.set_ylabel('Normalized Energy', fontsize=11)
                ax2.set_title('Time-Integrated Energy Profile with Detection Threshold', fontsize=12, fontweight='bold')
                ax2.legend(loc='upper right', fontsize=9)
                ax2.grid(True, alpha=0.3)
                
                # Panel 3: Original signal with detected segments
                ax3 = axes[2]
                time = np.arange(len(self.filtered_signal)) / self.fs
                ax3.plot(time, self.filtered_signal, 'k-', linewidth=0.5, alpha=0.5, label='Filtered Signal')
                
                # Highlight detected segments
                colors = plt.cm.Set1(np.linspace(0, 1, max(len(self.segments), 1)))
                for i, (start, end) in enumerate(self.segments):
                    ax3.axvspan(start/self.fs, end/self.fs, alpha=0.3, color=colors[i % len(colors)],
                              label=f'Segment {i+1}' if i < 5 else '')
                
                ax3.set_xlabel('Time (s)', fontsize=11)
                ax3.set_ylabel('Amplitude', fontsize=11)
                ax3.set_title(f'Detected Muscle Activity Segments ({len(self.segments)} segments)', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                if self.segments:
                    ax3.legend(loc='upper right', ncol=2, fontsize=9)
                
                plt.tight_layout()
                
                # Create info text with enhanced information
                progress(0.9, desc="Formatting results...")
                info = f"✅ **HHT Detection**: {len(self.segments)} muscle activity segments detected:\n\n"
                info += f"**Algorithm Parameters:**\n"
                info += f"- Sensitivity: {hht_sensitivity:.2f} (lower = more events)\n"
                info += f"- Energy threshold: {hht_energy_threshold*100:.0f}th percentile\n"
                info += f"- Local contrast weight: {hht_local_contrast_weight:.2f}\n"
                info += f"- Temporal compactness: {hht_temporal_compactness:.2f}\n"
                info += f"- Duration constraints: {min_duration:.2f}s - {max_dur if max_dur else '∞'}s\n\n"
                
                info += f"**Spectrum Info:**\n"
                info += f"- Resolution: {spectrum.shape[1]} time bins × {spectrum.shape[0]} freq bins\n"
                info += f"- Frequency range: {freq_axis[0]:.1f} - {freq_axis[-1]:.1f} Hz\n"
                
                if threshold_info:
                    info += f"- Effective threshold: {threshold_info.get('adaptive_threshold', 0):.3f}\n\n"
                else:
                    info += "\n"
                
                info += f"**Detected Segments:**\n"
                for i, seg in enumerate(self.segment_data[:10], 1):
                    info += f"**Segment {i}**: {seg['start_time']:.3f}s - {seg['end_time']:.3f}s "
                    info += f"(duration: {seg['duration']:.3f}s, peak: {seg['peak_amplitude']:.3f}, RMS: {seg['rms']:.3f})\n"
                
                if len(self.segment_data) > 10:
                    info += f"\n... and {len(self.segment_data) - 10} more segments"
            
            progress(1.0, desc="Complete!")
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error detecting activity: {str(e)}\n{traceback.format_exc()}", None
    
    def detect_batch_activity(self, detection_method, min_duration, max_duration, sensitivity, n_detectors, fusion_method, 
                             use_multi_detector, use_clustering, detector_sens_str, classification_threshold,
                             use_tkeo, merge_threshold, max_merge_count,
                             hht_energy_threshold, hht_temporal_compactness, hht_resolution,
                             hht_sensitivity, hht_local_contrast_weight, progress=gr.Progress()):
        """Detect muscle activity in all batch-filtered files using PELT or HHT algorithm."""
        try:
            if not self.batch_filtered:
                return "❌ Please apply batch filters first", None
            
            self.batch_segments = []
            total_segments = 0
            
            progress(0.1, desc="Starting batch detection...")
            
            # Convert max_duration
            max_dur = float(max_duration) if max_duration and max_duration > 0 else None
            
            if detection_method == "PELT":
                # Parse individual detector sensitivities if provided
                detector_sensitivities = None
                if use_multi_detector and detector_sens_str and detector_sens_str.strip():
                    try:
                        detector_sensitivities = [float(x.strip()) for x in detector_sens_str.split(',')]
                        if len(detector_sensitivities) != n_detectors:
                            return f"❌ Number of detector sensitivities ({len(detector_sensitivities)}) must match number of detectors ({n_detectors})", None
                    except ValueError:
                        return "❌ Invalid detector sensitivities format. Use comma-separated numbers", None
                
                for i, data in enumerate(self.batch_filtered):
                    progress((i + 1) / len(self.batch_filtered) * 0.7, desc=f"Detecting in {data['filename']} (PELT)...")
                    
                    # Detect muscle activity using PELT method
                    segments = detect_muscle_activity(
                        data['filtered'],
                        fs=self.fs,
                        method="combined",
                        min_duration=float(min_duration),
                        max_duration=max_dur,
                        sensitivity=float(sensitivity),
                        n_detectors=int(n_detectors),
                        detector_sensitivities=detector_sensitivities,
                        fusion_method=fusion_method,
                        use_multi_detector=use_multi_detector,
                        classify_segments=True,
                        use_clustering=use_clustering,
                        classification_threshold=float(classification_threshold),
                        use_tkeo=use_tkeo,
                        merge_threshold=float(merge_threshold),
                        max_merge_count=int(max_merge_count)
                    )
                    
                    # Get detailed segment information
                    segment_data = segment_signal(
                        data['filtered'],
                        segments,
                        fs=self.fs,
                        include_metadata=True
                    )
                    
                    self.batch_segments.append({
                        'filename': data['filename'],
                        'filtered': data['filtered'],
                        'segments': segments,
                        'segment_data': segment_data
                    })
                    total_segments += len(segments)
            
            else:  # HHT method
                for i, data in enumerate(self.batch_filtered):
                    progress((i + 1) / len(self.batch_filtered) * 0.7, desc=f"Detecting in {data['filename']} (HHT)...")
                    
                    # Detect muscle activity using HHT method with enhanced parameters
                    segments = detect_activity_hht(
                        data['filtered'],
                        fs=self.fs,
                        min_duration=float(min_duration),
                        max_duration=max_dur,
                        energy_threshold=float(hht_energy_threshold),
                        temporal_compactness=float(hht_temporal_compactness),
                        resolution_per_second=int(hht_resolution),
                        sensitivity=float(hht_sensitivity),
                        local_contrast_weight=float(hht_local_contrast_weight),
                        return_spectrum=False  # Don't return spectrum for batch processing
                    )
                    
                    # Get detailed segment information
                    segment_data = segment_signal(
                        data['filtered'],
                        segments,
                        fs=self.fs,
                        include_metadata=True
                    )
                    
                    self.batch_segments.append({
                        'filename': data['filename'],
                        'filtered': data['filtered'],
                        'segments': segments,
                        'segment_data': segment_data
                    })
                    total_segments += len(segments)
            
            progress(0.8, desc="Creating visualization...")
            
            # Also set single file for downstream if only one file
            if len(self.batch_segments) == 1:
                self.filtered_signal = self.batch_filtered[0]['filtered']
                self.segments = self.batch_segments[0]['segments']
                self.segment_data = self.batch_segments[0]['segment_data']
            
            # Create summary visualization using helper method
            fig, axes, n_show = self._create_batch_subplot_axes(len(self.batch_segments), max_show=6, figsize_per_row=3)
            
            for i, data in enumerate(self.batch_segments[:n_show]):
                time = np.arange(len(data['filtered'])) / self.fs
                axes[i].plot(time, data['filtered'], 'k-', linewidth=0.5, alpha=0.5)
                
                # Highlight detected segments
                colors = plt.cm.Set1(np.linspace(0, 1, max(len(data['segments']), 1)))
                for j, (start, end) in enumerate(data['segments']):
                    axes[i].axvspan(start/self.fs, end/self.fs, alpha=0.3, color=colors[j % len(colors)])
                
                axes[i].set_title(f"{data['filename']} ({len(data['segments'])} segs)", fontsize=9)
                axes[i].set_xlabel('Time (s)', fontsize=8)
                axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for j in range(n_show, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            
            progress(1.0, desc="Complete!")
            
            info = f"""
✅ Batch detection complete!
- Files processed: {len(self.batch_segments)}
- Total segments detected: {total_segments}
- Method: {detection_method}

**Detection results:**
"""
            for data in self.batch_segments:
                info += f"\n- {data['filename']}: {len(data['segments'])} segments"
                if data['segment_data']:
                    avg_duration = np.mean([s['duration'] for s in data['segment_data']])
                    avg_rms = np.mean([s['rms'] for s in data['segment_data']])
                    info += f" (avg duration: {avg_duration:.3f}s, avg RMS: {avg_rms:.4f})"
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error in batch detection: {str(e)}\n{traceback.format_exc()}", None
    
    def extract_batch_features(self, progress=gr.Progress()):
        """Extract features from all segments in batch-processed files."""
        try:
            if not self.batch_segments:
                return "❌ Please run batch detection first", None
            
            self.batch_features = []
            total_segments = 0
            
            progress(0.1, desc="Starting feature extraction...")
            
            for i, data in enumerate(self.batch_segments):
                progress((i + 1) / len(self.batch_segments) * 0.8, desc=f"Extracting from {data['filename']}...")
                
                file_features = []
                for seg in data['segment_data']:
                    features = extract_semg_features(seg['data'], self.fs)
                    features['segment_start'] = seg['start_time']
                    features['segment_end'] = seg['end_time']
                    features['segment_duration'] = seg['duration']
                    file_features.append(features)
                    total_segments += 1
                
                self.batch_features.append({
                    'filename': data['filename'],
                    'features': file_features
                })
            
            progress(1.0, desc="Complete!")
            
            # Create summary visualization of key features
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Collect all features for plotting
            all_rms = []
            all_mdf = []
            all_mnf = []
            all_di = []
            file_labels = []
            
            for data in self.batch_features:
                for feat in data['features']:
                    all_rms.append(feat['RMS'])
                    all_mdf.append(feat['MDF'])
                    all_mnf.append(feat['MNF'])
                    all_di.append(feat['DI'])
                    file_labels.append(data['filename'][:15])  # Truncate for display
            
            # Plot RMS distribution
            axes[0, 0].bar(range(len(all_rms)), all_rms, color='steelblue', alpha=0.7)
            axes[0, 0].set_title('RMS by Segment')
            axes[0, 0].set_xlabel('Segment Index')
            axes[0, 0].set_ylabel('RMS')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot MDF distribution
            axes[0, 1].bar(range(len(all_mdf)), all_mdf, color='coral', alpha=0.7)
            axes[0, 1].set_title('Median Frequency (MDF) by Segment')
            axes[0, 1].set_xlabel('Segment Index')
            axes[0, 1].set_ylabel('MDF (Hz)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot MNF distribution
            axes[1, 0].bar(range(len(all_mnf)), all_mnf, color='seagreen', alpha=0.7)
            axes[1, 0].set_title('Mean Frequency (MNF) by Segment')
            axes[1, 0].set_xlabel('Segment Index')
            axes[1, 0].set_ylabel('MNF (Hz)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot Dimitrov Index distribution
            axes[1, 1].bar(range(len(all_di)), all_di, color='purple', alpha=0.7)
            axes[1, 1].set_title('Dimitrov Index (DI) by Segment')
            axes[1, 1].set_xlabel('Segment Index')
            axes[1, 1].set_ylabel('DI')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            info = f"""
✅ Batch feature extraction complete!
- Files processed: {len(self.batch_features)}
- Total segments: {total_segments}

**Feature summary (averages across all segments):**
- RMS: {np.mean(all_rms):.4f} ± {np.std(all_rms):.4f}
- MDF: {np.mean(all_mdf):.2f} ± {np.std(all_mdf):.2f} Hz
- MNF: {np.mean(all_mnf):.2f} ± {np.std(all_mnf):.2f} Hz
- DI: {np.mean(all_di):.6e} ± {np.std(all_di):.6e}

**Per-file breakdown:**
"""
            for data in self.batch_features:
                if data['features']:
                    avg_rms = np.mean([f['RMS'] for f in data['features']])
                    avg_mdf = np.mean([f['MDF'] for f in data['features']])
                    avg_di = np.mean([f['DI'] for f in data['features']])
                    info += f"\n- {data['filename']}: {len(data['features'])} segs, RMS={avg_rms:.4f}, MDF={avg_mdf:.1f}Hz, DI={avg_di:.6e}"
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error extracting features: {str(e)}\n{traceback.format_exc()}", None
    
    def export_batch_features(self, output_path, subject_id, fatigue_level, progress=gr.Progress()):
        """Export all batch features to a CSV file."""
        try:
            if not self.batch_features:
                return "❌ Please extract batch features first"
            
            progress(0.2, desc="Preparing features for export...")
            
            all_features = []
            segment_names = []
            
            for file_data in self.batch_features:
                for i, feat in enumerate(file_data['features']):
                    feat_copy = feat.copy()
                    feat_copy['source_file'] = file_data['filename']
                    all_features.append(feat_copy)
                    segment_names.append(f"{file_data['filename']}_seg{i+1:03d}")
            
            progress(0.6, desc="Writing CSV...")
            
            # Build annotations
            annotations = {}
            if subject_id:
                annotations['Subject'] = subject_id
            if fatigue_level:
                annotations['Fatigue_Level'] = fatigue_level
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Export
            export_features_to_csv(all_features, output_path, segment_names, annotations)
            
            progress(1.0, desc="Complete!")
            
            return f"""
✅ Batch features exported to: {output_path}

- Total segments: {len(all_features)}
- Files included: {len(self.batch_features)}
- Features per segment: {len(all_features[0]) if all_features else 0}
"""
        except Exception as e:
            import traceback
            return f"❌ Error exporting features: {str(e)}\n{traceback.format_exc()}"
    
    def perform_hht_analysis(self, segment_index, n_freq_bins, max_freq, normalize_length, 
                             do_normalize, use_ceemdan, normalize_amplitude, progress=gr.Progress()):
        """Perform HHT analysis on a segment using CEEMDAN."""
        try:
            if self.segment_data is None or len(self.segment_data) == 0:
                return "❌ Please detect segments first", None
            
            seg_idx = int(segment_index) - 1
            if seg_idx < 0 or seg_idx >= len(self.segment_data):
                return f"❌ Invalid segment index. Valid range: 1-{len(self.segment_data)}", None
            
            progress(0.1, desc="Preparing segment...")
            segment = self.segment_data[seg_idx]['data']
            
            # Optionally normalize length
            norm_len = int(normalize_length) if do_normalize else None
            
            progress(0.3, desc="Computing CEEMDAN decomposition..." if use_ceemdan else "Computing EMD decomposition...")
            
            # Perform enhanced HHT analysis
            progress(0.5, desc="Computing Hilbert spectrum...")
            self.hht_results = hht_analysis_enhanced(
                segment,
                fs=self.fs,
                n_freq_bins=int(n_freq_bins),
                max_freq=float(max_freq) if max_freq > 0 else None,
                normalize_length=norm_len,
                normalize_time=do_normalize,
                normalize_amplitude=normalize_amplitude,
                use_ceemdan=use_ceemdan,
                return_imfs=True,
                extract_features=True
            )
            
            # Create spectrum plot
            progress(0.7, desc="Creating visualization...")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Hilbert spectrum with improved colormap
            ax1 = axes[0, 0]
            spectrum = self.hht_results['spectrum']
            time = self.hht_results['time']
            freq = self.hht_results['frequency']
            
            # Use jet colormap for better sEMG visualization
            im = ax1.pcolormesh(time, freq, spectrum, shading='auto', cmap='jet')
            ax1.set_xlabel('Time (normalized)' if do_normalize else 'Time (s)')
            ax1.set_ylabel('Frequency (Hz)')
            ax1.set_title(f'Hilbert Spectrum ({"CEEMDAN" if use_ceemdan else "EMD"})')
            plt.colorbar(im, ax=ax1, label='Log Amplitude')
            
            # Marginal spectrum
            ax2 = axes[0, 1]
            ax2.plot(freq, self.hht_results['marginal_spectrum'], 'b-', linewidth=1)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Amplitude')
            ax2.set_title('Marginal Hilbert Spectrum')
            ax2.grid(True, alpha=0.3)
            
            # Original segment
            ax3 = axes[1, 0]
            seg_time = np.arange(len(segment)) / self.fs
            ax3.plot(seg_time, segment, 'b-', linewidth=0.5)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Amplitude')
            ax3.set_title(f'Segment {seg_idx + 1} Signal')
            ax3.grid(True, alpha=0.3)
            
            # IMF components
            ax4 = axes[1, 1]
            imfs = self.hht_results.get('imfs', [])
            for i, imf in enumerate(imfs[:5]):  # Show first 5 IMFs
                imf_time = np.arange(len(imf)) / self.fs
                offset = i * np.max(np.abs(imf)) * 2 if np.max(np.abs(imf)) > 0 else i
                ax4.plot(imf_time, imf + offset, label=f'IMF {i+1}', alpha=0.7)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Amplitude (offset)')
            ax4.set_title('IMF Components (CEEMDAN)' if use_ceemdan else 'IMF Components (EMD)')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            progress(1.0, desc="Complete!")
            
            # Format features display
            features = self.hht_results.get('features', {})
            features_str = "\n**sEMG Features:**\n"
            for key, value in features.items():
                if isinstance(value, float):
                    features_str += f"- {key}: {value:.4f}\n"
                else:
                    features_str += f"- {key}: {value}\n"
            
            info = f"""
✅ HHT Analysis Complete!

**Segment {seg_idx + 1}:**
- Original length: {self.hht_results['original_length']} samples
- Analyzed length: {self.hht_results['normalized_length']} samples
- Number of IMFs: {len(imfs)}
- Mean frequency: {self.hht_results['mean_frequency']:.2f} Hz
- Decomposition: {"CEEMDAN" if use_ceemdan else "EMD"}
- Time normalized: {"Yes" if do_normalize else "No"}
- Amplitude normalized: {"Yes" if normalize_amplitude else "No"}
{features_str}
"""
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error in HHT analysis: {str(e)}\n{traceback.format_exc()}", None
    
    def perform_augmentation(self, augmentation_method, n_augmented, perturbation, 
                            segment_index, use_ceemdan, progress=gr.Progress()):
        """Perform data augmentation on a segment using CEEMDAN."""
        try:
            if self.segment_data is None or len(self.segment_data) == 0:
                return "❌ Please detect segments first", None
            
            seg_idx = int(segment_index) - 1
            if seg_idx < 0 or seg_idx >= len(self.segment_data):
                return f"❌ Invalid segment index. Valid range: 1-{len(self.segment_data)}", None
            
            progress(0.1, desc="Preparing segment...")
            segment = self.segment_data[seg_idx]['data']
            
            progress(0.3, desc=f"Performing {augmentation_method} augmentation with {'CEEMDAN' if use_ceemdan else 'EMD'}...")
            
            if augmentation_method == 'imf_mixing':
                self.augmented_signals = augment_by_imf_mixing(
                    segment, n_augmented=int(n_augmented), 
                    imf_perturbation=float(perturbation),
                    use_ceemdan=use_ceemdan
                )
            elif augmentation_method == 'imf_scaling':
                self.augmented_signals = augment_by_imf_scaling(
                    segment, n_augmented=int(n_augmented),
                    scale_range=(1-float(perturbation), 1+float(perturbation)),
                    use_ceemdan=use_ceemdan
                )
            elif augmentation_method == 'noise_injection':
                self.augmented_signals = augment_by_noise_injection(
                    segment, n_augmented=int(n_augmented),
                    noise_level=float(perturbation)
                )
            elif augmentation_method == 'time_warping':
                self.augmented_signals = augment_by_time_warping(
                    segment, n_augmented=int(n_augmented),
                    warp_factor=float(perturbation)
                )
            elif augmentation_method == 'ceemdan_random_imf':
                # Use all detected segments for CEEMDAN random IMF combination
                all_segments = [seg['data'] for seg in self.segment_data]
                if len(all_segments) < 2:
                    return "❌ Need at least 2 segments for CEEMDAN random IMF method", None
                
                self.augmented_signals = [segment.copy()]  # Include original
                generated = augment_ceemdan_random_imf(
                    all_segments, 
                    n_augmented=int(n_augmented),
                    use_ceemdan=use_ceemdan
                )
                self.augmented_signals.extend(generated)
            elif augmentation_method == 'comprehensive':
                results = comprehensive_augmentation(
                    segment, n_per_method=max(1, int(n_augmented) // 4),
                    use_ceemdan=use_ceemdan
                )
                self.augmented_signals = [segment.copy()]
                # Properly iterate through results without array comparison
                for method_name, method_signals in results.items():
                    if method_name != 'original':  # Skip original
                        self.augmented_signals.extend(method_signals)
            
            progress(0.7, desc="Creating visualization...")
            
            # Create visualization
            n_show = min(len(self.augmented_signals), 6)
            fig, axes = plt.subplots(n_show, 1, figsize=(12, 2*n_show))
            if n_show == 1:
                axes = [axes]
            
            for i, aug_signal in enumerate(self.augmented_signals[:n_show]):
                time = np.arange(len(aug_signal)) / self.fs
                label = 'Original' if i == 0 else f'Augmented {i}'
                color = 'blue' if i == 0 else 'green'
                axes[i].plot(time, aug_signal, color=color, linewidth=0.5, alpha=0.7)
                axes[i].set_title(label)
                axes[i].set_xlabel('Time (s)')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            progress(1.0, desc="Complete!")
            
            info = f"""
✅ Augmentation Complete!

**Method**: {augmentation_method}
**Decomposition**: {"CEEMDAN" if use_ceemdan else "EMD"}
**Original segment**: {seg_idx + 1}
**Generated signals**: {len(self.augmented_signals)} (including original)
**Perturbation factor**: {perturbation}
"""
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error in augmentation: {str(e)}\n{traceback.format_exc()}", None
    
    def perform_batch_augmentation(self, file_objs, n_augmented, fs, value_column, 
                                    has_header, skip_rows, use_ceemdan, 
                                    progress=gr.Progress()):
        """
        Perform batch data augmentation using CEEMDAN-based IMF recombination.
        
        This implements the algorithm:
        1. Load multiple signal files (CSV or MAT) as input signals
        2. Decompose each signal using CEEMDAN into IMFs (pad to 8)
        3. For each generated signal, randomly select m=8 signals
        4. From each selected signal, take one random IMF at a unique position
        5. Sum all selected IMFs to create the artificial signal
        """
        try:
            if file_objs is None or len(file_objs) == 0:
                return "❌ Please upload signal files for augmentation", None, None
            
            if len(file_objs) < 2:
                return "❌ Need at least 2 signal files for IMF recombination", None, None
            
            progress(0.1, desc="Loading input signals...")
            
            # Load all signal files
            input_signals = []
            filenames = []
            
            for file_obj in file_objs:
                try:
                    signal, df = load_signal_file(
                        file_obj.name,
                        value_column=int(value_column),
                        has_header=has_header,
                        skip_rows=int(skip_rows)
                    )
                    input_signals.append(signal)
                    filenames.append(os.path.basename(file_obj.name))
                except Exception as e:
                    print(f"Warning: Failed to load {file_obj.name}: {e}")
            
            if len(input_signals) < 2:
                return "❌ Failed to load enough valid signal files (need at least 2)", None, None
            
            progress(0.3, desc=f"Loaded {len(input_signals)} signals. Starting CEEMDAN decomposition...")
            
            # Store input signals for potential export
            self.batch_augmentation_inputs = input_signals
            self.batch_augmentation_fs = float(fs)
            
            # Perform CEEMDAN-based augmentation
            progress(0.5, desc=f"Generating {n_augmented} artificial signals using IMF recombination...")
            
            self.augmented_signals = augment_ceemdan_random_imf(
                input_signals,
                n_augmented=int(n_augmented),
                use_ceemdan=use_ceemdan
            )
            
            progress(0.8, desc="Creating visualization...")
            
            # Create visualization showing original signals and generated signals
            n_show_input = min(len(input_signals), 4)
            n_show_output = min(len(self.augmented_signals), 4)
            n_cols = max(n_show_input, n_show_output)
            
            fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 6))
            
            # Ensure axes is 2D array even when n_cols == 1
            if n_cols == 1:
                axes = axes.reshape(2, 1)
            
            # Plot input signals (top row)
            for i in range(n_cols):
                if i < n_show_input:
                    time = np.arange(len(input_signals[i])) / float(fs)
                    axes[0, i].plot(time, input_signals[i], 'b-', linewidth=0.5, alpha=0.7)
                    axes[0, i].set_title(f'Input {i+1}: {filenames[i][:15]}', fontsize=9)
                    axes[0, i].set_xlabel('Time (s)', fontsize=8)
                    axes[0, i].grid(True, alpha=0.3)
                else:
                    axes[0, i].axis('off')
            
            # Plot generated signals (bottom row)
            for i in range(n_cols):
                if i < n_show_output:
                    aug_signal = self.augmented_signals[i]
                    time = np.arange(len(aug_signal)) / float(fs)
                    axes[1, i].plot(time, aug_signal, 'g-', linewidth=0.5, alpha=0.7)
                    axes[1, i].set_title(f'Generated {i+1}', fontsize=9)
                    axes[1, i].set_xlabel('Time (s)', fontsize=8)
                    axes[1, i].grid(True, alpha=0.3)
                else:
                    axes[1, i].axis('off')
            
            axes[0, 0].set_ylabel('Input Signals', fontsize=10)
            axes[1, 0].set_ylabel('Generated Signals', fontsize=10)
            
            plt.tight_layout()
            
            progress(1.0, desc="Complete!")
            
            # Calculate statistics for generated signals
            gen_lengths = [len(s) for s in self.augmented_signals]
            gen_rms = [np.sqrt(np.mean(s**2)) for s in self.augmented_signals]
            
            info = f"""
✅ Batch Augmentation Complete!

**Input:**
- Files loaded: {len(input_signals)}
- Input file names: {', '.join(filenames[:5])}{'...' if len(filenames) > 5 else ''}
- Sampling frequency: {fs} Hz

**Output:**
- Generated signals: {len(self.augmented_signals)}
- Signal length: {gen_lengths[0]} samples ({gen_lengths[0]/float(fs):.3f} s)
- Average RMS: {np.mean(gen_rms):.4f} ± {np.std(gen_rms):.4f}

**Method:** CEEMDAN-based IMF recombination
- Decomposition: {'CEEMDAN' if use_ceemdan else 'EMD'}
- IMFs per signal: 8 (padded with zeros if fewer)
- Each generated signal combines 8 IMFs from 8 randomly selected input signals
"""
            
            return info.strip(), fig, f"Generated {len(self.augmented_signals)} signals"
        except Exception as e:
            import traceback
            return f"❌ Error in batch augmentation: {str(e)}\n{traceback.format_exc()}", None, None
    
    def export_augmented_signals(self, output_dir, prefix, progress=gr.Progress()):
        """Export generated augmented signals to CSV files."""
        try:
            if self.augmented_signals is None or len(self.augmented_signals) == 0:
                return "❌ No augmented signals to export. Please generate signals first."
            
            progress(0.1, desc="Creating output directory...")
            os.makedirs(output_dir, exist_ok=True)
            
            fs = getattr(self, 'batch_augmentation_fs', self.fs)
            saved_files = []
            
            for i, signal in enumerate(self.augmented_signals):
                progress((i + 1) / len(self.augmented_signals) * 0.9, 
                        desc=f"Saving signal {i+1}/{len(self.augmented_signals)}...")
                
                filename = f"{prefix}_generated_{i+1:03d}.csv"
                filepath = os.path.join(output_dir, filename)
                
                # Create dataframe with time and signal
                time = np.arange(len(signal)) / fs
                df = pd.DataFrame({
                    'Time': time,
                    'Signal': signal
                })
                df.to_csv(filepath, index=False)
                saved_files.append(filepath)
            
            progress(1.0, desc="Complete!")
            
            return f"""
✅ Export Complete!

- Output directory: {output_dir}
- Files exported: {len(saved_files)}
- File prefix: {prefix}_generated_XXX.csv
- Sampling frequency: {fs} Hz
"""
        except Exception as e:
            import traceback
            return f"❌ Error exporting signals: {str(e)}\n{traceback.format_exc()}"
    
    def batch_hht_all_segments(self, n_freq_bins, normalize_length, max_freq, 
                                use_ceemdan, progress=gr.Progress()):
        """Perform HHT analysis on all detected segments."""
        try:
            if self.segment_data is None or len(self.segment_data) == 0:
                return "❌ Please detect segments first", None
            
            progress(0.1, desc="Preparing segments...")
            
            # Collect all segment data
            segments = [seg['data'] for seg in self.segment_data]
            
            progress(0.3, desc=f"Performing batch HHT on {len(segments)} segments...")
            
            # Perform batch analysis
            self.batch_hht_results = batch_hht_analysis(
                segments,
                fs=self.fs,
                n_freq_bins=int(n_freq_bins),
                normalize_length=int(normalize_length),
                max_freq=float(max_freq) if max_freq > 0 else None,
                use_ceemdan=use_ceemdan,
                extract_features=True
            )
            
            # Store parameters for later validation during export
            self._batch_hht_params = {
                'n_freq_bins': int(n_freq_bins),
                'normalize_length': int(normalize_length),
                'use_ceemdan': use_ceemdan
            }
            
            progress(0.7, desc="Creating visualization...")
            
            # Create visualization showing first 4 spectra
            n_show = min(len(segments), 4)
            fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
            
            if n_show == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(n_show):
                # Spectrum
                spectrum = self.batch_hht_results['spectra'][i]
                time = self.batch_hht_results['time']
                freq = self.batch_hht_results['frequency']
                
                im = axes[0, i].pcolormesh(time, freq, spectrum, shading='auto', cmap='jet')
                axes[0, i].set_title(f'Segment {i+1}')
                axes[0, i].set_xlabel('Time (norm)')
                axes[0, i].set_ylabel('Frequency (Hz)')
                plt.colorbar(im, ax=axes[0, i])
                
                # Original signal
                seg = segments[i]
                seg_time = np.arange(len(seg)) / self.fs
                axes[1, i].plot(seg_time, seg, 'b-', linewidth=0.5)
                axes[1, i].set_title(f'Signal {i+1}')
                axes[1, i].set_xlabel('Time (s)')
                axes[1, i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            progress(1.0, desc="Complete!")
            
            # Format features summary
            features_summary = "\n**Features Summary (first 4 segments):**\n"
            features_list = self.batch_hht_results.get('features', [])
            for i, features in enumerate(features_list[:4]):
                features_summary += f"\n*Segment {i+1}:*\n"
                features_summary += f"  WL={features['WL']:.2f}, ZC={features['ZC']}, SSC={features['SSC']}\n"
                features_summary += f"  MDF={features['MDF']:.1f}Hz, MNF={features['MNF']:.1f}Hz, IMNF={features['IMNF']:.1f}Hz\n"
                features_summary += f"  RMS={features['RMS']:.4f}, WIRE51={features['WIRE51']:.2f}, DI={features.get('DI', 0):.6e}\n"
            
            info = f"""
✅ Batch HHT Analysis Complete!

**Summary:**
- Total segments: {len(segments)}
- Spectra shape: {self.batch_hht_results['spectra_array'].shape}
- Decomposition: {"CEEMDAN" if use_ceemdan else "EMD"}
- Normalized length: {normalize_length} samples
- Frequency bins: {n_freq_bins}

All spectra have uniform size {n_freq_bins}x{normalize_length}, ready for CNN input.
{features_summary}
"""
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error in batch HHT: {str(e)}\n{traceback.format_exc()}", None
    
    def export_features_csv(self, output_path, subject_id, fatigue_level, progress=gr.Progress()):
        """Export all segment features to a CSV file."""
        try:
            if self.segment_data is None or len(self.segment_data) == 0:
                return "❌ Please detect segments first"
            
            progress(0.2, desc="Extracting features from all segments...")
            
            # Extract features from all segments
            features_list = []
            segment_names = []
            
            for i, seg in enumerate(self.segment_data):
                features = extract_semg_features(seg['data'], self.fs)
                features_list.append(features)
                segment_names.append(f"Segment_{i+1:03d}")
            
            progress(0.7, desc="Exporting to CSV...")
            
            # Build annotations
            annotations = {}
            if subject_id:
                annotations['Subject'] = subject_id
            if fatigue_level:
                annotations['Fatigue_Level'] = fatigue_level
            
            # Handle output path - if it's a directory, append default filename
            if os.path.isdir(output_path) or (not output_path.endswith('.csv') and not os.path.exists(output_path)):
                # It's a directory path, add default filename
                default_filename = self._get_default_feature_filename()
                output_path = os.path.join(output_path, default_filename)
            
            # Export
            export_features_to_csv(features_list, output_path, segment_names, annotations)
            
            progress(1.0, desc="Complete!")
            
            return f"✅ Features exported to: {output_path}\n\nExported {len(features_list)} segments with {len(features_list[0])} features each."
        
        except Exception as e:
            import traceback
            return f"❌ Error exporting features: {str(e)}\n{traceback.format_exc()}"
    
    def analyze_segment_features(self, file_objs, fs, value_column, has_header, skip_rows, progress=gr.Progress()):
        """
        Analyze features from single or multiple sEMG segment files (CSV or MAT).
        Each file should contain a single segment of sEMG data.
        """
        try:
            if file_objs is None or len(file_objs) == 0:
                return "❌ Please upload segment files", None, None
            
            progress(0.1, desc="Loading segment files...")
            
            # Load all segment files
            segments = []
            filenames = []
            
            for file_obj in file_objs:
                try:
                    signal, df = load_signal_file(
                        file_obj.name,
                        value_column=int(value_column),
                        has_header=has_header,
                        skip_rows=int(skip_rows)
                    )
                    segments.append(signal)
                    filenames.append(os.path.basename(file_obj.name))
                except Exception as e:
                    print(f"Warning: Failed to load {file_obj.name}: {e}")
            
            if len(segments) == 0:
                return "❌ No valid segment files loaded", None, None
            
            progress(0.3, desc=f"Extracting features from {len(segments)} segment(s)...")
            
            # Extract features from all segments
            features_list = []
            for i, segment in enumerate(segments):
                progress(0.3 + (i / len(segments)) * 0.5, desc=f"Extracting features from segment {i+1}/{len(segments)}...")
                features = extract_semg_features(segment, float(fs))
                features['filename'] = filenames[i]
                features['segment_index'] = i + 1
                features_list.append(features)
            
            # Store for export
            self.segment_features_analysis = {
                'features': features_list,
                'filenames': filenames,
                'fs': float(fs)
            }
            
            progress(0.8, desc="Creating visualization...")
            
            # Create trend visualization
            fig = self._create_feature_trend_plots(features_list, filenames)
            
            progress(1.0, desc="Complete!")
            
            # Create summary info
            info = self._create_feature_summary(features_list, filenames)
            
            # Create DataFrame for display
            df_display = self._create_feature_dataframe(features_list, filenames)
            
            return info, fig, df_display
            
        except Exception as e:
            import traceback
            return f"❌ Error analyzing segments: {str(e)}\n{traceback.format_exc()}", None, None
    
    def _create_feature_trend_plots(self, features_list, filenames):
        """Create trend plots for sEMG features across segments."""
        # Define feature groups for visualization
        time_domain_features = ['WL', 'ZC', 'SSC', 'RMS', 'MAV', 'VAR']
        frequency_features = ['MDF', 'MNF', 'IMNF', 'PKF', 'TTP']
        fatigue_indicators = ['DI', 'WIRE51']
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        segment_indices = list(range(1, len(features_list) + 1))
        
        # Plot 1: Time domain features
        ax = axes[0]
        for feature in time_domain_features:
            values = [f[feature] for f in features_list]
            ax.plot(segment_indices, values, 'o-', label=feature, linewidth=2, markersize=6)
        ax.set_xlabel('Segment Index', fontsize=11)
        ax.set_ylabel('Feature Value', fontsize=11)
        ax.set_title('Time Domain Features Trend', fontsize=12, fontweight='bold')
        ax.legend(loc='best', ncol=3, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Frequency domain features
        ax = axes[1]
        for feature in frequency_features:
            values = [f[feature] for f in features_list]
            ax.plot(segment_indices, values, 'o-', label=feature, linewidth=2, markersize=6)
        ax.set_xlabel('Segment Index', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        ax.set_title('Frequency Domain Features Trend', fontsize=12, fontweight='bold')
        ax.legend(loc='best', ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Fatigue indicators
        ax = axes[2]
        # Use dual y-axes for different scales
        ax2 = ax.twinx()
        
        # Plot DI on left axis (very small values - scale to 1e14 for better display)
        di_values = [f['DI'] * 1e14 for f in features_list]  # Scale for display
        line1 = ax.plot(segment_indices, di_values, 'ro-', label='DI (×10⁻¹⁴)', linewidth=2, markersize=6)
        ax.set_ylabel('DI Value (×10⁻¹⁴)', fontsize=11, color='r')
        ax.tick_params(axis='y', labelcolor='r')
        
        # Plot WIRE51 on right axis
        wire51_values = [f['WIRE51'] for f in features_list]
        line2 = ax2.plot(segment_indices, wire51_values, 'bs-', label='WIRE51', linewidth=2, markersize=6)
        ax2.set_ylabel('WIRE51 Value', fontsize=11, color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        ax.set_xlabel('Segment Index', fontsize=11)
        ax.set_title('Fatigue Indicators Trend', fontsize=12, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_feature_summary(self, features_list, filenames):
        """Create summary text for features."""
        n_segments = len(features_list)
        
        info = f"""
✅ Feature Analysis Complete!

**Summary:**
- Total segments analyzed: {n_segments}
- Sampling frequency: {self.segment_features_analysis['fs']} Hz

**Feature Statistics (Mean ± Std):**

*Time Domain:*
"""
        
        # Calculate statistics for key features
        time_features = ['WL', 'ZC', 'SSC', 'RMS', 'MAV', 'VAR']
        for feat in time_features:
            values = [f[feat] for f in features_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            info += f"  {feat}: {mean_val:.4f} ± {std_val:.4f}\n"
        
        info += "\n*Frequency Domain:*\n"
        freq_features = ['MDF', 'MNF', 'IMNF', 'PKF']
        for feat in freq_features:
            values = [f[feat] for f in features_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            info += f"  {feat}: {mean_val:.2f} ± {std_val:.2f} Hz\n"
        
        info += "\n*Fatigue Indicators:*\n"
        di_values = [f['DI'] for f in features_list]
        wire51_values = [f['WIRE51'] for f in features_list]
        info += f"  DI: {np.mean(di_values):.6e} ± {np.std(di_values):.6e}\n"
        info += f"  WIRE51: {np.mean(wire51_values):.4f} ± {np.std(wire51_values):.4f}\n"
        
        if n_segments > 1:
            info += f"\n**Trend Analysis:**\n"
            # Check if features show increasing/decreasing trends
            mdf_trend = "decreasing" if features_list[-1]['MDF'] < features_list[0]['MDF'] else "increasing"
            di_trend = "increasing" if features_list[-1]['DI'] > features_list[0]['DI'] else "decreasing"
            info += f"  MDF: {mdf_trend} (first: {features_list[0]['MDF']:.1f} Hz → last: {features_list[-1]['MDF']:.1f} Hz)\n"
            info += f"  DI: {di_trend} (first: {features_list[0]['DI']:.6e} → last: {features_list[-1]['DI']:.6e})\n"
            
            if mdf_trend == "decreasing" and di_trend == "increasing":
                info += "  ⚠️ Possible fatigue pattern detected (MDF↓ + DI↑)\n"
        
        return info.strip()
    
    def _create_feature_dataframe(self, features_list, filenames):
        """Create a pandas DataFrame for feature display."""
        df_data = []
        for i, features in enumerate(features_list):
            row = {
                'Segment': i + 1,
                'Filename': filenames[i],
                'WL': features['WL'],
                'ZC': features['ZC'],
                'SSC': features['SSC'],
                'RMS': features['RMS'],
                'MAV': features['MAV'],
                'MDF': features['MDF'],
                'MNF': features['MNF'],
                'IMNF': features['IMNF'],
                'DI': features['DI'],
                'WIRE51': features['WIRE51'],
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def export_segment_features_csv(self, output_path, progress=gr.Progress()):
        """Export analyzed segment features to CSV."""
        try:
            if not hasattr(self, 'segment_features_analysis') or not self.segment_features_analysis:
                return "❌ Please analyze segments first"
            
            progress(0.3, desc="Preparing data for export...")
            
            features_list = self.segment_features_analysis['features']
            filenames = self.segment_features_analysis['filenames']
            
            # Create detailed CSV with all features
            df_data = []
            for i, features in enumerate(features_list):
                row = {
                    'Segment_Index': i + 1,
                    'Filename': filenames[i],
                }
                # Add all features
                for key, value in features.items():
                    if key not in ['filename', 'segment_index']:
                        row[key] = value
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            progress(0.7, desc="Writing CSV file...")
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Export to CSV
            df.to_csv(output_path, index=False)
            
            progress(1.0, desc="Complete!")
            
            return f"""
✅ Features exported successfully!

**Output file:** {output_path}
**Segments:** {len(features_list)}
**Features per segment:** {len(df.columns) - 2}

The CSV contains all extracted features for each segment.
"""
        except Exception as e:
            import traceback
            return f"❌ Error exporting features: {str(e)}\n{traceback.format_exc()}"
    
    def analyze_segment_features_hht(self, file_objs, fs, value_column, has_header, skip_rows,
                                     n_freq_bins, normalize_length, use_ceemdan, progress=gr.Progress()):
        """
        Perform batch HHT analysis on uploaded sEMG segment files.
        Each segment is independently processed with HHT and features are extracted.
        """
        try:
            if file_objs is None or len(file_objs) == 0:
                return "❌ Please upload segment files", None
            
            progress(0.1, desc="Loading segment files...")
            
            # Load all segment files
            segments = []
            filenames = []
            
            for file_obj in file_objs:
                try:
                    signal, df = load_signal_file(
                        file_obj.name,
                        value_column=int(value_column),
                        has_header=has_header,
                        skip_rows=int(skip_rows)
                    )
                    segments.append(signal)
                    filenames.append(os.path.basename(file_obj.name))
                except Exception as e:
                    print(f"Warning: Failed to load {file_obj.name}: {e}")
            
            if len(segments) == 0:
                return "❌ No valid segment files loaded", None
            
            progress(0.2, desc=f"Computing HHT for {len(segments)} segment(s)...")
            
            # Perform batch HHT analysis
            self.segment_features_hht = batch_hht_analysis(
                segments,
                fs=float(fs),
                n_freq_bins=int(n_freq_bins),
                normalize_length=int(normalize_length),
                use_ceemdan=use_ceemdan
            )
            
            # Store filenames for export
            self.segment_features_hht['filenames'] = filenames
            self.segment_features_hht['fs'] = float(fs)
            
            progress(0.8, desc="Creating visualization...")
            
            # Create visualization with a subset of spectra
            spectra_list = self.segment_features_hht['spectra']
            features_list = self.segment_features_hht['features']
            time_axis = self.segment_features_hht['time']
            freq_axis = self.segment_features_hht['frequency']
            
            n_show = min(6, len(spectra_list))
            n_cols = 2
            n_rows = (n_show + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(n_show):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col]
                
                spectrum = spectra_list[i]
                
                im = ax.pcolormesh(time_axis, freq_axis, spectrum, shading='auto', cmap='jet')
                ax.set_xlabel('Time (normalized)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title(f'{filenames[i]} (Seg {i+1})', fontsize=10)
                plt.colorbar(im, ax=ax, label='Log Amplitude')
            
            # Hide unused subplots
            for i in range(n_show, n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            progress(1.0, desc="Complete!")
            
            # Create summary info
            info = f"""
✅ Batch HHT Analysis Complete!

**Summary:**
- Total segments: {len(segments)}
- Decomposition: {"CEEMDAN" if use_ceemdan else "EMD"}
- Spectrum size: {n_freq_bins} × {normalize_length}
- Frequency range: 20-450 Hz

**Features Summary (first 4 segments):**
"""
            for i, (features, filename) in enumerate(zip(features_list[:4], filenames[:4])):
                info += f"\n*{filename}:*\n"
                info += f"  WL={features['WL']:.2f}, ZC={features['ZC']}, SSC={features['SSC']}\n"
                info += f"  MDF={features['MDF']:.1f}Hz, MNF={features['MNF']:.1f}Hz, IMNF={features['IMNF']:.1f}Hz\n"
                info += f"  RMS={features['RMS']:.4f}, WIRE51={features['WIRE51']:.2f}\n"
            
            if len(segments) > 4:
                info += f"\n... and {len(segments) - 4} more segments"
            
            return info.strip(), fig
            
        except Exception as e:
            import traceback
            return f"❌ Error in batch HHT analysis: {str(e)}\n{traceback.format_exc()}", None
    
    def export_segment_features_hht_csv(self, output_path, progress=gr.Progress()):
        """Export HHT features from uploaded segments to CSV."""
        try:
            if not hasattr(self, 'segment_features_hht') or self.segment_features_hht is None:
                return "❌ Please run HHT analysis first"
            
            progress(0.2, desc="Preparing HHT features for export...")
            
            features_list = self.segment_features_hht['features']
            filenames = self.segment_features_hht['filenames']
            
            # Create DataFrame
            df_data = []
            for i, features in enumerate(features_list):
                row = {
                    'Segment': i + 1,
                    'Filename': filenames[i],
                    'WL': features['WL'],
                    'ZC': features['ZC'],
                    'SSC': features['SSC'],
                    'RMS': features['RMS'],
                    'MAV': features['MAV'],
                    'VAR': features['VAR'],
                    'MDF': features['MDF'],
                    'MNF': features['MNF'],
                    'IMNF': features['IMNF'],
                    'PKF': features['PKF'],
                    'TTP': features['TTP'],
                    'DI': features['DI'],
                    'WIRE51': features['WIRE51']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            progress(0.6, desc="Writing CSV file...")
            
            # Handle output path - if it's a directory, append default filename
            if os.path.isdir(output_path) or (not output_path.endswith('.csv') and not os.path.exists(output_path)):
                # It's a directory path, add default filename
                output_path = os.path.join(output_path, 'segment_hht_features.csv')
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Export to CSV
            df.to_csv(output_path, index=False)
            
            progress(1.0, desc="Complete!")
            
            return f"""
✅ HHT features exported successfully!

**Output file:** {output_path}
**Segments:** {len(features_list)}
**Features per segment:** {len(df.columns) - 2}

The CSV contains HHT-extracted features for each segment.
"""
        except Exception as e:
            import traceback
            return f"❌ Error exporting HHT features: {str(e)}\n{traceback.format_exc()}"
    
    def export_data(self, output_dir, export_full, export_segments, 
                   subject_id, fatigue_level, quality_rating, action_type, notes,
                   export_hht, export_augmented, custom_prefix, progress=gr.Progress()):
        """Export processed data and segments with annotations."""
        try:
            if self.filtered_signal is None:
                return "Please process signal first"
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            results = []
            
            # Prepare annotations
            annotations = {}
            if subject_id:
                annotations['subject'] = subject_id
            if fatigue_level:
                annotations['fatigue_level'] = fatigue_level
            if quality_rating:
                annotations['quality_rating'] = quality_rating
            if action_type:
                annotations['action_type'] = action_type
            if notes:
                annotations['notes'] = notes
            
            prefix = custom_prefix if custom_prefix else 'segment'
            
            progress(0.1, desc="Exporting data...")
            
            # Export full processed signal
            if export_full:
                full_path = os.path.join(output_dir, f'{prefix}_processed_signal.csv')
                save_processed_data(full_path, self.filtered_signal, fs=self.fs)
                results.append(f"✅ Saved processed signal to: {full_path}")
            
            progress(0.3, desc="Exporting segments...")
            
            # Export segments with annotations
            if export_segments and self.segment_data:
                segment_dir = os.path.join(output_dir, 'segments')
                saved_files = export_segments_to_csv(
                    self.filtered_signal,
                    self.segment_data,
                    fs=self.fs,
                    output_dir=segment_dir,
                    prefix=prefix,
                    annotations=annotations
                )
                results.append(f"✅ Saved {len(saved_files)} segment files to: {segment_dir}")
            
            progress(0.5, desc="Exporting HHT results...")
            
            # Export HHT results - batch export for all segments
            if export_hht and self.segment_data:
                from semg_preprocessing.hht import export_activity_segments_hht
                
                # Create HHT directories
                hht_matrices_dir = os.path.join(output_dir, 'hht_matrices')
                hht_images_dir = os.path.join(output_dir, 'hht_images')
                os.makedirs(hht_matrices_dir, exist_ok=True)
                os.makedirs(hht_images_dir, exist_ok=True)
                
                # Export Hilbert spectra for all detected segments
                try:
                    # Extract segment boundaries - use correct keys from segment_signal()
                    segment_tuples = [(int(seg['start_index']), int(seg['end_index'])) 
                                     for seg in self.segment_data]
                    
                    # Check if we have precomputed HHT results from batch analysis
                    precomputed_hht = None
                    if hasattr(self, 'batch_hht_results') and self.batch_hht_results is not None:
                        # Verify the batch results match current segments
                        # Check both count and that segments have compatible parameters
                        batch_spectra = self.batch_hht_results.get('spectra', [])
                        if (len(batch_spectra) == len(segment_tuples) and
                            hasattr(self, '_batch_hht_params') and
                            self._batch_hht_params.get('n_freq_bins') == 256 and
                            self._batch_hht_params.get('normalize_length') == 256):
                            precomputed_hht = self.batch_hht_results
                            print("  ✓ Using precomputed HHT results (fast export)")
                        else:
                            print("  ⚠ Batch HHT results don't match current export parameters, will recompute")
                    else:
                        print("  ℹ No precomputed HHT results, computing now...")
                    
                    # Export matrices to hht_matrices directory
                    export_info_matrices = export_activity_segments_hht(
                        self.filtered_signal,
                        segment_tuples,
                        fs=self.fs,
                        output_dir=hht_matrices_dir,
                        base_filename=prefix if prefix else 'segment',
                        n_freq_bins=256,
                        normalize_length=256,
                        use_ceemdan=True,
                        save_visualization=False,  # We'll save images separately
                        precomputed_spectra=precomputed_hht
                    )
                    
                    # Save visualizations to hht_images directory
                    import matplotlib.pyplot as plt
                    for idx, (start, end) in enumerate(segment_tuples):
                        segment_num = idx + 1
                        segment_signal = self.filtered_signal[start:end]
                        
                        # Load the saved spectrum
                        npz_file = os.path.join(hht_matrices_dir, f'{prefix if prefix else "segment"}_{segment_num:03d}.npz')
                        data = np.load(npz_file)
                        spectrum = data['spectrum']
                        time = data['time']
                        frequency = data['frequency']
                        
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                        im = ax.pcolormesh(time, frequency, spectrum, shading='auto', cmap='jet')
                        ax.set_xlabel('Normalized Time', fontsize=12)
                        ax.set_ylabel('Frequency (Hz)', fontsize=12)
                        ax.set_title(f'Hilbert Spectrum - Segment {segment_num:03d}', fontsize=14, fontweight='bold')
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.set_label('Amplitude (Log Scale)', fontsize=11)
                        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                        plt.tight_layout()
                        
                        # Save to images directory
                        img_path = os.path.join(hht_images_dir, f'{prefix if prefix else "segment"}_{segment_num:03d}.png')
                        fig.savefig(img_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    
                    results.append(f"✅ Saved {len(segment_tuples)} Hilbert spectrum matrices to: {hht_matrices_dir}")
                    results.append(f"✅ Saved {len(segment_tuples)} Hilbert spectrum images to: {hht_images_dir}")
                except Exception as e:
                    results.append(f"⚠️  Error exporting HHT: {str(e)}")
            
            progress(0.7, desc="Exporting augmented signals...")
            
            # Export augmented signals
            if export_augmented and self.augmented_signals:
                aug_dir = os.path.join(output_dir, 'augmented')
                os.makedirs(aug_dir, exist_ok=True)
                
                for i, aug_signal in enumerate(self.augmented_signals):
                    aug_path = os.path.join(aug_dir, f'{prefix}_augmented_{i:03d}.csv')
                    aug_df = pd.DataFrame({
                        'Time (s)': np.arange(len(aug_signal)) / self.fs,
                        'Signal': aug_signal
                    })
                    
                    with open(aug_path, 'w') as f:
                        f.write(f"# Augmented signal {i}\n")
                        f.write(f"# Original: {'Yes' if i == 0 else 'No'}\n")
                        if annotations:
                            for key, value in annotations.items():
                                f.write(f"# {key}: {value}\n")
                        f.write("#\n")
                        aug_df.to_csv(f, index=False)
                
                results.append(f"✅ Saved {len(self.augmented_signals)} augmented signals to: {aug_dir}")
            
            progress(1.0, desc="Export complete!")
            
            return "\n".join(results) if results else "No export options selected"
        except Exception as e:
            import traceback
            return f"❌ Error exporting data: {str(e)}\n{traceback.format_exc()}"


def create_gui():
    """Create and configure the Gradio interface."""
    
    processor = EMGProcessorGUI()
    
    with gr.Blocks(title="sEMG Signal Preprocessing") as app:
        gr.Markdown("""
        # 🔬 sEMG Signal Preprocessing Toolkit v2.0
        ## 表面肌电信号预处理工具
        
        A comprehensive tool for sEMG signal preprocessing with filtering, detection, 
        HHT analysis, data augmentation, and batch processing.
        
        **New Features:**
        - 📁 Batch file processing for dataset building (全流程批处理)
        - 📊 Hilbert-Huang Transform (HHT) analysis
        - 🔄 EMD-based data augmentation
        - 🏷️ Data annotation support
        - 💾 Comprehensive export options
        - 📈 Enhanced visualization for all results
        """)
        
        with gr.Tabs():
            # Tab 1: Load Data
            with gr.Tab("📁 Load Data / 加载数据"):
                with gr.Tabs():
                    with gr.Tab("Single File"):
                        gr.Markdown("### Upload your sEMG CSV file")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                file_input = gr.File(label="Select Signal File (CSV or MAT)", file_types=['.csv', '.mat'])
                                fs_input = gr.Number(value=1000, label="Sampling Frequency (Hz)", precision=0)
                                column_input = gr.Number(value=1, label="Signal Column Index", precision=0,
                                                        info="Column containing signal values (0-indexed)")
                                header_input = gr.Checkbox(value=True, label="File has header row")
                                skip_rows_input = gr.Number(value=0, label="Skip rows (跳过行数)", precision=0,
                                                           info="Number of rows to skip before reading data (for files with 2+ header rows)")
                                load_btn = gr.Button("Load File", variant="primary")
                            
                            with gr.Column(scale=2):
                                load_info = gr.Textbox(label="Load Status", lines=7)
                                load_plot = gr.Plot(label="Signal Preview")
                        
                        load_btn.click(
                            fn=processor.load_file,
                            inputs=[file_input, fs_input, column_input, header_input, skip_rows_input],
                            outputs=[load_info, load_plot]
                        )
                    
                    with gr.Tab("Batch Processing / 批处理"):
                        gr.Markdown("""
                        ### Upload multiple CSV files for batch processing
                        **批量上传CSV文件进行处理** - 支持全流程批处理：滤波→检测→特征提取→导出
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_file_input = gr.File(
                                    label="Select Multiple Signal Files (选择多个信号文件 CSV/MAT)", 
                                    file_types=['.csv', '.mat'],
                                    file_count="multiple"
                                )
                                batch_fs_input = gr.Number(value=1000, label="Sampling Frequency (Hz)", precision=0)
                                batch_column_input = gr.Number(value=1, label="Signal Column Index", precision=0)
                                batch_header_input = gr.Checkbox(value=True, label="Files have header row")
                                batch_skip_rows_input = gr.Number(value=0, label="Skip rows (跳过行数)", precision=0,
                                                                  info="For files with 2 header rows, set this to 1")
                                batch_load_btn = gr.Button("📂 Load All Files", variant="primary")
                            
                            with gr.Column(scale=2):
                                batch_load_info = gr.Textbox(label="Batch Load Status", lines=12)
                                batch_load_plot = gr.Plot(label="Signal Previews (显示所有信号)")
                        
                        batch_load_btn.click(
                            fn=processor.load_batch_files,
                            inputs=[batch_file_input, batch_fs_input, batch_column_input, batch_header_input, batch_skip_rows_input],
                            outputs=[batch_load_info, batch_load_plot]
                        )
            
            # Tab 2: Apply Filters
            with gr.Tab("🔧 Apply Filters / 应用滤波器"):
                gr.Markdown("### Configure and apply preprocessing filters")
                
                with gr.Tabs():
                    with gr.Tab("Single File"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**Bandpass Filter Settings**")
                                lowcut_input = gr.Slider(5, 50, value=20, step=1, label="High-pass cutoff (Hz)")
                                highcut_input = gr.Slider(200, 500, value=450, step=10, label="Low-pass cutoff (Hz)")
                                order_input = gr.Slider(2, 6, value=4, step=1, label="Filter order")
                                
                                gr.Markdown("**Notch Filter Settings** (for power line interference)")
                                notch_freq_input = gr.Radio([0, 50, 60], value=50, label="Notch frequency (Hz, 0=disabled)")
                                harmonics_input = gr.Textbox(value="1,2,3", label="Harmonics (comma-separated)")
                                
                                filter_btn = gr.Button("Apply Filters", variant="primary")
                            
                            with gr.Column(scale=2):
                                filter_info = gr.Textbox(label="Filter Status", lines=4)
                                filter_plot = gr.Plot(label="Before/After Comparison")
                        
                        filter_btn.click(
                            fn=processor.apply_filters,
                            inputs=[lowcut_input, highcut_input, order_input, notch_freq_input, harmonics_input],
                            outputs=[filter_info, filter_plot]
                        )
                    
                    with gr.Tab("Batch Filtering / 批量滤波"):
                        gr.Markdown("### Apply filters to all batch-loaded files")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**Bandpass Filter Settings**")
                                batch_lowcut_input = gr.Slider(5, 50, value=20, step=1, label="High-pass cutoff (Hz)")
                                batch_highcut_input = gr.Slider(200, 500, value=450, step=10, label="Low-pass cutoff (Hz)")
                                batch_order_input = gr.Slider(2, 6, value=4, step=1, label="Filter order")
                                
                                gr.Markdown("**Notch Filter Settings**")
                                batch_notch_freq_input = gr.Radio([0, 50, 60], value=50, label="Notch frequency (Hz)")
                                batch_harmonics_input = gr.Textbox(value="1,2,3", label="Harmonics")
                                
                                batch_filter_btn = gr.Button("🔧 Apply Filters to All Files", variant="primary")
                            
                            with gr.Column(scale=2):
                                batch_filter_info = gr.Textbox(label="Batch Filter Status", lines=12)
                                batch_filter_plot = gr.Plot(label="Filtered Signals Preview")
                        
                        batch_filter_btn.click(
                            fn=processor.apply_batch_filters,
                            inputs=[batch_lowcut_input, batch_highcut_input, batch_order_input, 
                                   batch_notch_freq_input, batch_harmonics_input],
                            outputs=[batch_filter_info, batch_filter_plot]
                        )
            
            # Tab 3: Detect Activity
            with gr.Tab("🎯 Detect Activity / 检测肌肉活动"):
                gr.Markdown("""
                ### Muscle Activity Detection
                
                **Two Detection Methods Available:**
                
                **1. PELT Algorithm** (Pruned Exact Linear Time)
                - Energy-based adaptive penalty zones (low energy = more sensitive)
                - Multi-dimensional features (time, frequency, complexity domains)
                - Multi-detector ensemble with individual sensitivity control
                - Activity/Non-activity classification (filters out rest periods)
                - Intelligent merging of dense events (gaps < 50ms)
                
                **2. HHT Algorithm** (Hilbert-Huang Transform)
                - Computes full-signal Hilbert spectrum
                - Detects high-energy stripes characteristic of muscle activity
                - Maps detected patterns back to time domain
                - Frequency range: 20-450 Hz (sEMG effective range)
                - Dynamic resolution scaling with signal duration
                
                **Choose one method based on your needs:**
                - PELT: Fast, robust for general sEMG signals
                - HHT: More sensitive to frequency-domain features, good for complex patterns
                """)
                
                with gr.Tabs():
                    with gr.Tab("Single File"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                # Method selection
                                detection_method_input = gr.Radio(
                                    ["PELT", "HHT"],
                                    value="PELT",
                                    label="Detection Method",
                                    info="Choose between PELT and HHT algorithms"
                                )
                                
                                # Common parameters
                                gr.Markdown("**Common Parameters:**")
                                min_duration_input = gr.Slider(0.01, 10.0, value=0.1, step=0.01,
                                                              label="Minimum segment duration (s)")
                                max_duration_input = gr.Slider(0, 30.0, value=0, step=1.0,
                                                              label="Maximum segment duration (s, 0=no limit)")
                                
                                # PELT-specific parameters
                                gr.Markdown("**PELT Algorithm Parameters:**")
                                sensitivity_input = gr.Slider(0.1, 5.0, value=1.5, step=0.1,
                                                             label="Base Detection Sensitivity",
                                                             info="Only for PELT. Lower = more sensitive")
                                
                                gr.Markdown("**Multi-Detector Ensemble Settings:**")
                                use_multi_detector_input = gr.Checkbox(value=True, label="Enable Multi-Detector Ensemble")
                                n_detectors_input = gr.Slider(1, 5, value=3, step=1,
                                                             label="Number of Detectors")
                                detector_sens_input = gr.Textbox(
                                    label="Individual Detector Sensitivities (optional)",
                                    placeholder="e.g., 1.0, 1.5, 2.0 (leave blank for auto range)",
                                    value="",
                                    info="Comma-separated sensitivity values, one per detector"
                                )
                                fusion_method_input = gr.Radio(
                                    ["confidence", "voting", "union"],
                                    value="confidence",
                                    label="Fusion Method",
                                    info="confidence: weighted by scores, voting: majority, union: combine all"
                                )
                                
                                gr.Markdown("**Activity Classification:**")
                                classification_threshold_input = gr.Slider(
                                    -2.0, 2.0, value=0.5, step=0.1,
                                    label="Classification Strictness",
                                    info="Negative = very lenient, 0 = median, Positive = strict"
                                )
                                use_clustering_input = gr.Checkbox(
                                    value=False, 
                                    label="Use Clustering for Classification",
                                    info="K-means clustering (slower but automatic)"
                                )
                                
                                gr.Markdown("**Advanced PELT Settings:**")
                                use_tkeo_input = gr.Checkbox(
                                    value=True,
                                    label="Enable TKEO Preprocessing",
                                    info="Teager-Kaiser Energy Operator enhances changepoint detection"
                                )
                                merge_threshold_input = gr.Slider(
                                    0.3, 0.9, value=0.7, step=0.05,
                                    label="Segment Merge Threshold",
                                    info="Energy ratio for merging (lower = more aggressive, extended range 0.3-0.9)"
                                )
                                max_merge_count_input = gr.Slider(
                                    1, 5, value=3, step=1,
                                    label="Max Segments to Merge",
                                    info="Maximum PELT segments merged into one event (prevents merging independent actions)"
                                )
                                
                                # HHT-specific parameters
                                gr.Markdown("**HHT Algorithm Parameters:**")
                                hht_sensitivity_input = gr.Slider(
                                    0.1, 3.0, value=1.0, step=0.1,
                                    label="HHT Detection Sensitivity",
                                    info="Lower = detect more events (0.5=sensitive), Higher = stricter (2.0=strict)"
                                )
                                hht_energy_threshold_input = gr.Slider(
                                    0.1, 0.95, value=0.65, step=0.05,
                                    label="Energy Threshold Percentile",
                                    info="Only for HHT. EXTENDED RANGE: 0.1=very sensitive (near baseline), 0.95=very strict. Use low values for weak/extreme cases."
                                )
                                hht_local_contrast_weight_input = gr.Slider(
                                    0.0, 0.7, value=0.3, step=0.1,
                                    label="Local Contrast Weight",
                                    info="Weight for local energy contrast (0=global only, 0.5=balanced)"
                                )
                                hht_temporal_compactness_input = gr.Slider(
                                    0.1, 0.7, value=0.3, step=0.05,
                                    label="Temporal Compactness",
                                    info="Only for HHT. Minimum energy density in time (0.1-0.7)"
                                )
                                hht_resolution_input = gr.Slider(
                                    64, 256, value=128, step=32,
                                    label="Resolution per Second",
                                    info="Only for HHT. Time bins per second (64-256)"
                                )
                                
                                detect_btn = gr.Button("Detect Activity", variant="primary")
                            
                            with gr.Column(scale=2):
                                detect_info = gr.Textbox(label="Detection Results", lines=12)
                                detect_plot = gr.Plot(label="Detected Segments")
                        
                        detect_btn.click(
                            fn=processor.detect_activity,
                            inputs=[detection_method_input, min_duration_input, max_duration_input, sensitivity_input, 
                                   n_detectors_input, fusion_method_input, use_multi_detector_input,
                                   use_clustering_input, detector_sens_input, classification_threshold_input,
                                   use_tkeo_input, merge_threshold_input, max_merge_count_input,
                                   hht_energy_threshold_input, hht_temporal_compactness_input, hht_resolution_input,
                                   hht_sensitivity_input, hht_local_contrast_weight_input],
                            outputs=[detect_info, detect_plot]
                        )
                    
                    with gr.Tab("Batch Detection / 批量检测"):
                        gr.Markdown("### Detect activity in all batch-filtered files")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                # Method selection for batch
                                batch_detection_method_input = gr.Radio(
                                    ["PELT", "HHT"],
                                    value="PELT",
                                    label="Detection Method",
                                    info="Choose between PELT and HHT algorithms"
                                )
                                
                                batch_min_duration_input = gr.Slider(0.01, 10.0, value=0.1, step=0.01,
                                                                    label="Minimum segment duration (s)")
                                batch_max_duration_input = gr.Slider(0, 30.0, value=0, step=1.0,
                                                                    label="Maximum segment duration (s, 0=no limit)")
                                batch_sensitivity_input = gr.Slider(0.1, 5.0, value=1.5, step=0.1,
                                                                   label="Base Detection Sensitivity")
                                
                                gr.Markdown("**Multi-Detector Ensemble Settings:**")
                                batch_use_multi_detector_input = gr.Checkbox(value=True, label="Enable Multi-Detector Ensemble")
                                batch_n_detectors_input = gr.Slider(1, 5, value=3, step=1,
                                                                   label="Number of Detectors")
                                batch_detector_sens_input = gr.Textbox(
                                    label="Individual Detector Sensitivities (optional)",
                                    placeholder="e.g., 1.0, 1.5, 2.0",
                                    value=""
                                )
                                batch_fusion_method_input = gr.Radio(
                                    ["confidence", "voting", "union"],
                                    value="confidence",
                                    label="Fusion Method"
                                )
                                
                                gr.Markdown("**Activity Classification:**")
                                batch_classification_threshold_input = gr.Slider(
                                    -2.0, 2.0, value=0.5, step=0.1,
                                    label="Classification Strictness",
                                    info="Negative = very lenient, 0 = median, Positive = strict"
                                )
                                batch_use_clustering_input = gr.Checkbox(
                                    value=False,
                                    label="Use Clustering for Classification"
                                )
                                
                                gr.Markdown("**Advanced Detection Settings:**")
                                batch_use_tkeo_input = gr.Checkbox(
                                    value=True,
                                    label="Enable TKEO Preprocessing",
                                    info="Teager-Kaiser Energy Operator enhances changepoint detection"
                                )
                                batch_merge_threshold_input = gr.Slider(
                                    0.3, 0.9, value=0.7, step=0.05,
                                    label="Segment Merge Threshold",
                                    info="Energy ratio for merging (lower = more aggressive, range 0.3-0.9)"
                                )
                                batch_max_merge_count_input = gr.Slider(
                                    1, 5, value=3, step=1,
                                    label="Max Segments to Merge",
                                    info="Maximum PELT segments merged into one event"
                                )
                                
                                # HHT-specific parameters for batch
                                gr.Markdown("**HHT Algorithm Parameters:**")
                                batch_hht_sensitivity_input = gr.Slider(
                                    0.1, 3.0, value=1.0, step=0.1,
                                    label="HHT Detection Sensitivity",
                                    info="Lower = detect more events, Higher = stricter"
                                )
                                batch_hht_energy_threshold_input = gr.Slider(
                                    0.1, 0.95, value=0.65, step=0.05,
                                    label="Energy Threshold Percentile",
                                    info="Only for HHT. EXTENDED RANGE: 0.1=very sensitive (near baseline), 0.95=very strict"
                                )
                                batch_hht_local_contrast_weight_input = gr.Slider(
                                    0.0, 0.7, value=0.3, step=0.1,
                                    label="Local Contrast Weight",
                                    info="Weight for local energy contrast"
                                )
                                batch_hht_temporal_compactness_input = gr.Slider(
                                    0.1, 0.7, value=0.3, step=0.05,
                                    label="Temporal Compactness",
                                    info="Only for HHT. Minimum energy density"
                                )
                                batch_hht_resolution_input = gr.Slider(
                                    64, 256, value=128, step=32,
                                    label="Resolution per Second",
                                    info="Only for HHT. Time bins per second"
                                )
                                
                                batch_detect_btn = gr.Button("🎯 Detect in All Files", variant="primary")
                            
                            with gr.Column(scale=2):
                                batch_detect_info = gr.Textbox(label="Batch Detection Results", lines=15)
                                batch_detect_plot = gr.Plot(label="Detection Results Overview")
                        
                        batch_detect_btn.click(
                            fn=processor.detect_batch_activity,
                            inputs=[batch_detection_method_input, batch_min_duration_input, batch_max_duration_input, batch_sensitivity_input, 
                                   batch_n_detectors_input, batch_fusion_method_input, batch_use_multi_detector_input,
                                   batch_use_clustering_input, batch_detector_sens_input, batch_classification_threshold_input,
                                   batch_use_tkeo_input, batch_merge_threshold_input, batch_max_merge_count_input,
                                   batch_hht_energy_threshold_input, batch_hht_temporal_compactness_input, batch_hht_resolution_input,
                                   batch_hht_sensitivity_input, batch_hht_local_contrast_weight_input],
                            outputs=[batch_detect_info, batch_detect_plot]
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("### Extract Features from All Segments (批量特征提取)")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_extract_btn = gr.Button("📊 Extract Features from All Segments", variant="secondary")
                            with gr.Column(scale=2):
                                batch_extract_info = gr.Textbox(label="Feature Extraction Results", lines=15)
                                batch_extract_plot = gr.Plot(label="Features Overview")
                        
                        batch_extract_btn.click(
                            fn=processor.extract_batch_features,
                            inputs=[],
                            outputs=[batch_extract_info, batch_extract_plot]
                        )
            
            # Tab 4: HHT Analysis
            with gr.Tab("📊 HHT Analysis / 希尔伯特-黄变换"):
                gr.Markdown("""
                ### Hilbert-Huang Transform (HHT) Analysis
                
                Perform time-frequency analysis using **CEEMDAN** (Complete Ensemble EMD with Adaptive Noise) 
                for improved decomposition quality. Extracts sEMG features: WL, ZC, SSC, MDF, MNF, IMNF, WIRE51.
                """)
                
                with gr.Tabs():
                    with gr.Tab("Single Segment"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                hht_segment_input = gr.Number(value=1, label="Segment Index", precision=0,
                                                             info="Which segment to analyze (1-indexed)")
                                hht_freq_bins = gr.Number(value=256, label="Frequency Bins", precision=0)
                                hht_max_freq = gr.Number(value=500, label="Max Frequency (Hz, 0=auto)", precision=0)
                                hht_use_ceemdan = gr.Checkbox(value=True, label="Use CEEMDAN (recommended)")
                                hht_normalize = gr.Checkbox(value=True, label="Normalize time axis (for CNN)")
                                hht_norm_length = gr.Number(value=256, label="Target length (samples)", precision=0)
                                hht_norm_amp = gr.Checkbox(value=False, label="Normalize amplitude")
                                
                                hht_btn = gr.Button("Perform HHT Analysis", variant="primary")
                            
                            with gr.Column(scale=2):
                                hht_info = gr.Textbox(label="HHT Results & Features", lines=20)
                                hht_plot = gr.Plot(label="Hilbert Spectrum & IMFs")
                        
                        hht_btn.click(
                            fn=processor.perform_hht_analysis,
                            inputs=[hht_segment_input, hht_freq_bins, hht_max_freq, hht_norm_length, 
                                   hht_normalize, hht_use_ceemdan, hht_norm_amp],
                            outputs=[hht_info, hht_plot]
                        )
                    
                    with gr.Tab("Batch HHT (All Segments)"):
                        gr.Markdown("""
                        **One-Click Analysis**: Process all detected segments at once.
                        All spectra will have uniform size for CNN input.
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_hht_freq_bins = gr.Number(value=256, label="Frequency Bins", precision=0)
                                batch_hht_norm_length = gr.Number(value=256, label="Normalized Length", precision=0)
                                batch_hht_max_freq = gr.Number(value=500, label="Max Frequency (Hz)", precision=0)
                                batch_hht_ceemdan = gr.Checkbox(value=True, label="Use CEEMDAN")
                                
                                batch_hht_btn = gr.Button("🚀 Analyze All Segments", variant="primary")
                            
                            with gr.Column(scale=2):
                                batch_hht_info = gr.Textbox(label="Batch Results & Features", lines=20)
                                batch_hht_plot = gr.Plot(label="Spectra Overview")
                        
                        batch_hht_btn.click(
                            fn=processor.batch_hht_all_segments,
                            inputs=[batch_hht_freq_bins, batch_hht_norm_length, batch_hht_max_freq, batch_hht_ceemdan],
                            outputs=[batch_hht_info, batch_hht_plot]
                        )
            
            # Tab 4.5: Segment Feature Analysis
            with gr.Tab("📈 Feature Analysis / 特征分析"):
                gr.Markdown("""
                ### sEMG Segment Feature Analysis (sEMG分段特征分析)
                
                Analyze sEMG features from uploaded segment CSV/MAT files.
                Each uploaded file should contain one sEMG segment.
                
                **Two analysis modes:**
                1. **Basic Feature Analysis**: Extract time/frequency features (WL, ZC, SSC, MDF, MNF, etc.)
                2. **HHT Analysis**: Compute Hilbert-Huang Transform for each segment + features
                """)
                
                with gr.Tabs():
                    with gr.Tab("Basic Feature Analysis"):
                        gr.Markdown("""
                        **Features extracted:**
                        - Time domain: WL, ZC, SSC, RMS, MAV, VAR
                        - Frequency domain: MDF, MNF, IMNF, PKF, TTP
                        - Fatigue indicators: DI (Dimitrov Index), WIRE51
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**Upload Segment Files**")
                                segment_files_input = gr.File(
                                    label="Select Segment Files (选择分段文件 CSV/MAT)", 
                                    file_types=['.csv', '.mat'],
                                    file_count="multiple"
                                )
                                segment_fs_input = gr.Number(value=1000, label="Sampling Frequency (Hz)", precision=0)
                                segment_column_input = gr.Number(value=1, label="Signal Column Index", precision=0)
                                segment_header_input = gr.Checkbox(value=True, label="Files have header row")
                                segment_skip_rows_input = gr.Number(value=0, label="Skip rows (跳过行数)", precision=0,
                                                                   info="For files with 2 header rows, set this to 1")
                                
                                analyze_features_btn = gr.Button("📊 Analyze Features", variant="primary")
                            
                            with gr.Column(scale=2):
                                feature_analysis_info = gr.Textbox(label="Feature Analysis Results", lines=20)
                                feature_trend_plot = gr.Plot(label="Feature Trends Across Segments")
                        
                        gr.Markdown("---")
                        gr.Markdown("### Feature Data Table & Export")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                feature_table_display = gr.Dataframe(
                                    label="Extracted Features (可查看所有特征值)",
                                    interactive=False
                                )
                        
                        # Single click handler that updates all outputs
                        analyze_features_btn.click(
                            fn=processor.analyze_segment_features,
                            inputs=[segment_files_input, segment_fs_input, segment_column_input, 
                                   segment_header_input, segment_skip_rows_input],
                            outputs=[feature_analysis_info, feature_trend_plot, feature_table_display]
                        )
                        
                        gr.Markdown("### Export Features to CSV")
                        with gr.Row():
                            with gr.Column(scale=1):
                                feature_csv_output_path = gr.Textbox(
                                    value="./output/segment_features_analysis.csv", 
                                    label="Output CSV Path",
                                    info="Directory path or full file path (.csv)"
                                )
                                export_segment_features_btn = gr.Button("💾 Export Features to CSV", variant="secondary")
                            with gr.Column(scale=2):
                                feature_export_status = gr.Textbox(label="Export Status", lines=5)
                        
                        export_segment_features_btn.click(
                            fn=processor.export_segment_features_csv,
                            inputs=[feature_csv_output_path],
                            outputs=[feature_export_status]
                        )
                    
                    with gr.Tab("HHT Batch Analysis"):
                        gr.Markdown("""
                        ### Hilbert-Huang Transform Batch Processing
                        
                        **Compute HHT for each uploaded segment independently:**
                        - Each segment gets its own HHT decomposition
                        - Hilbert spectrum computed for each segment
                        - Features extracted from HHT results
                        - All segments processed with uniform resolution
                        
                        **Note:** This is independent from event detection HHT (they are decoupled).
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**Upload Segment Files**")
                                hht_segment_files_input = gr.File(
                                    label="Select Segment Files (选择分段文件 CSV/MAT)", 
                                    file_types=['.csv', '.mat'],
                                    file_count="multiple"
                                )
                                hht_segment_fs_input = gr.Number(value=1000, label="Sampling Frequency (Hz)", precision=0)
                                hht_segment_column_input = gr.Number(value=1, label="Signal Column Index", precision=0)
                                hht_segment_header_input = gr.Checkbox(value=True, label="Files have header row")
                                hht_segment_skip_rows_input = gr.Number(value=0, label="Skip rows (跳过行数)", precision=0)
                                
                                gr.Markdown("**HHT Parameters**")
                                hht_segment_freq_bins = gr.Number(value=256, label="Frequency Bins", precision=0)
                                hht_segment_norm_length = gr.Number(value=256, label="Normalized Length", precision=0)
                                hht_segment_ceemdan = gr.Checkbox(value=True, label="Use CEEMDAN (recommended)")
                                
                                analyze_hht_btn = gr.Button("🔬 Compute HHT for All Segments", variant="primary")
                            
                            with gr.Column(scale=2):
                                hht_analysis_info = gr.Textbox(label="HHT Analysis Results", lines=20)
                                hht_spectra_plot = gr.Plot(label="Hilbert Spectra (First 6 segments)")
                        
                        analyze_hht_btn.click(
                            fn=processor.analyze_segment_features_hht,
                            inputs=[hht_segment_files_input, hht_segment_fs_input, hht_segment_column_input,
                                   hht_segment_header_input, hht_segment_skip_rows_input,
                                   hht_segment_freq_bins, hht_segment_norm_length, hht_segment_ceemdan],
                            outputs=[hht_analysis_info, hht_spectra_plot]
                        )
                        
                        gr.Markdown("### Export HHT Features to CSV")
                        with gr.Row():
                            with gr.Column(scale=1):
                                hht_csv_output_path = gr.Textbox(
                                    value="./output", 
                                    label="Output Directory or CSV Path",
                                    info="Directory path or full file path (.csv). Default filename: segment_hht_features.csv"
                                )
                                export_hht_features_btn = gr.Button("💾 Export HHT Features to CSV", variant="secondary")
                            with gr.Column(scale=2):
                                hht_export_status = gr.Textbox(label="Export Status", lines=5)
                        
                        export_hht_features_btn.click(
                            fn=processor.export_segment_features_hht_csv,
                            inputs=[hht_csv_output_path],
                            outputs=[hht_export_status]
                        )
            
            # Tab 5: Segment Feature Analysis (already added above)
            
            # Tab 6: Data Augmentation
            with gr.Tab("🔄 Augmentation / 数据增强"):
                gr.Markdown("""
                ### CEEMDAN-Based Data Augmentation (基于CEEMDAN的数据增强)
                
                Generate synthetic sEMG signals using **CEEMDAN** decomposition for more stable
                and physically meaningful IMF components.
                """)
                
                with gr.Tabs():
                    with gr.Tab("Single Segment Augmentation"):
                        gr.Markdown("**Augment from detected segments** - Use segments from the loaded/processed signal")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                aug_segment_input = gr.Number(value=1, label="Segment Index", precision=0)
                                aug_method_input = gr.Radio(
                                    ["imf_mixing", "imf_scaling", "noise_injection", "time_warping", 
                                     "ceemdan_random_imf", "comprehensive"],
                                    value="ceemdan_random_imf",
                                    label="Augmentation Method",
                                    info="ceemdan_random_imf: Uses all segments to generate new signals"
                                )
                                aug_use_ceemdan = gr.Checkbox(value=True, label="Use CEEMDAN (recommended)")
                                aug_n_input = gr.Slider(1, 50, value=10, step=1, label="Number of augmented signals")
                                aug_perturbation = gr.Slider(0.01, 0.5, value=0.1, step=0.01,
                                                            label="Perturbation factor (for mixing/scaling/noise)")
                                
                                aug_btn = gr.Button("Generate Augmented Data", variant="primary")
                            
                            with gr.Column(scale=2):
                                aug_info = gr.Textbox(label="Augmentation Results", lines=10)
                                aug_plot = gr.Plot(label="Augmented Signals")
                        
                        aug_btn.click(
                            fn=processor.perform_augmentation,
                            inputs=[aug_method_input, aug_n_input, aug_perturbation, aug_segment_input, aug_use_ceemdan],
                            outputs=[aug_info, aug_plot]
                        )
                    
                    with gr.Tab("Batch File Augmentation / 批量文件增强"):
                        gr.Markdown("""
                        ### Batch IMF Recombination Data Augmentation (批量IMF重组数据增强)
                        
                        **Algorithm / 算法说明:**
                        1. Load multiple CSV sEMG signal files as input (加载多个CSV信号文件作为输入)
                        2. Decompose each signal using CEEMDAN into IMFs, pad to 8 IMFs (使用CEEMDAN分解为IMF，补齐到8个)
                        3. For each generated signal, randomly select m=8 signals (对于每个生成信号，随机选择m=8个信号)
                        4. From each selected signal, take one random IMF at a unique position (从每个选中信号取一个不同位置的IMF)
                        5. Sum all selected IMFs to create the artificial signal (将所有选中的IMF相加生成人工信号)
                        
                        **Note:** This method requires at least 2 input signal files. More input files = more variety in generated signals.
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_aug_file_input = gr.File(
                                    label="Select Signal Files for Augmentation (选择信号文件 CSV/MAT)", 
                                    file_types=['.csv', '.mat'],
                                    file_count="multiple"
                                )
                                batch_aug_fs_input = gr.Number(value=1000, label="Sampling Frequency (Hz)", precision=0)
                                batch_aug_column_input = gr.Number(value=1, label="Signal Column Index", precision=0)
                                batch_aug_header_input = gr.Checkbox(value=True, label="Files have header row")
                                batch_aug_skip_rows_input = gr.Number(value=0, label="Skip rows (跳过行数)", precision=0,
                                                                      info="For files with 2 header rows, set this to 1")
                                batch_aug_n_input = gr.Slider(1, 100, value=20, step=1, 
                                                              label="Number of signals to generate (目标生成数量)")
                                batch_aug_ceemdan_input = gr.Checkbox(value=True, label="Use CEEMDAN (recommended)")
                                
                                batch_aug_btn = gr.Button("🚀 Generate Artificial Signals", variant="primary")
                            
                            with gr.Column(scale=2):
                                batch_aug_info = gr.Textbox(label="Batch Augmentation Results", lines=15)
                                batch_aug_plot = gr.Plot(label="Input vs Generated Signals")
                                batch_aug_status = gr.Textbox(label="Status", lines=1)
                        
                        batch_aug_btn.click(
                            fn=processor.perform_batch_augmentation,
                            inputs=[batch_aug_file_input, batch_aug_n_input, batch_aug_fs_input, 
                                   batch_aug_column_input, batch_aug_header_input, batch_aug_skip_rows_input, 
                                   batch_aug_ceemdan_input],
                            outputs=[batch_aug_info, batch_aug_plot, batch_aug_status]
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("### Export Generated Signals (导出生成的信号)")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_aug_output_dir = gr.Textbox(value="./output/augmented", 
                                                                  label="Output Directory")
                                batch_aug_prefix = gr.Textbox(value="artificial_semg", 
                                                              label="File prefix")
                                batch_aug_export_btn = gr.Button("💾 Export Generated Signals", variant="secondary")
                            with gr.Column(scale=2):
                                batch_aug_export_info = gr.Textbox(label="Export Status", lines=5)
                        
                        batch_aug_export_btn.click(
                            fn=processor.export_augmented_signals,
                            inputs=[batch_aug_output_dir, batch_aug_prefix],
                            outputs=[batch_aug_export_info]
                        )
            
            # Tab 7: Export Results
            with gr.Tab("💾 Export Results / 导出结果"):
                gr.Markdown("### Export processed data, segments, analysis results, and features")
                
                with gr.Tabs():
                    with gr.Tab("Single File Export"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**Output Settings**")
                                output_dir_input = gr.Textbox(value="./output", label="Output Directory")
                                custom_prefix_input = gr.Textbox(value="", label="Custom filename prefix (optional)")
                                
                                gr.Markdown("**Export Options**")
                                export_full_input = gr.Checkbox(value=True, label="Export full processed signal")
                                export_segments_input = gr.Checkbox(value=True, label="Export individual segments")
                                export_hht_input = gr.Checkbox(value=False, label="Export HHT results")
                                export_augmented_input = gr.Checkbox(value=False, label="Export augmented signals")
                                
                                gr.Markdown("**Annotations / 数据标注**")
                                subject_input = gr.Textbox(value="", label="Subject ID (受试者)")
                                fatigue_input = gr.Dropdown(
                                    choices=["", "fresh", "mild_fatigue", "moderate_fatigue", "severe_fatigue"],
                                    value="", label="Fatigue Level (疲劳程度)"
                                )
                                quality_input = gr.Slider(1, 5, value=3, step=1, label="Quality Rating (动作质量评级)")
                                action_input = gr.Textbox(value="", label="Action Type (动作类型)")
                                notes_input = gr.Textbox(value="", label="Notes (备注)", lines=2)
                                
                                export_btn = gr.Button("Export Data", variant="primary")
                                
                                gr.Markdown("---")
                                gr.Markdown("**Feature Export / 特征导出**")
                                features_output_path = gr.Textbox(value="./output", 
                                                                 label="Output Directory (输出目录)",
                                                                 placeholder="Enter directory path (e.g., ./output). Filename will be auto-generated from signal file.")
                                features_subject = gr.Textbox(value="", label="Subject ID for features")
                                features_fatigue = gr.Dropdown(
                                    choices=["", "fresh", "mild_fatigue", "moderate_fatigue", "severe_fatigue"],
                                    value="", label="Fatigue Level for features"
                                )
                                export_features_btn = gr.Button("📊 Export All Segment Features", variant="secondary")
                                features_export_info = gr.Textbox(label="Features Export Status", lines=3)
                            
                            with gr.Column(scale=2):
                                export_info = gr.Textbox(label="Export Status", lines=15)
                        
                        export_btn.click(
                            fn=processor.export_data,
                            inputs=[output_dir_input, export_full_input, export_segments_input,
                                   subject_input, fatigue_input, quality_input, action_input, notes_input,
                                   export_hht_input, export_augmented_input, custom_prefix_input],
                            outputs=[export_info]
                        )
                        
                        export_features_btn.click(
                            fn=processor.export_features_csv,
                            inputs=[features_output_path, features_subject, features_fatigue],
                            outputs=[features_export_info]
                        )
                    
                    with gr.Tab("Batch Export / 批量导出"):
                        gr.Markdown("""
                        ### Export all batch-processed features to CSV
                        **导出所有批处理结果的特征到CSV文件**
                        
                        Prerequisites: Run batch filtering → batch detection → batch feature extraction first.
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_features_output_path = gr.Textbox(
                                    value="./output/batch_features.csv", 
                                    label="Batch Features CSV Path"
                                )
                                batch_features_subject = gr.Textbox(value="", label="Subject ID (optional)")
                                batch_features_fatigue = gr.Dropdown(
                                    choices=["", "fresh", "mild_fatigue", "moderate_fatigue", "severe_fatigue"],
                                    value="", label="Fatigue Level (optional)"
                                )
                                batch_export_features_btn = gr.Button("📊 Export All Batch Features", variant="primary")
                            
                            with gr.Column(scale=2):
                                batch_features_export_info = gr.Textbox(label="Batch Export Status", lines=10)
                        
                        batch_export_features_btn.click(
                            fn=processor.export_batch_features,
                            inputs=[batch_features_output_path, batch_features_subject, batch_features_fatigue],
                            outputs=[batch_features_export_info]
                        )
            
            # Tab 8: Help
            with gr.Tab("ℹ️ Help / 帮助"):
                gr.Markdown("""
                ## Quick Start Guide / 快速入门
                
                ### 1. Load Data (加载数据)
                - **Single file**: Upload a single CSV for analysis
                - **Batch processing**: Upload multiple files for full pipeline processing
                - **Skip rows**: For CSV files with 2+ header rows, set skip_rows appropriately
                
                ### 2. Apply Filters (应用滤波)
                - **Single File**: Apply to single loaded file
                - **Batch Filtering**: Apply to all batch-loaded files at once
                - **Bandpass**: 20-450 Hz recommended for sEMG
                - **Notch**: 50 Hz (Europe/Asia) or 60 Hz (Americas)
                
                ### 3. Detect Activity (检测活动)
                - **Single File**: Detect in single file
                - **Batch Detection**: Detect in all batch-filtered files at once
                - **Extract Features**: Extract features from all detected segments
                - **Advanced PELT algorithm** with multi-detector ensemble
                
                ### 4. HHT Analysis (希尔伯特-黄变换)
                - **CEEMDAN** decomposition for stable IMFs (recommended)
                - Generates Hilbert spectrum (time-frequency representation)
                - **Batch HHT**: One-click analysis of all segments
                - **Time normalization**: Uniform spectra sizes for CNN input
                - **Feature extraction**: WL, ZC, SSC, MDF, MNF, IMNF, WIRE51, DI
                
                ### 5. Feature Analysis (特征分析) **NEW!**
                - **Upload segment CSV files**: Single or multiple sEMG segments
                - **Automatic feature extraction**: All time/frequency domain features
                - **Trend visualization**: See how features change across segments
                - **CSV export**: Save all extracted features for further analysis
                - **Fatigue detection**: Identify fatigue patterns (MDF↓, DI↑)
                
                ### 6. Data Augmentation (数据增强)
                - **Single Segment**: Augment from detected segments in the loaded signal
                - **Batch File Augmentation** (推荐): 
                  - Upload multiple CSV signal files directly
                  - Specify target number of generated signals
                  - Uses CEEMDAN-based IMF recombination algorithm:
                    1. Decompose each input signal into IMFs (pad to 8)
                    2. For each generated signal, randomly select 8 source signals
                    3. Each source contributes 1 IMF at a unique position
                    4. Sum all 8 IMFs to create artificial signal
                  - Export generated signals to CSV files
                
                ### 7. Export (导出)
                - **Single File Export**: Export processed signal, segments, and features
                - **Batch Export**: Export all batch-processed features to CSV
                - **Augmented Signals Export**: Export generated artificial signals to CSV
                - Add annotations: subject ID, fatigue level, quality rating
                
                ---
                
                ## Batch Processing Workflow / 批处理工作流程
                
                1. **Load**: Batch Processing tab → Load All Files
                2. **Filter**: Batch Filtering tab → Apply Filters to All Files  
                3. **Detect**: Batch Detection tab → Detect in All Files
                4. **Features**: Batch Detection tab → Extract Features from All Segments
                5. **Export**: Batch Export tab → Export All Batch Features
                
                ## Feature Analysis Workflow / 特征分析工作流程
                
                1. **Upload**: Feature Analysis tab → Select segment CSV files
                2. **Configure**: Set sampling frequency and CSV format options
                3. **Analyze**: Click "Analyze Features" to extract all features
                4. **Visualize**: View feature trends and statistics
                5. **Export**: Click "Export Features to CSV" to save results
                
                ## Data Augmentation Workflow / 数据增强工作流程
                
                1. **Upload**: Augmentation → Batch File Augmentation tab → Select CSV files
                2. **Configure**: Set target number of signals to generate
                3. **Generate**: Click "Generate Artificial Signals"
                4. **Export**: Click "Export Generated Signals" to save to CSV
                
                ---
                
                ## sEMG Features Guide / sEMG特征指南
                
                - **WL** (Waveform Length): 波形长度，反映信号复杂度
                - **ZC** (Zero Crossings): 过零点数，反映频率
                - **SSC** (Slope Sign Changes): 斜率变化数
                - **MDF** (Median Frequency): 中值频率，疲劳指标
                - **MNF** (Mean Frequency): 平均频率
                - **IMNF** (Instantaneous Mean Freq): 瞬时平均频率
                - **WIRE51**: 小波可靠性指数 (基于DWT离散小波变换)
                - **DI** (Dimitrov Index): 迪米特罗夫指数，疲劳指标 (高值=更疲劳)
                - **RMS**: 均方根值，反映信号幅度
                
                ## Parameter Guide / 参数指南
                
                - **Skip rows**: 跳过CSV开头的行数 (如有2个标题行则设为1)
                - **Sensitivity**: 0.5 (more segments) to 2.0 (fewer segments)
                - **Perturbation**: 0.05 (subtle) to 0.3 (significant variation)
                - **Quality Rating**: 1 (poor) to 5 (excellent)
                - **CEEMDAN**: More stable than EMD, recommended for sEMG
                """)
        
        gr.Markdown("""
        ---
        **sEMG Preprocessing Toolkit v0.5.0** | [GitHub](https://github.com/PRIMOCOSMOS/sEMG-pre-processing) | MIT License
        """)
    
    return app


if __name__ == "__main__":
    app = create_gui()
    app.queue(default_concurrency_limit=4)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
