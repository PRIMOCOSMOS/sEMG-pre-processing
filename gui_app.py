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
import tempfile
import zipfile
from io import BytesIO

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
    apply_bandpass_filter,
    apply_notch_filter,
    detect_muscle_activity,
    segment_signal,
    save_processed_data,
    export_segments_to_csv,
    # HHT functions
    emd_decomposition,
    compute_hilbert_spectrum,
    hht_analysis,
    save_hilbert_spectrum,
    # Augmentation functions
    augment_by_imf_mixing,
    augment_by_imf_recombination,
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
        # HHT results
        self.hht_results = None
        # Augmentation results
        self.augmented_signals = None
        
    def load_file(self, file_obj, fs, value_column, has_header):
        """Load CSV file and extract signal."""
        try:
            if file_obj is None:
                return "Please upload a file", None
            
            # Load the data
            self.fs = float(fs)
            self.signal, self.df = load_csv_data(
                file_obj.name,
                value_column=int(value_column),
                has_header=has_header
            )
            self.current_filename = os.path.basename(file_obj.name)
            
            # Create preview plot
            fig, ax = plt.subplots(figsize=(12, 4))
            time = np.arange(len(self.signal)) / self.fs
            ax.plot(time, self.signal, 'b-', linewidth=0.5, alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Loaded sEMG Signal: {self.current_filename}')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            info = f"""
✅ File loaded successfully!
- Filename: {self.current_filename}
- Samples: {len(self.signal)}
- Duration: {len(self.signal)/self.fs:.2f} seconds
- Sampling frequency: {self.fs} Hz
- Signal range: [{self.signal.min():.3f}, {self.signal.max():.3f}]
            """
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error loading file: {str(e)}\n{traceback.format_exc()}", None
    
    def load_batch_files(self, file_objs, fs, value_column, has_header, progress=gr.Progress()):
        """Load multiple CSV files for batch processing."""
        try:
            if file_objs is None or len(file_objs) == 0:
                return "Please upload files", None
            
            self.fs = float(fs)
            self.batch_data = []
            
            progress(0.1, desc="Loading files...")
            
            for i, file_obj in enumerate(file_objs):
                progress((i + 1) / len(file_objs) * 0.8, desc=f"Loading {os.path.basename(file_obj.name)}...")
                
                try:
                    signal, df = load_csv_data(
                        file_obj.name,
                        value_column=int(value_column),
                        has_header=has_header
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
            
            # Create summary plot
            fig, axes = plt.subplots(min(len(self.batch_data), 4), 1, figsize=(12, 3*min(len(self.batch_data), 4)))
            if len(self.batch_data) == 1:
                axes = [axes]
            
            for i, data in enumerate(self.batch_data[:4]):
                time = np.arange(len(data['signal'])) / self.fs
                axes[i].plot(time, data['signal'], 'b-', linewidth=0.5, alpha=0.7)
                axes[i].set_title(data['filename'])
                axes[i].set_xlabel('Time (s)')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            info = f"""
✅ Batch loading complete!
- Files loaded: {len(self.batch_data)} / {len(file_objs)}
- Sampling frequency: {self.fs} Hz

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
    
    def detect_activity(self, method, min_duration, sensitivity, use_clustering, adaptive_pen, progress=gr.Progress()):
        """Detect muscle activity segments."""
        try:
            if self.filtered_signal is None:
                return "❌ Please apply filters first", None
            
            progress(0.1, desc="Initializing detection...")
            
            # Detect muscle activity
            progress(0.3, desc="Detecting muscle activity...")
            self.segments = detect_muscle_activity(
                self.filtered_signal,
                fs=self.fs,
                method=method,
                min_duration=float(min_duration),
                sensitivity=float(sensitivity),
                use_clustering=use_clustering,
                adaptive_pen=adaptive_pen
            )
            
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
            
            # Highlight detected segments
            colors = plt.cm.Set1(np.linspace(0, 1, max(len(self.segments), 1)))
            for i, (start, end) in enumerate(self.segments):
                ax.axvspan(start/self.fs, end/self.fs, alpha=0.3, color=colors[i % len(colors)],
                          label=f'Segment {i+1}' if i < 5 else '')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Muscle Activity Detection ({len(self.segments)} segments detected)')
            ax.grid(True, alpha=0.3)
            if self.segments:
                ax.legend(loc='upper right', ncol=2)
            
            plt.tight_layout()
            
            # Create info text
            progress(0.9, desc="Formatting results...")
            info = f"✅ Detected {len(self.segments)} muscle activity segments:\n\n"
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
    
    def perform_hht_analysis(self, segment_index, n_freq_bins, max_freq, normalize_length, 
                             do_normalize, progress=gr.Progress()):
        """Perform HHT analysis on a segment."""
        try:
            if self.segment_data is None or len(self.segment_data) == 0:
                return "❌ Please detect segments first", None, None
            
            seg_idx = int(segment_index) - 1
            if seg_idx < 0 or seg_idx >= len(self.segment_data):
                return f"❌ Invalid segment index. Valid range: 1-{len(self.segment_data)}", None, None
            
            progress(0.1, desc="Preparing segment...")
            segment = self.segment_data[seg_idx]['data']
            
            # Optionally normalize length
            norm_len = int(normalize_length) if do_normalize else None
            
            progress(0.3, desc="Computing EMD decomposition...")
            
            # Perform HHT analysis
            progress(0.5, desc="Computing Hilbert spectrum...")
            self.hht_results = hht_analysis(
                segment,
                fs=self.fs,
                n_freq_bins=int(n_freq_bins),
                max_freq=float(max_freq) if max_freq > 0 else None,
                normalize_length=norm_len,
                return_imfs=True
            )
            
            # Create spectrum plot
            progress(0.7, desc="Creating visualization...")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Hilbert spectrum
            ax1 = axes[0, 0]
            spectrum = self.hht_results['spectrum']
            time = self.hht_results['time']
            freq = self.hht_results['frequency']
            
            im = ax1.pcolormesh(time, freq, spectrum, shading='auto', cmap='hot')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Frequency (Hz)')
            ax1.set_title('Hilbert Spectrum')
            plt.colorbar(im, ax=ax1, label='Amplitude')
            
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
                ax4.plot(imf_time, imf + i * np.max(np.abs(imf)) * 2, label=f'IMF {i+1}', alpha=0.7)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Amplitude (offset)')
            ax4.set_title('IMF Components')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            progress(1.0, desc="Complete!")
            
            info = f"""
✅ HHT Analysis Complete!

**Segment {seg_idx + 1}:**
- Original length: {self.hht_results['original_length']} samples
- Analyzed length: {len(segment)} samples
- Number of IMFs: {len(imfs)}
- Mean frequency: {self.hht_results['mean_frequency']:.2f} Hz
- Frequency bins: {n_freq_bins}
- Max frequency: {max_freq if max_freq > 0 else self.fs/2} Hz
"""
            
            return info.strip(), fig, None
        except Exception as e:
            import traceback
            return f"❌ Error in HHT analysis: {str(e)}\n{traceback.format_exc()}", None, None
    
    def perform_augmentation(self, augmentation_method, n_augmented, perturbation, 
                            segment_index, progress=gr.Progress()):
        """Perform data augmentation on a segment."""
        try:
            if self.segment_data is None or len(self.segment_data) == 0:
                return "❌ Please detect segments first", None
            
            seg_idx = int(segment_index) - 1
            if seg_idx < 0 or seg_idx >= len(self.segment_data):
                return f"❌ Invalid segment index. Valid range: 1-{len(self.segment_data)}", None
            
            progress(0.1, desc="Preparing segment...")
            segment = self.segment_data[seg_idx]['data']
            
            progress(0.3, desc=f"Performing {augmentation_method} augmentation...")
            
            if augmentation_method == 'imf_mixing':
                self.augmented_signals = augment_by_imf_mixing(
                    segment, n_augmented=int(n_augmented), 
                    imf_perturbation=float(perturbation)
                )
            elif augmentation_method == 'imf_scaling':
                self.augmented_signals = augment_by_imf_scaling(
                    segment, n_augmented=int(n_augmented),
                    scale_range=(1-float(perturbation), 1+float(perturbation))
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
            elif augmentation_method == 'comprehensive':
                results = comprehensive_augmentation(
                    segment, n_per_method=max(1, int(n_augmented) // 4)
                )
                self.augmented_signals = [segment]
                for method_signals in results.values():
                    if method_signals != [segment]:
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
**Original segment**: {seg_idx + 1}
**Generated signals**: {len(self.augmented_signals)} (including original)
**Perturbation factor**: {perturbation}
"""
            
            return info.strip(), fig
        except Exception as e:
            import traceback
            return f"❌ Error in augmentation: {str(e)}\n{traceback.format_exc()}", None
    
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
            
            # Export HHT results
            if export_hht and self.hht_results:
                hht_dir = os.path.join(output_dir, 'hht')
                os.makedirs(hht_dir, exist_ok=True)
                
                # Save spectrum as NPZ
                spectrum_path = os.path.join(hht_dir, f'{prefix}_hilbert_spectrum.npz')
                save_hilbert_spectrum(
                    self.hht_results['spectrum'],
                    self.hht_results['time'],
                    self.hht_results['frequency'],
                    spectrum_path,
                    format='npz'
                )
                results.append(f"✅ Saved Hilbert spectrum to: {spectrum_path}")
                
                # Save spectrum as image
                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.pcolormesh(
                    self.hht_results['time'],
                    self.hht_results['frequency'],
                    self.hht_results['spectrum'],
                    shading='auto', cmap='hot'
                )
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title('Hilbert Spectrum')
                plt.colorbar(im, ax=ax, label='Amplitude')
                plt.tight_layout()
                
                img_path = os.path.join(hht_dir, f'{prefix}_hilbert_spectrum.png')
                fig.savefig(img_path, dpi=150)
                plt.close(fig)
                results.append(f"✅ Saved spectrum image to: {img_path}")
            
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
        - 📁 Batch file processing for dataset building
        - 📊 Hilbert-Huang Transform (HHT) analysis
        - 🔄 EMD-based data augmentation
        - 🏷️ Data annotation support
        - 💾 Comprehensive export options
        """)
        
        with gr.Tabs():
            # Tab 1: Load Data
            with gr.Tab("📁 Load Data / 加载数据"):
                with gr.Tabs():
                    with gr.Tab("Single File"):
                        gr.Markdown("### Upload your sEMG CSV file")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                file_input = gr.File(label="Select CSV File", file_types=['.csv'])
                                fs_input = gr.Number(value=1000, label="Sampling Frequency (Hz)", precision=0)
                                column_input = gr.Number(value=1, label="Signal Column Index", precision=0,
                                                        info="Column containing signal values (0-indexed)")
                                header_input = gr.Checkbox(value=True, label="File has header row")
                                load_btn = gr.Button("Load File", variant="primary")
                            
                            with gr.Column(scale=2):
                                load_info = gr.Textbox(label="Load Status", lines=6)
                                load_plot = gr.Plot(label="Signal Preview")
                        
                        load_btn.click(
                            fn=processor.load_file,
                            inputs=[file_input, fs_input, column_input, header_input],
                            outputs=[load_info, load_plot]
                        )
                    
                    with gr.Tab("Batch Processing"):
                        gr.Markdown("### Upload multiple CSV files for batch processing")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_file_input = gr.File(
                                    label="Select Multiple CSV Files", 
                                    file_types=['.csv'],
                                    file_count="multiple"
                                )
                                batch_fs_input = gr.Number(value=1000, label="Sampling Frequency (Hz)", precision=0)
                                batch_column_input = gr.Number(value=1, label="Signal Column Index", precision=0)
                                batch_header_input = gr.Checkbox(value=True, label="Files have header row")
                                batch_load_btn = gr.Button("Load All Files", variant="primary")
                            
                            with gr.Column(scale=2):
                                batch_load_info = gr.Textbox(label="Batch Load Status", lines=10)
                                batch_load_plot = gr.Plot(label="Signal Previews")
                        
                        batch_load_btn.click(
                            fn=processor.load_batch_files,
                            inputs=[batch_file_input, batch_fs_input, batch_column_input, batch_header_input],
                            outputs=[batch_load_info, batch_load_plot]
                        )
            
            # Tab 2: Apply Filters
            with gr.Tab("🔧 Apply Filters / 应用滤波器"):
                gr.Markdown("### Configure and apply preprocessing filters")
                
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
            
            # Tab 3: Detect Activity
            with gr.Tab("🎯 Detect Activity / 检测肌肉活动"):
                gr.Markdown("""
                ### Detect muscle activity segments
                
                **Sensitivity**: Lower values = more sensitive (detects more segments)
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        method_input = gr.Radio(
                            ["multi_feature", "combined", "amplitude", "ruptures"],
                            value="multi_feature",
                            label="Detection Method"
                        )
                        min_duration_input = gr.Slider(0.05, 1.0, value=0.1, step=0.05,
                                                      label="Minimum segment duration (s)")
                        sensitivity_input = gr.Slider(0.1, 3.0, value=1.0, step=0.1,
                                                     label="Detection Sensitivity")
                        clustering_input = gr.Checkbox(value=True, label="Use clustering")
                        adaptive_pen_input = gr.Checkbox(value=True, label="Use adaptive penalty")
                        
                        detect_btn = gr.Button("Detect Activity", variant="primary")
                    
                    with gr.Column(scale=2):
                        detect_info = gr.Textbox(label="Detection Results", lines=12)
                        detect_plot = gr.Plot(label="Detected Segments")
                
                detect_btn.click(
                    fn=processor.detect_activity,
                    inputs=[method_input, min_duration_input, sensitivity_input, clustering_input, adaptive_pen_input],
                    outputs=[detect_info, detect_plot]
                )
            
            # Tab 4: HHT Analysis
            with gr.Tab("📊 HHT Analysis / 希尔伯特-黄变换"):
                gr.Markdown("""
                ### Hilbert-Huang Transform (HHT) Analysis
                
                Perform time-frequency analysis on detected segments using EMD and Hilbert Transform.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        hht_segment_input = gr.Number(value=1, label="Segment Index", precision=0,
                                                     info="Which segment to analyze (1-indexed)")
                        hht_freq_bins = gr.Number(value=256, label="Frequency Bins", precision=0)
                        hht_max_freq = gr.Number(value=500, label="Max Frequency (Hz, 0=auto)", precision=0)
                        hht_normalize = gr.Checkbox(value=False, label="Normalize segment length")
                        hht_norm_length = gr.Number(value=1000, label="Target length (samples)", precision=0,
                                                    visible=True)
                        
                        hht_btn = gr.Button("Perform HHT Analysis", variant="primary")
                    
                    with gr.Column(scale=2):
                        hht_info = gr.Textbox(label="HHT Results", lines=10)
                        hht_plot = gr.Plot(label="Hilbert Spectrum")
                        hht_download = gr.File(label="Download Results", visible=False)
                
                hht_btn.click(
                    fn=processor.perform_hht_analysis,
                    inputs=[hht_segment_input, hht_freq_bins, hht_max_freq, hht_norm_length, hht_normalize],
                    outputs=[hht_info, hht_plot, hht_download]
                )
            
            # Tab 5: Data Augmentation
            with gr.Tab("🔄 Augmentation / 数据增强"):
                gr.Markdown("""
                ### EMD-Based Data Augmentation
                
                Generate synthetic sEMG signals by decomposing and recombining IMF components.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        aug_segment_input = gr.Number(value=1, label="Segment Index", precision=0)
                        aug_method_input = gr.Radio(
                            ["imf_mixing", "imf_scaling", "noise_injection", "time_warping", "comprehensive"],
                            value="imf_mixing",
                            label="Augmentation Method"
                        )
                        aug_n_input = gr.Slider(1, 20, value=5, step=1, label="Number of augmented signals")
                        aug_perturbation = gr.Slider(0.01, 0.5, value=0.1, step=0.01,
                                                    label="Perturbation factor")
                        
                        aug_btn = gr.Button("Generate Augmented Data", variant="primary")
                    
                    with gr.Column(scale=2):
                        aug_info = gr.Textbox(label="Augmentation Results", lines=8)
                        aug_plot = gr.Plot(label="Augmented Signals")
                
                aug_btn.click(
                    fn=processor.perform_augmentation,
                    inputs=[aug_method_input, aug_n_input, aug_perturbation, aug_segment_input],
                    outputs=[aug_info, aug_plot]
                )
            
            # Tab 6: Export Results
            with gr.Tab("💾 Export Results / 导出结果"):
                gr.Markdown("### Export processed data, segments, and analysis results")
                
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
                    
                    with gr.Column(scale=2):
                        export_info = gr.Textbox(label="Export Status", lines=15)
                
                export_btn.click(
                    fn=processor.export_data,
                    inputs=[output_dir_input, export_full_input, export_segments_input,
                           subject_input, fatigue_input, quality_input, action_input, notes_input,
                           export_hht_input, export_augmented_input, custom_prefix_input],
                    outputs=[export_info]
                )
            
            # Tab 7: Help
            with gr.Tab("ℹ️ Help / 帮助"):
                gr.Markdown("""
                ## Quick Start Guide / 快速入门
                
                ### 1. Load Data (加载数据)
                - **Single file**: Upload a single CSV for analysis
                - **Batch processing**: Upload multiple files for dataset building
                
                ### 2. Apply Filters (应用滤波)
                - **Bandpass**: 20-450 Hz recommended for sEMG
                - **Notch**: 50 Hz (Europe/Asia) or 60 Hz (Americas)
                
                ### 3. Detect Activity (检测活动)
                - **multi_feature** method recommended
                - Adjust sensitivity for more/fewer detections
                
                ### 4. HHT Analysis (希尔伯特-黄变换)
                - Performs EMD decomposition
                - Generates Hilbert spectrum (time-frequency representation)
                - Option to normalize segment lengths for uniform spectra
                
                ### 5. Data Augmentation (数据增强)
                - **imf_mixing**: Perturb IMF components
                - **imf_scaling**: Scale IMF amplitudes
                - **noise_injection**: Add controlled noise
                - **time_warping**: Apply time distortion
                - **comprehensive**: Apply all methods
                
                ### 6. Export (导出)
                - Add annotations: subject ID, fatigue level, quality rating
                - Export segments, HHT spectra, and augmented signals
                - Custom filename prefixes for organization
                
                ## Parameter Guide / 参数指南
                
                - **Sensitivity**: 0.5 (more segments) to 2.0 (fewer segments)
                - **Perturbation**: 0.05 (subtle) to 0.3 (significant variation)
                - **Quality Rating**: 1 (poor) to 5 (excellent)
                """)
        
        gr.Markdown("""
        ---
        **sEMG Preprocessing Toolkit v2.0** | [GitHub](https://github.com/PRIMOCOSMOS/sEMG-pre-processing) | MIT License
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
