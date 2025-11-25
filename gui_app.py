"""
Gradio GUI Application for sEMG Signal Preprocessing

This provides a user-friendly web interface for:
- Loading CSV files with sEMG data
- Applying preprocessing filters
- Detecting muscle activity
- Visualizing results
- Exporting processed data and segments
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
    apply_bandpass_filter,
    apply_notch_filter,
    detect_muscle_activity,
    segment_signal,
    save_processed_data,
    export_segments_to_csv,
)


class EMGProcessorGUI:
    """GUI application for sEMG signal preprocessing."""
    
    def __init__(self):
        self.signal = None
        self.filtered_signal = None
        self.fs = 1000.0
        self.segments = None
        self.df = None
        
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
            
            # Create preview plot
            fig, ax = plt.subplots(figsize=(12, 4))
            time = np.arange(len(self.signal)) / self.fs
            ax.plot(time, self.signal, 'b-', linewidth=0.5, alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Loaded sEMG Signal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            info = f"""
✅ File loaded successfully!
- Samples: {len(self.signal)}
- Duration: {len(self.signal)/self.fs:.2f} seconds
- Sampling frequency: {self.fs} Hz
- Signal range: [{self.signal.min():.3f}, {self.signal.max():.3f}]
            """
            
            return info.strip(), fig
        except Exception as e:
            return f"❌ Error loading file: {str(e)}", None
    
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
            return f"❌ Error applying filters: {str(e)}", None
    
    def detect_activity(self, method, min_duration, sensitivity, use_clustering, adaptive_pen, progress=gr.Progress()):
        """Detect muscle activity segments."""
        try:
            if self.filtered_signal is None:
                return "❌ Please apply filters first", None
            
            # Debug info
            print(f"[DEBUG] Starting detection with method={method}, min_duration={min_duration}")
            print(f"[DEBUG] Signal length: {len(self.filtered_signal)}, fs={self.fs}")
            print(f"[DEBUG] sensitivity={sensitivity}, use_clustering={use_clustering}, adaptive_pen={adaptive_pen}")
            
            # Update progress
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
            
            print(f"[DEBUG] Detection complete. Found {len(self.segments)} segments")
            
            # Get detailed segment information
            progress(0.6, desc="Extracting segment information...")
            segment_info = segment_signal(
                self.filtered_signal,
                self.segments,
                fs=self.fs,
                include_metadata=True
            )
            
            print(f"[DEBUG] Segment info created")
            
            # Create visualization
            progress(0.8, desc="Creating visualization...")
            fig, ax = plt.subplots(figsize=(12, 6))
            time = np.arange(len(self.filtered_signal)) / self.fs
            
            ax.plot(time, self.filtered_signal, 'k-', linewidth=0.5, alpha=0.5, label='Filtered Signal')
            
            # Highlight detected segments
            for i, (start, end) in enumerate(self.segments):
                ax.axvspan(start/self.fs, end/self.fs, alpha=0.3, color='red',
                          label='Detected Activity' if i == 0 else '')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Muscle Activity Detection ({len(self.segments)} segments detected)')
            ax.grid(True, alpha=0.3)
            if self.segments:
                ax.legend(loc='upper right')
            
            plt.tight_layout()
            
            print(f"[DEBUG] Plot created")
            
            # Create info text
            progress(0.9, desc="Formatting results...")
            info = f"✅ Detected {len(self.segments)} muscle activity segments:\n\n"
            for i, seg in enumerate(segment_info[:10], 1):  # Show first 10
                info += f"Segment {i}: {seg['start_time']:.3f}s - {seg['end_time']:.3f}s "
                info += f"(duration: {seg['duration']:.3f}s, peak: {seg['peak_amplitude']:.3f}, RMS: {seg['rms']:.3f})\n"
            
            if len(segment_info) > 10:
                info += f"\n... and {len(segment_info) - 10} more segments"
            
            progress(1.0, desc="Complete!")
            print(f"[DEBUG] Returning results")
            return info.strip(), fig
        except Exception as e:
            import traceback
            error_msg = f"❌ Error detecting activity: {str(e)}\n\n"
            error_msg += "Traceback:\n" + traceback.format_exc()
            print(f"[ERROR] {error_msg}")
            return error_msg, None
    
    def export_data(self, output_dir, export_full, export_segments):
        """Export processed data and segments."""
        try:
            if self.filtered_signal is None:
                return "Please process signal first"
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            results = []
            
            # Export full processed signal
            if export_full:
                full_path = os.path.join(output_dir, 'processed_signal.csv')
                save_processed_data(full_path, self.filtered_signal, fs=self.fs)
                results.append(f"✅ Saved processed signal to: {full_path}")
            
            # Export segments
            if export_segments and self.segments:
                segment_dir = os.path.join(output_dir, 'segments')
                segment_info = segment_signal(
                    self.filtered_signal,
                    self.segments,
                    fs=self.fs,
                    include_metadata=True
                )
                saved_files = export_segments_to_csv(
                    self.filtered_signal,
                    segment_info,
                    fs=self.fs,
                    output_dir=segment_dir,
                    prefix='segment'
                )
                results.append(f"✅ Saved {len(saved_files)} segment files to: {segment_dir}")
            
            return "\n".join(results) if results else "No export options selected"
        except Exception as e:
            return f"❌ Error exporting data: {str(e)}"


def create_gui():
    """Create and configure the Gradio interface."""
    
    processor = EMGProcessorGUI()
    
    with gr.Blocks(title="sEMG Signal Preprocessing") as app:
        gr.Markdown("""
        # 🔬 sEMG Signal Preprocessing Toolkit
        ## 表面肌电信号预处理工具
        
        A comprehensive tool for surface electromyography signal preprocessing with filtering, 
        muscle activity detection, and segment extraction.
        """)
        
        with gr.Tabs():
            # Tab 1: Load Data
            with gr.Tab("📁 Load Data / 加载数据"):
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
                
                **Note**: Detection time varies by method:
                - `multi_feature`: ~1-2 seconds (recommended)
                - `amplitude`: <1 second (fast)
                - `combined`: ~8-10 seconds (slower but accurate)
                - `ruptures`: ~8-10 seconds (change-point only)
                
                **Sensitivity**: Lower values = more sensitive (detects more segments), Higher values = stricter (fewer segments)
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        method_input = gr.Radio(
                            ["multi_feature", "combined", "amplitude", "ruptures"],
                            value="multi_feature",
                            label="Detection Method",
                            info="multi_feature: Most robust (recommended)"
                        )
                        min_duration_input = gr.Slider(0.05, 1.0, value=0.1, step=0.05,
                                                      label="Minimum segment duration (s)")
                        sensitivity_input = gr.Slider(0.1, 3.0, value=1.0, step=0.1,
                                                     label="Detection Sensitivity (lower=more sensitive)")
                        clustering_input = gr.Checkbox(value=True, label="Use clustering for classification")
                        adaptive_pen_input = gr.Checkbox(value=True, label="Use adaptive penalty parameter")
                        
                        detect_btn = gr.Button("Detect Activity", variant="primary")
                    
                    with gr.Column(scale=2):
                        detect_info = gr.Textbox(label="Detection Results", lines=12)
                        detect_plot = gr.Plot(label="Detected Segments")
                
                detect_btn.click(
                    fn=processor.detect_activity,
                    inputs=[method_input, min_duration_input, sensitivity_input, clustering_input, adaptive_pen_input],
                    outputs=[detect_info, detect_plot]
                )
            
            # Tab 4: Export Results
            with gr.Tab("💾 Export Results / 导出结果"):
                gr.Markdown("### Export processed data and segments")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        output_dir_input = gr.Textbox(value="./output", label="Output Directory")
                        export_full_input = gr.Checkbox(value=True, label="Export full processed signal")
                        export_segments_input = gr.Checkbox(value=True, label="Export individual segments")
                        
                        export_btn = gr.Button("Export Data", variant="primary")
                    
                    with gr.Column(scale=2):
                        export_info = gr.Textbox(label="Export Status", lines=8)
                
                export_btn.click(
                    fn=processor.export_data,
                    inputs=[output_dir_input, export_full_input, export_segments_input],
                    outputs=[export_info]
                )
            
            # Tab 5: Help
            with gr.Tab("ℹ️ Help / 帮助"):
                gr.Markdown("""
                ## Quick Start Guide / 快速入门
                
                ### 1. Load Data (加载数据)
                - Upload your CSV file containing sEMG signal
                - CSV should have signal values in the 2nd column (index 1) by default
                - Set the correct sampling frequency
                
                ### 2. Apply Filters (应用滤波)
                - **Bandpass filter**: Removes low-frequency artifacts and high-frequency noise
                  - Recommended: 20-450 Hz with order 4
                - **Notch filter**: Removes power line interference
                  - Use 50 Hz for Europe/Asia, 60 Hz for Americas
                  - Harmonics: Include multiples (e.g., 1,2,3 for 50, 100, 150 Hz)
                
                ### 3. Detect Activity (检测活动)
                - **multi_feature** method (recommended): Uses multiple signal features for robust detection
                  - RMS, envelope, variance, energy
                  - Adaptive penalty parameter
                  - Optional clustering for activity/rest classification
                - **combined**: Balance of speed and accuracy
                - **amplitude**: Fast, threshold-based
                - **ruptures**: Change-point detection only
                
                ### 4. Export Results (导出结果)
                - Export full processed signal as CSV
                - Export individual segments as separate CSV files
                - Each segment includes metadata (duration, peak, RMS)
                
                ## Parameters Guide / 参数指南
                
                - **Sampling Frequency**: Your data's sampling rate (Hz)
                - **Signal Column**: Which column contains the signal (0-indexed)
                - **High-pass cutoff**: 10-20 Hz (removes motion artifacts)
                - **Low-pass cutoff**: 450-500 Hz (removes high-frequency noise)
                - **Filter order**: 2-4 (higher = sharper but may distort)
                - **Minimum duration**: Minimum segment length (seconds)
                - **Use clustering**: Classify segments as activity vs rest
                - **Adaptive penalty**: Automatically adjust detection sensitivity
                
                ## Tips / 提示
                
                - Always filter your signal before detection
                - Use multi_feature method for best results
                - Adjust minimum duration to filter out noise
                - Check the visualization to verify detection quality
                - Export segments for further analysis
                """)
        
        gr.Markdown("""
        ---
        **sEMG Preprocessing Toolkit** | [GitHub](https://github.com/PRIMOCOSMOS/sEMG-pre-processing) | MIT License
        """)
    
    return app


if __name__ == "__main__":
    app = create_gui()
    
    # Enable queue for better handling of concurrent requests
    app.queue(default_concurrency_limit=4)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False  # Set to True for more verbose logging
    )
