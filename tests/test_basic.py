"""
Basic tests for sEMG preprocessing toolkit.

Run with: python -m pytest tests/test_basic.py
Or: python tests/test_basic.py
"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from semg_preprocessing import (
    apply_highpass_filter,
    apply_lowpass_filter,
    apply_bandpass_filter,
    apply_notch_filter,
    remove_powerline_dft,
    detect_muscle_activity,
    segment_signal,
    export_segments_to_csv,
)


def test_filters():
    """Test that filters run without errors."""
    # Create synthetic signal
    fs = 1000.0
    duration = 1.0
    t = np.arange(0, duration, 1/fs)
    signal = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
    
    # Test high-pass filter
    filtered = apply_highpass_filter(signal, fs, cutoff=20, order=4)
    assert len(filtered) == len(signal)
    assert not np.isnan(filtered).any()
    
    # Test low-pass filter
    filtered = apply_lowpass_filter(signal, fs, cutoff=450, order=4)
    assert len(filtered) == len(signal)
    assert not np.isnan(filtered).any()
    
    # Test bandpass filter
    filtered = apply_bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4)
    assert len(filtered) == len(signal)
    assert not np.isnan(filtered).any()
    
    # Test notch filter
    filtered = apply_notch_filter(signal, fs, freq=50, harmonics=[1, 2])
    assert len(filtered) == len(signal)
    assert not np.isnan(filtered).any()
    
    # Test DFT powerline removal
    filtered = remove_powerline_dft(signal, fs, freq=50, harmonics=[1, 2])
    assert len(filtered) == len(signal)
    assert not np.isnan(filtered).any()
    
    print("✓ All filter tests passed")


def test_detection():
    """Test muscle activity detection."""
    # Create signal with clear activity bursts
    fs = 1000.0
    duration = 2.0
    t = np.arange(0, duration, 1/fs)
    
    # Background noise
    signal = 0.1 * np.random.randn(len(t))
    
    # Add two activity bursts with higher amplitude
    burst1_start = int(0.5 * fs)
    burst1_end = int(1.0 * fs)
    signal[burst1_start:burst1_end] += 3.0 * np.random.randn(burst1_end - burst1_start)
    
    burst2_start = int(1.3 * fs)
    burst2_end = int(1.7 * fs)
    signal[burst2_start:burst2_end] += 2.5 * np.random.randn(burst2_end - burst2_start)
    
    # Test amplitude method with manual threshold
    segments = detect_muscle_activity(
        signal, fs, 
        method='amplitude', 
        min_duration=0.1,
        amplitude_threshold=0.5  # Manual threshold
    )
    
    assert len(segments) > 0, "Should detect at least one segment"
    assert all(isinstance(s, tuple) and len(s) == 2 for s in segments), "Segments should be tuples of (start, end)"
    assert all(s[1] > s[0] for s in segments), "End should be after start"
    
    print(f"✓ Amplitude detection test passed ({len(segments)} segments detected)")
    
    # Test multi-feature method
    segments_multi = detect_muscle_activity(
        signal, fs,
        method='multi_feature',
        min_duration=0.1,
        use_clustering=False,
        adaptive_pen=True
    )
    
    assert len(segments_multi) > 0, "Multi-feature should detect at least one segment"
    print(f"✓ Multi-feature detection test passed ({len(segments_multi)} segments detected)")


def test_segmentation():
    """Test signal segmentation."""
    fs = 1000.0
    signal = np.random.randn(5000)
    segments = [(500, 1000), (2000, 3000), (4000, 4500)]
    
    segmented = segment_signal(signal, segments, fs, include_metadata=True)
    
    assert len(segmented) == len(segments)
    
    for i, seg_dict in enumerate(segmented):
        assert 'data' in seg_dict
        assert 'start_index' in seg_dict
        assert 'end_index' in seg_dict
        assert 'duration' in seg_dict
        assert 'peak_amplitude' in seg_dict
        assert 'rms' in seg_dict
        
        # Check data length
        expected_length = segments[i][1] - segments[i][0]
        assert len(seg_dict['data']) == expected_length
        
    print(f"✓ Segmentation test passed ({len(segmented)} segments)")


def test_filter_parameters():
    """Test filter parameter validation."""
    fs = 1000.0
    signal = np.random.randn(1000)
    
    # Test invalid filter type
    try:
        apply_highpass_filter(signal, fs, filter_type='invalid')
        assert False, "Should raise error for invalid filter type"
    except ValueError:
        pass
    
    # Test cutoff >= Nyquist
    try:
        apply_lowpass_filter(signal, fs, cutoff=600)  # Above Nyquist (500Hz)
        assert False, "Should raise error for cutoff >= Nyquist"
    except ValueError:
        pass
    
    print("✓ Parameter validation tests passed")


def test_export_segments():
    """Test segment export functionality."""
    fs = 1000.0
    signal = np.random.randn(5000)
    segments = [(500, 1000), (2000, 3000)]
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Export segments
        files = export_segments_to_csv(
            signal, segments, fs, temp_dir, prefix='test'
        )
        
        assert len(files) == len(segments), "Should create one file per segment"
        
        # Verify files exist
        for f in files:
            assert os.path.exists(f), f"File should exist: {f}"
        
        print(f"✓ Export segments test passed ({len(files)} files created)")
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running sEMG Preprocessing Toolkit Tests")
    print("=" * 60)
    print()
    
    test_filters()
    test_detection()
    test_segmentation()
    test_filter_parameters()
    test_export_segments()
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
