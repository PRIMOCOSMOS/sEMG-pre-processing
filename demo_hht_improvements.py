"""
Demo script showing the HHT improvements in action.

This demonstrates:
1. Creating a test sEMG signal
2. Computing Hilbert spectrum with new method (20-450Hz, average pooling)
3. Visualizing the results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semg_preprocessing.hht import (
    compute_hilbert_spectrum_production,
    SEMG_LOW_FREQ_CUTOFF,
    SEMG_HIGH_FREQ_CUTOFF,
)

print("="*70)
print("HHT Algorithm Improvements Demonstration")
print("="*70)

# Create a realistic sEMG-like signal
print("\n1. Creating test sEMG signal...")
fs = 1000.0  # 1000 Hz sampling rate
duration = 2.0  # 2 seconds
t = np.linspace(0, duration, int(fs * duration))

# Create signal with components in valid sEMG range (20-450Hz)
# Simulate muscle fiber firing patterns
signal = np.zeros_like(t)
for freq in [50, 80, 120, 180, 250, 350]:  # Different motor unit frequencies
    signal += np.sin(2 * np.pi * freq * t) * np.exp(-((t - 1.0)**2) / 0.5)

# Add realistic noise
signal += 0.2 * np.random.randn(len(signal))

print(f"   Signal duration: {duration}s")
print(f"   Sampling rate: {fs}Hz")
print(f"   Signal length: {len(signal)} samples")

# Compute Hilbert spectrum
print("\n2. Computing Hilbert spectrum with improved method...")
spectrum, time_axis, freq_axis, validation_info = compute_hilbert_spectrum_production(
    signal, fs, 
    n_freq_bins=256, 
    target_length=256,
    validate_energy=True
)

print(f"   Spectrum shape: {spectrum.shape}")
print(f"   Time axis: {time_axis[0]:.3f} to {time_axis[-1]:.3f} (normalized)")
print(f"   Frequency axis: {freq_axis[0]:.1f}Hz to {freq_axis[-1]:.1f}Hz")
print(f"   Energy error: {validation_info['energy_error']:.4f} ({validation_info['energy_error']*100:.2f}%)")
print(f"   Energy conservation: {'✓ PASS' if validation_info['energy_conservation_ok'] else '✗ FAIL'}")

# Create visualization
print("\n3. Creating visualization...")
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Original signal
axes[0].plot(t, signal, 'b-', linewidth=0.5)
axes[0].set_xlabel('Time (s)', fontsize=12)
axes[0].set_ylabel('Amplitude', fontsize=12)
axes[0].set_title('Test sEMG Signal', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Hilbert Spectrum
im = axes[1].pcolormesh(
    time_axis, 
    freq_axis, 
    spectrum,
    shading='auto',
    cmap='jet'
)
axes[1].set_xlabel('Normalized Time', fontsize=12)
axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
axes[1].set_title(
    f'Hilbert Spectrum (20-450Hz, No Interpolation)\nEnergy Error: {validation_info["energy_error"]*100:.2f}%',
    fontsize=14, fontweight='bold'
)
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add colorbar
cbar = fig.colorbar(im, ax=axes[1])
cbar.set_label('Normalized Amplitude', fontsize=11)

# Add text annotations
axes[1].text(
    0.02, 0.98, 
    'Improvements:\n' +
    '✓ Average pooling (no interpolation)\n' +
    '✓ 20-450Hz frequency range\n' +
    f'✓ Energy conserved ({validation_info["energy_error"]*100:.2f}% error)',
    transform=axes[1].transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.savefig('demo_hht_improvements.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Visualization saved to: demo_hht_improvements.png")

print("\n" + "="*70)
print("Demo Complete!")
print("="*70)
print("\nKey Improvements Demonstrated:")
print("  1. Frequency axis maps to 20-450Hz (valid sEMG range)")
print("  2. Average pooling used instead of interpolation")
print("  3. Energy conservation validated")
print("  4. Original signal processed at full resolution, then pooled")
print("="*70)
