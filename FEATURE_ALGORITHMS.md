# sEMG Feature Extraction Algorithms - Mathematical Formulas and Physical Meanings

This document provides comprehensive documentation of all feature extraction algorithms implemented in this toolkit, including mathematical formulas, physical interpretations, and typical value ranges for surface electromyography (sEMG) signals.

## Table of Contents
1. [Time Domain Features](#time-domain-features)
2. [Frequency Domain Features](#frequency-domain-features)
3. [Advanced Time-Frequency Features](#advanced-time-frequency-features)
4. [Fatigue Indicators](#fatigue-indicators)
5. [Hilbert-Huang Transform (HHT)](#hilbert-huang-transform-hht)

---

## Time Domain Features

Time domain features capture signal characteristics in the temporal domain without frequency transformation.

### 1. Waveform Length (WL)

**Formula:**
```
WL = Σ|x[n+1] - x[n]|
```

**Physical Meaning:**
- Measures the cumulative length of the waveform over the time segment
- Represents signal complexity and variability
- Increases with muscle contraction intensity and firing rate

**Typical Range (for 1-second sEMG segment):**
- Rest: 50-200 mV
- Light contraction: 200-500 mV
- Strong contraction: 500-2000 mV

**Code Implementation:**
```python
wl = np.sum(np.abs(np.diff(signal)))
```

---

### 2. Zero Crossings (ZC)

**Formula:**
```
ZC = Σ sgn(x[n] × x[n+1] < 0 ∧ |x[n] - x[n+1]| > threshold)
where sgn() is the sign function
```

**Physical Meaning:**
- Counts number of times signal crosses zero amplitude
- Approximation of dominant frequency content
- Related to motor unit firing rate
- Threshold prevents noise-induced false crossings

**Typical Range (for 1-second segment at 1000 Hz):**
- Rest: 10-50 crossings
- Light contraction: 50-150 crossings
- Strong contraction: 150-400 crossings

**Code Implementation:**
```python
threshold = 0.01 * np.std(signal)
zc = sum(1 for i in range(n-1) 
         if ((signal[i] > 0 and signal[i+1] < 0) or 
             (signal[i] < 0 and signal[i+1] > 0)) and
             abs(signal[i] - signal[i+1]) > threshold)
```

---

### 3. Slope Sign Changes (SSC)

**Formula:**
```
SSC = Σ sgn((x[n] - x[n-1]) × (x[n] - x[n+1]) > 0 ∧ 
             |x[n] - x[n+1]| > threshold)
```

**Physical Meaning:**
- Counts frequency of slope reversals in the signal
- Indicates signal complexity and oscillation frequency
- Related to motor unit action potential (MUAP) firing pattern

**Typical Range (for 1-second segment):**
- Rest: 20-100
- Light contraction: 100-300
- Strong contraction: 300-800

---

### 4. Root Mean Square (RMS)

**Formula:**
```
RMS = √(Σx²[n] / N)
```

**Physical Meaning:**
- Represents average signal power
- Directly related to muscle force production
- Most commonly used amplitude indicator in sEMG
- Proportional to number of active motor units

**Typical Range:**
- Rest: 0.01-0.05 mV
- Light contraction (20% MVC): 0.05-0.2 mV
- Strong contraction (80% MVC): 0.3-1.5 mV

**Code Implementation:**
```python
rms = np.sqrt(np.mean(signal ** 2))
```

---

### 5. Mean Absolute Value (MAV)

**Formula:**
```
MAV = (Σ|x[n]|) / N
```

**Physical Meaning:**
- Average rectified signal amplitude
- Similar to RMS but less sensitive to outliers
- Linear relationship with muscle contraction level at low forces

**Typical Range:**
- Rest: 0.008-0.04 mV
- Light contraction: 0.04-0.15 mV
- Strong contraction: 0.2-1.0 mV

---

### 6. Variance (VAR)

**Formula:**
```
VAR = Σ(x[n] - μ)² / N
where μ = mean(x)
```

**Physical Meaning:**
- Measures signal power around the mean
- Indicates variability of muscle activation
- Useful for detecting changes in motor unit recruitment

**Typical Range:**
- Rest: 0.0001-0.001 mV²
- Light contraction: 0.001-0.04 mV²
- Strong contraction: 0.04-0.5 mV²

---

## Frequency Domain Features

Frequency domain features are computed from the power spectrum using **Welch's method** for robust estimation.

### Power Spectrum Estimation (Welch Method)

**Method:**
```
P(f) = Welch(x, fs, nperseg=256, scaling='density')
```

**Advantages over FFT:**
- Reduces variance by averaging multiple periodograms
- Provides smoother spectral estimates
- Better for non-stationary signals like sEMG

**Valid Frequency Range:**
- All features exclude DC and low frequencies (< 20 Hz)
- Reason: DC offset, electrode impedance, motion artifacts, baseline drift
- sEMG physiological content: 20-450 Hz range

---

### 7. Median Frequency (MDF)

**Formula:**
```
MDF: ∫₀^MDF P(f)df = ∫_MDF^∞ P(f)df = TTP/2
where TTP = Total Power
```

**Physical Meaning:**
- Frequency that divides power spectrum into two equal halves
- Robust indicator of spectral characteristics
- **Decreases with muscle fatigue** (5-10% per minute during sustained contraction)
- Reflects muscle fiber conduction velocity

**Typical Range:**
- Fresh muscle: 80-120 Hz
- Fatigued muscle: 50-80 Hz
- Decline rate: 0.5-1.5 Hz per 10s during sustained 50% MVC

**Fatigue Response:**
- ↓ Conduction velocity → ↓ MDF
- ↑ Type II fiber recruitment → ↑ MDF initially
- Overall: MDF decreases with fatigue

**Code Implementation:**
```python
# Using Welch PSD
freqs, psd = welch(signal, fs=fs, nperseg=256)
valid_mask = freqs >= 20  # Exclude low frequencies
cumsum = np.cumsum(psd[valid_mask])
half_power = cumsum[-1] / 2
mdf_idx = np.searchsorted(cumsum, half_power)
mdf = freqs[valid_mask][mdf_idx]
```

---

### 8. Mean Frequency (MNF)

**Formula:**
```
MNF = ∫f·P(f)df / ∫P(f)df
```

**Physical Meaning:**
- Power-weighted average frequency
- Center of gravity of power spectrum
- **Decreases with muscle fatigue** similar to MDF
- More sensitive to spectral changes than MDF

**Typical Range:**
- Fresh muscle: 90-130 Hz
- Fatigued muscle: 60-90 Hz

**Comparison with MDF:**
- MNF more sensitive to spectral shape
- MDF more robust to noise
- Both decrease with fatigue but MNF typically shows larger changes

**Code Implementation:**
```python
freqs, psd = welch(signal, fs=fs, nperseg=256)
valid_mask = freqs >= 20
valid_freqs = freqs[valid_mask]
valid_psd = psd[valid_mask]
mnf = np.trapz(valid_freqs * valid_psd, valid_freqs) / np.trapz(valid_psd, valid_freqs)
```

---

### 9. Peak Frequency (PKF)

**Formula:**
```
PKF = argmax P(f)  for f ≥ 20 Hz
```

**Physical Meaning:**
- Frequency with maximum power density
- Indicates dominant motor unit firing rate
- Reflects primary frequency component of muscle activation

**Typical Range:**
- Slow-twitch dominant muscles: 60-100 Hz
- Fast-twitch dominant muscles: 100-150 Hz
- During fatigue: Shifts to lower frequencies

**Code Implementation:**
```python
freqs, psd = welch(signal, fs=fs, nperseg=256)
valid_mask = freqs >= 20
pkf = freqs[valid_mask][np.argmax(psd[valid_mask])]
```

---

### 10. Total Power (TTP)

**Formula:**
```
TTP = ∫P(f)df
```

**Physical Meaning:**
- Integrated power across all frequencies
- Related to overall signal energy
- Increases with muscle activation level

**Typical Range:**
- Rest: 0.0001-0.001 mV²
- Light contraction: 0.001-0.05 mV²
- Strong contraction: 0.05-0.5 mV²

---

## Advanced Time-Frequency Features

### 11. Instantaneous Mean Frequency (IMNF)

**Method: Choi-Williams Distribution (CWD) Approximation**

**Mathematical Background:**

The Choi-Williams Distribution is a time-frequency representation that provides better concentration and reduced cross-term interference compared to other methods.

**Theoretical CWD Formula:**
```
CWD(t,f) = ∫∫ A(θ,τ) · x(u+τ/2) · x*(u-τ/2) · e^(-j2πfτ) dτ du

where A(θ,τ) = (1/(4π|θ|σ))^0.5 · exp(-τ²/(4σθ))  (Choi-Williams kernel)
σ: scaling parameter (typically σ = 1)
```

**Practical Implementation:**

For computational efficiency, we use a pseudo-CWD approach:
1. Short-Time Fourier Transform (STFT) for time-frequency decomposition
2. Gaussian smoothing in both time and frequency (emulates CWD kernel)
3. Power-weighted mean frequency at each time point

**Implementation Formula:**
```
STFT: S(t,f) = ∫ x(τ)·w(τ-t)·e^(-j2πfτ) dτ
Power: P(t,f) = |S(t,f)|²
Smoothed: P_smooth(t,f) = GaussianFilter(P(t,f), σ_f=1.5, σ_t=1.0)

Instantaneous mean frequency:
IMF(t) = ∫ f·P_smooth(t,f) df / ∫ P_smooth(t,f) df  for f ≥ 20 Hz

IMNF = Time-averaged IMF(t) weighted by total power at each time
```

**Physical Meaning:**
- Represents time-varying center frequency of sEMG signal
- Captures dynamic changes in motor unit recruitment
- **Decreases with fatigue** similar to MNF
- More sensitive to transient changes than MNF

**Typical Range:**
- Fresh muscle: 85-125 Hz
- Fatigued muscle: 55-85 Hz
- Usually within ±10% of MNF

**Advantages over Hilbert Transform:**
- Better handling of multi-component signals
- Reduced cross-term interference
- More accurate instantaneous frequency estimation
- Robust to noise

**Code Implementation:**
```python
from scipy.signal import stft
from scipy.ndimage import gaussian_filter1d

# STFT
f_stft, t_stft, Zxx = stft(signal, fs=fs, nperseg=256, noverlap=128)
power_tf = np.abs(Zxx) ** 2

# Smooth (CWD-like behavior)
power_smooth = gaussian_filter1d(power_tf, sigma=1.5, axis=0)  # Frequency
power_smooth = gaussian_filter1d(power_smooth, sigma=1.0, axis=1)  # Time

# Compute IMNF
valid_mask = f_stft >= 20
for t_idx in range(power_smooth.shape[1]):
    power_at_t = power_smooth[valid_mask, t_idx]
    if power_at_t.sum() > 0:
        imf_at_t = np.sum(f_stft[valid_mask] * power_at_t) / power_at_t.sum()
        
# Time-weighted average
imnf = np.average(imf_values, weights=time_weights)
```

---

## Fatigue Indicators

### 12. Dimitrov Index (DI)

**Formula:**
```
DI = M₋₁ / M₅

where spectral moments:
M_k = (Σ f^k · P(f)) / (Σ P(f))  for f ≥ 20 Hz

Specifically:
M₋₁ = Σ(f^(-1) · P(f)) / Σ P(f)  (emphasizes low frequencies)
M₅ = Σ(f^5 · P(f)) / Σ P(f)      (emphasizes high frequencies)
```

**Physical Meaning:**

**Spectral Moments:**
- M₋₁: Inverse frequency weighting (1/f) → gives more weight to low frequencies
- M₅: Fifth power weighting (f⁵) → gives extreme weight to high frequencies

**DI Behavior:**
- When muscle fatigues, power spectrum shifts toward lower frequencies
- M₋₁ increases (more power at low f, 1/f weighting helps)
- M₅ decreases dramatically (less power at high f, f⁵ weighting hurts)
- Therefore: DI = M₋₁ / M₅ **increases with fatigue**

**Typical Absolute Values:**
- Non-fatigued state: DI ≈ 1-5 × 10⁻¹² to 10⁻¹⁴
- Fatigued state: DI ≈ 5-20 × 10⁻¹² to 10⁻¹²
- Range varies with muscle, electrode placement, and individual

**What Matters:**
- **Relative change (ratio)** between states: typically 2-10× increase
- **Trend** during sustained contraction: monotonic increase
- Not the absolute value (which depends on many factors)

**Example for Biceps Brachii:**
```
Start of contraction: DI = 2.3 × 10⁻¹²
After 30s at 50% MVC: DI = 6.8 × 10⁻¹²
After 60s at 50% MVC: DI = 14.2 × 10⁻¹²
Ratio (60s/0s): 6.2× increase → Clear fatigue indication
```

**Reference:**
Dimitrov GV, et al. (2006). "Muscle fatigue during dynamic contractions assessed by new spectral indices." Med Sci Sports Exerc 38(11):1971-1979.

**Code Implementation:**
```python
# Use Welch PSD
freqs, psd = welch(signal, fs=fs, nperseg=256)
valid_mask = freqs >= 20
valid_freqs = freqs[valid_mask]
valid_psd = psd[valid_mask]

# Normalize to probability distribution
norm_psd = valid_psd / valid_psd.sum()

# Compute moments
M_minus1 = np.sum((valid_freqs ** -1) * norm_psd)
M_5 = np.sum((valid_freqs ** 5) * norm_psd)

# DI
DI = M_minus1 / M_5 if M_5 > 0 else 0
```

**Interpretation Guidelines:**
- DI < 2× baseline: Non-fatigued
- DI = 2-5× baseline: Moderate fatigue
- DI > 5× baseline: Significant fatigue
- Monitor rate of change: faster increase = faster fatigue development

---

### 13. WIRE51 - Wavelet Index of Reliability Estimation

**Formula:**
```
WIRE51 = E(D5) / E(D1)

where:
E(Di) = Σ D_i²[n]  (energy of detail coefficients at scale i)
Di: Detail coefficients from Discrete Wavelet Transform (DWT)
```

**Wavelet: sym5 (5th-order Symlet)**
- Compact support, good time-frequency localization
- Suitable for biomedical signals
- Filter length: 10

**Maximum Decomposition Level:**
```
max_level = floor(log₂(N / (filter_length - 1)))
For sym5: max_level ≈ floor(log₂(N/9))
```

**Frequency Band Mapping:**

For a signal sampled at fs Hz, DWT detail level i corresponds to frequency band:
```
Band_i ≈ [fs/(2^(i+1)), fs/(2^i)] Hz
```

**Example for fs = 1000 Hz:**
```
D1: [250, 500] Hz    - Highest frequency details
D2: [125, 250] Hz    - High-mid frequency details
D3: [62.5, 125] Hz   - Mid frequency details
D4: [31.2, 62.5] Hz  - Low-mid frequency details
D5: [15.6, 31.2] Hz  - Low frequency details (partially below 20 Hz cutoff)
```

**Physical Meaning:**

**Energy Distribution:**
- D1 energy: High-frequency muscle activity
  - Fast motor unit firing
  - Sharp action potentials
  - Dominant in fresh muscle

- D5 energy: Low-frequency components
  - Slow motor unit firing
  - Broader action potentials
  - Increases with fatigue

**WIRE51 = E(D5) / E(D1):**
- Fresh muscle: High E(D1), low E(D5) → Low WIRE51
- Fatigued muscle: Lower E(D1), higher E(D5) → High WIRE51
- **WIRE51 increases with fatigue**

**Typical Values:**
- Non-fatigued: WIRE51 = 0.1-0.5
- Moderately fatigued: WIRE51 = 0.5-2.0
- Significantly fatigued: WIRE51 = 2.0-10.0
- Varies with muscle type and individual

**Advantages:**
- Time-localized frequency analysis
- Robust to non-stationarity
- Clear physical interpretation of frequency shifts

**Limitations for Short Signals:**
- Requires minimum signal length for 5-level decomposition
- Typical minimum: N ≥ 512 samples at 1000 Hz
- Fallback strategy: Use available levels (max_level < 5)

**Code Implementation:**
```python
import pywt

wavelet = 'sym5'
max_level = pywt.dwt_max_level(len(signal), wavelet)

if max_level >= 5:
    # Full 5-level decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=5)
    # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]
    d5 = coeffs[1]  # Detail level 5
    d1 = coeffs[5]  # Detail level 1
    
    E_d5 = np.sum(d5 ** 2)
    E_d1 = np.sum(d1 ** 2)
    
    WIRE51 = E_d5 / E_d1 if E_d1 > 0 else 0
else:
    # Adaptive fallback for short signals
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    high_scale = coeffs[1]   # Lowest freq detail available
    low_scale = coeffs[-1]   # Highest freq detail available
    WIRE51 = np.sum(high_scale**2) / np.sum(low_scale**2)
```

**Interpretation:**
- WIRE51 < 1: High-frequency dominant (fresh muscle)
- WIRE51 ≈ 1: Balanced frequency distribution
- WIRE51 > 1: Low-frequency dominant (fatigued muscle)
- Monitor rate of change for fatigue progression

---

## Hilbert-Huang Transform (HHT)

The HHT provides adaptive time-frequency analysis through Empirical Mode Decomposition (EMD) followed by Hilbert spectral analysis.

### CEEMDAN Decomposition

**Complete Ensemble EMD with Adaptive Noise (CEEMDAN)**

**Algorithm:**
```
1. For each ensemble i (i = 1 to N_ensembles):
   - Add white noise: x_i(t) = x(t) + ε_i(t)
   - Decompose x_i(t) into IMFs using EMD
   
2. Average over ensembles:
   IMF_k = (1/N) Σ IMF_k,i
   
3. Adaptive noise decreases with decomposition level
```

**Intrinsic Mode Functions (IMFs):**

Properties:
1. Number of extrema and zero crossings differ by at most 1
2. Mean of upper and lower envelopes is approximately zero
3. Each IMF represents a characteristic oscillation mode

**Physical Interpretation for sEMG:**
- IMF 1-3: High-frequency components (fast motor units, noise)
- IMF 4-6: Mid-frequency components (main muscle activity)
- IMF 7-8: Low-frequency components (slow motor units, force modulation)
- Residue: Overall trend, baseline

---

### Production-Ready HHT Spectrum

**Function: `compute_hilbert_spectrum_production()`**

**Key Improvements:**

#### 1. Fixed IMF Count
```
target_imfs = 8 (default)
```
- Ensures consistent decomposition across all signals
- Zero-padding if n_imfs < 8
- Truncation if n_imfs > 8
- Enables standardized feature extraction and comparison

#### 2. Signal Normalization
```
x_norm = (x - mean(x)) / std(x)
```
**Benefits:**
- Removes DC offset
- Standardizes amplitude scale
- Improves EMD convergence
- Enables fair comparison across recordings

#### 3. Unified Time-Frequency Axes
```
Time: [0, 1] normalized, N = 256 samples
Frequency: [0, fs/2] Hz, M = 256 bins
```
- All spectra have identical dimensions (256 × 256)
- Suitable for machine learning (CNN input)
- Enables batch processing

#### 4. Energy Conservation Validation
```
E_original = ||x||²
E_reconstructed = ||Σ IMFs||²
error = |E_original - E_reconstructed| / E_original
```
**Acceptance Criterion:**
- error < 5%: Good decomposition
- error 5-10%: Acceptable
- error > 10%: Poor decomposition, investigate

**Physical Meaning:**
- Validates completeness of IMF representation
- Detects decomposition failures
- Ensures no information loss

#### 5. Instantaneous Frequency Calculation
```
For each IMF:
1. Compute analytic signal: z(t) = x(t) + j·H[x(t)]
   where H[·] is Hilbert transform
   
2. Extract phase: φ(t) = arctan(Im(z) / Re(z))

3. Unwrap phase: φ_unwrapped = unwrap(φ)

4. Instantaneous frequency:
   f_inst(t) = (1/2π) · dφ/dt = (1/2π) · Δφ/Δt · fs
```

**Accuracy Considerations:**
- Minimal frequency spreading (σ = 0.5 bins)
- Conservative Gaussian weighting
- Clipping to valid range [0, fs/2]

#### 6. Amplitude Thresholding
```
threshold = Percentile(spectrum[spectrum > 0], P)
spectrum[spectrum < threshold] = 0
```
**Purpose:**
- Remove low-amplitude noise
- Enhance significant components
- Improve visual clarity

**Typical Value:** P = 10% (removes bottom 10% amplitudes)

#### 7. Amplitude Normalization
```
spectrum_norm = spectrum / max(spectrum)
```
**Physical Meaning:**
- Represents relative muscle activity
- 0: No activity at that time-frequency point
- 1: Maximum activity
- Independent of absolute recording scale

---

### Hilbert Spectrum Interpretation

**Time-Frequency Representation:**
```
H(t, f): Amplitude at time t and frequency f
```

**Marginal Spectrum:**
```
h(f) = ∫ H(t,f) dt
```
- Frequency-amplitude representation
- Similar to Fourier spectrum but adaptive
- Better time resolution

**Instantaneous Energy:**
```
E(t) = ∫ H²(t,f) df
```
- Energy at each time instant
- Tracks muscle activation dynamics

---

## Summary: Feature Relationships

### Amplitude Features
- RMS ≈ MAV × 1.11 (for sinusoidal signals)
- VAR ≈ RMS² (for zero-mean signals)
- All increase with muscle force

### Frequency Features
- MNF ≥ MDF (typically MNF 5-10% higher)
- IMNF ≈ MNF ± 10%
- All decrease with fatigue (0.5-1.5 Hz per 10s at 50% MVC)

### Fatigue Indicators
- MDF ↓, MNF ↓, IMNF ↓: Decrease with fatigue
- WIRE51 ↑, DI ↑: Increase with fatigue
- Typical fatigue detection: 2-10× change over 60s sustained contraction

### Correlation Patterns
```
High correlation:
- RMS ↔ MAV (r > 0.95)
- MDF ↔ MNF (r > 0.90)
- WIRE51 ↔ DI (r > 0.85)

Moderate correlation:
- RMS ↔ MNF (r ≈ 0.5-0.7, depends on fatigue state)
- WL ↔ RMS (r ≈ 0.6-0.8)

Low correlation:
- ZC ↔ RMS (r ≈ 0.2-0.4)
- Time domain ↔ Fatigue indices (r < 0.3)
```

---

## Best Practices

### 1. Feature Selection
- **Amplitude:** RMS (most common)
- **Frequency:** MDF or MNF (both reliable)
- **Fatigue:** WIRE51 + DI (complementary information)
- **Advanced:** IMNF + HHT (research applications)

### 2. Signal Requirements
- **Minimum length:** 0.5-1 second (500-1000 samples at 1 kHz)
- **Recommended:** 1-2 seconds for stable estimates
- **Frequency features:** Need longer segments (≥1s)
- **WIRE51:** Minimum 512 samples for 5-level DWT

### 3. Sampling Frequency
- **Minimum:** 1000 Hz (Nyquist covers up to 500 Hz)
- **Recommended:** 2000 Hz (better temporal resolution)
- **Practical:** 1000-2000 Hz for most applications

### 4. Preprocessing Requirements
- **Bandpass filter:** 20-450 Hz (remove DC and high-freq noise)
- **Notch filter:** Remove 50/60 Hz power line interference
- **Normalization:** Applied automatically in production HHT

### 5. Validation
- Check energy conservation (HHT): error < 5%
- Verify frequency values in physiological range (20-450 Hz)
- Compare with known reference values
- Use multiple features for robust analysis

---

## References

1. **General sEMG Features:**
   - Phinyomark A, et al. (2012). "Feature extraction of the first difference of EMG time series for EMG pattern recognition." Computer Methods and Programs in Biomedicine 117(2):247-256.

2. **Dimitrov Index:**
   - Dimitrov GV, et al. (2006). "Muscle fatigue during dynamic contractions assessed by new spectral indices." Med Sci Sports Exerc 38(11):1971-1979.

3. **WIRE51:**
   - Karlsson S, et al. (2000). "Mean frequency and signal amplitude of the surface EMG of the quadriceps muscles increase with increasing torque." J Electromyogr Kinesiol 10(2):133-140.

4. **Choi-Williams Distribution:**
   - Choi HI, Williams WJ (1989). "Improved time-frequency representation of multicomponent signals using exponential kernels." IEEE Trans Acoust Speech Signal Process 37(6):862-871.

5. **Hilbert-Huang Transform:**
   - Huang NE, et al. (1998). "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis." Proc R Soc Lond A 454:903-995.

6. **CEEMDAN:**
   - Torres ME, et al. (2011). "A complete ensemble empirical mode decomposition with adaptive noise." IEEE ICASSP 2011:4144-4147.

---

## Appendix: Validation Examples

### Example 1: Synthetic Signal Validation
```python
# Generate test signal: 100 Hz + 150 Hz
fs = 1000
t = np.linspace(0, 1, fs)
signal = 0.5*np.sin(2*np.pi*100*t) + 0.3*np.sin(2*np.pi*150*t)

features = extract_semg_features(signal, fs)

# Expected results:
# MNF ≈ 110 Hz (weighted average of 100 and 150)
# MDF ≈ 110-120 Hz
# IMNF ≈ 110 ± 10 Hz
# DI ≈ 1e-13 to 1e-12
```

### Example 2: Fatigue Detection
```python
# Compare fresh vs. fatigued muscle
fresh_mnf = 110  # Hz
fatigued_mnf = 75  # Hz
decline = (fresh_mnf - fatigued_mnf) / fresh_mnf  # 32% decline

fresh_di = 2.5e-12
fatigued_di = 12.3e-12
ratio = fatigued_di / fresh_di  # 4.9× increase

# Both indicators confirm fatigue
```

---

*Last Updated: 2025-12-11*
*Version: 1.0 (Production Release)*
