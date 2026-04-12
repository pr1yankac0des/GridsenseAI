"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 1: DATASET GENERATOR                                     ║
║  Creates training data for GridSense AI                        ║
║                                                                ║
║  WHY WE NEED THIS:                                             ║
║  - Kaggle data alone may not cover all edge cases              ║
║  - Synthetic data lets us control exactly what we generate     ║
║  - More data = better accuracy                                 ║
║  - Shows judges we understand the physics of power systems     ║
║                                                                ║
║  RUN: python generate_dataset.py                               ║
║  OUTPUT: power_quality_data.csv (15,000 samples)               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import os
from collections import namedtuple

SignalFeatures = namedtuple(
    "SignalFeatures",
    ["rms", "peak", "thd", "duration", "dwt_energy", "dwt_entropy", "snr"]
)

# ─── WHAT EACH FEATURE MEANS ───────────────────────────────────
#
# RMS_Voltage:
#   "Root Mean Square" voltage. Think of it as the "effective" voltage.
#   For a perfect 230V supply, RMS = 230V.
#   If it drops below ~207V = voltage sag
#   If it rises above ~253V = voltage swell
#
# Peak_Voltage:
#   The maximum instantaneous voltage.
#   For perfect 230V: Peak = 230 × √2 = 325.27V
#   High peaks = transients or spikes
#
# THD (Total Harmonic Distortion):
#   Measures how "distorted" the sine wave is.
#   0% = perfect sine wave
#   >5% = significant harmonics (IEEE limit)
#   >15% = dangerous (fire hazard from overheating)
#
# Duration:
#   How long (in seconds) the disturbance lasts.
#   Sag: typically 0.01 to 1 second
#   Transient: microseconds to milliseconds
#
# DWT_Energy (Discrete Wavelet Transform Energy):
#   Wavelet decomposition captures both time AND frequency info.
#   High energy in detail coefficients = transient events.
#   This is more advanced than FFT alone.
#
# DWT_Entropy:
#   Measures the "randomness" or "complexity" of the signal.
#   Normal signal = low entropy (predictable sine wave)
#   Disturbed signal = high entropy (unpredictable)
#
# Signal_Noise_Ratio_dB:
#   How clean the signal is. Higher = cleaner.
#   >30 dB = clean signal
#   <20 dB = noisy (harder to classify)
#
# Phase:
#   Which phase of the 3-phase system (A, B, or C).
#   Some faults only affect one phase.
#
# Fault_Type (TARGET - what we predict):
#   0 = Normal
#   1 = Voltage Sag (voltage drops)
#   2 = Voltage Swell (voltage rises)
#   3 = Harmonics (wave distortion)
#   4 = Transients (spikes/oscillations)
# ─────────────────────────────────────────────────────────────────

np.random.seed(42)  # For reproducibility

SAMPLES_PER_CLASS = 3000
CLASSES = {
    0: "Normal",
    1: "Voltage_Sag",
    2: "Voltage_Swell",
    3: "Harmonics",
    4: "Transients"
}


def generate_normal(n):
    """
    Normal operation: everything within acceptable limits.
    
    RMS: 220-240V (Indian standard: 230V ± ~5%)
    THD: 0-3% (clean signal)
    Peak: proportional to RMS (× √2)
    """
    rms = np.random.normal(230, 4, n)           # centered at 230V, small variation
    thd = np.random.uniform(0.5, 3.0, n)        # low distortion
    peak = rms * np.sqrt(2) * (1 + np.random.normal(0, 0.02, n))  # ideal peak
    duration = np.zeros(n)                       # no disturbance duration
    dwt_energy = np.random.uniform(0.1, 1.0, n) # low energy (stable)
    dwt_entropy = np.random.uniform(0.1, 0.5, n)# low entropy (predictable)
    snr = np.random.uniform(30, 45, n)           # clean signal
    return SignalFeatures(rms, peak, thd, duration, dwt_energy, dwt_entropy, snr)


def generate_sag(n):
    """
    Voltage Sag: voltage drops to 10-90% of nominal.
    
    REAL WORLD: Motor starts, short circuits on nearby feeders.
    RMS drops significantly (below 207V for 230V system).
    Duration: 0.5 cycles to 1 minute (we use seconds).
    """
    rms = np.random.uniform(120, 207, n)         # DROPPED voltage
    thd = np.random.uniform(1.0, 5.0, n)         # slight increase in THD
    peak = rms * np.sqrt(2) * (1 + np.random.normal(0, 0.03, n))
    duration = np.random.uniform(0.01, 1.0, n)   # short to medium duration
    dwt_energy = np.random.uniform(1.0, 5.0, n)  # elevated energy
    dwt_entropy = np.random.uniform(0.5, 2.0, n) # moderate entropy
    snr = np.random.uniform(20, 35, n)            # slightly noisier
    return SignalFeatures(rms, peak, thd, duration, dwt_energy, dwt_entropy, snr)


def generate_swell(n):
    """
    Voltage Swell: voltage rises to 110-180% of nominal.
    
    REAL WORLD: Large load suddenly disconnected, capacitor switching.
    RMS increases above 253V.
    """
    rms = np.random.uniform(253, 380, n)          # ELEVATED voltage
    thd = np.random.uniform(1.0, 6.0, n)          # moderate THD
    peak = rms * np.sqrt(2) * (1 + np.random.normal(0, 0.03, n))
    duration = np.random.uniform(0.01, 0.8, n)    # typically shorter than sag
    dwt_energy = np.random.uniform(1.5, 6.0, n)   # elevated
    dwt_entropy = np.random.uniform(0.4, 1.8, n)  # moderate
    snr = np.random.uniform(22, 35, n)
    return SignalFeatures(rms, peak, thd, duration, dwt_energy, dwt_entropy, snr)


def generate_harmonics(n):
    """
    Harmonics: wave contains extra frequencies (3rd, 5th, 7th...).
    
    REAL WORLD: VFDs, rectifiers, LED drivers, UPS systems.
    THD is the KEY indicator — will be HIGH (>5%).
    RMS may be near normal, but the SHAPE is wrong.
    """
    rms = np.random.normal(228, 8, n)              # near-normal RMS
    thd = np.random.uniform(8.0, 35.0, n)          # HIGH THD (key feature!)
    peak = rms * np.sqrt(2) * (1 + thd/100 * np.random.uniform(0.3, 0.8, n))
    duration = np.random.uniform(0.5, 10.0, n)     # continuous (harmonics persist)
    dwt_energy = np.random.uniform(2.0, 8.0, n)    # elevated (extra frequencies)
    dwt_entropy = np.random.uniform(1.0, 3.5, n)   # HIGH entropy (complex signal)
    snr = np.random.uniform(15, 28, n)              # lower SNR
    return SignalFeatures(rms, peak, thd, duration, dwt_energy, dwt_entropy, snr)


def generate_transients(n):
    """
    Transients: sudden, short-duration spikes or oscillations.
    
    REAL WORLD: Lightning, switching surges, capacitor bank switching.
    Peak voltage is VERY HIGH relative to RMS.
    Duration is VERY SHORT.
    DWT energy is HIGH (wavelet catches transients well).
    """
    rms = np.random.normal(232, 10, n)              # near-normal RMS
    thd = np.random.uniform(2.0, 10.0, n)           # moderate THD
    # Transients have extreme peaks!
    peak = rms * np.sqrt(2) * (1.5 + np.random.uniform(0.5, 3.0, n))
    duration = np.random.uniform(0.0001, 0.01, n)   # VERY short (microseconds)
    dwt_energy = np.random.uniform(5.0, 15.0, n)    # VERY HIGH (spike energy)
    dwt_entropy = np.random.uniform(2.0, 5.0, n)    # HIGH (unpredictable)
    snr = np.random.uniform(10, 25, n)               # noisy
    return SignalFeatures(rms, peak, thd, duration, dwt_energy, dwt_entropy, snr)


def main():
    print("="*60)
    print("GridSense AI — Dataset Generator")
    print("="*60)
    
    generators = {
        0: ("Normal",        generate_normal),
        1: ("Voltage_Sag",   generate_sag),
        2: ("Voltage_Swell", generate_swell),
        3: ("Harmonics",     generate_harmonics),
        4: ("Transients",    generate_transients),
    }
    
    all_data = []
    
    for class_id, (class_name, gen_func) in generators.items():
        print(f"\n  Generating {SAMPLES_PER_CLASS} samples for: {class_name}")
        
        sig = gen_func(SAMPLES_PER_CLASS)
        
        # Random phase assignment (A, B, or C)
        phases = np.random.choice(["A", "B", "C"], SAMPLES_PER_CLASS)
        
        for i in range(SAMPLES_PER_CLASS):
            all_data.append({
                "RMS_Voltage":          round(sig.rms[i], 2),
                "Peak_Voltage":         round(sig.peak[i], 2),
                "THD":                  round(sig.thd[i], 2),
                "Duration":             round(sig.duration[i], 6),
                "DWT_Energy_Levels":    round(sig.dwt_energy[i], 4),
                "DWT_Entropy":          round(sig.dwt_entropy[i], 4),
                "Signal_Noise_Ratio_dB": round(sig.snr[i], 2),
                "Phase":                phases[i],
                "Fault_Type":           class_id,
            })
        
        print(f"    ✅ RMS range: {sig.rms.min():.1f} - {sig.rms.max():.1f} V")
        print(f"    ✅ THD range: {sig.thd.min():.1f} - {sig.thd.max():.1f} %")
        print(f"    ✅ Peak range: {sig.peak.min():.1f} - {sig.peak.max():.1f} V")
    
    # Shuffle the data
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), "power_quality_data.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ DATASET SAVED: {output_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"   Target: Fault_Type")
    print(f"\n   Class distribution:")
    for cls, name in CLASSES.items():
        count = len(df[df.Fault_Type == cls])
        print(f"     {cls} ({name}): {count} samples")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
