import numpy as np
from scipy.signal import find_peaks as scipy_signal_find_peaks
import librosa

def freq_to_bark(f):
    """Convert Frequency (Hz) to Bark scale."""
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500)**2)

def ath_threshold(f):
    """Absolute Threshold of Hearing (ATH) - Terhardt formula."""
    f = np.maximum(f, 0.01) 
    f_khz = f / 1000
    ath = 3.64 * (f_khz**-3.9) - 6.5 * np.exp(-0.6 * (f_khz - 3.3)**2) + 10**-3 * (f_khz**4)
    return ath - 90

def compute_masking_threshold(magnitudes, freqs, threshold_offset):
    """Calculate the Global Masking Threshold for a single STFT frame."""
    ath = ath_threshold(freqs)
    masking_curve = np.full_like(magnitudes, -100.0)
    bark_freqs = freq_to_bark(freqs)
    
    mag_db = librosa.amplitude_to_db(magnitudes, ref=np.max)
    peaks, _ = scipy_signal_find_peaks(mag_db, height=-60)
    
    for p in peaks:
        target_bark = bark_freqs[p]
        target_mag = mag_db[p]
        delta_bark = bark_freqs - target_bark
        
        # Spreading function: simultaneous masking effect
        spread = np.where(delta_bark < 0, 27 * delta_bark, -10 * delta_bark)
        masking_curve = np.maximum(masking_curve, target_mag + spread - 7)

    global_threshold_db = np.maximum(ath, masking_curve) + threshold_offset
    return np.clip(global_threshold_db, -100, 0)