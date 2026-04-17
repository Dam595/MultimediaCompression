import librosa
import numpy as np
from src.psychoacoustics import compute_masking_threshold

def process_compression(y, sr, n_fft, threshold_offset):
    """Execute the full psychoacoustic compression pipeline."""
    D = librosa.stft(y, n_fft=n_fft)
    D_mag, D_phase = librosa.magphase(D)
    S_db = librosa.amplitude_to_db(D_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Compute thresholds for all frames
    global_mask_db = np.zeros_like(S_db)
    for i in range(S_db.shape[1]):
        global_mask_db[:, i] = compute_masking_threshold(
            D_mag[:, i], freqs, threshold_offset, librosa.amplitude_to_db
        )

    # Apply masking
    mask = S_db > global_mask_db
    D_processed = D.copy()
    D_processed[~mask] = 0
    
    y_compressed = librosa.istft(D_processed)
    y_removed = librosa.istft(D * (~mask))
    
    return y_compressed, y_removed, S_db, global_mask_db, mask, freqs

def calculate_snr(original, compressed):
    """Compute Signal-to-Noise Ratio (SNR)."""
    min_len = min(len(original), len(compressed))
    noise = original[:min_len] - compressed[:min_len]
    original_power = np.sum(original[:min_len]**2)
    noise_power = np.sum(noise**2) + 1e-8
    return 10 * np.log10(original_power / noise_power)