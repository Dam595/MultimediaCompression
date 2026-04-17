import librosa
import numpy as np
from .psychoacoustics import compute_masking_threshold

def compress_audio(y, sr, n_fft, threshold_offset):
    """Execute the psychoacoustic compression workflow."""
    D = librosa.stft(y, n_fft=n_fft)
    D_mag, D_phase = librosa.magphase(D)
    S_db = librosa.amplitude_to_db(D_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    global_mask_db = np.zeros_like(S_db)
    for i in range(S_db.shape[1]):
        global_mask_db[:, i] = compute_masking_threshold(D_mag[:, i], freqs, threshold_offset)

    mask = S_db > global_mask_db
    D_processed = D.copy()
    D_processed[~mask] = 0

    y_compressed = librosa.istft(D_processed)
    y_removed = librosa.istft(D * (~mask))
    
    return y_compressed, y_removed, S_db, global_mask_db, mask, freqs, D, D_mag

def calculate_snr(y_orig, y_comp):
    """Compute the Signal-to-Noise Ratio (SNR) between original and compressed signals."""
    min_len = min(len(y_orig), len(y_comp))
    noise = y_orig[:min_len] - y_comp[:min_len]
    original_power = np.sum(y_orig[:min_len]**2)
    noise_power = np.sum(noise**2) + 1e-8
    return 10 * np.log10(original_power / noise_power)

def run_evaluation_suite(D, D_mag, S_db, freqs, sr):
    """Run evaluation loop across multiple threshold values."""
    threshold_values = np.linspace(-40, 40, 10)
    snr_list = []
    compression_list = []

    for t in threshold_values:
        temp_mask_db = np.zeros_like(S_db)
        for i in range(S_db.shape[1]):
            temp_mask_db[:, i] = compute_masking_threshold(D_mag[:, i], freqs, t)

        temp_mask = S_db > temp_mask_db
        D_temp = D.copy()
        D_temp[~temp_mask] = 0
        y_temp = librosa.istft(D_temp)

        # y_orig derived from original D
        snr_val = calculate_snr(librosa.istft(D), y_temp) 
        comp_ratio = (np.sum(~temp_mask) / temp_mask.size) * 100

        snr_list.append(snr_val)
        compression_list.append(comp_ratio)
        
    return threshold_values, snr_list, compression_list