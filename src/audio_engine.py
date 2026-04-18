import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .psychoacoustics import compute_masking_threshold

def compress_audio(y, sr, n_fft, threshold_offset):
    """Execute the psychoacoustic compression workflow with temporal smoothing."""
    D = librosa.stft(y, n_fft=n_fft)
    D_mag, D_phase = librosa.magphase(D)
    S_db = librosa.amplitude_to_db(D_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Compute initial masking threshold
    global_mask_db = np.zeros_like(S_db)
    for i in range(S_db.shape[1]):
        global_mask_db[:, i] = compute_masking_threshold(D_mag[:, i], freqs, threshold_offset)

    # Temporal Smoothing
    # Apply Gaussian filter along the time axis (axis=1) to prevent "musical noise"
    global_mask_db = gaussian_filter1d(global_mask_db, sigma=1.2, axis=1)

    # Soft Thresholding (Gain-based)
    diff = S_db - global_mask_db
    
    # Frequencies below threshold are attenuated by 30dB instead of being cut to zero
    # This preserves natural background floor and reduces artifacts
    gain = np.where(diff > 0, 1.0, 10**(-30/20)) 
    
    D_processed_mag = D_mag * gain
    D_processed = D_processed_mag * D_phase

    y_compressed = librosa.istft(D_processed)
    
    # For residual audio (the "removed" parts)
    y_removed = librosa.istft(D_mag * (1 - gain) * D_phase)
    
    # mask_binary is used for calculating compression ratio in UI
    mask_binary = diff > 0
    
    return y_compressed, y_removed, S_db, global_mask_db, mask_binary, freqs, D, D_mag

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
        # Use simple masking for evaluation metrics
        D_temp = D.copy()
        D_temp[~temp_mask] = 0
        y_temp = librosa.istft(D_temp)

        snr_val = calculate_snr(librosa.istft(D), y_temp) 
        comp_ratio = (np.sum(~temp_mask) / temp_mask.size) * 100

        snr_list.append(snr_val)
        compression_list.append(comp_ratio)
        
    return threshold_values, snr_list, compression_list