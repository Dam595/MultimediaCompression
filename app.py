import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
from scipy.fftpack import fft
from scipy.signal import find_peaks as scipy_signal_find_peaks

st.set_page_config(layout="wide")
st.title("Visualizing Human Perception in Audio Compression")

def freq_to_bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500)**2)

def ath_threshold(f):
    f = np.maximum(f, 0.01) 
    f_khz = f / 1000
    ath = 3.64 * (f_khz**-3.9) - 6.5 * np.exp(-0.6 * (f_khz - 3.3)**2) + 10**-3 * (f_khz**4)
    return ath - 90

def compute_masking_threshold(magnitudes, freqs, threshold_offset):
    ath = ath_threshold(freqs)
    masking_curve = np.full_like(magnitudes, -100.0)
    bark_freqs = freq_to_bark(freqs)
    
    mag_db = librosa.amplitude_to_db(magnitudes, ref=np.max)
    
    peaks, _ = scipy_signal_find_peaks(mag_db, height=-60)
    
    for p in peaks:
        target_bark = bark_freqs[p]
        target_mag = mag_db[p]
        delta_bark = bark_freqs - target_bark
        spread = np.where(delta_bark < 0, 27 * delta_bark, -10 * delta_bark)
        masking_curve = np.maximum(masking_curve, target_mag + spread - 7)

    global_threshold_db = np.maximum(ath, masking_curve) + threshold_offset
    return np.clip(global_threshold_db, -100, 0)

uploaded_file = st.file_uploader("Chọn file âm thanh (.wav)", type=["wav"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, duration=10)
    y = librosa.util.normalize(y)
    
    st.sidebar.header("Cài đặt thuật toán")
    n_fft = st.sidebar.select_slider("Window Size (n_fft)", options=[512, 1024, 2048, 4096], value=2048)
    threshold_offset = st.sidebar.slider("Tăng/Giảm ngưỡng nhạy (dB)", -40, 40, 0)

    D = librosa.stft(y, n_fft=n_fft)
    D_mag, D_phase = librosa.magphase(D)
    S_db = librosa.amplitude_to_db(D_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    with st.spinner('Đang tính toán mô hình tâm lý âm học...'):
        global_mask_db = np.zeros_like(S_db)
        for i in range(S_db.shape[1]):
            global_mask_db[:, i] = compute_masking_threshold(D_mag[:, i], freqs, threshold_offset)

    mask = S_db > global_mask_db
    D_processed = D.copy()
    D_processed[~mask] = 0

    y_compressed = librosa.istft(D_processed)
    y_removed = librosa.istft(D * (~mask))

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔊 File Gốc")
        st.audio(uploaded_file)
        fig_orig = go.Figure(data=go.Heatmap(z=S_db, colorscale='Viridis', zmin=-80, zmax=0))
        fig_orig.update_layout(title="Spectrogram Gốc", xaxis_title="Time", yaxis_title="Frequency")
        st.plotly_chart(fig_orig, use_container_width=True)

    with col2:
        st.subheader("🔇 File Sau Khi Nén")
        st.audio(y_compressed, sample_rate=sr)
        S_db_processed = librosa.amplitude_to_db(np.abs(D_processed), ref=np.max)
        fig_proc = go.Figure(data=go.Heatmap(z=S_db_processed, colorscale='Viridis', zmin=-80, zmax=0))
        fig_proc.update_layout(title="Spectrogram Đã Nén", xaxis_title="Time", yaxis_title="Frequency")
        st.plotly_chart(fig_proc, use_container_width=True)

    st.divider()
    
    st.subheader("📊 Phân tích hiệu quả nén")
    total_elements = mask.size
    removed_elements = np.sum(~mask)
    compression_ratio = (removed_elements / total_elements) * 100
    
    c1, c2 = st.columns(2)
    c1.metric("Tần số bị loại bỏ", f"{removed_elements}", f"{compression_ratio:.2f}% data")
    c2.write("### Nghe phần âm thanh bị loại bỏ:")
    c2.audio(y_removed, sample_rate=sr)

    st.subheader("🔍 Chi tiết mô hình che lấp (Frame trung tâm)")
    middle_frame = int(min(S_db.shape[1] // 2, S_db.shape[1] - 1))
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=freqs, y=S_db[:, middle_frame], name="Phổ tín hiệu"))
    fig_line.add_trace(go.Scatter(x=freqs, y=global_mask_db[:, middle_frame], name="Ngưỡng che lấp toàn cục", line=dict(dash='dash', color='red')))
    fig_line.update_layout(xaxis_type="log", xaxis_title="Tần số (Hz)", yaxis_title="Biên độ (dB)", yaxis_range=[-100, 5])
    st.plotly_chart(fig_line, use_container_width=True)