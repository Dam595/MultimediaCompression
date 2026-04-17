import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from src.audio_engine import compress_audio, calculate_snr, run_evaluation_suite

st.set_page_config(layout="wide", page_title="Audio Perception Lab")
st.title("🎧 Visualizing Human Perception in Audio Compression")
st.info("🎛️ Adjust the slider to see how compression affects audio quality")

uploaded_file = st.file_uploader("📂 Choose an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Load & Normalize
    y, sr = librosa.load(uploaded_file, duration=20)
    y = librosa.util.normalize(y)
    
    # Sidebar
    st.sidebar.header("⚙️ Algorithm Settings")
    n_fft = st.sidebar.select_slider("Window Size (n_fft)", options=[512, 1024, 2048, 4096], value=2048)
    threshold_offset = st.sidebar.slider("Masking Sensitivity (dB)", -40, 40, 0)
    st.sidebar.caption("Dam The Anh 202414607, Le Cong Hai Quan 202414659")

    with st.spinner('🔄 Processing signal...'):
        y_compressed, y_removed, S_db, global_mask_db, mask, freqs, D_orig, D_mag = compress_audio(
            y, sr, n_fft, threshold_offset
        )

    # --- 1. Audio Players ---
    st.subheader("🎧 Before vs After Compression")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original Audio")
        st.audio(y, sample_rate=sr)
    with col2:
        st.write("Compressed Audio")
        st.audio(y_compressed, sample_rate=sr)

    # --- 2. Spectrograms ---
    st.divider()
    st.subheader("📊 Spectrogram Comparison")
    # Recalculate processed STFT for accurate visualization
    S_db_processed = librosa.amplitude_to_db(np.abs(librosa.stft(y_compressed, n_fft=n_fft)), ref=np.max)
    c3, c4 = st.columns(2)
    for col, data, title in zip([c3, c4], [S_db, S_db_processed], ["Original Spectrogram", "Compressed Spectrogram"]):
        fig = go.Figure(data=go.Heatmap(z=data, colorscale='Viridis', zmin=-80, zmax=0))
        fig.update_layout(title=title, xaxis_title="Time Frames", yaxis_title="Frequency Bin")
        col.plotly_chart(fig, use_container_width=True)

    # --- 3. Metrics ---
    st.divider()
    st.subheader("📈 Compression Analysis")
    removed_elements = np.sum(~mask)
    compression_ratio = (removed_elements / mask.size) * 100
    snr = calculate_snr(y, y_compressed)

    m1, m2, m3 = st.columns(3)
    m1.metric("Removed Frequencies", f"{removed_elements}", f"{compression_ratio:.2f}%")
    m2.metric("SNR (dB)", f"{snr:.2f}")
    m3.write("🎧 Removed Sound (Residual)")
    m3.audio(y_removed, sample_rate=sr)

    # --- 4. Masking Curve Detail ---
    st.divider()
    st.subheader("🔍 Masking Model (Middle Frame)")
    middle_frame = int(min(S_db.shape[1] // 2, S_db.shape[1] - 1))
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=freqs, y=S_db[:, middle_frame], name="Signal"))
    fig_line.add_trace(go.Scatter(x=freqs, y=global_mask_db[:, middle_frame], name="Masking Threshold", line=dict(dash='dash', color='red')))
    fig_line.update_layout(xaxis_type="log", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude (dB)", yaxis_range=[-100, 5], title="Masking Curve Analysis")
    st.plotly_chart(fig_line, use_container_width=True)

    # --- 5. Evaluation & Analysis Loop ---
    st.divider()
    st.header("📊 Evaluation & Analysis")
    
    t_vals, snr_vals, comp_vals = run_evaluation_suite(D_orig, D_mag, S_db, freqs, sr)

    # Plot SNR vs Threshold
    fig_snr = go.Figure()
    fig_snr.add_trace(go.Scatter(x=t_vals, y=snr_vals, mode='lines+markers'))
    fig_snr.update_layout(xaxis_title="Threshold (dB)", yaxis_title="SNR (dB)", title="SNR vs Threshold")
    st.plotly_chart(fig_snr, use_container_width=True)

    # Plot Compression vs Threshold
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=t_vals, y=comp_vals, mode='lines+markers'))
    fig_comp.update_layout(xaxis_title="Threshold (dB)", yaxis_title="Compression (%)", title="Compression Ratio vs Threshold")
    st.plotly_chart(fig_comp, use_container_width=True)

    # DataFrame and Download
    df = pd.DataFrame({"Threshold": t_vals, "SNR": snr_vals, "Compression (%)": comp_vals})
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Evaluation Results (CSV)", csv, "evaluation.csv", "text/csv")