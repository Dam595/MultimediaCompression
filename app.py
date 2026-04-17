import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from src.audio_engine import process_compression, calculate_snr

st.set_page_config(page_title="Audio Perception Lab", layout="wide")
st.title("🎧 Visualizing Human Perception in Audio Compression")

uploaded_file = st.file_uploader("📂 Upload .wav file", type=["wav"])

if uploaded_file is not None:
    # 1. Load and Normalize
    y, sr = librosa.load(uploaded_file, duration=10)
    y = librosa.util.normalize(y)
    
    # 2. Sidebar Settings
    st.sidebar.header("⚙️ Algorithm Settings")
    n_fft = st.sidebar.select_slider("Window Size (n_fft)", [512, 1024, 2048, 4096], 2048)
    threshold_offset = st.sidebar.slider("Masking Sensitivity (dB)", -40, 40, 0)

    # 3. Processing
    with st.spinner('🔄 Processing signal...'):
        y_comp, y_rem, S_db, g_mask, mask, freqs = process_compression(
            y, sr, n_fft, threshold_offset
        )

    # 4. Audio Players
    c1, c2 = st.columns(2)
    c1.write("### Original Audio")
    c1.audio(y, sample_rate=sr)
    c2.write("### Compressed Audio")
    c2.audio(y_comp, sample_rate=sr)

    # 5. Spectrograms
    st.divider()
    st.subheader("📊 Spectrogram Comparison")
    col3, col4 = st.columns(2)
    
    for col, data, title in zip([col3, col4], [S_db, librosa.amplitude_to_db(np.abs(y_comp), ref=np.max)], ["Original", "Compressed"]):
        fig = go.Figure(data=go.Heatmap(z=data, colorscale='Viridis', zmin=-80, zmax=0))
        fig.update_layout(title=title)
        col.plotly_chart(fig, use_container_width=True)

    # 6. Evaluation Metrics
    st.divider()
    m1, m2, m3 = st.columns(3)
    comp_ratio = (np.sum(~mask) / mask.size) * 100
    m1.metric("Removed Data", f"{comp_ratio:.2f}%")
    m2.metric("SNR (dB)", f"{calculate_snr(y, y_comp):.2f}")
    m3.write("🎧 Residual Sound (Removed)")
    m3.audio(y_rem, sample_rate=sr)

    # 7. Masking Detail
    st.divider()
    st.subheader("🔍 Masking Model (Middle Frame)")
    mid = S_db.shape[1] // 2
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=freqs, y=S_db[:, mid], name="Signal"))
    fig_line.add_trace(go.Scatter(x=freqs, y=g_mask[:, mid], name="Threshold", line=dict(dash='dash', color='red')))
    fig_line.update_layout(xaxis_type="log", xaxis_title="Freq (Hz)", yaxis_title="dB", yaxis_range=[-100, 5])
    st.plotly_chart(fig_line, use_container_width=True)