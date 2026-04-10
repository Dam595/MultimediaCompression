import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf

st.title("Audio Compression Visualization")

uploaded_file = st.file_uploader("Chọn file âm thanh (.wav)", type=["wav"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file)
    st.audio(uploaded_file, format='audio/wav')

    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    st.subheader("Phổ tần số gốc (Spectrogram)")
    fig = go.Figure(data=go.Heatmap(
        z=S_db,
        colorscale='Viridis'
    ))
    fig.update_layout(title="Spectrogram", xaxis_title="Time Frames", yaxis_title="Frequency Bins")
    st.plotly_chart(fig)
    st.subheader("Cài đặt mô hình tâm lý âm học")
    threshold = st.slider("Ngưỡng che lấp (dB)", -80, 0, -40)
    
    st.info(f"Hệ thống sẽ loại bỏ các tần số có biên độ dưới {threshold} dB")