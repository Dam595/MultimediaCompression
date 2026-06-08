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
    threshold_offset = st.sidebar.slider("Masking Sensitivity (dB)", -60, 40, 0)
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

    # --- 5. Critical Bands Visualization ---
    st.divider()
    st.subheader("🎼 Critical Bands (Bark Scale)")
    st.caption("The 24 critical bands represent the ear's frequency resolution. Each band is one 'Bark' unit wide — sounds within the same band mask each other strongly.")

    # Define the 24 critical band boundaries in Hz (standard Zwicker values)
    critical_band_edges_hz = [
        20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
        1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
    ]

    # --- 5a. Masking Curve with Critical Band Overlay ---
    fig_cb = go.Figure()

    # Add critical band shaded regions (alternating for visibility)
    for i in range(len(critical_band_edges_hz) - 1):
        f_low = critical_band_edges_hz[i]
        f_high = critical_band_edges_hz[i + 1]
        fill_color = "rgba(100, 180, 255, 0.06)" if i % 2 == 0 else "rgba(100, 180, 255, 0.13)"
        fig_cb.add_vrect(
            x0=f_low, x1=f_high,
            fillcolor=fill_color,
            line_width=0.5,
            line_color="rgba(100,180,255,0.3)",
            annotation_text=str(i + 1),
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="rgba(150,200,255,0.7)"
        )

    # Signal and masking threshold traces
    fig_cb.add_trace(go.Scatter(
        x=freqs, y=S_db[:, middle_frame],
        name="Signal", line=dict(color='#5BC8F5', width=1.2)
    ))
    fig_cb.add_trace(go.Scatter(
        x=freqs, y=global_mask_db[:, middle_frame],
        name="Masking Threshold", line=dict(dash='dash', color='red', width=1.5)
    ))

    fig_cb.update_layout(
        title="Masking Curve with 24 Critical Bands Overlay",
        xaxis_title="Frequency (Hz)",
        xaxis_type="log",
        xaxis=dict(range=[np.log10(20), np.log10(15500)]),
        yaxis_title="Amplitude (dB)",
        yaxis_range=[-100, 5],
        legend=dict(x=0.01, y=0.01),
        height=420
    )
    st.plotly_chart(fig_cb, use_container_width=True)

    # --- 5b. Energy per Critical Band bar chart ---
    st.markdown("**Energy per Critical Band** — how much signal energy falls in each of the 24 bands")

    band_energies = []
    band_masked_ratios = []
    band_labels = []

    for i in range(len(critical_band_edges_hz) - 1):
        f_low = critical_band_edges_hz[i]
        f_high = critical_band_edges_hz[i + 1]
        band_mask = (freqs >= f_low) & (freqs < f_high)
        if np.sum(band_mask) == 0:
            band_energies.append(-100)
            band_masked_ratios.append(0)
        else:
            energy = np.mean(S_db[band_mask, middle_frame])
            threshold_energy = np.mean(global_mask_db[band_mask, middle_frame])
            masked_ratio = np.mean(S_db[band_mask, middle_frame] < global_mask_db[band_mask, middle_frame]) * 100
            band_energies.append(float(energy))
            band_masked_ratios.append(float(masked_ratio))
        band_labels.append(f"CB{i+1}<br>{f_low}–{f_high}Hz")

    fig_energy = go.Figure()
    fig_energy.add_trace(go.Bar(
        x=list(range(1, 25)),
        y=band_energies,
        name="Signal Energy (dB)",
        marker_color=[
            f"rgba(91,200,245,{0.4 + 0.6*(1 - r/100)})" for r in band_masked_ratios
        ],
        hovertemplate="Band %{x}<br>Energy: %{y:.1f} dB<extra></extra>"
    ))
    fig_energy.add_trace(go.Bar(
        x=list(range(1, 25)),
        y=band_masked_ratios,
        name="% Masked",
        marker_color="rgba(220,90,80,0.6)",
        yaxis="y2",
        hovertemplate="Band %{x}<br>Masked: %{y:.1f}%<extra></extra>"
    ))
    fig_energy.update_layout(
        xaxis_title="Critical Band number",
        yaxis_title="Avg energy (dB)",
        yaxis2=dict(title="% bins masked", overlaying="y", side="right", range=[0, 100]),
        barmode="overlay",
        height=340,
        legend=dict(x=0.01, y=0.99),
        xaxis=dict(tickmode="linear", tick0=1, dtick=1)
    )
    st.plotly_chart(fig_energy, use_container_width=True)

    # --- 5c. Bark scale mapping info ---
    with st.expander("ℹ️ What is the Bark scale?"):
        st.markdown("""
        The **Bark scale** maps Hz to perceptual units used by the human auditory system.
        
        - 1 Bark ≈ 1 critical band width
        - Below ~500 Hz: bands are ~100 Hz wide (ear is very precise here)  
        - Above 500 Hz: bands grow wider with frequency (ear becomes less precise)
        - 24 Bark units cover the full audible range (20 Hz – ~15.5 kHz)
        
        Masking calculations in this project use Bark distance rather than Hz distance, 
        which is why the spreading function (`15 dB/Bark` downward, `10 dB/Bark` upward) 
        feels natural to human listeners.
        """)

    # --- 6. Evaluation & Analysis Loop ---
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