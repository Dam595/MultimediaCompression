# Audio Compression

This project is an interactive visualization tool designed to demonstrate the principles of **Lossy Audio Compression** (like MP3). It utilizes **Psychoacoustic Models** to identify and remove frequencies that are "invisible" to the human ear, effectively reducing data size while maintaining perceived audio quality.

## Key Features

- **Psychoacoustic Modeling**: Implements the **Absolute Threshold of Hearing (ATH)** and **Simultaneous Masking** algorithms.
- **Bark Scale Integration**: Frequency analysis based on human critical bands (Bark Scale) rather than linear Hz.
- **Dual Spectrogram Visualization**: Side-by-side comparison of original vs. compressed signals.
- **Interactive Controls**: Real-time adjustment of Window Size (n_fft) and Masking Sensitivity (dB).
- **Technical Metrics**: Calculation of **Compression Ratio** and **Signal-to-Noise Ratio (SNR)**.
- **Residual Audio**: Capability to listen to the "removed" sounds to evaluate the transparency of the compression.

## Folder Structure

```Project 1
.
├── app.py                 # Main Streamlit UI application
├── src/
│   ├── __init__.py        # Package initializer
│   ├── psychoacoustics.py # Mathematical formulas (ATH, Bark, Masking)
│   └── audio_engine.py    # Signal processing logic (STFT/ISTFT, Filtering)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

## Installation & Setup
git clone <your-repository-url>
cd <project-folder>

python3 -m venv multimedia_env

source multimedia_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py