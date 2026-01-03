import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="EchoGuard",
    layout="wide"
)

# Custom CSS for Professional Industrial Look
st.markdown("""
    <style>
    /* Background Image - Acoustic/Tech Theme */
    .stApp {
        background-image: linear-gradient(rgba(10, 15, 30, 0.9), rgba(10, 15, 30, 0.9)), 
                          url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Clean Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 6px;
        color: white;
    }
    
    /* Professional Result Banners (No Emojis) */
    .status-banner {
        padding: 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        color: white;
        font-family: 'Segoe UI', sans-serif;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .status-safe {
        background-color: rgba(16, 185, 129, 0.2);
        border-left: 4px solid #10b981;
    }
    .status-danger {
        background-color: rgba(239, 68, 68, 0.2);
        border-left: 4px solid #ef4444;
    }
    
    /* Typography */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -1px;
    }
    h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
SAMPLE_RATE = 16000
DURATION = 10
N_MELS = 64
N_FFT = 1024
HOP_LEN = 512
LATENT_DIM = 256

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.encoder_lin = nn.Sequential(
            nn.Linear(128 * 4 * 20, LATENT_DIM),
            nn.Dropout(p=0.3) 
        )
        self.decoder_lin = nn.Linear(LATENT_DIM, 128 * 4 * 20)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 4, 20))
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=(1,1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(1,0)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1,0)), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=(1,0)), nn.Sigmoid() 
        )

    def forward(self, x):
        x_enc = self.encoder_cnn(x)
        x_flat = self.flatten(x_enc)
        latent_vector = self.encoder_lin(x_flat)
        x_dec = self.decoder_lin(latent_vector)
        x_unflat = self.unflatten(x_dec)
        reconstruction = self.decoder_cnn(x_unflat)
        return latent_vector, reconstruction

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    model = ConvolutionalAutoencoder()
    
    # Priority Loading
    paths = ['echo_guard_v10_generalist.pth', 'echo_guard_v9.pth']
    loaded_path = None
    for p in paths:
        if os.path.exists(p):
            try:
                model.load_state_dict(torch.load(p, map_location=device))
                model.eval()
                loaded_path = p
                break
            except: continue
            
    if not loaded_path:
        return None, None, "Error: Neural Network model file not found."

    # Load IsoForest
    iso_paths = ['iso_forest_v10_generalist.joblib', 'iso_forest_v9.joblib']
    iso_forest = None
    for p in iso_paths:
        if os.path.exists(p):
            try:
                iso_forest = joblib.load(p)
                break
            except: continue
            
    if iso_forest is None:
        return None, None, "Error: Isolation Forest model file not found."
        
    return model, iso_forest, None

def process_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LEN)
        log_mels = librosa.power_to_db(mels, ref=1.0)
        log_mels = (log_mels + 80.0) / 80.0
        log_mels = np.clip(log_mels, 0.0, 1.0)
        tensor = torch.tensor(log_mels, dtype=torch.float32).unsqueeze(0)
        return y, sr, log_mels, tensor, None
    except Exception as e:
        return None, None, None, None, f"DSP Error: {e}"

# ==========================================
# 4. MAIN APP INTERFACE
# ==========================================

# -- Main Header --
st.title("EchoGuard")
st.markdown("##### Acoustic Anomaly Detection System")

# -- Configuration Section (Top of Page) --
st.markdown("### Configuration")
col_conf1, col_conf2 = st.columns([2, 1])

with col_conf1:
    uploaded_file = st.file_uploader("Upload Audio Input (.wav)", type=["wav"])

with col_conf2:
    st.markdown("**Calibration**")
    THRESHOLD = st.slider("Anomaly Threshold", 0.30, 0.60, 0.4718, 0.001)
    st.caption(f"Model: Hybrid Autoencoder v10")

st.markdown("---")

# Load Brain
model, iso_forest, err_msg = load_models()

if err_msg:
    st.error(err_msg)
    st.stop()

if uploaded_file is None:
    # Idle State
    st.markdown("""
    <div style="text-align: center; padding: 40px; color: #888;">
        Await Sensor Data Input
    </div>
    """, unsafe_allow_html=True)

else:
    # --- ANALYSIS ENGINE ---
    with st.spinner("Processing signal..."):
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        y, sr, spec_img, tensor, proc_err = process_audio("temp.wav")
        
        if proc_err:
            st.error(f"Processing Failed: {proc_err}")
            st.stop()

        device = torch.device("cpu")
        input_tensor = tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            latent_vec, reconstruction = model(input_tensor)
            latent_vec_flat = latent_vec.cpu().numpy().flatten()
            
            # Hybrid Feature Extraction
            # MSE Error calculated for display only
            mse_error = torch.mean((input_tensor - reconstruction) ** 2).item()
            
        # Scoring - FIX: Use only the 256 latent features to match training
        score = -iso_forest.score_samples(latent_vec_flat.reshape(1, -1))[0]
        is_anomaly = score > THRESHOLD

    # --- INSIGHTS ENGINE ---
    diff = score - THRESHOLD
    
    # --- DASHBOARD LAYOUT ---
    
    # 1. Status Banner
    if is_anomaly:
        st.markdown(f"""
        <div class="status-banner status-danger">
            <div>
                <h3 style="margin:0; color:#fca5a5;">ANOMALY DETECTED</h3>
                <p style="margin:0; font-size: 0.9rem; opacity:0.8;">Spectral signature deviation exceeds limit.</p>
            </div>
            <div style="font-weight:bold; font-size:1.5rem;">{score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-banner status-safe">
            <div>
                <h3 style="margin:0; color:#6ee7b7;">SYSTEM NORMAL</h3>
                <p style="margin:0; font-size: 0.9rem; opacity:0.8;">Acoustic signature within healthy parameters.</p>
            </div>
            <div style="font-weight:bold; font-size:1.5rem;">{score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Key Metrics Grid
    m1, m2, m3 = st.columns(3)
    m1.metric("Deviation Score", f"{diff:+.4f}", help="Difference between Score and Threshold")
    m2.metric("Active Threshold", f"{THRESHOLD:.4f}")
    m3.metric("Reconstruction Error", f"{mse_error:.5f}")

    st.markdown("---")

    # 3. Technical Visuals (Clean Tabs)
    tab1, tab2 = st.tabs(["Signal", "Neural Analysis"])

    with tab1:
        st.audio("temp.wav", format="audio/wav")
        
        fig, ax = plt.subplots(figsize=(12, 3))
        color = '#ef4444' if is_anomaly else '#10b981'
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color=color, alpha=0.9, linewidth=0.6)
        
        # Minimalist Dark Theme
        ax.set_facecolor('#00000000') # Transparent
        fig.patch.set_facecolor('#00000000')
        ax.axis('off')
        
        # Use container width for mobile responsiveness
        st.pyplot(fig, use_container_width=True)
        st.caption("Amplitude/Time Waveform")

    with tab2:
        in_img = tensor.squeeze().numpy()
        out_img = reconstruction.squeeze().numpy()
        err_img = np.abs(in_img - out_img)
        
        # Helper for dark mode plots - Increased size and aspect ratio management
        def plot_full(data, title, cmap, vmax=1.0):
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(title, color='#94a3b8', fontsize=12, pad=10, loc='left')
            ax.set_xlabel("Time", color='#505050')
            ax.set_ylabel("Frequency", color='#505050')
            ax.tick_params(axis='both', colors='#505050')
            
            # Remove frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            fig.patch.set_facecolor('#00000000')
            ax.set_facecolor('#00000000')
            
            cbar = plt.colorbar(im, ax=ax, pad=0.02)
            cbar.ax.yaxis.set_tick_params(color='#505050', labelcolor='#505050')
            cbar.outline.set_visible(False)
            
            return fig

        st.markdown("#### A. Sensor Input")
        st.pyplot(plot_full(in_img, "Raw Spectrogram", 'viridis'), use_container_width=True)
        
        st.markdown("#### B. Model Reconstruction")
        st.pyplot(plot_full(out_img, "AI Baseline (Expected Normal)", 'viridis'), use_container_width=True)
            
        st.markdown("#### C. Residual Error Map")
        st.pyplot(plot_full(err_img, "Difference (Anomalies highlighted)", 'magma', vmax=0.3), use_container_width=True)