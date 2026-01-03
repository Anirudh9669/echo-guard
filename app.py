import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Echo-Guard Diagnostics",
    page_icon="🔊",
    layout="centered"
)

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
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    model = ConvolutionalAutoencoder()
    
    # 1. Load Neural Network (Prioritize v10)
    model_name = 'echo_guard_v10_generalist.pth'
    if not os.path.exists(model_name):
        model_name = 'echo_guard_v9.pth' # Fallback

    try:
        model.load_state_dict(torch.load(model_name, map_location=device))
        model.eval()
    except FileNotFoundError:
        return None, None, f"❌ Missing Model: Could not find '{model_name}'"
    except Exception as e:
        return None, None, f"❌ Neural Net Error: {e}"

    # 2. Load Isolation Forest (Prioritize v10)
    iso_name = 'iso_forest_v10_generalist.joblib'
    if not os.path.exists(iso_name):
        iso_name = 'iso_forest_v9.joblib' # Fallback

    try:
        iso_forest = joblib.load(iso_name)
    except FileNotFoundError:
        return None, None, f"❌ Missing Judge: Could not find '{iso_name}'"
    except Exception as e:
        return None, None, f"❌ IsoForest Error: {e}"
        
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
        return None, None, None, None, f"Error processing audio: {e}"

# ==========================================
# 4. MAIN APP INTERFACE
# ==========================================

# -- Sidebar --
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])
    
    st.markdown("---")
    st.subheader("Calibration")
    # Optimal Threshold from v10 Evaluation
    THRESHOLD = st.slider("Anomaly Threshold", 0.30, 0.60, 0.4718, 0.01, 
                          help="Fine-tune sensitivity. If getting false alarms, increase this.")

# -- Main Content --
st.title("🔊 Echo-Guard")
st.subheader("Industrial Acoustic Anomaly Detection")
st.write("Upload a machine sound file to analyze its health status.")

# Load Models
model, iso_forest, err_msg = load_models()

if err_msg:
    st.error(err_msg)
    st.info("💡 Tip: Make sure you downloaded the .pth and .joblib files from Google Drive to this folder!")
    st.stop()

if uploaded_file is None:
    st.info("👈 Please upload a .wav file from the sidebar.")
else:
    st.divider()
    
    # --- ANALYSIS ---
    with st.spinner("Analyzing signal patterns..."):
        # Save temp file
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        y, sr, spec_img, tensor, proc_err = process_audio("temp.wav")
        
        if proc_err:
            st.error(proc_err)
            st.stop()

        # AI Inference
        device = torch.device("cpu")
        input_tensor = tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            latent_vec, reconstruction = model(input_tensor)
            latent_vec_flat = latent_vec.cpu().numpy().flatten()
            
            # We calculate MSE just for display, but DO NOT add it to features for scoring
            # because the saved Isolation Forest expects only 256 features.
            mse_error = torch.mean((input_tensor - reconstruction) ** 2).item()
            
        # Scoring (Using only the 256 latent features)
        score = -iso_forest.score_samples(latent_vec_flat.reshape(1, -1))[0]
        is_anomaly = score > THRESHOLD

    # --- RESULTS DISPLAY ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_anomaly:
            st.error(f"### 🚨 ANOMALY DETECTED")
            st.write(f"**Assessment:** The machine sound deviates significantly from normal patterns.")
        else:
            st.success(f"### ✅ SYSTEM NORMAL")
            st.write(f"**Assessment:** The machine sound matches the healthy baseline.")
            
    with col2:
        st.metric("Anomaly Score", f"{score:.4f}", delta=f"Limit: {THRESHOLD}", delta_color="inverse")

    # --- VISUALIZATION (Expandable) ---
    st.divider()
    with st.expander("🔎 View Technical Analysis (Spectrograms)"):
        st.write("Compare the **Input** (what we heard) with the **Reconstruction** (what a normal machine *should* sound like). Differences appear in the **Residual Map**.")
        st.audio("temp.wav", format="audio/wav")
        
        # Prepare Visuals
        in_img = tensor.squeeze().numpy()
        out_img = reconstruction.squeeze().numpy()
        err_img = np.abs(in_img - out_img)
        
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        
        # 1. Input
        im1 = axes[0].imshow(in_img, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title("Input Spectrogram (Actual Sound)")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # 2. Output
        im2 = axes[1].imshow(out_img, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title("AI Reconstruction (Normal Baseline)")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # 3. Residual
        im3 = axes[2].imshow(err_img, aspect='auto', origin='lower', cmap='magma')
        axes[2].set_title("Residual Map (Difference)")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        st.pyplot(fig)