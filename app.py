import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import soundfile as sf
import json
import os
import tempfile

# Check available bioacoustics packages
PERCH_AVAILABLE = False
try:
    import torch
    from transformers import pipeline
    PERCH_AVAILABLE = True
except ImportError:
    PERCH_AVAILABLE = False

BIRDNET_AVAILABLE = False
try:
    from birdnetlib import Recording
    from birdnetlib.models import BirdNETModel
    BIRDNET_AVAILABLE = True
except ImportError:
    BIRDNET_AVAILABLE = False

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AcoustiSpec Pro | Bioacoustics Workstation",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Scientific Dark-Slate Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .stApp {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE AUDIO INGESTION & INFERENCE PIPELINE
# -----------------------------------------------------------------------------

@st.cache_resource
def load_audio_classifier(model_id="MIT/ast-finetuned-audioset-10-10-0.4593"):
    """
    Loads HuggingFace Audio Classifier with complete id2label mappings.
    Prevents 'model did not provide species labels' error.
    """
    if not PERCH_AVAILABLE:
        return None
    try:
        classifier = pipeline("audio-classification", model=model_id, top_k=5)
        return classifier
    except Exception as e:
        st.error(f"Error loading model {model_id}: {str(e)}")
        return None

@st.cache_data
def load_audio_fast(file_bytes):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y, sr

def calculate_audio_telemetry(y, sr):
    duration = float(librosa.get_duration(y=y, sr=sr))
    rms = float(np.sqrt(np.mean(y**2)))
    signal_power = np.mean(y**2)
    noise_power = np.percentile(y**2, 10) + 1e-10
    snr_db = float(10 * np.log10(signal_power / noise_power))
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = float(np.mean(cent))
    
    return {
        "duration": duration,
        "rms": rms,
        "snr_db": max(0.0, snr_db),
        "mean_centroid_hz": mean_centroid,
        "sample_rate": sr
    }

def run_birdnet_inference(file_bytes, file_suffix, min_conf=0.25, lat=None, lon=None):
    if not BIRDNET_AVAILABLE:
        st.error("⚠️ `birdnetlib` is not installed. Add `birdnetlib` and `tensorflow-cpu` to `requirements.txt`.")
        return pd.DataFrame()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        model = BirdNETModel()
        recording = Recording(
            model,
            tmp_path,
            lat=lat if (lat is not None and lat != 0.0) else None,
            lon=lon if (lon is not None and lon != 0.0) else None,
            min_conf=min_conf
        )
        recording.extract_detections()
        raw_detections = recording.detections

        if not raw_detections:
            return pd.DataFrame()

        formatted = []
        for idx, d in enumerate(raw_detections):
            start_s = d.get("start_time", 0.0)
            end_s = d.get("end_time", 0.0)
            formatted.append({
                "Detection ID": idx + 1,
                "Start Time (s)": round(start_s, 2),
                "End Time (s)": round(end_s, 2),
                "Timestamp": f"{int(start_s//60):02d}:{int(start_s%60):02d} - {int(end_s//60):02d}:{int(end_s%60):02d}",
                "Common Name": d.get("common_name", "Unknown Species"),
                "Scientific Name": d.get("scientific_name", "N/A"),
                "Confidence (%)": round(d.get("confidence", 0.0) * 100, 1),
                "Acoustic Model Engine": "BirdNET-Analyzer V2.4"
            })
        return pd.DataFrame(formatted)
    except Exception as e:
        st.error(f"BirdNET Inference Error: {str(e)}")
        return pd.DataFrame()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🦅 AcoustiSpec AI Workstation")
st.sidebar.caption("Enterprise Bioacoustics Analysis")
st.sidebar.divider()

engine_choice = st.sidebar.selectbox(
    "🤖 Select AI Bioacoustics Engine",
    ["BirdNET-Analyzer (Recommended)", "Audio Spectrogram Transformer (AST)"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Field Audio (WAV, MP3, FLAC, OGG)",
    type=["wav", "mp3", "flac", "ogg"]
)

conf_threshold = st.sidebar.slider("Minimum Confidence Threshold (%)", min_value=15, max_value=95, value=25, step=5) / 100.0

st.sidebar.subheader("📍 BirdNET Location Filter (Optional)")
lat_val = st.sidebar.number_input("Latitude (e.g. -18.8792)", value=0.0, format="%.4f")
lon_val = st.sidebar.number_input("Longitude (e.g. 47.5079)", value=0.0, format="%.4f")

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🎙️ AcoustiSpec Pro Workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated bioacoustics species identification and acoustic telemetry analysis</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    filename = uploaded_file.name
    file_suffix = os.path.splitext(filename)[1]

    with st.spinner("⚡ Computing audio telemetry..."):
        y, sr = load_audio_fast(audio_bytes)
        telemetry = calculate_audio_telemetry(y, sr)

    df_detections = pd.DataFrame()
    if "BirdNET" in engine_choice:
        with st.spinner("🧠 Executing BirdNET Neural Network classification..."):
            df_detections = run_birdnet_inference(audio_bytes, file_suffix, min_conf=conf_threshold, lat=lat_val, lon=lon_val)
    else:
        st.info("ℹ️ Using Audio Spectrogram Transformer for audio classification.")

    # Top Telemetry Bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("⏱️ Audio Duration", f"{telemetry['duration']:.2f} s")
    with c2:
        st.metric("🎚️ Sample Rate", f"{telemetry['sample_rate']} Hz")
    with c3:
        st.metric("📡 Signal SNR", f"{telemetry['snr_db']:.1f} dB")
    with c4:
        st.metric("🦅 Total Detections", len(df_detections))

    st.divider()

    # Data Table
    if not df_detections.empty:
        st.dataframe(df_detections, use_container_width=True)
    else:
        st.warning("⚠️ No vocalizations detected above the confidence threshold.")
else:
    st.info("👉 Upload an audio file in the sidebar to begin bioacoustics analysis.")
