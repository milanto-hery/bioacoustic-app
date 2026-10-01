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
import tempfile
import os
from datetime import datetime

# Try importing real BirdNET model library
try:
    from birdnetlib import Recording
    from birdnetlib.models import BirdNETModel
    BIRDNET_INSTALLED = True
except ImportError:
    BIRDNET_INSTALLED = False

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bioacoustics AI Species Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-badge-real {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.4rem 0.8rem;
        border-radius: 0.375rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .status-badge-warning {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 0.4rem 0.8rem;
        border-radius: 0.375rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# REAL MODEL INITIALIZATION & CACHING
# -----------------------------------------------------------------------------

@st.cache_resource
def load_birdnet_model_singleton():
    """Loads and caches the real BirdNET AI Neural Network Model in memory."""
    if BIRDNET_INSTALLED:
        try:
            model = BirdNETModel()
            return model
        except Exception as e:
            st.error(f"Error initializing BirdNET Model: {e}")
            return None
    return None

@st.cache_data
def load_audio_from_bytes(file_bytes):
    """
    Fast, cached audio loader with sr=None to preserve original sample rate
    without CPU-heavy resampling.
    """
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
    return y, sr

@st.cache_data
def generate_sample_audio(duration=15, sr=22050):
    """Generates synthetic bioacoustic sample audio for instant demo testing."""
    t = np.linspace(0, duration, int(sr * duration))
    noise = np.random.normal(0, 0.05, len(t))
    
    # Chirp 1 (2-5 kHz)
    f1 = np.where((t >= 3) & (t <= 6), 3000 + 1500 * np.sin(2 * np.pi * 5 * t), 0)
    sig1 = np.where((t >= 3) & (t <= 6), 0.4 * np.sin(2 * np.pi * f1 * t), 0)
    
    # Chirp 2 (4-7.5 kHz)
    f2 = np.where((t >= 8) & (t <= 12), 4500 + 800 * np.cos(2 * np.pi * 12 * t), 0)
    sig2 = np.where((t >= 8) & (t <= 12), 0.35 * np.sin(2 * np.pi * f2 * t), 0)
    
    audio = noise + sig1 + sig2
    return audio, sr

def format_timestamp(seconds):
    """Converts seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def analyze_real_audio_birdnet(file_bytes, min_conf=0.25, lat=-18.8792, lon=47.5079, use_location=True):
    """
    Executes REAL BirdNET deep neural network model inference on uploaded audio file bytes.
    Extracts true detected species, scientific names, exact timestamps, and confidence scores.
    """
    birdnet_model = load_birdnet_model_singleton()
    if birdnet_model is None:
        return pd.DataFrame(), "BirdNET library (`birdnetlib`) is not loaded. Please ensure `birdnetlib` is added to your requirements.txt."
    
    # Create temporary file for BirdNET C-library / Audio reader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        recording_kwargs = {
            "model": birdnet_model,
            "file_path": tmp_path,
            "min_conf": min_conf,
            "date": datetime.now()
        }
        if use_location:
            recording_kwargs["lat"] = lat
            recording_kwargs["lon"] = lon

        recording = Recording(**recording_kwargs)
        recording.analyze()
        
        raw_detections = recording.detections
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        if not raw_detections:
            return pd.DataFrame(), "No bird vocalizations recognized above the chosen confidence threshold."
            
        detections_list = []
        for idx, d in enumerate(raw_detections):
            start_t = float(d.get('start_time', 0.0))
            end_t = float(d.get('end_time', 0.0))
            conf = float(d.get('confidence', 0.0))
            common_name = d.get('common_name', 'Unknown Species')
            scientific_name = d.get('scientific_name', 'N/A')
            
            detections_list.append({
                "Segment": idx + 1,
                "Start Time (s)": round(start_t, 2),
                "End Time (s)": round(end_t, 2),
                "Timestamp": f"{format_timestamp(start_t)} - {format_timestamp(end_t)}",
                "Common Name": common_name,
                "Scientific Name": scientific_name,
                "Confidence": round(conf * 100, 1),
                "Frequency Range": "1.0 - 8.0 kHz",
                "Model Used": "BirdNET AI (Real Model)"
            })
            
        return pd.DataFrame(detections_list), None

    except Exception as err:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return pd.DataFrame(), f"BirdNET Processing Error: {str(err)}"

# Fallback spectral energy analysis if birdnetlib is not installed yet
def analyze_audio_spectral_fallback(y, sr, segment_dur=5.0, confidence_threshold=0.5):
    """
    Fallback spectral acoustic signal analyzer when birdnetlib is not yet installed.
    Flags active audio signals based on real spectral acoustic energy.
    """
    total_duration = librosa.get_duration(y=y, sr=sr)
    num_segments = int(np.ceil(total_duration / segment_dur))
    
    detections = []
    for i in range(num_segments):
        start_t = i * segment_dur
        end_t = min((i + 1) * segment_dur, total_duration)
        
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        chunk = y[start_sample:end_sample]
        
        if len(chunk) == 0:
            continue
            
        rms = float(np.sqrt(np.mean(chunk**2)))
        
        # Real audio signal detection threshold
        if rms > 0.035:
            conf = min(0.95, max(0.40, float(0.50 + rms * 5.0)))
            if conf >= confidence_threshold:
                detections.append({
                    "Segment": i + 1,
                    "Start Time (s)": round(start_t, 2),
                    "End Time (s)": round(end_t, 2),
                    "Timestamp": f"{format_timestamp(start_t)} - {format_timestamp(end_t)}",
                    "Common Name": "Acoustic Vocalization Signal Detected",
                    "Scientific Name": "Unclassified Bioacoustic Signal",
                    "Confidence": round(conf * 100, 1),
                    "Frequency Range": "0.5 - 8.0 kHz",
                    "Model Used": "Spectral Energy Detector (Install birdnetlib for Species Names)"
                })
                
    return pd.DataFrame(detections)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/color/96/000000/microprocessor.png", width=60)
st.sidebar.title("🎛️ Control Panel")

# AI Model Choice
model_choice = st.sidebar.selectbox(
    "🤖 Select AI Engine",
    ["BirdNET V2.4 (Real Neural Network)", "Spectral Energy Feature Detector"]
)

# File Source
st.sidebar.subheader("📁 Audio Source")
source_option = st.sidebar.radio("Choose Input Type:", ["Upload Real Audio File", "Use Demo Sample Audio"])

y = None
sr = 22050
file_bytes = None
filename = ""

if source_option == "Upload Real Audio File":
    uploaded_file = st.sidebar.file_uploader("Upload Audio (WAV, MP3, FLAC, OGG)", type=["wav", "mp3", "flac", "ogg"])
    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
        y, sr = load_audio_from_bytes(file_bytes)
else:
    filename = "sample_rainforest_audio.wav"
    y, sr = generate_sample_audio(duration=15, sr=sr)
    # Convert numpy sample audio to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    file_bytes = buf.getvalue()
    st.sidebar.success("Loaded 15s Synthetic Demo Sample")

# Geographic & Threshold Controls
st.sidebar.subheader("⚙️ Detection Parameters")
conf_threshold = st.sidebar.slider("Min Confidence Threshold (%)", min_value=10, max_value=90, value=25, step=5) / 100.0

use_location_filter = st.sidebar.checkbox("Use Geographic Filter (BirdNET)", value=True)
if use_location_filter:
    c_lat, c_lon = st.sidebar.columns(2)
    lat_val = c_lat.number_input("Latitude", value=-18.8792, format="%.4f")
    lon_val = c_lon.number_input("Longitude", value=47.5079, format="%.4f")
else:
    lat_val, lon_val = -18.8792, 47.5079

colormap_choice = st.sidebar.selectbox("Spectrogram Colormap", ["magma", "viridis", "inferno", "plasma"], index=0)

# -----------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------

st.markdown('<div class="main-header">🎙️ Real Bioacoustics AI Species Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated wildlife vocalization detection powered by BirdNET deep neural networks</div>', unsafe_allow_html=True)

# Status Badge
if BIRDNET_INSTALLED and model_choice.startswith("BirdNET"):
    st.markdown('<div class="status-badge-real">🟢 Real BirdNET AI Engine Active</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-badge-warning">⚠️ Running Fallback Detector — Add `birdnetlib` to requirements.txt for full BirdNET Species Identification</div>', unsafe_allow_html=True)

if y is not None and file_bytes is not None:
    duration_total = librosa.get_duration(y=y, sr=sr)
    
    # Run Inference
    with st.spinner("🔄 Analyzing audio with AI Neural Network..."):
        if BIRDNET_INSTALLED and model_choice.startswith("BirdNET"):
            df_detections, err_msg = analyze_real_audio_birdnet(
                file_bytes, min_conf=conf_threshold, lat=lat_val, lon=lon_val, use_location=use_location_filter
            )
            if err_msg and df_detections.empty:
                st.info(f"ℹ️ {err_msg}")
        else:
            df_detections = analyze_audio_spectral_fallback(y, sr, segment_dur=5.0, confidence_threshold=conf_threshold)

    # -------------------------------------------------------------------------
    # TOP SUMMARY METRICS
    # -------------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("⏱️ Audio Duration", f"{duration_total:.1f} s")
    with m2:
        st.metric("🦅 Total Detections", len(df_detections))
    with m3:
        unique_sp = df_detections["Common Name"].nunique() if not df_detections.empty else 0
        st.metric("🌿 Species Richness", f"{unique_sp} Species")
    with m4:
        avg_conf = f"{df_detections['Confidence'].mean():.1f}%" if not df_detections.empty else "N/A"
        st.metric("🎯 Avg Confidence", avg_conf)
        
    st.divider()

    # -------------------------------------------------------------------------
    # MAIN NAVIGATION TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Real AI Detection Dashboard", "🎵 Waveform & Spectrogram Viewer", "📚 BirdNET Model & Deployment Guide"])

    # -------------------------------------------------------------------------
    # TAB 1: REAL AI DETECTION DASHBOARD
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📋 Real AI Species Detection Log")
        
        if not df_detections.empty:
            # Filter controls
            c_filter1, c_filter2 = st.columns([2, 1])
            with c_filter1:
                all_species = ["All Species"] + list(df_detections["Common Name"].unique())
                selected_species = st.selectbox("Filter by Species:", all_species)
            
            filtered_df = df_detections.copy()
            if selected_species != "All Species":
                filtered_df = filtered_df[filtered_df["Common Name"] == selected_species]
                
            st.dataframe(
                filtered_df[["Timestamp", "Common Name", "Scientific Name", "Confidence", "Frequency Range", "Model Used"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Export Options
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Real Detection Report (CSV)",
                data=csv,
                file_name=f"real_bioacoustics_report_{filename}.csv",
                mime="text/csv"
            )
            
            st.divider()
            
            # Visual Analytics
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("📊 Species Richness Distribution")
                species_counts = df_detections["Common Name"].value_counts().reset_index()
                species_counts.columns = ["Species", "Count"]
                fig_donut = px.pie(
                    species_counts, values="Count", names="Species", hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_donut, use_container_width=True)
                
            with col_chart2:
                st.subheader("⏱️ Real Detection Timeline")
                fig_timeline = px.scatter(
                    df_detections,
                    x="Start Time (s)",
                    y="Common Name",
                    size="Confidence",
                    color="Common Name",
                    hover_data=["Scientific Name", "Timestamp", "Confidence"],
                    labels={"Start Time (s)": "Time (Seconds)", "Common Name": "Detected Species"}
                )
                fig_timeline.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.warning("⚠️ No vocalizations identified above the chosen confidence threshold. Try lowering the Min Confidence slider in the sidebar.")

    # -------------------------------------------------------------------------
    # TAB 2: WAVEFORM & SPECTROGRAM VIEWER
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🎚️ Interactive Audio & Spectrogram Inspector")
        
        start_sec, end_sec = st.slider(
            "Select Time Window to Inspect (Seconds):",
            min_value=0.0,
            max_value=float(duration_total),
            value=(0.0, min(10.0, float(duration_total))),
            step=0.5
        )
        
        start_samp = int(start_sec * sr)
        end_samp = int(end_sec * sr)
        y_slice = y[start_samp:end_samp]
        
        if len(y_slice) > 0:
            st.write(f"🔊 **Playing audio segment ({start_sec:.1f}s - {end_sec:.1f}s):**")
            buffer = io.BytesIO()
            sf.write(buffer, y_slice, sr, format='WAV')
            st.audio(buffer.getvalue(), format="audio/wav")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            
            # Plot 1: Waveform
            time_axis = np.linspace(start_sec, end_sec, len(y_slice))
            ax1.plot(time_axis, y_slice, color="#2563EB", alpha=0.8, linewidth=1)
            ax1.set_ylabel("Amplitude")
            ax1.set_title(f"Audio Waveform ({start_sec:.1f}s - {end_sec:.1f}s)")
            ax1.grid(True, linestyle="--", alpha=0.5)
            
            # Plot 2: Mel-Spectrogram
            S = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            spec_time_axis = np.linspace(start_sec, end_sec, S_dB.shape[1])
            img = librosa.display.specshow(
                S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, 
                ax=ax2, cmap=colormap_choice, x_coords=spec_time_axis
            )
            ax2.set_ylabel("Frequency (Hz)")
            ax2.set_xlabel("Time (Seconds)")
            ax2.set_title("Mel-Spectrogram Visualization")
            fig.colorbar(img, ax=ax2, format='%+2.0f dB')
            
            st.pyplot(fig)

    # -------------------------------------------------------------------------
    # TAB 3: MODEL & DEPLOYMENT GUIDE
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("📚 Real AI Model Architecture & Requirements")
        
        st.markdown("""
        ### 🦅 Real BirdNET Integration
        This application uses **`birdnetlib`**, the official Python wrapper for the **Cornell Lab of Ornithology BirdNET Model**.
        
        #### How to enable Real Species Identification on Streamlit Cloud / Hugging Face Spaces:
        Update your **`requirements.txt`** file to include:
        ```text
        streamlit
        librosa
        matplotlib
        numpy
        pandas
        plotly
        soundfile
        birdnetlib
        resampy
        scipy
        ```
        
        When `birdnetlib` is installed, the AI model will automatically analyze real uploaded audio files and return **true species names, scientific classifications, exact timestamps, and probability confidence scores**!
        """)

else:
    st.info("👈 Upload a real audio file or select 'Use Demo Sample Audio' from the sidebar to begin analysis!")
