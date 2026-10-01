import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import io
import soundfile as sf
import os
import tempfile

# Try importing birdnetlib for real AI inference when deployed
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
    page_title="AcoustiSpec Pro | BirdNET AI Bioacoustics Workstation",
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
# CORE AUDIO INGESTION & BIRDNET INFERENCE PIPELINE
# -----------------------------------------------------------------------------

@st.cache_data
def load_audio_fast(file_bytes):
    """
    Fast cached audio ingestion using Native Sample Rate to avoid CPU resampling bottlenecks.
    """
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y, sr

def calculate_audio_telemetry(y, sr):
    """
    Computes audio quality and ecoacoustic metrics.
    """
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
    """
    Executes true BirdNET model inference via birdnetlib.
    Extracts raw species predictions directly from BirdNET neural network weights.
    """
    if not BIRDNET_AVAILABLE:
        st.error("⚠️ `birdnetlib` library is not installed in the environment. Please ensure `birdnetlib` is listed in your `requirements.txt`.")
        return pd.DataFrame()

    # Save audio bytes to a temporary file on disk as required by birdnetlib
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # Load BirdNET Model Weights
        model = BirdNETModel()
        
        # Configure Recording
        recording = Recording(
            model,
            tmp_path,
            lat=lat if (lat is not None and lat != 0.0) else None,
            lon=lon if (lon is not None and lon != 0.0) else None,
            min_conf=min_conf
        )
        
        # Run neural network extraction
        recording.extract_detections()
        raw_detections = recording.detections

        if not raw_detections:
            return pd.DataFrame()

        # Format detections into clean DataFrame
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
                "Acoustic Model Engine": "BirdNET-Analyzer V2.4 (Cornell Lab)"
            })

        return pd.DataFrame(formatted)

    except Exception as e:
        st.error(f"Error during BirdNET inference: {str(e)}")
        return pd.DataFrame()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & MODEL CONFIGURATION
# -----------------------------------------------------------------------------

st.sidebar.markdown("### 🦅 AcoustiSpec AI Workstation")
st.sidebar.caption("Enterprise Bioacoustics Analysis & Avian Monitoring")
st.sidebar.divider()

st.sidebar.markdown("**🤖 AI Classifier Engine**")
st.sidebar.info("Using **BirdNET-Analyzer V2.4** (Cornell Lab of Ornithology)")

# Audio Upload
st.sidebar.subheader("📁 Audio Source Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Upload Field Audio (WAV, MP3, FLAC, OGG)",
    type=["wav", "mp3", "flac", "ogg"]
)

# Detection Hyperparameters
st.sidebar.subheader("⚙️ Detection Hyperparameters")
conf_threshold = st.sidebar.slider("Minimum Confidence Threshold (%)", min_value=10, max_value=95, value=25, step=5) / 100.0
spectrogram_cmap = st.sidebar.selectbox("Spectrogram Colormap", ["magma", "viridis", "inferno", "plasma", "cividis"], index=0)

# Optional Geographic Filter Inputs
st.sidebar.subheader("📍 Geographic Location Filter (Optional)")
use_geo = st.sidebar.checkbox("Enable Coordinates Filter", value=False)
lat_input = 0.0
lon_input = 0.0
if use_geo:
    lat_input = st.sidebar.number_input("Latitude", value=-18.8792, format="%.4f")
    lon_input = st.sidebar.number_input("Longitude", value=47.5079, format="%.4f")
    st.sidebar.caption("Filters BirdNET predictions to species native to specified coordinates.")

# -----------------------------------------------------------------------------
# MAIN APPLICATION WORKSPACE
# -----------------------------------------------------------------------------

st.markdown('<div class="main-header">🎙️ AcoustiSpec Pro — Bioacoustics AI Workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated wildlife vocalization identification, acoustic feature telemetry, and high-density spectrogram analysis</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    _, file_ext = os.path.splitext(filename)
    if not file_ext:
        file_ext = ".wav"
    
    with st.spinner("⚡ Ingesting audio stream and computing acoustic telemetry..."):
        y, sr = load_audio_fast(file_bytes)
        telemetry = calculate_audio_telemetry(y, sr)
        
    with st.spinner("🧠 Executing BirdNET-Analyzer V2.4 inference..."):
        df_detections = run_birdnet_inference(
            file_bytes=file_bytes,
            file_suffix=file_ext,
            min_conf=conf_threshold,
            lat=lat_input if use_geo else None,
            lon=lon_input if use_geo else None
        )
        
    # -------------------------------------------------------------------------
    # TOP TELEMETRY METRICS BAR
    # -------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("⏱️ Audio Duration", f"{telemetry['duration']:.2f} s")
    with col2:
        st.metric("🎚️ Sample Rate", f"{telemetry['sample_rate']} Hz")
    with col3:
        st.metric("📡 Signal SNR", f"{telemetry['snr_db']:.1f} dB")
    with col4:
        st.metric("🦅 Total Detections", len(df_detections))
    with col5:
        unique_species = df_detections["Common Name"].nunique() if not df_detections.empty else 0
        st.metric("🌿 Species Richness", f"{unique_species} Species")
        
    st.divider()

    # -------------------------------------------------------------------------
    # MAIN WORKSPACE TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Detection Log & Species Identification",
        "🎵 High-Res Waveform & Spectrogram Viewer",
        "📊 Soundscape Analytics & Abundance",
        "📚 AI Architecture & Taxonomy Reference"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: DETECTION LOG & SPECIES IDENTIFICATION
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("🦅 BirdNET Species Identification Log")
        
        if not df_detections.empty:
            f_col1, f_col2 = st.columns([3, 1])
            with f_col1:
                species_filter = st.multiselect(
                    "Filter by Detected Species:",
                    options=list(df_detections["Common Name"].unique()),
                    default=list(df_detections["Common Name"].unique())
                )
            with f_col2:
                sort_order = st.selectbox("Sort By:", ["Timestamp", "Confidence (%)"])
                
            filtered_df = df_detections[df_detections["Common Name"].isin(species_filter)].copy()
            
            if sort_order == "Confidence (%)":
                filtered_df = filtered_df.sort_values(by="Confidence (%)", ascending=False)

            # Display Data Table directly returned by BirdNET
            st.dataframe(
                filtered_df[[
                    "Timestamp", "Common Name", "Scientific Name", 
                    "Confidence (%)", "Acoustic Model Engine"
                ]],
                use_container_width=True,
                hide_index=True
            )
            
            # Export Section
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Detection Report (CSV)",
                    data=csv_data,
                    file_name=f"BirdNET_Report_{filename}.csv",
                    mime="text/csv"
                )
            with exp_col2:
                json_data = filtered_df.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 Export Detections (JSON)",
                    data=json_data,
                    file_name=f"BirdNET_Metadata_{filename}.json",
                    mime="application/json"
                )
                
            st.divider()
            
            # Visual Charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.subheader("📊 Species Relative Abundance")
                species_counts = filtered_df["Common Name"].value_counts().reset_index()
                species_counts.columns = ["Species", "Detections"]
                fig_pie = px.pie(
                    species_counts, values="Detections", names="Species", hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with chart_col2:
                st.subheader("⏱️ Temporal Detection Timeline")
                fig_scatter = px.scatter(
                    filtered_df,
                    x="Start Time (s)",
                    y="Common Name",
                    size="Confidence (%)",
                    color="Common Name",
                    hover_data=["Scientific Name", "Timestamp", "Confidence (%)"],
                    labels={"Start Time (s)": "Time (Seconds)", "Common Name": "Detected Species"}
                )
                fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("⚠️ No vocalizations recognized by BirdNET above the chosen confidence threshold. Try adjusting the Minimum Confidence Threshold slider in the sidebar.")

    # -------------------------------------------------------------------------
    # TAB 2: WAVEFORM & SPECTROGRAM VIEWER
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🎚️ Interactive Time-Frequency Inspector")
        
        duration_total = telemetry["duration"]
        start_sec, end_sec = st.slider(
            "Select Time Slice to Inspect (Seconds):",
            min_value=0.0,
            max_value=float(duration_total),
            value=(0.0, min(10.0, float(duration_total))),
            step=0.5
        )
        
        start_samp = int(start_sec * sr)
        end_samp = int(end_sec * sr)
        y_slice = y[start_samp:end_samp]
        
        st.write(f"🔊 **Playing Audio Segment ({start_sec:.1f}s - {end_sec:.1f}s):**")
        buf = io.BytesIO()
        sf.write(buf, y_slice, sr, format='WAV')
        st.audio(buf.getvalue(), format="audio/wav")
        
        # Generate Dual Figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        time_axis = np.linspace(start_sec, end_sec, len(y_slice))
        ax1.plot(time_axis, y_slice, color="#0284C7", alpha=0.85, linewidth=1)
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"Audio Waveform Segment ({start_sec:.1f}s - {end_sec:.1f}s)")
        ax1.grid(True, linestyle="--", alpha=0.4)
        
        S = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=128, fmax=min(12000, sr//2))
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        spec_time_axis = np.linspace(start_sec, end_sec, S_dB.shape[1])
        img = librosa.display.specshow(
            S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=min(12000, sr//2),
            ax=ax2, cmap=spectrogram_cmap, x_coords=spec_time_axis
        )
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (Seconds)")
        ax2.set="Mel-Spectrogram Energy Distribution"
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # TAB 3: SOUNDSCAPE ANALYTICS
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("📊 Acoustic Feature Telemetry & Metrics")
        
        s1, s2, s3 = st.columns(3)
        with s1:
            st.info(f"**Root Mean Square (RMS):** `{telemetry['rms']:.5f}`")
        with s2:
            st.info(f"**Spectral Centroid:** `{telemetry['mean_centroid_hz']:.1f} Hz`")
        with s3:
            st.info(f"**Estimated Signal SNR:** `{telemetry['snr_db']:.2f} dB`")
            
        st.markdown("""
        ### 🌿 Bioacoustic Soundscape Indices
        - **Acoustic Complexity Index (ACI):** Measures the fluctuation in amplitude in spectrogram frequency bins.
        - **Bioacoustic Index (BI):** Area under the sound level curve across avian frequency bands (2 kHz - 8 kHz).
        - **Normalized Difference Soundscape Index (NDSI):** Ratio of biophony (biological sound) to anthrophony (human noise).
        """)

    # -------------------------------------------------------------------------
    # TAB 4: AI ARCHITECTURE REFERENCE
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("📚 BirdNET-Analyzer V2.4 Architecture")
        st.markdown("""
        ### 🦅 BirdNET Neural Network Overview
        - **Developer:** Cornell Lab of Ornithology & Chemnitz University of Technology.
        - **Taxonomy Scope:** 6,000+ global avian species.
        - **Model Inputs:** 3.0-second sliding audio windows converted to mel-spectrograms.
        - **Inference Pipeline:** Executes convolutional feature extraction (`birdnetlib`) and returns calibrated confidence logits.
        """)

else:
    st.info("👈 Upload an audio file in the sidebar to begin automated BirdNET AI species analysis!")
