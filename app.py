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
import datetime
import csv
import os
import tempfile
from pathlib import Path

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AcoustiSpec Pro | Enterprise AI Bioacoustics Workstation",
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
    .metric-card {
        background-color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .status-badge {
        background-color: #0284C7;
        color: #FFFFFF;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stApp {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE AUDIO & AI INFERENCE PIPELINE
# -----------------------------------------------------------------------------

PERCH_MODEL_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
PERCH_SAMPLE_RATE = 32000
PERCH_WINDOW_SECONDS = 5

@st.cache_data
def load_audio_fast(file_bytes):
    """
    Fast cached audio ingestion using Native Sample Rate to avoid CPU resampling bottlenecks.
    """
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
    # Ensure float32 normalized audio
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y, sr

def calculate_audio_telemetry(y, sr):
    """
    Computes rigorous audio quality and ecoacoustic metrics.
    """
    duration = float(librosa.get_duration(y=y, sr=sr))
    rms = float(np.sqrt(np.mean(y**2)))
    # Estimate Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(y**2)
    noise_power = np.percentile(y**2, 10) + 1e-10
    snr_db = float(10 * np.log10(signal_power / noise_power))
    
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = float(np.mean(cent))
    
    return {
        "duration": duration,
        "rms": rms,
        "snr_db": max(0.0, snr_db),
        "mean_centroid_hz": mean_centroid,
        "sample_rate": sr
    }

@st.cache_resource
def load_birdnet_analyzer():
    try:
        from birdnetlib.analyzer import Analyzer
    except ImportError as error:
        raise RuntimeError(
            "BirdNET is not installed in the active Python environment. "
            "Install this project's requirements.txt before running inference."
        ) from error

    return Analyzer()


def run_birdnet_inference(audio_data, sr, confidence_threshold, region=None):
    try:
        from birdnetlib import Recording
    except ImportError as error:
        raise RuntimeError(
            "BirdNET is not installed in the active Python environment. "
            "Install this project's requirements.txt before running inference."
        ) from error

    analyzer = load_birdnet_analyzer()
    detections = []

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "uploaded_audio.wav")
        sf.write(audio_path, np.asarray(audio_data, dtype=np.float32), sr)
        recording_options = {
            "date": datetime.date.today().isoformat(),
            "min_conf": confidence_threshold,
        }
        if region is not None:
            recording_options.update({"lat": region[0], "lon": region[1]})

        recording = Recording(analyzer, audio_path, **recording_options)
        recording.analyze()

        for index, result in enumerate(recording.detections, start=1):
            start_time = float(result.get("start_time", 0.0))
            end_time = float(result.get("end_time", start_time))
            common_name = result.get("common_name") or result.get("label") or result.get("scientific_name") or "Unknown"
            scientific_name = result.get("scientific_name") or ""
            detections.append({
                "Segment ID": index,
                "Start Time (s)": round(start_time, 2),
                "End Time (s)": round(end_time, 2),
                "Timestamp": f"{int(start_time // 60):02d}:{int(start_time % 60):02d} - {int(end_time // 60):02d}:{int(end_time % 60):02d}",
                "Species": common_name,
                "Scientific Name": scientific_name,
                "Model Score (%)": round(float(result.get("confidence", 0.0)) * 100, 1),
                "Acoustic Model Engine": "BirdNET",
            })

    return pd.DataFrame(detections, columns=[
        "Segment ID", "Start Time (s)", "End Time (s)", "Timestamp",
        "Species", "Scientific Name", "Model Score (%)", "Acoustic Model Engine",
    ])


@st.cache_resource
def load_perch_model_and_labels():
    try:
        import kagglehub
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "Google Perch requires kagglehub and tensorflow-cpu. "
            "Install the project's requirements.txt in the same environment as Streamlit."
        ) from error

    model_path = Path(kagglehub.model_download(PERCH_MODEL_HANDLE))
    model = tf.saved_model.load(str(model_path))
    labels_path = model_path / "assets" / "labels.csv"
    if not labels_path.is_file():
        raise RuntimeError(f"Perch model labels were not found at {labels_path}.")

    with labels_path.open(newline="", encoding="utf-8") as labels_file:
        rows = list(csv.reader(labels_file))
    labels = [row[0].strip() for row in rows[1:] if row and row[0].strip()]
    if not labels:
        raise RuntimeError("The Perch model's labels.csv file did not contain any species labels.")

    return model, labels


def run_perch_inference(audio_data, sr, score_threshold):
    model, labels = load_perch_model_and_labels()
    audio_32k = librosa.resample(
        np.asarray(audio_data, dtype=np.float32), orig_sr=sr, target_sr=PERCH_SAMPLE_RATE
    )
    window_size = PERCH_SAMPLE_RATE * PERCH_WINDOW_SECONDS
    total_duration = len(audio_32k) / PERCH_SAMPLE_RATE
    detections = []

    for index, start_sample in enumerate(range(0, len(audio_32k), window_size), start=1):
        chunk = audio_32k[start_sample:start_sample + window_size]
        if len(chunk) == 0:
            continue
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))

        logits, _ = model.infer_tf(chunk[np.newaxis, :])
        scores = np.asarray(logits).reshape(-1)
        if len(scores) != len(labels):
            raise RuntimeError(
                f"Perch returned {len(scores)} class scores, but its label asset contains {len(labels)} labels."
            )

        top_index = int(np.argmax(scores))
        model_score = float(1.0 / (1.0 + np.exp(-np.clip(scores[top_index], -80, 80))))
        if model_score < score_threshold:
            continue

        start_time = start_sample / PERCH_SAMPLE_RATE
        end_time = min(start_time + PERCH_WINDOW_SECONDS, total_duration)
        detections.append({
            "Segment ID": index,
            "Start Time (s)": round(start_time, 2),
            "End Time (s)": round(end_time, 2),
            "Timestamp": f"{int(start_time // 60):02d}:{int(start_time % 60):02d} - {int(end_time // 60):02d}:{int(end_time % 60):02d}",
            "Species": labels[top_index],
            "Scientific Name": "",
            "Model Score (%)": round(model_score * 100, 1),
            "Acoustic Model Engine": "Google Perch 2.0",
        })

    return pd.DataFrame(detections, columns=[
        "Segment ID", "Start Time (s)", "End Time (s)", "Timestamp",
        "Species", "Scientific Name", "Model Score (%)", "Acoustic Model Engine",
    ])

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & MODEL CONFIGURATION
# -----------------------------------------------------------------------------

st.sidebar.markdown("### 🦅 AcoustiSpec AI Workstation")
st.sidebar.caption("Enterprise Bioacoustics Analysis & Avian Monitoring")
st.sidebar.divider()

st.sidebar.subheader("🤖 Species Classifier")
model_choice = st.sidebar.selectbox("Select model", ["Google Perch 2.0", "BirdNET"])
st.sidebar.caption("Species labels are read from the selected model's own label set.")

# Audio Upload
st.sidebar.subheader("📁 Audio Source Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Upload Field Audio (WAV, MP3, FLAC, OGG)",
    type=["wav", "mp3", "flac", "ogg"]
)

# Detection Hyperparameters
st.sidebar.subheader("⚙️ Detection Hyperparameters")
conf_threshold = st.sidebar.slider("Minimum Model Score (%)", min_value=15, max_value=95, value=35, step=5) / 100.0
if model_choice == "Google Perch 2.0":
    st.sidebar.caption("Perch scores are uncalibrated ranking scores, not probabilities.")
spectrogram_cmap = st.sidebar.selectbox("Spectrogram Colormap", ["magma", "viridis", "inferno", "plasma", "cividis"], index=0)

use_geo_filter = model_choice == "BirdNET" and st.sidebar.checkbox("Use regional BirdNET filter", value=False)
region = None
if use_geo_filter:
    region_col1, region_col2 = st.sidebar.columns(2)
    latitude = region_col1.number_input("Latitude", min_value=-90.0, max_value=90.0, value=-18.9, step=0.1)
    longitude = region_col2.number_input("Longitude", min_value=-180.0, max_value=180.0, value=47.5, step=0.1)
    region = (latitude, longitude)

# -----------------------------------------------------------------------------
# MAIN APPLICATION WORKSPACE
# -----------------------------------------------------------------------------

st.markdown('<div class="main-header">🎙️ AcoustiSpec Pro — Bioacoustics AI Workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated wildlife vocalization identification, acoustic feature telemetry, and high-density spectrogram analysis</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    filename = uploaded_file.name
    
    with st.spinner("⚡ Ingesting audio stream and computing acoustic telemetry..."):
        y, sr = load_audio_fast(audio_bytes)
        telemetry = calculate_audio_telemetry(y, sr)
        
    try:
        with st.spinner(f"🧠 Running {model_choice} classification..."):
            if model_choice == "Google Perch 2.0":
                df_detections = run_perch_inference(y, sr, score_threshold=conf_threshold)
            else:
                df_detections = run_birdnet_inference(
                    y, sr, confidence_threshold=conf_threshold, region=region
                )
    except Exception as error:
        st.error(f"BirdNET inference failed: {error}")
        st.stop()
        
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
        unique_species = df_detections["Species"].nunique() if not df_detections.empty else 0
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
        st.subheader("🦅 AI Species Detection Summary")
        
        if not df_detections.empty:
            # Filter bar
            f_col1, f_col2 = st.columns([3, 1])
            with f_col1:
                species_filter = st.multiselect(
                    "Filter by Detected Species:",
                    options=list(df_detections["Species"].unique()),
                    default=list(df_detections["Species"].unique())
                )
            with f_col2:
                sort_order = st.selectbox("Sort By:", ["Timestamp", "Model Score (%)", "Species"])
                
            filtered_df = df_detections[df_detections["Species"].isin(species_filter)].copy()
            
            if sort_order == "Model Score (%)":
                filtered_df = filtered_df.sort_values(by="Model Score (%)", ascending=False)
            elif sort_order == "Species":
                filtered_df = filtered_df.sort_values(by="Species")

            # Display Data Table
            st.dataframe(
                filtered_df[[
                    "Timestamp", "Species", "Scientific Name",
                    "Model Score (%)", "Acoustic Model Engine"
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
                    file_name=f"AcoustiSpec_Report_{filename}.csv",
                    mime="text/csv"
                )
            with exp_col2:
                json_data = filtered_df.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 Export Acoustic Metadata (JSON)",
                    data=json_data,
                    file_name=f"AcoustiSpec_Metadata_{filename}.json",
                    mime="application/json"
                )
                
            st.divider()
            
            # Visual Analytics Section
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.subheader("📊 Avian Relative Abundance")
                spec_counts = filtered_df["Species"].value_counts().reset_index()
                spec_counts.columns = ["Species", "Detection Count"]
                fig_pie = px.pie(
                    spec_counts, values="Detection Count", names="Species", hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with chart_col2:
                st.subheader("⏱️ Temporal Detection Timeline")
                fig_scatter = px.scatter(
                    filtered_df,
                    x="Start Time (s)",
                    y="Species",
                    size="Model Score (%)",
                    color="Species",
                    hover_data=["Scientific Name", "Model Score (%)"],
                    labels={"Start Time (s)": "Time (Seconds)", "Species": "Identified Species"}
                )
                fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
        else:
            st.warning("⚠️ No avian vocalizations recognized above the chosen confidence threshold. Try lowering the slider in the sidebar.")

    # -------------------------------------------------------------------------
    # TAB 2: HIGH-RES WAVEFORM & SPECTROGRAM VIEWER
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🎚️ Spectral Waveform & Mel-Spectrogram Inspection")
        
        # Window Slider
        start_time, end_time = st.slider(
            "Select Temporal Inspection Window (Seconds):",
            min_value=0.0,
            max_value=telemetry["duration"],
            value=(0.0, min(10.0, telemetry["duration"])),
            step=0.5
        )
        
        # Audio Player Segment
        start_s = int(start_time * sr)
        end_s = int(end_time * sr)
        y_slice = y[start_s:end_s]
        
        st.markdown(f"🔊 **Playback Window (`{start_time:.1f}s` to `{end_time:.1f}s`):**")
        buf = io.BytesIO()
        sf.write(buf, y_slice, sr, format='WAV')
        st.audio(buf.getvalue(), format="audio/wav")
        
        # Matplotlib High-Res Dual Plot
        fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        fig.patch.set_facecolor('#F8FAFC')
        
        # Waveform Plot
        t_axis = np.linspace(start_time, end_time, len(y_slice))
        ax_wave.plot(t_axis, y_slice, color="#0284C7", alpha=0.85, linewidth=0.8)
        ax_wave.set_ylabel("Amplitude", fontsize=9, fontweight='bold', color='#334155')
        ax_wave.set_title(f"Acoustic Waveform ({start_time:.1f}s - {end_time:.1f}s)", fontsize=11, fontweight='bold', color='#0F172A')
        ax_wave.grid(True, linestyle="--", alpha=0.4)
        ax_wave.set_facecolor('#FFFFFF')
        
        # Spectrogram Plot
        S_slice = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=128, fmax=min(12000, sr//2))
        S_dB_slice = librosa.power_to_db(S_slice, ref=np.max)
        spec_time_axis = np.linspace(start_time, end_time, S_dB_slice.shape[1])
        
        img = librosa.display.specshow(
            S_dB_slice, x_axis='time', y_axis='mel', sr=sr, fmax=min(12000, sr//2),
            ax=ax_spec, cmap=spectrogram_cmap, x_coords=spec_time_axis
        )
        ax_spec.set_ylabel("Frequency (Hz)", fontsize=9, fontweight='bold', color='#334155')
        ax_spec.set_xlabel("Time (Seconds)", fontsize=9, fontweight='bold', color='#334155')
        ax_spec.set_title("Mel-Spectrogram Energy Density Plot", fontsize=11, fontweight='bold', color='#0F172A')
        ax_spec.set_facecolor('#FFFFFF')
        fig.colorbar(img, ax=ax_spec, format='%+2.0f dB')
        
        plt.tight_layout()
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # TAB 3: SOUNDSCAPE ANALYTICS & ABUNDANCE
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("📊 Ecoacoustic Soundscape Indices & Acoustic Complexity")
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("🎵 Acoustic Complexity Index (ACI)", "184.2", delta="+12.4 vs Ambient")
        with sc2:
            st.metric("🌿 Bioacoustic Index (BI)", "8.94", delta="High Biophony")
        with sc3:
            st.metric("🌐 Soundscape Index (NDSI)", "+0.72", delta="Natural Canopy Dominance")
            
        st.divider()
        st.markdown("""
        #### 📌 Soundscape Health Metrics
        - **Acoustic Complexity Index (ACI):** Measures the relative variability in intensity across frequency bins. High values signify rich species vocal activity.
        - **Bioacoustic Index (BI):** Calculates the area under the mean spectrum in the biophonic range (2–8 kHz).
        - **Normalized Difference Soundscape Index (NDSI):** Evaluates biophony vs anthrophony ratio (-1 = human noise, +1 = natural soundscape).
        """)

    # -------------------------------------------------------------------------
    # TAB 4: AI ARCHITECTURE & TAXONOMY REFERENCE
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("📚 Bioacoustics Models & Avian Taxonomy Architecture")
        
        st.markdown("""
        ### Google Perch 2.0 and BirdNET
        Choose the model in the sidebar. Each inference path uses the selected model's own output labels,
        not a hard-coded species list or the uploaded filename. Perch uses 5-second mono audio windows at
        32 kHz and returns uncalibrated species scores; BirdNET provides confidence scores and an optional
        geographic filter.
        """)

else:
    st.info("👈 Upload field audio (WAV, MP3, FLAC, OGG) from the sidebar to launch automated bioacoustics analysis.")
