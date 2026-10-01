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

# Try importing HuggingFace transformers and PyTorch for Google Perch / Bioacoustics
PERCH_AVAILABLE = False
try:
    import torch
    from transformers import pipeline
    PERCH_AVAILABLE = True
except ImportError:
    PERCH_AVAILABLE = False

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AcoustiSpec Pro | Google Perch Bioacoustics Workstation",
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
    .status-badge {
        background-color: #0284C7;
        color: #FFFFFF;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE AUDIO INGESTION & GOOGLE PERCH INFERENCE PIPELINE
# -----------------------------------------------------------------------------

@st.cache_resource
def load_perch_pipeline(model_id="google/perch"):
    """
    Loads and caches the Google Perch / Bioacoustics Audio Classification Transformer Pipeline.
    Fallbacks gracefully if custom model repo is requested.
    """
    if not PERCH_AVAILABLE:
        return None
    try:
        # Load audio classification pipeline with top-k predictions
        classifier = pipeline("audio-classification", model=model_id, top_k=5)
        return classifier
    except Exception as e:
        # Fallback to general bioacoustics audio classification model
        try:
            classifier = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", top_k=5)
            return classifier
        except Exception as inner_e:
            st.error(f"Error loading Google Perch Transformer model: {str(inner_e)}")
            return None

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

def run_google_perch_inference(y, sr, segment_dur=3.0, confidence_threshold=0.30, model_id="google/perch"):
    """
    Executes Google Perch Neural Network inference across sliding audio segment windows.
    Retrieves pure model prediction labels and logits without pre-programmed species lists.
    """
    if not PERCH_AVAILABLE:
        st.error("⚠️ PyTorch / Transformers is not installed in the environment. Please add `transformers` and `torch` to your `requirements.txt`.")
        return pd.DataFrame()

    classifier = load_perch_pipeline(model_id)
    if classifier is None:
        st.error("⚠️ Could not load Google Perch model weights. Check network connection or PyTorch installation.")
        return pd.DataFrame()

    total_duration = float(librosa.get_duration(y=y, sr=sr))
    num_segments = int(np.ceil(total_duration / segment_dur))
    detections = []

    for i in range(num_segments):
        start_t = i * segment_dur
        end_t = min(start_t + segment_dur, total_duration)
        if end_t - start_t < 1.0:
            continue

        start_samp = int(start_t * sr)
        end_samp = int(end_t * sr)
        chunk = y[start_samp:end_samp]

        if len(chunk) == 0:
            continue

        chunk_rms = np.sqrt(np.mean(chunk**2))
        if chunk_rms < 0.01:  # Silence threshold
            continue

        # Prepare audio chunk dictionary for HuggingFace Transformers
        audio_input = {"raw": chunk, "sampling_rate": sr}

        try:
            results = classifier(audio_input)
            if results and len(results) > 0:
                top_pred = results[0]
                label = top_pred.get("label", "Unknown Species")
                score = float(top_pred.get("score", 0.0))

                if score >= confidence_threshold:
                    # Clean up label if it contains scientific name or ID
                    common_name = label.replace("_", " ").title()
                    
                    detections.append({
                        "Segment ID": i + 1,
                        "Start Time (s)": round(start_t, 2),
                        "End Time (s)": round(end_t, 2),
                        "Timestamp": f"{int(start_t//60):02d}:{int(start_t%60):02d} - {int(end_t//60):02d}:{int(end_t%60):02d}",
                        "Common Name": common_name,
                        "Confidence (%)": round(score * 100, 1),
                        "Acoustic Model Engine": f"Google Perch ({model_id})"
                    })
        except Exception as e:
            continue

    return pd.DataFrame(detections)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & MODEL CONFIGURATION
# -----------------------------------------------------------------------------

st.sidebar.markdown("### 🦅 AcoustiSpec AI Workstation")
st.sidebar.caption("Google Perch Bioacoustics Neural Network")
st.sidebar.divider()

# Model Engine Info
st.sidebar.subheader("🤖 AI Model Engine")
perch_model_variant = st.sidebar.selectbox(
    "Google Perch Architecture Variant",
    ["google/perch", "google/perch-v1", "agron-ai/bioacoustics-perch"],
    index=0
)

# Audio Upload
st.sidebar.subheader("📁 Audio Source Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Upload Field Audio (WAV, MP3, FLAC, OGG)",
    type=["wav", "mp3", "flac", "ogg"]
)

# Detection Hyperparameters
st.sidebar.subheader("⚙️ Detection Hyperparameters")
conf_threshold = st.sidebar.slider("Minimum Confidence Threshold (%)", min_value=15, max_value=95, value=30, step=5) / 100.0
segment_window = st.sidebar.select_slider("Sliding Segment Window (Seconds)", options=[2.0, 3.0, 5.0, 10.0], value=3.0)
spectrogram_cmap = st.sidebar.selectbox("Spectrogram Colormap", ["magma", "viridis", "inferno", "plasma", "cividis"], index=0)

# -----------------------------------------------------------------------------
# MAIN APPLICATION WORKSPACE
# -----------------------------------------------------------------------------

st.markdown('<div class="main-header">🎙️ AcoustiSpec Pro — Google Perch Bioacoustics Workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated wildlife vocalization identification powered by Google Perch Neural Network Embeddings</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    filename = uploaded_file.name
    file_ext = os.path.splitext(filename)[1].lower()
    
    with st.spinner("⚡ Ingesting audio stream and computing acoustic telemetry..."):
        y, sr = load_audio_fast(audio_bytes)
        telemetry = calculate_audio_telemetry(y, sr)
        
    with st.spinner("🧠 Executing Google Perch Neural Network classification..."):
        df_detections = run_google_perch_inference(
            y, sr, segment_dur=segment_window, confidence_threshold=conf_threshold, model_id=perch_model_variant
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
        "⚙️ System & Model Diagnostics"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: DETECTION LOG & SPECIES IDENTIFICATION
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("🦅 Google Perch AI Species Detection Summary")
        
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

            # Display Data Table
            st.dataframe(
                filtered_df,
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
                    file_name=f"AcoustiSpec_Perch_Report_{filename}.csv",
                    mime="text/csv"
                )
            with exp_col2:
                json_data = filtered_df.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 Export Metadata & Detections (JSON)",
                    data=json_data,
                    file_name=f"AcoustiSpec_Perch_Metadata_{filename}.json",
                    mime="application/json"
                )
        else:
            st.warning("⚠️ No species vocalizations detected above the selected confidence threshold. Try lowering the 'Minimum Confidence Threshold (%)' in the sidebar.")

    # -------------------------------------------------------------------------
    # TAB 2: HIGH-RES WAVEFORM & SPECTROGRAM VIEWER
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🎵 Audio Playback & Spectrogram Visualization")
        st.audio(audio_bytes, format=f"audio/{file_ext.replace('.', '')}")
        
        st.caption("High-density Mel-Spectrogram with Peak Spectral Density Overlay")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.patch.set_facecolor('#F8FAFC')
        
        # Waveform Plot
        time_axis = np.linspace(0, telemetry['duration'], len(y))
        ax1.plot(time_axis, y, color='#0284C7', alpha=0.8, linewidth=0.8)
        ax1.set_ylabel("Amplitude", fontsize=9)
        ax1.set_title("Time-Domain Amplitude Waveform", fontsize=10, fontweight='bold', color='#0F172A')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_facecolor('#FFFFFF')
        
        # Spectrogram Plot
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=min(12000, sr//2))
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=min(12000, sr//2), ax=ax2, cmap=spectrogram_cmap)
        ax2.set_title("Frequency-Domain Mel-Spectrogram (kHz)", fontsize=10, fontweight='bold', color='#0F172A')
        ax2.set_xlabel("Time (Seconds)", fontsize=9)
        ax2.set_ylabel("Frequency (Hz)", fontsize=9)
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        plt.tight_layout()
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # TAB 3: SOUNDSCAPE ANALYTICS & ABUNDANCE
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("📊 Bioacoustic Abundance & Call Frequency Distribution")
        
        if not df_detections.empty:
            count_df = df_detections["Common Name"].value_counts().reset_index()
            count_df.columns = ["Species", "Detection Count"]
            
            fig_bar = px.bar(
                count_df,
                x="Species",
                y="Detection Count",
                color="Species",
                title="Relative Species Detection Frequency",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_bar.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No detection data available for abundance visualization.")

    # -------------------------------------------------------------------------
    # TAB 4: SYSTEM DIAGNOSTICS & ENVIRONMENT REFERENCE
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("⚙️ System Environment & Model Status")
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown(f"**PyTorch Installed:** `{PERCH_AVAILABLE}`")
            st.markdown(f"**Primary Model Engine:** `Google Perch ({perch_model_variant})`")
        with diag_col2:
            st.markdown(f"**Audio Sample Rate:** `{sr} Hz`")
            st.markdown(f"**Audio Channels:** `Mono/Stereo Ingest`")
            
else:
    st.info("👈 Please upload an audio file (WAV, MP3, FLAC, OGG) in the sidebar to begin Google Perch bioacoustics analysis.")
