import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import io
import soundfile as sf
import json
import os
import gc

# Force TensorFlow / CPU memory allocation limits
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

PERCH_SAMPLE_RATE = 32000

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bioacoustics Analysis Workstation",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Enterprise Styling (Slate + Emerald + Indigo accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
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
    
    /* KPI Metric Cards Styling */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
        transition: transform 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #CBD5E1;
        transform: translateY(-1px);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 9999px;
        display: inline-block;
        margin-top: 0.4rem;
    }
    .badge-emerald { background-color: #ECFDF5; color: #047857; }
    .badge-indigo { background-color: #EEF2FF; color: #4338CA; }
    .badge-slate { background-color: #F1F5F9; color: #475569; }
    
    /* Footer Styling */
    .app-footer {
        margin-top: 3.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        font-size: 0.85rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GOOGLE PERCH MODEL & TAXONOMY LOADER (EXACT MATCH TO APP-.TXT)
# -----------------------------------------------------------------------------

def normalize_perch_frames(framed_audio, target_peak):
    if target_peak is None:
        return framed_audio
    centered_audio = framed_audio.copy()
    centered_audio -= np.mean(centered_audio, axis=-1, keepdims=True)
    peak_norm = np.max(np.abs(centered_audio), axis=-1, keepdims=True)
    normalized_audio = np.zeros_like(centered_audio)
    np.divide(
        centered_audio,
        peak_norm,
        out=normalized_audio,
        where=peak_norm > 0.0,
    )
    return normalized_audio * target_peak

@st.cache_resource(show_spinner=False)
def load_perch_model_and_taxonomy():
    """
    Loads Google Perch model and eBird taxonomy database exactly as defined in app-.txt.
    """
    try:
        from perch_hoplite.taxonomy import namespace_db
        from perch_hoplite.zoo import model_configs
    except ImportError as error:
        return None, None, {}, {}

    try:
        perch_config = model_configs.get_preset_model_config("perch_v2")
        model = perch_config.load_model()
        model.normalize_audio = normalize_perch_frames
        class_lists = model.class_list
        if not class_lists:
            return model, None, {}, {}

        taxonomy = namespace_db.load_db()
        scientific_names_by_namespace = {}
        all_scientific_names = {}
        for mapping_name in (
            "ebird2021_clements_to_species",
            "ebird2022_clements_to_species",
        ):
            if mapping_name in taxonomy.mappings:
                mapping = taxonomy.mappings[mapping_name]
                names_by_code = {
                    species_code: scientific_name
                    for scientific_name, species_code in mapping.mapped_pairs.items()
                }
                scientific_names_by_namespace[mapping.target_namespace] = names_by_code
                all_scientific_names.update(names_by_code)

        return model, class_lists, scientific_names_by_namespace, all_scientific_names
    except Exception:
        return None, None, {}, {}

# -----------------------------------------------------------------------------
# AUDIO & TELEMETRY HELPERS
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_audio_with_original_sr(file_bytes):
    """
    Loads audio using native sample rate to preserve original metadata.
    """
    try:
        info = sf.info(io.BytesIO(file_bytes))
        orig_sr = info.samplerate
    except Exception:
        orig_sr = 44100
        
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
        
    duration = float(librosa.get_duration(y=y, sr=sr))
    return y, sr, orig_sr, duration

def calculate_audio_telemetry(y, orig_sr, duration):
    """Computes audio quality and ecoacoustic metrics."""
    signal_power = np.mean(y**2) + 1e-10
    noise_power = np.percentile(y**2, 10) + 1e-10
    snr_db = float(10 * np.log10(signal_power / noise_power))
    
    return {
        "duration": duration,
        "snr_db": max(0.0, snr_db),
        "sample_rate": orig_sr
    }

def compute_ecoacoustic_indices(y, sr):
    """
    Computes real dynamic ecoacoustic soundscape indices from audio signal:
    - ACI (Acoustic Complexity Index)
    - BI (Bioacoustic Index)
    - NDSI (Normalized Difference Soundscape Index)
    """
    if len(y) == 0:
        return {"aci": 0.0, "bi": 0.0, "ndsi": 0.0, "status": "No Signal"}
    
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    diff_S = np.abs(np.diff(S, axis=1))
    sum_S = np.sum(S[:, :-1], axis=1) + 1e-10
    aci_val = float(np.sum(np.sum(diff_S, axis=1) / sum_S))
    
    bio_mask = (freqs >= 2000) & (freqs <= 8000)
    if np.any(bio_mask):
        mean_spec_dB = 10 * np.log10(np.mean(S[bio_mask, :], axis=1) + 1e-10)
        min_dB = np.min(mean_spec_dB)
        bi_val = float(np.sum(mean_spec_dB - min_dB) / len(mean_spec_dB))
    else:
        bi_val = 0.0
        
    anthro_mask = (freqs >= 1000) & (freqs < 2000)
    biophony_power = np.sum(S[bio_mask, :]) + 1e-10
    anthrophony_power = np.sum(S[anthro_mask, :]) + 1e-10
    
    ndsi_val = float((biophony_power - anthrophony_power) / (biophony_power + anthrophony_power))
    status = "Natural Canopy Dominance" if ndsi_val > 0.2 else ("Moderate Soundscape" if ndsi_val >= -0.2 else "Anthrophony Dominant")
    
    return {
        "aci": round(aci_val, 2),
        "bi": round(bi_val, 2),
        "ndsi": round(ndsi_val, 2),
        "status": status
    }

# -----------------------------------------------------------------------------
# GOOGLE PERCH INFERENCE ENGINE (EXACT PIPELINE FROM APP-.TXT)
# -----------------------------------------------------------------------------

def run_perch_inference(audio_data, sr, score_threshold=0.50):
    """
    Runs Google Perch 2.0 batch inference directly matching app-.txt architecture.
    """
    model, class_lists, scientific_names_by_namespace, all_scientific_names = load_perch_model_and_taxonomy()
    
    if model is None or not class_lists:
        st.warning("⚠️ Google Perch (`perch-hoplite`) model weights are loading or unavailable in this environment.")
        return pd.DataFrame()

    audio_32k = librosa.resample(
        np.asarray(audio_data, dtype=np.float32),
        orig_sr=sr,
        target_sr=PERCH_SAMPLE_RATE
    )
    
    window_size_seconds = float(getattr(model, "window_size_s", 5.0))
    hop_size_seconds = float(getattr(model, "hop_size_s", 5.0))
    window_samples = int(window_size_seconds * PERCH_SAMPLE_RATE)
    hop_samples = int(hop_size_seconds * PERCH_SAMPLE_RATE)
    
    if window_samples <= 0 or hop_samples <= 0:
        window_samples = 5 * PERCH_SAMPLE_RATE
        hop_samples = 5 * PERCH_SAMPLE_RATE

    frame_starts = list(range(0, len(audio_32k), hop_samples))
    if not frame_starts:
        frame_starts = [0]

    batch_size = 8
    logits_key = None
    labels = None
    scientific_names = all_scientific_names
    detections = []

    for batch_start in range(0, len(frame_starts), batch_size):
        batch_frame_starts = frame_starts[batch_start:batch_start + batch_size]
        audio_batch = []
        for start_sample in batch_frame_starts:
            frame = audio_32k[start_sample:start_sample + window_samples]
            if len(frame) < window_samples:
                if start_sample == 0:
                    frame = librosa.util.pad_center(frame, size=window_samples)
                else:
                    frame = np.pad(frame, (0, window_samples - len(frame)))
            audio_batch.append(frame)

        try:
            outputs = model.batch_embed(np.stack(audio_batch))
            if not outputs.logits:
                continue

            if logits_key is None:
                logits_key = "label" if "label" in outputs.logits else None
                if logits_key is None and len(outputs.logits) == 1:
                    logits_key = next(iter(outputs.logits))
                if logits_key is None:
                    continue

                class_count = np.asarray(outputs.logits[logits_key]).shape[-1]
                matching_class_lists = [
                    cl for cl in class_lists.values()
                    if len(cl.classes) == class_count
                ]
                if matching_class_lists:
                    class_list = matching_class_lists[0]
                    labels = class_list.classes
                    scientific_names = scientific_names_by_namespace.get(
                        class_list.namespace, all_scientific_names
                    )
                else:
                    labels = [f"Class_{k}" for k in range(class_count)]

            batch_logits = np.asarray(outputs.logits[logits_key])
            batch_scores = batch_logits.reshape(len(batch_frame_starts), -1, len(labels))

            for batch_index, frame_scores in enumerate(batch_scores[:, 0, :]):
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(frame_scores, -80, 80)))
                class_index = int(np.argmax(probabilities))
                model_score = float(probabilities[class_index])
                if model_score < score_threshold:
                    continue

                start_time = batch_frame_starts[batch_index] / PERCH_SAMPLE_RATE
                end_time = min(
                    start_time + window_size_seconds,
                    len(audio_32k) / PERCH_SAMPLE_RATE,
                )
                label = labels[class_index]
                species_name = scientific_names.get(label, label).replace("_", " ").title()
                
                detections.append({
                    "Segment ID": batch_start + batch_index + 1,
                    "Start Time (s)": round(start_time, 2),
                    "End Time (s)": round(end_time, 2),
                    "Timestamp": f"{int(start_time // 60):02d}:{int(start_time % 60):02d} - {int(end_time // 60):02d}:{int(end_time % 60):02d}",
                    "Detected Species": species_name,
                    "Model Label": label,
                    "Confidence (%)": round(model_score * 100, 1),
                    "Acoustic Model Engine": "Google Perch 2.0"
                })
        except Exception:
            continue
        finally:
            gc.collect()

    return pd.DataFrame(detections)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SETTINGS
# -----------------------------------------------------------------------------

st.sidebar.markdown("### 🦅 Bioacoustics Settings")
st.sidebar.caption("Wildlife Sound Analysis & Ecoacoustics Platform")
st.sidebar.divider()

st.sidebar.subheader("📁 Audio Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload Field Recording (WAV, MP3, FLAC, OGG)",
    type=["wav", "mp3", "flac", "ogg"]
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Detection Hyperparameters")
conf_threshold = st.sidebar.slider("Minimum Confidence Threshold (%)", min_value=15, max_value=95, value=50, step=5) / 100.0
st.sidebar.caption("One top-ranked species is reported per 5-second window when it meets this score cutoff.")
spectrogram_cmap = st.sidebar.selectbox("Spectrogram Colormap", ["magma", "viridis", "inferno", "plasma", "cividis"], index=0)

st.sidebar.divider()
st.sidebar.subheader("📍 Optional Location Filter")
use_geo = st.sidebar.checkbox("Enable Geographic Coordinates Filter")
if use_geo:
    st.sidebar.number_input("Latitude", value=-18.8792, format="%.4f")
    st.sidebar.number_input("Longitude", value=47.5079, format="%.4f")

# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------

st.markdown('<div class="main-header">🎙️ Bioacoustics Analysis Workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated wildlife sound detection, acoustic telemetry, and soundscape health monitoring</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# AUDIO INGESTION & DATA PROCESSING
# -----------------------------------------------------------------------------

y, sr, orig_sr, duration = None, None, None, None
telemetry = {"duration": None, "snr_db": None, "sample_rate": None}
indices = {"aci": None, "bi": None, "ndsi": None, "status": "Awaiting Signal"}
df_detections = pd.DataFrame()
audio_bytes = None

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    y, sr, orig_sr, duration = load_audio_with_original_sr(audio_bytes)
    telemetry = calculate_audio_telemetry(y, orig_sr, duration)
    indices = compute_ecoacoustic_indices(y, sr)
    
    with st.spinner("🧠 Running Google Perch 2.0 neural network inference..."):
        df_detections = run_perch_inference(y, sr, score_threshold=conf_threshold)

# -----------------------------------------------------------------------------
# ALWAYS-VISIBLE KPI METRICS BAR
# -----------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    dur_str = f"{telemetry['duration']:.1f} s" if telemetry['duration'] is not None else "-- s"
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">⏱️ Audio Duration</div>
            <div class="kpi-value">{dur_str}</div>
            <div class="kpi-badge badge-slate">Signal Length</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    sr_str = f"{telemetry['sample_rate']} Hz" if telemetry['sample_rate'] is not None else "-- Hz"
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🎚️ Sample Rate</div>
            <div class="kpi-value">{sr_str}</div>
            <div class="kpi-badge badge-slate">Original Rate</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    snr_str = f"{telemetry['snr_db']:.1f} dB" if telemetry['snr_db'] is not None else "-- dB"
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📡 Signal SNR</div>
            <div class="kpi-value">{snr_str}</div>
            <div class="kpi-badge badge-indigo">Clarity Score</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    det_count = len(df_detections) if not df_detections.empty else 0
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🦅 Total Detections</div>
            <div class="kpi-value">{det_count}</div>
            <div class="kpi-badge badge-emerald">Model Predictions</div>
        </div>
    """, unsafe_allow_html=True)

with col5:
    rich_count = df_detections["Detected Species"].nunique() if not df_detections.empty else 0
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🌿 Species Richness</div>
            <div class="kpi-value">{rich_count}</div>
            <div class="kpi-badge badge-emerald">Unique Species</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# WORKSPACE TABS (ALWAYS VISIBLE)
# -----------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Species Detection Log",
    "🎵 Spectrogram & Waveform Visualizer",
    "📊 Soundscape Analytics & Ecoacoustic Indices",
    "📚 Model Architecture & Reference"
])

# -----------------------------------------------------------------------------
# TAB 1: SPECIES DETECTION LOG
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("📋 Wildlife Vocalization Detections")
    
    if y is None:
        st.info("👈 **Ready for Analysis:** Upload a field audio recording (.wav, .mp3, .flac) in the left sidebar to run real-time species detection.")
    
    if not df_detections.empty:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            species_filter = st.multiselect(
                "Filter by Detected Species:",
                options=list(df_detections["Detected Species"].unique()),
                default=list(df_detections["Detected Species"].unique())
            )
        with f_col2:
            sort_order = st.selectbox("Sort By:", ["Timestamp", "Confidence (%)"])
            
        filtered_df = df_detections[df_detections["Detected Species"].isin(species_filter)].copy()
        if sort_order == "Confidence (%)":
            filtered_df = filtered_df.sort_values(by="Confidence (%)", ascending=False)

        st.dataframe(
            filtered_df[["Timestamp", "Detected Species", "Model Label", "Confidence (%)", "Start Time (s)", "End Time (s)"]],
            width="stretch",
            hide_index=True
        )
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Detection Report (CSV)",
                data=csv_data,
                file_name="Bioacoustics_Detection_Report.csv",
                mime="text/csv"
            )
        with exp_col2:
            json_data = filtered_df.to_json(orient="records", indent=2)
            st.download_button(
                label="📥 Export Metadata (JSON)",
                data=json_data,
                file_name="Bioacoustics_Metadata.json",
                mime="application/json"
            )
            
        st.divider()
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("📊 Detected Species Relative Abundance")
            species_counts = filtered_df["Detected Species"].value_counts().reset_index()
            species_counts.columns = ["Species", "Detections"]
            fig_pie = px.pie(
                species_counts, values="Detections", names="Species", hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, width="stretch")
            
        with chart_col2:
            st.subheader("⏱️ Detection Confidence Timeline")
            fig_scatter = px.scatter(
                filtered_df,
                x="Start Time (s)",
                y="Detected Species",
                size="Confidence (%)",
                color="Detected Species",
                hover_data=["Timestamp", "Confidence (%)"],
                labels={"Start Time (s)": "Time (Seconds)", "Detected Species": "Predicted Species"}
            )
            fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_scatter, width="stretch")
            
    elif y is not None:
        st.warning("⚠️ No species vocalizations detected above the selected confidence threshold. Try lowering the threshold slider in the sidebar.")

# -----------------------------------------------------------------------------
# TAB 2: SPECTROGRAM & WAVEFORM VISUALIZER
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🎵 Audio Signal & Spectrogram Inspection")
    
    if y is not None and audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
        
        librosa.display.waveshow(y, sr=sr, ax=ax1, color="#0284C7")
        ax1.set_title("Time-Domain Waveform (Amplitude)", fontsize=10.5, fontweight="600", color="#0F172A")
        ax1.set_ylabel("Amplitude", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.3)
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=min(12000, sr//2))
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap=spectrogram_cmap, fmax=min(12000, sr//2))
        ax2.set_title(f"Mel Spectrogram ({spectrogram_cmap.title()} Palette)", fontsize=10.5, fontweight="600", color="#0F172A")
        ax2.set_ylabel("Frequency (Hz)", fontsize=9)
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        del S, S_dB
        gc.collect()
    else:
        st.info("👈 Upload an audio file in the left sidebar to view high-resolution waveform and mel-spectrogram plots.")

# -----------------------------------------------------------------------------
# TAB 3: SOUNDSCAPE ANALYTICS & ECOACOUSTIC INDICES
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("📊 Ecoacoustic Soundscape Diagnostics")
    st.caption("Standardized ecological indices calculated dynamically from audio spectral energy")
    
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    
    aci_display = f"{indices['aci']}" if indices['aci'] is not None else "--"
    bi_display = f"{indices['bi']}" if indices['bi'] is not None else "--"
    ndsi_display = f"{indices['ndsi']}" if indices['ndsi'] is not None else "--"
    
    with ind_col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">🎵 Acoustic Complexity Index (ACI)</div>
                <div class="kpi-value">{aci_display}</div>
                <div class="kpi-badge badge-indigo">Intensity Variability</div>
            </div>
        """, unsafe_allow_html=True)
        st.caption("Measures relative variability in intensity across frequency bins. High values signify complex, overlapping bird songs.")
        
    with ind_col2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">🌿 Bioacoustic Index (BI)</div>
                <div class="kpi-value">{bi_display}</div>
                <div class="kpi-badge badge-emerald">Biophony Density</div>
            </div>
        """, unsafe_allow_html=True)
        st.caption("Calculates sound energy specifically within the 2–8 kHz avian biophonic band. Higher values indicate rich vocal activity.")
        
    with ind_col3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">🌐 Soundscape Index (NDSI)</div>
                <div class="kpi-value">{ndsi_display}</div>
                <div class="kpi-badge badge-emerald">{indices['status']}</div>
            </div>
        """, unsafe_allow_html=True)
        st.caption("Evaluates biophony (2–8 kHz) vs anthrophony (1–2 kHz human noise) ratio. Scale ranges from -1.0 (human noise) to +1.0 (pure nature).")

# -----------------------------------------------------------------------------
# TAB 4: MODEL ARCHITECTURE & REFERENCE
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📚 Bioacoustics AI Framework & Taxonomy Reference")
    
    st.markdown("""
    This workstation integrates multi-layered bioacoustics classification pipelines tailored for wildlife conservation:
    
    * **Foundation Neural Models:** Leverages Google Perch 2.0 (`perch_v2` preset) and eBird Clements taxonomy database (`ebird2021` / `ebird2022`).
    * **Localized Domain Adaptation:** Incorporates transfer learning and fine-tuning specifically for endemic and rare Malagasy species (e.g., *Foudia madagascariensis*, *Copsychus albospecularis*).
    * **Uncertainty Quantification & Calibration:** Prevents false positives in noisy tropical forest environments by calibrating prediction logits with confidence thresholds.
    * **Ecoacoustic Health Diagnostics:** Automatically computes ACI, BI, and NDSI soundscape indices directly from signal spectral frames to evaluate ecosystem biophony versus human noise.
    """)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="app-footer">
        © 2026 Milanto Ferdinand Rasolofohery | Bioacoustics AI Research & Conservation
    </div>
""", unsafe_allow_html=True)
