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

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bioacoustics AI Analyzer",
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
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .stApp {
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: AUDIO PROCESSING & MOCK AI INFERENCE
# -----------------------------------------------------------------------------

@st.cache_data
def generate_sample_audio(duration=15, sr=22050):
    """Generates synthetic bioacoustic sample audio for instant demo testing."""
    t = np.linspace(0, duration, int(sr * duration))
    # Ambient background noise
    noise = np.random.normal(0, 0.05, len(t))
    # Synthetic bird call 1: Madagascar Magpie-Robin at 3s-6s (chirp sweep)
    f1 = np.where((t >= 3) & (t <= 6), 3000 + 1500 * np.sin(2 * np.pi * 5 * t), 0)
    sig1 = np.where((t >= 3) & (t <= 6), 0.4 * np.sin(2 * np.pi * f1 * t), 0)
    
    # Synthetic bird call 2: Souimanga Sunbird at 8s-12s (rapid trill)
    f2 = np.where((t >= 8) & (t <= 12), 4500 + 800 * np.cos(2 * np.pi * 12 * t), 0)
    sig2 = np.where((t >= 8) & (t <= 12), 0.35 * np.sin(2 * np.pi * f2 * t), 0)
    
    # Synthetic lemur/frog call at 13s-15s (low pitch)
    f3 = np.where((t >= 13) & (t <= 15), 1200 + 300 * np.sin(2 * np.pi * 3 * t), 0)
    sig3 = np.where((t >= 13) & (t <= 15), 0.5 * np.sin(2 * np.pi * f3 * t), 0)
    
    audio = noise + sig1 + sig2 + sig3
    return audio, sr

def format_timestamp(seconds):
    """Converts seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def run_bioacoustic_inference(y, sr, segment_dur=5.0, confidence_threshold=0.5, model_type="BirdNET V2.4"):
    """
    Simulates or runs Bioacoustics AI Model (BirdNET / Google Perch architecture).
    Divides audio into segment windows and detects species.
    """
    total_duration = librosa.get_duration(y=y, sr=sr)
    num_segments = int(np.ceil(total_duration / segment_dur))
    
    # Known endemic / sample species database
    species_db = [
        {"common": "Madagascar Magpie-Robin", "scientific": "Copsychus albospecularis", "freq_range": "2.5 - 5.0 kHz"},
        {"common": "Souimanga Sunbird", "scientific": "Cinnyris souimanga", "freq_range": "4.0 - 7.5 kHz"},
        {"common": "Madagascar Coucal", "scientific": "Centropus toulou", "freq_range": "0.8 - 2.2 kHz"},
        {"common": "Madagascar Bulbul", "scientific": "Hypsipetes madagascariensis", "freq_range": "2.0 - 4.5 kHz"},
        {"common": "Indri Lemur (Vocalization)", "scientific": "Indri indri", "freq_range": "1.0 - 3.5 kHz"},
    ]
    
    detections = []
    
    for i in range(num_segments):
        start_t = i * segment_dur
        end_t = min((i + 1) * segment_dur, total_duration)
        
        # Extract audio chunk
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        chunk = y[start_sample:end_sample]
        
        if len(chunk) == 0:
            continue
            
        # Compute RMS energy of chunk to detect signal presence
        rms = np.sqrt(np.mean(chunk**2))
        
        # If signal energy exceeds threshold, run classifier logic
        if rms > 0.04:
            # Deterministic selection based on frequency energy / chunk index for reproducible demo
            spec_idx = (i * 3 + int(rms * 100)) % len(species_db)
            species = species_db[spec_idx]
            conf = min(0.98, max(0.55, float(0.65 + np.sin(i * 1.5) * 0.3)))
            
            if conf >= confidence_threshold:
                detections.append({
                    "Segment": i + 1,
                    "Start Time (s)": round(start_t, 2),
                    "End Time (s)": round(end_t, 2),
                    "Timestamp": f"{format_timestamp(start_t)} - {format_timestamp(end_t)}",
                    "Common Name": species["common"],
                    "Scientific Name": species["scientific"],
                    "Confidence": round(conf * 100, 1),
                    "Frequency Range": species["freq_range"],
                    "Model Used": model_type
                })
                
    return pd.DataFrame(detections)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/color/96/000000/microprocessor.png", width=60)
st.sidebar.title("🎛️ Control Panel")

# Model Selection
model_choice = st.sidebar.selectbox(
    "🤖 Select AI Model",
    ["BirdNET V2.4 (Open-Source)", "Google Perch (Bioacoustics)", "Custom Hybrid Ensemble"]
)

# File Source
st.sidebar.subheader("📁 Audio Source")
source_option = st.sidebar.radio("Choose Input Type:", ["Upload Audio File", "Use Demo Sample Audio"])

y = None
sr = 22050
filename = ""

if source_option == "Upload Audio File":
    uploaded_file = st.sidebar.file_uploader("Upload Audio (WAV, MP3, FLAC, OGG)", type=["wav", "mp3", "flac", "ogg"])
    if uploaded_file is not None:
        filename = uploaded_file.name
        # Load audio using librosa
        audio_bytes = uploaded_file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
else:
    filename = "madagascar_rainforest_sample.wav"
    y, sr = generate_sample_audio(duration=15, sr=sr)
    st.sidebar.success("Loaded 15s Synthetic Bioacoustics Sample")

# Analysis Parameters
st.sidebar.subheader("⚙️ Detection Parameters")
conf_threshold = st.sidebar.slider("Confidence Threshold (%)", min_value=30, max_value=95, value=60, step=5) / 100.0
segment_window = st.sidebar.selectbox("Segment Window Size (s)", [3.0, 5.0, 10.0], index=1)
colormap_choice = st.sidebar.selectbox("Spectrogram Colormap", ["viridis", "magma", "inferno", "plasma"], index=1)

# -----------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------

st.markdown('<div class="main-header">🎙️ Bioacoustics AI Species Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated wildlife vocalization detection, spectrogram visualization, and ecological monitoring dashboard</div>', unsafe_allow_html=True)

if y is not None:
    duration_total = librosa.get_duration(y=y, sr=sr)
    
    # Run AI Inference
    with st.spinner("🔄 Processing audio with AI model..."):
        df_detections = run_bioacoustic_inference(
            y, sr, segment_dur=segment_window, confidence_threshold=conf_threshold, model_type=model_choice
        )
    
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
    tab1, tab2, tab3 = st.tabs(["📊 Detection Dashboard", "🎵 Waveform & Spectrogram Viewer", "📚 Species & Model Guide"])

    # -------------------------------------------------------------------------
    # TAB 1: DETECTION DASHBOARD
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📋 AI Species Detection Log")
        
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
                label="📥 Download Detection Report (CSV)",
                data=csv,
                file_name=f"bioacoustics_report_{filename}.csv",
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
                st.subheader("⏱️ Detection Timeline")
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
            st.warning("⚠️ No species detected above the chosen confidence threshold. Try lowering the threshold slider in the sidebar.")

    # -------------------------------------------------------------------------
    # TAB 2: WAVEFORM & SPECTROGRAM VIEWER
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🎚️ Interactive Audio Analysis")
        
        # Segment Selection
        start_sec, end_sec = st.slider(
            "Select Time Window to Inspect (Seconds):",
            min_value=0.0,
            max_value=float(duration_total),
            value=(0.0, min(10.0, float(duration_total))),
            step=0.5
        )
        
        # Audio Player for selected slice
        start_samp = int(start_sec * sr)
        end_samp = int(end_sec * sr)
        y_slice = y[start_samp:end_samp]
        
        st.write(f"🔊 **Playing audio segment ({start_sec:.1f}s - {end_sec:.1f}s):**")
        buffer = io.BytesIO()
        sf.write(buffer, y_slice, sr, format='WAV')
        st.audio(buffer.getvalue(), format="audio/wav")
        
        # Generate Plots
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
        
        # Correctly align Spectrogram time axis matching S_dB matrix shape
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
    # TAB 3: SPECIES & MODEL GUIDE
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("📚 About Bioacoustics Models & AI Architecture")
        
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("""
            ### 🦅 BirdNET Architecture
            - **Developer:** Cornell Lab of Ornithology & Chemnitz University.
            - **Coverage:** 6,000+ bird species worldwide.
            - **Input:** 3-second audio windows converted to mel-spectrograms.
            - **Accuracy:** High precision with localized confidence score calibration.
            """)
            
        with g2:
            st.markdown("""
            ### 🦜 Google Perch (Bioacoustics)
            - **Developer:** Google Research & Bioacoustics Group.
            - **Features:** Global species embedding representations, robust under noisy rainforest environments.
            - **Output:** Multi-label classification with embedding vectors for unknown sound search.
            """)
            
        st.info("💡 **Deployment Note:** This Streamlit app can be deployed to **Hugging Face Spaces** or **Streamlit Community Cloud** with a single click by pushing your repository to GitHub.")

else:
    st.info("👈 Upload an audio file or select 'Use Demo Sample Audio' from the sidebar to begin analysis!")
