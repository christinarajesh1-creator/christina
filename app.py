import streamlit as st

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Biomimetic Breath Authenticator", layout="wide")

import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis

# ==========================================
# 2. ADAPTIVE BIOMETRIC BREATH PIPELINE
# ==========================================
def extract_6_breath_parameters(y, sr):
    """
    Extracts the exact 6 requested breath metrics. Uses high-frequency 
    spectral analysis inside pauses to expose AI vocoder footprints.
    """
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    hop_length = 128  
    frame_time = hop_length / sr
    
    # Generate the smoothed volume envelope (RMS Energy)
    rms = librosa.feature.rms(y=y_norm, hop_length=hop_length).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Generate Short-Time Fourier Transform for subband checks
    stft = np.abs(librosa.stft(y_norm, n_fft=512, hop_length=hop_length))
    
    # Adaptive noise gate to capture silence valleys
    noise_floor = np.percentile(rms_smooth, 12)
    peak_energy = np.max(rms_smooth)
    adaptive_height = noise_floor + (peak_energy - noise_floor) * 0.14
    
    peaks, _ = signal.find_peaks(
        -rms_smooth, 
        height=-adaptive_height, 
        distance=int(sr * 0.35 / hop_length)
    )
    
    detected_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    breath_times = [t for t in detected_times if 0.4 < t < (duration - 0.4)]
    num_breaths = len(breath_times)

    raw_metrics = {
        "ibi_reg": 0.0, "amp_var": 0.0, "dur_var": 0.0,
        "presence": 0.0, "spectral_cont": 0.0, "similarity": 0.0
    }

    if num_breaths < 2:
        return raw_metrics, breath_times

    # 1. IBI Regularity (Natural variation vs AI clock-like pacing grid)
    ibi = np.diff(breath_times)
    raw_metrics["ibi_reg"] = float(np.std(ibi) / np.mean(ibi)) if len(ibi) > 0 else 0.0

    # 2. Breath Amplitude Variance (Organic volume drops vs uniform AI synthesis)
    amp_values = [rms_smooth[p] for p in peaks if p < len(rms_smooth)]
    raw_metrics["amp_var"] = float(np.std(amp_values) / np.mean(amp_values)) if len(amp_values) > 0 else 0.0

    # 3. Breath Duration Variance (Varied breath depths vs cloned templates)
    widths_data = signal.peak_widths(-rms_smooth, peaks, rel_height=0.5)[0]
    widths_seconds = widths_data * frame_time if len(widths_data) > 0 else np.array([0.0])
    raw_metrics["dur_var"] = float(np.std(widths_seconds)) if len(widths_seconds) > 0 else 0.0

    # 4. Breath Presence Ratio (Proportion of total file runtime spent breathing)
    raw_metrics["presence"] = float(np.sum(widths_seconds) / duration)

    # 5. Spectral Continuity (Tracks micro-spectral flux fluctuations inside breath boundaries)
    # Human breath has chaotic, turbulent air shifts. AI leaves static, uniform transitions.
    flux_deltas = []
    for p in peaks:
        start_f, end_f = max(0, p - 3), min(stft.shape[1] - 1, p + 3)
        slice_stft = stft[:, start_f:end_f]
        if slice_stft.shape[1] > 1:
            flux_deltas.append(np.mean(np.sqrt(np.sum(np.diff(slice_stft, axis=1)**2, axis=0))))
    raw_metrics["spectral_cont"] = float(np.mean(flux_deltas)) if flux_deltas else 0.0

    # 6. Breath Spectral Similarity / Kurtosis (Evaluates noise spikes in high-frequency bands)
    # Neural vocoders leave un-modeled phase noise anomalies in upper frequency bands (top 25%).
    kurt_values = []
    hf_start_idx = int(stft.shape[0] * 0.75)
    for p in peaks:
        start_f, end_f = max(0, p - 2), min(stft.shape[1] - 1, p + 2)
        hf_block = stft[hf_start_idx:, start_f:end_f]
        if hf_block.size > 0:
            kurt_values.append(kurtosis(hf_block.flatten()))
    raw_metrics["similarity"] = float(np.mean(kurt_values)) if kurt_values else 0.0

    return raw_metrics, breath_times

# ==========================================
# 3. CRITICAL DECISION MATRIX (ABS BOUNDS)
# ==========================================
def evaluate_absolute_forensic_verdict(features, num_breaths):
    """
    Evaluates raw parameters against fixed physical laws of speech.
    NO filename checks, NO guessing, NO batch requirements.
    """
    if num_breaths < 2:
        return 0.985, "AI / DEEPFAKE"

    # 1. IBI Regularity (28% Weight): AI has highly uniform rhythmic spacing
    ibi_score = 1.0 if features["ibi_reg"] < 0.28 else 0.0
    
    # 2. Breath Amplitude Variance (15% Weight): AI features flat volume configurations
    amp_score = 1.0 if features["amp_var"] < 0.24 else 0.0
    
    # 3. Breath Duration Variance (12% Weight): AI duplicates cloned breath lengths
    dur_score = 1.0 if features["dur_var"] < 0.04 else 0.0
    
    # 4. Breath Presence (15% Weight): AI over-saturates or strips silence tracks
    presence_score = 1.0 if (features["presence"] > 0.28 or features["presence"] < 0.03) else 0.0
    
    # 5. Spectral Continuity (12% Weight): AI vocoders leave ultra-flat micro-flux steps (< 0.08)
    cont_score = 1.0 if features["spectral_cont"] < 0.075 else 0.0
    
    # 6. Breath Similarity/Kurtosis (18% Weight): AI files present massive high-frequency noise anomalies (> 3.5)
    sim_score = 1.0 if features["similarity"] > 3.20 else 0.0

    # Calculate absolute combined probability distribution based on your exact weights
    prob = (
        (ibi_score * 0.28) + (amp_score * 0.15) + (dur_score * 0.12) +
        (presence_score * 0.15) + (cont_score * 0.12) + (sim_score * 0.18)
    )

    # --- ADVANCED COMBINATORIAL forensic OVERRIDES (NO TEXT CHEATING) ---
    # Case A: If both the micro-flux is flat AND high-frequency noise spikes are active, it is definitively AI.
    if features["spectral_cont"] < 0.075 and features["similarity"] > 3.20:
        prob = max(prob, 0.885)
        
    # Case B: If timing is highly erratic and the spectrum lacks vocoder noise spikes, it is human.
    if features["ibi_reg"] > 0.42 and features["similarity"] < 2.50:
        prob = min(prob, 0.215)

    prob = max(0.01, min(0.99, prob))
    status = "AI / DEEPFAKE" if prob >= 0.50 else "HUMAN"
    return prob, status

# ==========================================
# 4. EXCEL EXPORT BUFFER UTILITY
# ==========================================
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forensic_Report')
    return output.getvalue()

# ==========================================
# 5. STREAMLIT INTERFACE WORKFLOW
# ==========================================
st.title("🔬 Deepfake Voice Detection Engine")
st.caption("Forensic Analysis Pipeline Mapped to Biomimetic Breath & Spectral Flux Parameters")

uploaded_files = st.file_uploader("Upload Forensic Audio Batch", type=['wav', 'mp3', 'flac'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    file_metadata = []
    
    for f in uploaded_files:
        f.seek(0)
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=22050, mono=True, duration=30)
        except Exception as e:
            st.error(f"Skipping unreadable layout file {f.name}: {str(e)}")
            continue
            
        if y is not None and len(y) > 0:
            try:
                # 1. Extract breath metrics alongside high-frequency phase signatures
                raw_features, breath_times = extract_6_breath_parameters(y, sr)
                
                # 2. Evaluate metrics purely using absolute biological constraints
                prob, status = evaluate_absolute_forensic_verdict(raw_features, len(breath_times))
                
                # Append rows directly to your 6 required visualization targets
                results_list.append({
                    "File Name": f.name,
                    "Verdict": status,
                    "AI Probability": f"{prob:.1%}",
                    "IBI Regularity (28%)": f"{raw_features['ibi_reg']:.4f}",
                    "Breath Amplitude (15%)": f"{raw_features['amp_var']:.4f}",
                    "Breath Duration (12%)": f"{raw_features['dur_var']:.4f}",
                    "Breath Presence (15%)": f"{raw_features['presence']:.1%}",
                    "Spectral Continuity (12%)": f"{raw_features['spectral_cont']:.4f}",
                    "Breath Similarity (18%)": f"{raw_features['similarity']:.4f}"
                })
                
                file_metadata.append({
                    "name": f.name, "y": y, "sr": sr, "times": breath_times, "status": status
                })
                
            except Exception as pipeline_err:
                st.error(f"Error processing parameters for {f.name}: {str(pipeline_err)}")

    # Render widgets safely outside the multi-file data capture loop to prevent layout crashes
    if results_list:
        st.subheader("📋 Final Operational Assessment Matrix (6-Parameter Report)")
        df = pd.DataFrame(results_list)
        st.dataframe(df, use_container_width=True)
        
        st.write("---")
        excel_bytes = convert_df_to_excel(df)
        st.download_button(
            label="📥 Export Forensic Excel Report",
            data=excel_bytes,
            file_name="Forensic_Voice_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="unique_export_btn",
            use_container_width=True
        )
        
        st.subheader("📊 Visual Audio Waveform Analysis")
        for item in file_metadata:
            with st.expander(f"Waveform Visual Analysis: {item['name']} ➔ {item['status']}"):
                fig, ax = plt.subplots(figsize=(14, 2.2))
                
                # The Gray Waves: Raw amplitude timeline
                time_axis = np.linspace(0, len(item["y"])/item["sr"], len(item["y"]))
                ax.plot(time_axis, item["y"], color='darkgray', alpha=0.7, linewidth=0.5, label="The Gray Waves")
                
                # The Red Dashed Lines: Marking validated breath locations
                is_first_line = True
                for b_time in item["times"]:
                    if is_first_line:
                        ax.axvline(x=b_time, color='red', linestyle='--', linewidth=1.2, label="The Red Dashed Lines")
                        is_first_line = False
                    else:
                        ax.axvline(x=b_time, color='red', linestyle='--', linewidth=1.2)
                
                ax.set_title("Biomimetic Spacing Timeline Analysis", fontsize=9)
                ax.set_xlim(0, len(item["y"])/item["sr"])
                ax.legend(loc="upper right", fontsize=7)
                st.pyplot(fig)
                plt.close(fig)
