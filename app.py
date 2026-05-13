import streamlit as st

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Forensic Voice Authenticator", layout="wide")

import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis

# ==========================================
# 2. ADVANCED SPECTRAL SIGNAL CLEANER
# ==========================================
def extract_6_breath_parameters(y, sr):
    """
    Extracts the exact 6 requested metrics by applying a strict 
    spectral energy mask to filter out background room noise bias.
    """
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    hop_length = 128  
    frame_time = hop_length / sr
    
    # Calculate Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(y_norm, n_fft=512, hop_length=hop_length))
    
    # Track raw Root-Mean-Square Volume Energy
    rms = librosa.feature.rms(y=y_norm, hop_length=hop_length).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # --- PHYSICAL ENERGY FLOOR MASK ---
    # Isolate the quietest segments to identify the true room noise floor
    noise_floor = np.percentile(rms_smooth, 15)
    
    # Locate valleys corresponding to conversational pauses
    peaks, _ = signal.find_peaks(
        -rms_smooth, 
        height=-noise_floor * 1.5, 
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

    # 1. IBI Regularity (Rhythmic variation coefficient)
    ibi = np.diff(breath_times)
    raw_metrics["ibi_reg"] = float(np.std(ibi) / np.mean(ibi)) if len(ibi) > 0 else 0.0

    # 2. Breath Amplitude Variance (Volume differences across intervals)
    amp_values = [rms_smooth[p] for p in peaks if p < len(rms_smooth)]
    raw_metrics["amp_var"] = float(np.std(amp_values) / np.mean(amp_values)) if len(amp_values) > 0 else 0.0

    # 3. Breath Duration Variance (Fixes tuple matrix unpacking errors)
    # Extracts widths directly by slicing index 0 from the returned Scipy dataset
    widths_data = signal.peak_widths(-rms_smooth, peaks, rel_height=0.5)[0]
    widths_seconds = widths_data * frame_time if len(widths_data) > 0 else np.array([0.0])
    raw_metrics["dur_var"] = float(np.std(widths_seconds)) if len(widths_seconds) > 0 else 0.0

    # 4. Breath Presence Ratio (Percentage of total speech spent pausing)
    raw_metrics["presence"] = float(np.sum(widths_seconds) / duration)

    # 5. Spectral Continuity (Acoustic flux transitions inside speech boundaries)
    flux_deltas = []
    for p in peaks:
        start_f, end_f = max(0, p - 3), min(stft.shape[1] - 1, p + 3)
        slice_stft = stft[:, start_f:end_f]
        if slice_stft.shape[1] > 1:
            flux_deltas.append(np.mean(np.sqrt(np.sum(np.diff(slice_stft, axis=1)**2, axis=0))))
    raw_metrics["spectral_cont"] = float(np.mean(flux_deltas)) if flux_deltas else 0.0

    # 6. Breath Spectral Similarity / Upper Band Kurtosis
    # Neural vocoders leave un-modeled mathematical phase noise spikes in top bands
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
# 3. ROBUST ABSOLUTE FORENSIC VERDICT MATRIX
# ==========================================
def evaluate_absolute_forensic_verdict(features, num_breaths):
    """
    Evaluates acoustic measurements using pure biomimetic speech parameters.
    """
    if num_breaths < 2:
        return 0.985, "AI / DEEPFAKE"

    # Evaluate individual metrics against structural speech parameters
    ibi_score = 1.0 if features["ibi_reg"] < 0.28 else 0.0
    amp_score = 1.0 if features["amp_var"] < 0.24 else 0.0
    dur_score = 1.0 if features["dur_var"] < 0.04 else 0.0
    presence_score = 1.0 if (features["presence"] > 0.26 or features["presence"] < 0.03) else 0.0
    cont_score = 1.0 if features["spectral_cont"] < 0.075 else 0.0
    sim_score = 1.0 if features["similarity"] > 3.20 else 0.0

    # Aggregate weighted index values matching your design metrics
    prob = (
        (ibi_score * 0.28) + (amp_score * 0.15) + (dur_score * 0.12) +
        (presence_score * 0.15) + (cont_score * 0.12) + (sim_score * 0.18)
    )

    # --- FORENSIC HYBRID MATRIX CALIBRATION (NO TEXT CHEATING) ---
    # Real human speakers exhibit dynamic spacing alongside low spectral kurtosis noise spikes
    if features["ibi_reg"] > 0.35 and features["similarity"] < 2.50:
        prob = min(prob, 0.245)
        
    # AI models exhibit flat amplitude variance alongside high-frequency noise spikes
    if features["amp_var"] < 0.25 and features["similarity"] > 3.10:
        prob = max(prob, 0.895)

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
                # 1. Extract parameters using the corrected Scipy width arrays
                raw_features, breath_times = extract_6_breath_parameters(y, sr)
                
                # 2. Evaluate metrics purely using absolute acoustic criteria
                prob, status = evaluate_absolute_forensic_verdict(raw_features, len(breath_times))
                
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

    # Render visualizations outside loop boundary targets to secure platform stability
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
                
                # The Red Dashed Lines: Marking validated breath metrics locations
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
