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

# ==========================================
# 2. FORENSIC SPECTRAL NOISE SUBTRACTOR
# ==========================================
def apply_spectral_noise_subtraction(y, sr):
    """
    Scientifically removes background microphone hiss, room hum, and static noise
    by profiling the noise floor in the spectral domain and subtracting it.
    """
    stft = librosa.stft(y, n_fft=512, hop_length=128)
    stft_mag, stft_phase = librosa.magphase(stft)
    
    # Profile the noise floor by looking at the lowest 15% energy frames (silence gaps)
    frame_energies = np.sum(stft_mag**2, axis=0)
    noise_thresh = np.percentile(frame_energies, 15)
    noise_frames = stft_mag[:, frame_energies <= noise_thresh]
    
    if noise_frames.shape[0] > 0:
        mean_noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)
    else:
        mean_noise_spectrum = np.median(stft_mag, axis=1, keepdims=True) * 0.1
        
    # Apply Spectral Subtraction (Wiener-style soft thresholding)
    subtracted_mag = np.maximum(stft_mag - (mean_noise_spectrum * 1.5), 0.0)
    
    # Reconstruct the clean time-domain audio signal
    clean_stft = subtracted_mag * stft_phase
    y_clean = librosa.istft(clean_stft, hop_length=128)
    return y_clean

# ==========================================
# 3. BIOMETRIC FEATURE EXTRACTION ENGINE
# ==========================================
def extract_6_breath_parameters(y_clean, sr):
    """
    Extracts the exact 6 biometric breath parameters from the noise-subtracted voice signal.
    """
    y_norm = librosa.util.normalize(y_clean)
    duration = len(y_norm) / sr
    hop_length = 128  
    frame_time = hop_length / sr
    
    # Track the clean vocal volume envelope using Root-Mean-Square (RMS) Energy
    rms = librosa.feature.rms(y=y_norm, hop_length=hop_length).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Locate valleys corresponding to breath/speech transitions
    peaks, _ = signal.find_peaks(
        -rms_smooth, 
        height=-np.median(rms_smooth)*0.9, 
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

    # 1. IBI Regularity (Coefficient of Variation of Inter-Breath Intervals)
    ibi = np.diff(breath_times)
    raw_metrics["ibi_reg"] = float(np.std(ibi) / np.mean(ibi)) if len(ibi) > 0 else 0.0

    # 2. Breath Amplitude Variance (Coefficient of Variation of peak power)
    amp_values = [rms_smooth[p] for p in peaks if p < len(rms_smooth)]
    raw_metrics["amp_var"] = float(np.std(amp_values) / np.mean(amp_values)) if len(amp_values) > 0 else 0.0

    # 3. Breath Duration Variance (Standard Deviation of widths)
    widths_data = signal.peak_widths(-rms_smooth, peaks, rel_height=0.5)[0]
    widths_seconds = widths_data * frame_time if len(widths_data) > 0 else np.array([0.0])
    raw_metrics["dur_var"] = float(np.std(widths_seconds)) if len(widths_seconds) > 0 else 0.0

    # 4. Breath Presence (Total breath time vs recording length)
    raw_metrics["presence"] = float(np.sum(widths_seconds) / duration)

    # 5. Spectral Continuity (Zero-Crossing Rate delta shifts at speech boundaries)
    zcr = librosa.feature.zero_crossing_rate(y=y_norm, hop_length=hop_length).flatten()
    zcr_deltas = []
    for p in peaks:
        start_f, end_f = max(0, p - 4), min(len(zcr) - 1, p + 4)
        zcr_deltas.append(np.max(np.abs(np.diff(zcr[start_f:end_f]))))
    raw_metrics["spectral_cont"] = float(np.max(zcr_deltas)) if len(zcr_deltas) > 0 else 0.0

    # 6. Breath Spectral Similarity (Cross-correlation of MFCC spectral arrays)
    breath_mfccs = []
    for t in breath_times:
        start_sample, end_sample = int((t - 0.12) * sr), int((t + 0.12) * sr)
        segment = y_norm[max(0, start_sample):min(len(y_norm), end_sample)]
        if len(segment) > 128:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=6)
            breath_mfccs.append(np.mean(mfcc, axis=1))
            
    if len(breath_mfccs) >= 2:
        matrix = np.corrcoef(breath_mfccs)
        np.fill_diagonal(matrix, 0)
        raw_metrics["similarity"] = float(np.max(matrix))
    else:
        raw_metrics["similarity"] = 0.0

    return raw_metrics, breath_times

# ==========================================
# 4. EXCEL EXPORT BUFFER UTILITY
# ==========================================
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forensic_Report')
    return output.getvalue()

# ==========================================
# 5. STREAMLIT INTERFACE & RUNTIME EXECUTION
# ==========================================
st.title("🔬 Deepfake Voice Detection Engine")
st.caption("Forensic Analysis Pipeline Powered by Absolute Acoustic Boundary Verification")

uploaded_files = st.file_uploader("Upload Forensic Audio Batch", type=['wav', 'mp3', 'flac'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    file_metadata = []
    
    # Process files independently to gather acoustic bounds
    for f in uploaded_files:
        f.seek(0)
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=22050, mono=True, duration=30)
        except Exception as e:
            st.error(f"Skipping unreadable asset {f.name}: {str(e)}")
            continue
            
        if y is not None and len(y) > 0:
            y_clean = apply_spectral_noise_subtraction(y, sr)
            features, breath_times = extract_6_breath_parameters(y_clean, sr)
            num_breaths = len(breath_times)
            
            # ABSOLUTE BIOMETRIC CLASSIFICATION ENGINE (No batch averages, no filenames)
            
            # 1. IBI Regularity (28% Weight): Mechanically uniform spacing = AI indicator
            ibi_score = 1.0 if (num_breaths < 2 or features["ibi_reg"] <= 0.32) else 0.0
                
            # 2. Breath Amplitude (15% Weight): Flattened artificial volume envelopes = AI indicator
            amp_score = 1.0 if features["amp_var"] < 0.22 else 0.0
            
            # 3. Breath Duration (12% Weight): Cloned, identical length breaths = AI indicator
            dur_score = 1.0 if features["dur_var"] < 0.04 else 0.0
            
            # 4. Breath Presence (15% Weight): Over-saturated or empty presence maps = AI indicator
            presence_score = 1.0 if (features["presence"] > 0.28 or features["presence"] < 0.03) else 0.0
                
            # 5. Spectral Continuity (12% Weight): Artificial processing boundaries = AI indicator
            cont_score = 1.0 if features["spectral_cont"] < 0.052 else 0.0
            
            # 6. Breath Similarity (18% Weight): Repeated copy-paste audio fragments = AI indicator
            sim_score = 1.0 if features["similarity"] > 0.72 else 0.0

            if num_breaths < 2:
                prob = 0.985
            else:
                prob = (
                    (ibi_score * 0.28) + (amp_score * 0.15) + (dur_score * 0.12) +
                    (presence_score * 0.15) + (cont_score * 0.12) + (sim_score * 0.18)
                )

            # Scientific Calibration Baseline Override:
            # If the cadence spacing metrics are highly erratic, human lungs are validated.
            if features["ibi_reg"] > 0.35 and features["similarity"] < 0.60:
                prob = min(prob, 0.245)

            prob = max(0.01, min(0.99, prob))
            status = "AI / DEEPFAKE" if prob >= 0.50 else "HUMAN"
            
            results_list.append({
                "File Name": f.name,
                "Verdict": status,
                "AI Probability": f"{prob:.1%}",
                "IBI Regularity (28%)": f"{features['ibi_reg']:.4f}",
                "Breath Amplitude (15%)": f"{features['amp_var']:.4f}",
                "Breath Duration (12%)": f"{features['dur_var']:.4f}",
                "Breath Presence (15%)": f"{features['presence']:.1%}",
                "Spectral Continuity (12%)": f"{features['spectral_cont']:.4f}",
                "Breath Similarity (18%)": f"{features['similarity']:.1%}"
            })
            
            file_metadata.append({
                "name": f.name, "y": y, "sr": sr, "times": breath_times, "status": status
            })

        # Render visualizations below the data processing block
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
                    time_axis = np.linspace(0, len(item["y"])/item["sr"], len(item["y"]))
                    ax.plot(time_axis, item["y"], color='darkgray', alpha=0.7, linewidth=0.5, label="The Gray Waves")
                    
                    is_first = True
                    for b_time in item["times"]:
                        if is_first:
                            ax.axvline(x=b_time, color='red', linestyle='--', linewidth=1.2, label="The Red Dashed Lines")
                            is_first = False
                        else:
                            ax.axvline(x=b_time, color='red', linestyle='--', linewidth=1.2)
                    
                    ax.set_title("Biomimetic Spacing Timeline Analysis", fontsize=9)
                    ax.set_xlim(0, len(item["y"])/item["sr"]))
                    ax.legend(loc="upper right", fontsize=7)
                    st.pyplot(fig)
                    plt.close(fig)
