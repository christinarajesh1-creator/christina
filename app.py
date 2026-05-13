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
# 2. HYBRID BIOMETRIC & SPECTRAL ENGINE
# ==========================================
def extract_hybrid_forensic_features(y, sr):
    """
    Extracts your 6 required breath parameters as the primary matrix, 
    supplemented by high-dimensional spectral safety vectors.
    """
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    hop_length = 128  
    frame_time = hop_length / sr
    
    # Generate the smoothed volume envelope (RMS Energy)
    rms = librosa.feature.rms(y=y_norm, hop_length=hop_length).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Generate Short-Time Fourier Transform for the spectral engine
    stft = np.abs(librosa.stft(y_norm, n_fft=512, hop_length=hop_length))
    
    # Adaptive noise gate to capture silence valleys cleanly
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

    # Dictionary containing your 6 mandatory primary design parameters
    raw_metrics = {
        "ibi_reg": 0.0, "amp_var": 0.0, "dur_var": 0.0,
        "presence": 0.0, "spectral_cont": 0.0, "similarity": 0.0,
        "guardrail_flux": 0.0, "guardrail_mfcc": 0.0, "guardrail_kurt": 0.0
    }

    # --- EXTRACT VOCAL SPECTRUM COFACTOR HANDLES ---
    flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
    raw_metrics["guardrail_flux"] = float(np.std(flux)) if len(flux) > 0 else 0.0
    
    mfcc = librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=13)
    raw_metrics["guardrail_mfcc"] = float(np.mean(np.std(mfcc, axis=1)))
    
    hf_start_idx = int(stft.shape[0] * 0.75)
    hf_band = stft[hf_start_idx:, :]
    raw_metrics["guardrail_kurt"] = float(kurtosis(hf_band.flatten())) if hf_band.size > 0 else 0.0

    if num_breaths < 2:
        # Static baseline fallbacks if tracking limits are completely flattened by vocoders
        raw_metrics["ibi_reg"] = 0.15
        raw_metrics["amp_var"] = 0.08
        return raw_metrics, breath_times

    # 1. IBI Regularity (Rhythmic variation coefficient)
    ibi = np.diff(breath_times)
    raw_metrics["ibi_reg"] = float(np.std(ibi) / np.mean(ibi)) if len(ibi) > 0 else 0.0

    # 2. Breath Amplitude Variance (Volume differences across pauses)
    amp_values = [rms_smooth[p] for p in peaks if p < len(rms_smooth)]
    raw_metrics["amp_var"] = float(np.std(amp_values) / np.mean(amp_values)) if len(amp_values) > 0 else 0.0

    # 3. Breath Duration Variance
    widths_data = signal.peak_widths(-rms_smooth, peaks, rel_height=0.5)[0]
    widths_seconds = widths_data * frame_time if len(widths_data) > 0 else np.array([0.0])
    raw_metrics["dur_var"] = float(np.std(widths_seconds)) if len(widths_seconds) > 0 else 0.0

    # 4. Breath Presence Ratio (Percentage of total speech spent pausing)
    raw_metrics["presence"] = float(np.sum(widths_seconds) / duration)

    # 5. Spectral Continuity (Zero-Crossing Rate delta changes at speech boundaries)
    zcr = librosa.feature.zero_crossing_rate(y=y_norm, hop_length=hop_length).flatten()
    zcr_deltas = []
    for p in peaks:
        start_f, end_f = max(0, p - 4), min(len(zcr) - 1, p + 4)
        zcr_deltas.append(np.max(np.abs(np.diff(zcr[start_f:end_f]))))
    raw_metrics["spectral_cont"] = float(np.max(zcr_deltas)) if len(zcr_deltas) > 0 else 0.0

    # 6. Breath Spectral Similarity (Cross-correlation matrix of MFCC profiles across segments)
    breath_mfccs = []
    for t in breath_times:
        start_sample, end_sample = int((t - 0.12) * sr), int((t + 0.12) * sr)
        segment = y_norm[max(0, start_sample):min(len(y_norm), end_sample)]
        if len(segment) > 128:
            mfcc_seg = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=6)
            breath_mfccs.append(np.mean(mfcc_seg, axis=1))
            
    if len(breath_mfccs) >= 2:
        matrix = np.corrcoef(breath_mfccs)
        np.fill_diagonal(matrix, 0)
        raw_metrics["similarity"] = float(np.max(matrix))
    else:
        raw_metrics["similarity"] = 0.0

    return raw_metrics, breath_times

# ==========================================
# 3. EXCEL EXPORT BUFFER UTILITY
# ==========================================
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forensic_Report')
    return output.getvalue()

# ==========================================
# 4. STREAMLIT INTERFACE WORKFLOW
# ==========================================
st.title("🔬 Deepfake Voice Detection Engine")
st.caption("Forensic Hybrid Fusion Pipeline: Biomimetic Breath Patterns & Statistical Batch Normalization")

uploaded_files = st.file_uploader("Upload Forensic Audio Batch", type=['wav', 'mp3', 'flac'], accept_multiple_files=True)

if uploaded_files:
    raw_batch_data = []
    file_metadata = []
    
    # Step A: Loop through uploaded files to extract parameters safely
    for f in uploaded_files:
        f.seek(0)
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=22050, mono=True, duration=30)
        except Exception as e:
            st.error(f"Skipping unreadable layout file {f.name}: {str(e)}")
            continue
            
        if y is not None and len(y) > 0:
            try:
                features, breath_times = extract_hybrid_forensic_features(y, sr)
                raw_batch_data.append(features)
                file_metadata.append({
                    "name": f.name, "y": y, "sr": sr, "times": breath_times, "feats": features
                })
            except Exception as pipeline_err:
                st.error(f"Error processing parameters for {f.name}: {str(pipeline_err)}")

    # Step B: Batch-Level Statistical Variance Matrix Calculation
    if raw_batch_data:
        # Extract lists of the primary values across the current uploaded batch
        batch_ibis = [item["ibi_reg"] for item in raw_batch_data]
        batch_sims = [item["similarity"] for item in raw_batch_data]
        batch_flux = [item["guardrail_flux"] for item in raw_batch_data]
        
        # Calculate localized batch medians to adapt cleanly to your current hardware noise floor
        median_ibi = np.median(batch_ibis) if batch_ibis else 0.35
        median_sim = np.median(batch_sims) if batch_sims else 0.50
        median_flux = np.median(batch_flux) if batch_flux else 0.70
        
        results_list = []
        
        # Step C: Evaluate each file dynamically relative to the batch baseline
        for item in file_metadata:
            name = item["name"]
            features = item["feats"]
            num_breaths = len(item["times"])
            
            # 1. IBI Regularity Score (28% Weight)
            # Flag if the file's pacing rhythm drops drastically below the current batch baseline
            ibi_score = 1.0 if features["ibi_reg"] < 0.25 or features["ibi_reg"] < (median_ibi * 0.75) else 0.0
                
            # 2. Breath Amplitude Score (15% Weight)
            amp_score = 1.0 if features["amp_var"] < 0.22 else 0.0
            
            # 3. Breath Duration Score (12% Weight)
            dur_score = 1.0 if features["dur_var"] < 0.04 else 0.0
            
            # 4. Breath Presence Score (15% Weight)
            presence_score = 1.0 if (features["presence"] > 0.28 or features["presence"] < 0.03) else 0.0
                
            # 5. Spectral Continuity Score (12% Weight)
            cont_score = 1.0 if features["spectral_cont"] < 0.052 else 0.0
            
            # 6. Breath Similarity Score (18% Weight)
            sim_score = 1.0 if features["similarity"] > 0.74 or features["similarity"] > (median_sim * 1.25) else 0.0

            if num_breaths < 2:
                prob = 0.985
            else:
                prob = (
                    (ibi_score * 0.28) + (amp_score * 0.15) + (dur_score * 0.12) +
                    (presence_score * 0.15) + (cont_score * 0.12) + (sim_score * 0.18)
                )

            # --- DYNAMIC BIOMETRIC SAFETY OVERRIDE (NO TEXT CHEATING) ---
            # If the current file's pitch velocity velocity is higher than the batch median, 
            # and it shows dynamic cadence spacing variation, it is verified as Human.
            if features["guardrail_flux"] >= median_flux and features["ibi_reg"] > 0.35:
                prob = min(prob, 0.245)
                
            # Conversely, lock files that display rigid, uniform vocoder shapes
            if features["guardrail_flux"] < (median_flux * 0.70) or features["guardrail_kurt"] > 4.50:
                prob = max(prob, 0.895)

            prob = max(0.01, min(0.99, prob))
            status = "AI / DEEPFAKE" if prob >= 0.50 else "HUMAN"
            
            results_list.append({
                "File Name": name,
                "Verdict": status,
                "AI Probability": f"{prob:.1%}",
                "IBI Regularity (28%)": f"{features['ibi_reg']:.4f}",
                "Breath Amplitude (15%)": f"{features['amp_var']:.4f}",
                "Breath Duration (12%)": f"{features['dur_var']:.4f}",
                "Breath Presence (15%)": f"{features['presence']:.1%}",
                "Spectral Continuity (12%)": f"{features['spectral_cont']:.4f}",
                "Breath Similarity (18%)": f"{features['similarity']:.1%}"
            })
            item["status"] = status

        # Step D: Render widgets safely outside data loop blocks to secure platform stability
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
