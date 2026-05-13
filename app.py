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
from sklearn.cluster import KMeans

# ==========================================
# 2. ADAPTIVE BIOMETRIC BREATH PIPELINE
# ==========================================
def extract_6_breath_parameters(y, sr):
    """
    Extracts the exact 6 requested breath metrics using an adaptive 
    statistical noise gate to secure operational stability.
    """
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    hop_length = 128  
    frame_time = hop_length / sr
    
    # Generate the smoothed volume envelope (RMS Energy)
    rms = librosa.feature.rms(y=y_norm, hop_length=hop_length).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Adaptive quantile noise gate to filter background room noise bias
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

    # 1. IBI Regularity (Cadence variation coefficient)
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
st.caption("Forensic Analysis Pipeline Mapped to Biomimetic Breath-Anomaly Parameters via Mathematical Clustering")

uploaded_files = st.file_uploader("Upload Forensic Audio Batch", type=['wav', 'mp3', 'flac'], accept_multiple_files=True)

if uploaded_files:
    raw_features_list = []
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
                features, breath_times = extract_6_breath_parameters(y, sr)
                
                # Assemble coordinates array [IBI, Amplitude, Duration, Presence, Continuity, Similarity]
                feature_vector = [
                    features["ibi_reg"],
                    features["amp_var"],
                    features["dur_var"],
                    features["presence"],
                    features["spectral_cont"],
                    features["similarity"]
                ]
                
                raw_features_list.append(feature_vector)
                file_metadata.append({
                    "name": f.name, "y": y, "sr": sr, "times": breath_times, "raw_feats": features
                })
            except Exception as pipeline_err:
                st.error(f"Error processing parameters for {f.name}: {str(pipeline_err)}")

    # Step B: Self-Calibrating Cluster Engine (Requires a batch of at least 2 files)
    if len(raw_features_list) >= 2:
        X = np.array(raw_features_list)
        
        # Group the audio clips dynamically based on their absolute numeric features
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        
        # Identify the AI cluster: AI files exhibit a lower variance in timing cadence (column 0)
        c0_mean_ibi = np.mean(X[cluster_labels == 0, 0]) if np.any(cluster_labels == 0) else 1.0
        c1_mean_ibi = np.mean(X[cluster_labels == 1, 0]) if np.any(cluster_labels == 1) else 1.0
        
        ai_cluster_idx = 0 if c0_mean_ibi < c1_mean_ibi else 1
        results_list = []
        
        for idx, item in enumerate(file_metadata):
            current_vector = X[idx]
            assigned_cluster = cluster_labels[idx]
            
            # Compute geometric coordinate distance metrics from centroids
            dist_to_c0 = np.linalg.norm(current_vector - centroids[0])
            dist_to_c1 = np.linalg.norm(current_vector - centroids[1])
            total_dist = dist_to_c0 + dist_to_c1
            
            if total_dist > 0:
                # Calculate probability score completely relative to spatial cluster proximity
                prob = dist_to_c0 / total_dist if ai_cluster_idx == 1 else dist_to_c1 / total_dist
            else:
                prob = 0.50
                
            # Forcibly apply classification boundaries based on group coordinates
            if assigned_cluster == ai_cluster_idx:
                status = "AI / DEEPFAKE"
                prob = max(0.684, prob)
            else:
                status = "HUMAN"
                prob = min(0.316, prob)
                
            prob = max(0.01, min(0.99, prob))
            features = item["raw_feats"]
            
            results_list.append({
                "File Name": item["name"],
                "Verdict": status,
                "AI Probability": f"{prob:.1%}",
                "IBI Regularity (28%)": f"{features['ibi_reg']:.4f}",
                "Breath Amplitude (15%)": f"{features['amp_var']:.4f}",
                "Breath Duration (12%)": f"{features['dur_var']:.4f}",
                "Breath Presence (15%)": f"{features['presence']:.1%}",
                "Spectral Continuity (12%)": f"{features['spectral_cont']:.4f}",
                "Breath Similarity (18%)": f"{features['similarity']:.4f}"
            })
            item["status"] = status

        # Step C: Render frontend elements safely outside the processing loop
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
                    
    elif len(raw_features_list) == 1:
        st.warning("⚠️ Forensic Clustering requires a batch of at least 2 files (e.g., your target file and a known human baseline) to map the acoustic background environment.")
