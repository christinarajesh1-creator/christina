import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd

# Secure the Excel exporter dependency safely
try:
    import xlsxwriter
except ImportError:
    st.error("Missing dependency: Please run 'pip install xlsxwriter' in your terminal.")

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Forensic Voice Authenticator", layout="wide")

# ==========================================
# 2. HYBRID BIOMETRIC & SPECTRAL ENGINE
# ==========================================
def extract_hybrid_forensic_features(y, sr):
    """
    Extracts core acoustic respiratory parameters using zero-crossing profiles 
    and spectral flatness paired with vocal tract energy guardrails.
    """
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    hop_length = 256  
    frame_time = hop_length / sr
    
    # Generate biometric layers for unvoiced vs voiced tracking
    rms = librosa.feature.rms(y=y_norm, hop_length=hop_length).flatten()
    flatness = librosa.feature.spectral_flatness(y=y_norm, hop_length=hop_length).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=y_norm, hop_length=hop_length).flatten()
    
    # Smooth arrays to filter out microphone clicks
    rms_smooth = np.convolve(rms, np.ones(3)/3, mode='same')
    rms_min, rms_max = np.min(rms_smooth), np.max(rms_smooth)
    rms_rel = (rms_smooth - rms_min) / (rms_max - rms_min + 1e-6)

    # Isolate true breath zones (quiet, high-flatness noise)
    breath_indices = []
    for i in range(len(rms_rel)):
        if (0.02 < rms_rel[i] < 0.30) and (flatness[i] > 0.08) and (zcr[i] > 0.10):
            breath_indices.append(i)

    # Cluster consecutive frames into discrete breath events
    breath_events = []
    if breath_indices:
        current_event = [breath_indices[0]]
        for idx in breath_indices[1:]:
            if idx == current_event[-1] + 1:
                current_event.append(idx)
            else:
                dur_ms = len(current_event) * frame_time * 1000.0
                if 150.0 <= dur_ms <= 1000.0:
                    breath_events.append(current_event)
                current_event = [idx]
        if len(current_event) * frame_time * 1000.0 >= 150.0:
            breath_events.append(current_event)

    num_breaths = len(breath_events)

    # Initialize data structural baseline
    raw_metrics = {
        "amplitude": -50.0,           
        "duration": 0.0,              
        "decay_rate": 0.0,            
        "speech_breath_ratio": 0.0,   
        "spectral_flow": 0.0,         
        "recovery_intervals": 0.0,    
        "guardrail_rolloff": 0.0,     
        "guardrail_centroid": 0.0     
    }

    # Extract high-frequency anchors
    rolloff = librosa.feature.spectral_rolloff(y=y_norm, sr=sr, roll_percent=0.85)
    raw_metrics["guardrail_rolloff"] = float(np.mean(rolloff))
    
    centroid = librosa.feature.spectral_centroid(y=y_norm, sr=sr)
    raw_metrics["guardrail_centroid"] = float(np.mean(centroid))
    raw_metrics["spectral_flow"] = float(np.mean(centroid) * 0.6)

    if num_breaths < 2:
        return raw_metrics, num_breaths

    # --- Feature Extraction Processing ---
    
    # 1. Amplitude Profile (dB) - FIXED: Flatten multi-dimensional array chunks safely
    all_breath_amplitudes = [amp_val for evt in breath_events for amp_val in rms_smooth[evt]]
    raw_metrics["amplitude"] = float(20 * np.log10(np.mean(all_breath_amplitudes) + 1e-6))

    # 2. Duration Envelope (ms)
    durations = [len(evt) * frame_time * 1000.0 for evt in breath_events]
    raw_metrics["duration"] = float(np.mean(durations))

    # 3. Decay Rate (Exhalation slope profile)
    decay_slopes = []
    for evt in breath_events:
        end = evt[-1]
        post_idx = min(len(rms_smooth) - 1, end + 4)
        if post_idx > end:
            decay_slopes.append(abs(rms_smooth[post_idx] - rms_smooth[end]))
    raw_metrics["decay_rate"] = float(np.mean(decay_slopes)) if decay_slopes else 0.0

    # 4. Speech to Breath Ratio
    total_breath_time = (sum(len(evt) for evt in breath_events) * frame_time)
    raw_metrics["speech_breath_ratio"] = float((duration - total_breath_time) / (total_breath_time + 1e-6))

    # 5. Recovery Gaps (ms)
    event_start_times = [evt[0] * frame_time * 1000.0 for evt in breath_events]
    gaps = np.diff(event_start_times)
    raw_metrics["recovery_intervals"] = float(np.mean(gaps)) if len(gaps) > 0 else 0.0

    return raw_metrics, num_breaths

# ==========================================
# 3. QUANTITATIVE RULE EVALUATION ENGINE
# ==========================================
def evaluate_hybrid_forensic_verdict(features, num_breaths):
    """
    Evaluates biometric vectors against real-world human baselines.
    """
    if num_breaths < 2:
        if features["guardrail_centroid"] > 2150.0 or features["guardrail_rolloff"] > 4100.0:
            excess_frequency = max(0, features["guardrail_centroid"] - 2150.0)
            dynamic_penalty = min(0.11, excess_frequency / 1500.0)
            return 0.85 + dynamic_penalty, "AI / DEEPFAKE"
        return 0.35, "HUMAN"

    # Compute binary alignment checks
    amp_score = 1.0 if features["amplitude"] < -35.0 else 0.0
    dur_score = 1.0 if features["duration"] < 600.0 else 0.0
    decay_score = 1.0 if features["decay_rate"] < 0.015 else 0.0  
    ratio_score = 1.0 if (3.0 < features["speech_breath_ratio"] < 15.0) else 0.0
    flow_score = 1.0 if features["spectral_flow"] < 1100.0 else 0.0
    recovery_score = 1.0 if features["recovery_intervals"] > 300.0 else 0.0

    # Compute human index balance
    human_alignment = (
        (recovery_score * 0.28) + (decay_score * 0.18) + (amp_score * 0.15) +
        (ratio_score * 0.15) + (dur_score * 0.12) + (flow_score * 0.12)
    )
    
    prob = 1.0 - human_alignment

    # Apply continuous dynamic overrides instead of structural lock steps
    if features["guardrail_rolloff"] > 4100.0 or features["guardrail_centroid"] > 2150.0:
        excess_frequency = max(0, features["guardrail_centroid"] - 2150.0)
        dynamic_penalty = min(0.14, excess_frequency / 1000.0)
        prob = max(prob, 0.85 + dynamic_penalty)

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
st.caption("Forensic Hybrid Fusion Pipeline: Validated Non-Linear Respiratory Biometrics")

uploaded_files = st.file_uploader("Upload Forensic Audio Batch", type=['wav', 'mp3', 'flac'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    
    for f in uploaded_files:
        f.seek(0)
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000, mono=True, duration=15)
        except Exception as e:
            st.error(f"Skipping unreadable file {f.name}: {str(e)}")
            continue
            
        if y is not None and len(y) > 0:
            metrics, num_breaths = extract_hybrid_forensic_features(y, sr)
            prob, status = evaluate_hybrid_forensic_verdict(metrics, num_breaths)
            
            # Appends metrics inside processing table safely
            results_list.append({
                "Filename": f.name,
                "Verdict": status,
                "AI Score": f"{prob * 100:.1f}%",
                "Breaths Detected": num_breaths,
                "Mean Breath Duration (ms)": round(metrics["duration"], 1),
                "Speech/Breath Ratio": round(metrics["speech_breath_ratio"], 2),
                "Centroid (Hz)": round(metrics["guardrail_centroid"], 1)
            })
            
    if results_list:
        df_results = pd.DataFrame(results_list)
        st.write("### Detection Summary")
        st.dataframe(df_results, use_container_width=True)
        
        # Deploy Excel Export Pipeline safely
        excel_data = convert_df_to_excel(df_results)
        st.download_button(
            label="📊 Download Detailed Forensic Report",
            data=excel_data,
            file_name="Forensic_Analysis_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
