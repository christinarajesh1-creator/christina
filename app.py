import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
from scipy.optimize import curve_fit

# Secure Excel exporter dependency
try:
    import xlsxwriter
except ImportError:
    st.error("Missing dependency: run 'pip install xlsxwriter'")

# ==========================================
# 1. INITIALIZATION
# ==========================================
st.set_page_config(page_title="Forensic Voice Authenticator", layout="wide")

# ==========================================
# 2. BOUNDARY CRITERIA (Appendix A - Exact)
# ==========================================
BOUNDARIES = {
    "amplitude_drop_min_db": 30.0,
    "amplitude_drop_max_db": 60.0,
    "duration_min_ms": 100.0,
    "duration_max_ms": 2500.0,
    "decay_r_squared": 0.85,
    "ratio_min": 0.05,
    "ratio_max": 0.35,
    "spectral_min_hz": 300.0,
    "spectral_max_hz": 2000.0,
    "recovery_min_ms": 300.0,
    "centroid_max_hz": 2150.0,
    "rolloff_max_hz": 4100.0
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-9))

def compute_spectral_band_energy(y, sr, hop_length, band_min, band_max):
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    band_min_idx = np.searchsorted(freqs, band_min)
    band_max_idx = np.searchsorted(freqs, band_max)
    band_energy = np.sum(S[band_min_idx:band_max_idx, :] ** 2)
    total_energy = np.sum(S ** 2) + 1e-9
    return band_energy / total_energy

# ==========================================
# 4. FEATURE EXTRACTION ENGINE
# ==========================================
def extract_hybrid_forensic_features(y, sr):
    """
    Extracts the 6-parameter biological breath matrix:
    1. Amplitude (dB drop from speech)
    2. Duration (ms)
    3. Decay Rate (R² exponential fit)
    4. Speech-to-Breath Ratio (balance index)
    5. Spectral Flow (300-2000 Hz band)
    6. Recovery Intervals (ms)
    """
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    hop_length = 256
    frame_time = hop_length / sr
    
    # Global speech amplitude reference
    rms = librosa.feature.rms(y=y_norm, hop_length=hop_length).flatten()
    rms_smooth = np.convolve(rms, np.ones(3)/3, mode='same')
    peak_speech_rms = np.percentile(rms_smooth, 95)
    peak_speech_db = 20 * np.log10(peak_speech_rms + 1e-6)
    
    # Acoustic features
    flatness = librosa.feature.spectral_flatness(y=y_norm, hop_length=hop_length).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=y_norm, hop_length=hop_length).flatten()
    
    rms_min, rms_max = np.min(rms_smooth), np.max(rms_smooth)
    rms_rel = (rms_smooth - rms_min) / (rms_max - rms_min + 1e-6)
    
    # Identify breath zones
    breath_indices = []
    for i in range(len(rms_rel)):
        if (0.02 < rms_rel[i] < 0.30) and (flatness[i] > 0.08) and (zcr[i] > 0.10):
            breath_indices.append(i)
    
    # Cluster into events
    breath_events = []
    if breath_indices:
        current_event = [breath_indices[0]]
        for idx in breath_indices[1:]:
            if idx == current_event[-1] + 1:
                current_event.append(idx)
            else:
                dur_ms = len(current_event) * frame_time * 1000.0
                if BOUNDARIES["duration_min_ms"] <= dur_ms <= BOUNDARIES["duration_max_ms"]:
                    breath_events.append(current_event)
                current_event = [idx]
        
        dur_ms = len(current_event) * frame_time * 1000.0
        if BOUNDARIES["duration_min_ms"] <= dur_ms <= BOUNDARIES["duration_max_ms"]:
            breath_events.append(current_event)
    
    num_breaths = len(breath_events)
    
    # Initialize 6-parameter matrix
    raw_metrics = {
        "amplitude": -60.0,
        "duration": 0.0,
        "decay_rate": 0.0,
        "speech_breath_ratio": 0.0,
        "spectral_flow": 0.0,
        "recovery_intervals": 0.0,
        "guardrail_rolloff": 0.0,
        "guardrail_centroid": 0.0
    }
    
    # Spectral guardrails
    rolloff = librosa.feature.spectral_rolloff(y=y_norm, sr=sr, roll_percent=0.85)
    raw_metrics["guardrail_rolloff"] = float(np.mean(rolloff))
    
    centroid = librosa.feature.spectral_centroid(y=y_norm, sr=sr)
    raw_metrics["guardrail_centroid"] = float(np.mean(centroid))
    
    if num_breaths < 2:
        return raw_metrics, num_breaths
    
    # PARAMETER 1: Amplitude Drop
    all_breath_frames = [frame for evt in breath_events for frame in evt]
    breath_rms_vals = rms_smooth[all_breath_frames]
    breath_amplitude_db = 20 * np.log10(np.mean(breath_rms_vals) + 1e-6)
    amplitude_drop = peak_speech_db - breath_amplitude_db
    raw_metrics["amplitude"] = float(amplitude_drop)
    
    # PARAMETER 2: Duration
    durations = [len(evt) * frame_time * 1000.0 for evt in breath_events]
    raw_metrics["duration"] = float(np.mean(durations))
    
    # PARAMETER 3: Decay Rate (R²)
    r_squared_values = []
    for evt in breath_events:
        if len(evt) >= 5:
            env = rms_smooth[evt]
            x = np.arange(len(env))
            try:
                env_norm = env / (np.max(env) + 1e-9)
                popt, _ = curve_fit(exponential_decay, x, env_norm, p0=[1.0, 0.1], maxfev=1000)
                y_pred = exponential_decay(x, *popt)
                r2 = compute_r_squared(env_norm, y_pred)
                r_squared_values.append(r2)
            except:
                r_squared_values.append(0.0)
    
    raw_metrics["decay_rate"] = float(np.mean(r_squared_values)) if r_squared_values else 0.0
    
    # PARAMETER 4: Speech-to-Breath Ratio
    total_breath_time = sum(len(evt) * frame_time for evt in breath_events)
    total_speech_time = duration - total_breath_time
    raw_metrics["speech_breath_ratio"] = float(total_speech_time / (duration + 1e-6))
    
    # PARAMETER 5: Spectral Flow
    band_energy_ratios = []
    for evt in breath_events:
        breath_y = y_norm[int(evt[0] * hop_length):int(evt[-1] * hop_length) + hop_length]
        if len(breath_y) > hop_length:
            band_ratio = compute_spectral_band_energy(
                breath_y, sr, hop_length,
                BOUNDARIES["spectral_min_hz"],
                BOUNDARIES["spectral_max_hz"]
            )
            band_energy_ratios.append(band_ratio)
    
    raw_metrics["spectral_flow"] = float(np.mean(band_energy_ratios)) if band_energy_ratios else 0.0
    
    # PARAMETER 6: Recovery Intervals
    event_times = [evt[0] * frame_time for evt in breath_events]
    if len(event_times) >= 2:
        gaps_ms = np.diff(event_times) * 1000.0
        raw_metrics["recovery_intervals"] = float(np.mean(gaps_ms))
    
    return raw_metrics, num_breaths

# ==========================================
# 5. VERDICT ENGINE
# ==========================================
def evaluate_hybrid_forensic_verdict(features, num_breaths):
    """
    Evaluates against paper's exact boundary criteria.
    """
    # Fallback for < 2 breaths
    if num_breaths < 2:
        if features["guardrail_centroid"] > BOUNDARIES["centroid_max_hz"] or \
           features["guardrail_rolloff"] > BOUNDARIES["rolloff_max_hz"]:
            excess_freq = max(0, features["guardrail_centroid"] - BOUNDARIES["centroid_max_hz"])
            dynamic_penalty = min(0.11, excess_freq / 1500.0)
            return 0.85 + dynamic_penalty, "AI / DEEPFAKE"
        return 0.35, "HUMAN"
    
    # Binary alignment checks
    amp_score = 1.0 if (BOUNDARIES["amplitude_drop_min_db"] <= features["amplitude"] <= 
                       BOUNDARIES["amplitude_drop_max_db"]) else 0.0
    
    dur_score = 1.0 if (BOUNDARIES["duration_min_ms"] <= features["duration"] <= 
                       BOUNDARIES["duration_max_ms"]) else 0.0
    
    decay_score = 1.0 if features["decay_rate"] > BOUNDARIES["decay_r_squared"] else 0.0
    
    ratio_score = 1.0 if (BOUNDARIES["ratio_min"] <= features["speech_breath_ratio"] <= 
                         BOUNDARIES["ratio_max"]) else 0.0
    
    flow_score = 1.0 if features["spectral_flow"] > 0.30 else 0.0
    
    recovery_score = 1.0 if features["recovery_intervals"] >= BOUNDARIES["recovery_min_ms"] else 0.0
    
    # Weighted human alignment
    human_alignment = (
        (recovery_score * 0.28) +
        (decay_score * 0.18) +
        (amp_score * 0.15) +
        (ratio_score * 0.15) +
        (dur_score * 0.12) +
        (flow_score * 0.12)
    )
    
    prob = 1.0 - human_alignment
    
    # Dynamic overrides
    if features["guardrail_rolloff"] > BOUNDARIES["rolloff_max_hz"] or \
       features["guardrail_centroid"] > BOUNDARIES["centroid_max_hz"]:
        excess_freq = max(0, features["guardrail_centroid"] - BOUNDARIES["centroid_max_hz"])
        dynamic_penalty = min(0.14, excess_freq / 1000.0)
        prob = max(prob, 0.85 + dynamic_penalty)
    
    prob = max(0.01, min(0.99, prob))
    status = "AI / DEEPFAKE" if prob >= 0.50 else "HUMAN"
    
    return prob, status

# ==========================================
# 6. EXPORT UTILITY
# ==========================================
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forensic_Report')
    return output.getvalue()

# ==========================================
# 7. STREAMLIT UI
# ==========================================
st.title("🔬 Deepfake Voice Detection Engine")
st.caption("Forensic Hybrid Fusion Pipeline - 6-Parameter Biological Breath Matrix")
st.markdown("---")

uploaded_files = st.file_uploader(
    "Upload audio files (WAV, MP3, FLAC)",
    type=['wav', 'mp3', 'flac'],
    accept_multiple_files=True
)

if uploaded_files:
    results_list = []
    progress_bar = st.progress(0)
    
    for i, f in enumerate(uploaded_files):
        f.seek(0)
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000, mono=True, duration=15)
        except Exception as e:
            st.error(f"Error loading {f.name}: {e}")
            continue
        
        if y is not None and len(y) > 0:
            metrics, num_breaths = extract_hybrid_forensic_features(y, sr)
            prob, status = evaluate_hybrid_forensic_verdict(metrics, num_breaths)
            
            results_list.append({
                "Filename": f.name,
                "Verdict": status,
                "AI Score": f"{prob * 100:.1f}%",
                "Breaths": num_breaths,
                "Amp Drop (dB)": round(metrics["amplitude"], 1),
                "Duration (ms)": round(metrics["duration"], 1),
                "Decay R²": round(metrics["decay_rate"], 3),
                "Speech/Breath": round(metrics["speech_breath_ratio"], 3),
                "Spectral Flow": round(metrics["spectral_flow"], 3),
                "Recovery (ms)": round(metrics["recovery_intervals"], 1),
                "Centroid (Hz)": round(metrics["guardrail_centroid"], 1),
                "Rolloff (Hz)": round(metrics["guardrail_rolloff"], 1)
            })
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    if results_list:
        df_results = pd.DataFrame(results_list)
        
        st.markdown("---")
        st.markdown("### 📊 Detection Summary")
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        total = len(df_results)
        ai_count = df_results[df_results["Verdict"] == "AI / DEEPFAKE"].shape[0]
        human_count = total - ai_count
        
        with col1:
            st.metric("Total Files", total)
        with col2:
            st.metric("AI Detected", ai_count, delta_color="inverse")
        with col3:
            st.metric("Human Verified", human_count)
        
        # Excel Export
        excel_data = convert_df_to_excel(df_results)
        st.download_button(
            label="📥 Download Forensic Report (Excel)",
            data=excel_data,
            file_name="Forensic_Analysis_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Requirements
"""
# requirements.txt
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
xlsxwriter>=3.0.0
scipy>=1.10.0
streamlit>=1.28.0
"""
