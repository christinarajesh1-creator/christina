import streamlit as st

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Forensic Breath Authenticator", layout="wide")

import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# 2. ROBUST FORENSIC BREATH FEATURE ENGINE
# ==========================================
def extract_6_breath_parameters(y, sr):
    """
    Extracts the exact 6 biometric breath parameters.
    Implements an adaptive energy gate to ignore background room hiss.
    """
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # Track the voice volume envelope using Root-Mean-Square (RMS) Energy
    hop_length = 128  
    rms = librosa.feature.rms(y=y_norm, hop_length=hop_length).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # --- ADAPTIVE NOISE GATE OVERRIDE ---
    # Captures the noise floor of the bottom 15% lowest frames of your audio file
    noise_floor = np.percentile(rms_smooth, 15)
    speech_peak_max = np.max(rms_smooth)
    
    # Adaptive threshold: Look for real physical gaps, ignoring continuous static noise
    adaptive_height = noise_floor + (speech_peak_max - noise_floor) * 0.12
    
    # Locate inhalation valleys using the custom file threshold
    peaks, _ = signal.find_peaks(
        -rms_smooth, 
        height=-adaptive_height, 
        distance=int(sr * 0.35 / hop_length) # Real breaths occur at least 350ms apart
    )
    
    # Filter edge artifacts from the start/end boundaries of the clip
    detected_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    breath_times = [t for t in detected_times if 0.4 < t < (duration - 0.4)]
    num_breaths = len(breath_times)

    # Dictionary mapping your exact 6 requested parameters
    raw_metrics = {
        "ibi_reg": 0.0,       
        "amp_var": 0.0,       
        "dur_var": 0.0,       
        "presence": 0.0,      
        "spectral_cont": 0.0, 
        "similarity": 0.0     
    }

    if num_breaths < 2:
        return raw_metrics, breath_times

    # 1. IBI Regularity (Coefficient of Variation of Inter-Breath Intervals)
    ibi = np.diff(breath_times)
    raw_metrics["ibi_reg"] = float(np.std(ibi) / np.mean(ibi)) if len(ibi) > 0 else 0.0

    # 2. Breath Amplitude (Coefficient of Variation of peak RMS power)
    amp_values = [rms_smooth[p] for p in peaks if p < len(rms_smooth)]
    raw_metrics["amp_var"] = float(np.std(amp_values) / np.mean(amp_values)) if len(amp_values) > 0 else 0.0

    # 3. Breath Duration (Standard Deviation of pulse peak width mappings)
    widths_data = signal.peak_widths(-rms_smooth, peaks, rel_height=0.5)
    widths_seconds = widths_data[0] * (hop_length / sr) if len(widths_data) > 0 else np.array([0.0])
    raw_metrics["dur_var"] = float(np.std(widths_seconds)) if len(widths_seconds) > 0 else 0.0

    # 4. Breath Presence (Ratio of breath frames relative to total timeline duration)
    total_breath_duration = np.sum(widths_seconds)
    raw_metrics["presence"] = float(total_breath_duration / duration)

    # 5. Spectral Continuity (Zero-Crossing Rate delta changes at breath boundaries)
    zcr = librosa.feature.zero_crossing_rate(y=y_norm, hop_length=hop_length).flatten()
    zcr_deltas = []
    for p in peaks:
        start_f, end_f = max(0, p - 4), min(len(zcr) - 1, p + 4)
        zcr_deltas.append(np.max(np.abs(np.diff(zcr[start_f:end_f]))))
    raw_metrics["spectral_cont"] = float(np.max(zcr_deltas)) if len(zcr_deltas) > 0 else 0.0

    # 6. Breath Similarity (Cross-correlation matrix of MFCC profiles across segments)
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
# 3. HIGH-ACCURACY SCORING LOGIC
# ==========================================
def calculate_forensic_verdict(raw_features, num_breaths):
    """
    Computes an objective AI Probability based on physiological boundaries.
    """
    if num_breaths < 2:
        return 0.950, "AI / DEEPFAKE"

    # Score component 1: Timing regularity
    if raw_features["ibi_reg"] < 0.20: # Unnaturally perfect mechanical grid spacing
        ibi_score = 1.0
    elif raw_features["ibi_reg"] > 0.40: # Dynamic natural human speech pattern variations
        ibi_score = 0.0
    else:
        ibi_score = 0.5

    # Score component 2: Volume changes
    amp_score = 1.0 if raw_features["amp_var"] < 0.15 else 0.0

    # Score component 3: Uniform length clones
    dur_score = 1.0 if raw_features["dur_var"] < 0.05 else 0.0

    # Score component 4: Density footprint check
    if raw_features["presence"] > 0.28 or raw_features["presence"] < 0.02:
        presence_score = 1.0
    else:
        presence_score = 0.0

    # Score component 5: Computational spectral discontinuities
    cont_score = 1.0 if raw_features["spectral_cont"] < 0.05 else 0.0

    # Score component 6: Reused voice generation assets
    sim_score = 1.0 if raw_features["similarity"] > 0.82 else 0.0

    # Calculate absolute probability distribution based on requested weights
    ai_probability = (
        (ibi_score * 0.28) +
        (amp_score * 0.15) +
        (dur_score * 0.12) +
        (presence_score * 0.15) +
        (cont_score * 0.12) +
        (sim_score * 0.18)
    )

    ai_probability = max(0.01, min(0.99, ai_probability))
    status = "AI / DEEPFAKE" if ai_probability >= 0.50 else "HUMAN"
    
    return ai_probability, status

# ==========================================
# 4. STREAMLIT INTERFACE & RUNTIME
# ==========================================
st.title("🔬 Deepfake Voice Detection Engine")
st.caption("Forensic Analysis Pipeline Mapped to Biomimetic Breath-Anomaly Parameters")

uploaded_files = st.file_uploader("Upload Forensic Audio Batch", type=['wav', 'mp3', 'flac'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    
    for f in uploaded_files:
        f.seek(0)
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=22050, mono=True, duration=30)
        except Exception as e:
            st.error(f"Skipping unreadable file layout {f.name}: {str(e)}")
            continue
            
        if y is not None and len(y) > 0:
            try:
                # 1. Run signal processing layer to extract parameters
                raw_features, breath_times = extract_6_breath_parameters(y, sr)
                
                # 2. Run deterministic forensic weight metrics logic
                ai_probability, status = calculate_forensic_verdict(raw_features, len(breath_times))
                
                # Map directly to your 6 specific reporting fields
                results_list.append({
                    "File Name": f.name,
                    "Verdict": status,
                    "AI Probability": f"{ai_probability:.1%}",
                    "IBI Regularity (28%)": f"{raw_features['ibi_reg']:.4f}",
                    "Breath Amplitude (15%)": f"{raw_features['amp_var']:.4f}",
                    "Breath Duration (12%)": f"{raw_features['dur_var']:.4f}",
                    "Breath Presence (15%)": f"{raw_features['presence']:.1%}",
                    "Spectral Continuity (12%)": f"{raw_features['spectral_cont']:.4f}",
                    "Breath Similarity (18%)": f"{raw_features['similarity']:.1%}"
                })
                
                # --- VISUAL GRAPH GENERATION LOOP ---
                with st.expander(f"Waveform Visual Analysis: {f.name} ➔ {status}"):
                    fig, ax = plt.subplots(figsize=(14, 2.2))
                    
                    # The Gray Waves: Representing raw voice structure
                    time_axis = np.linspace(0, len(y)/sr, len(y))
                    ax.plot(time_axis, y, color='darkgray', alpha=0.7, linewidth=0.5, label="The Gray Waves (Speech Profile)")
                    
                    # The Red Dashed Lines: Marking each detected breath peak event location
                    is_first_marker = True
                    for b_time in breath_times:
                        if is_first_marker:
                            ax.axvline(x=b_time, color='red', linestyle='--', linewidth=1.2, label="The Red Dashed Lines (Breath Event)")
                            is_first_marker = False
                        else:
                            ax.axvline(x=b_time, color='red', linestyle='--', linewidth=1.2)
                    
                    ax.set_title("Biomimetic Spacing Timeline Analysis", fontsize=9)
                    ax.set_xlim(0, len(y)/sr)
                    ax.set_xlabel("Time Timeline (Seconds)", fontsize=8)
                    ax.set_ylabel("Amplitude", fontsize=8)
                    ax.legend(loc="upper right", fontsize=7)
                    ax.grid(False)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
            except Exception as pipeline_err:
                st.error(f"Error processing parameters for {f.name}: {str(pipeline_err)}")

    if results_list:
        st.subheader("📋 Final Operational Assessment Matrix (6-Parameter Report)")
        df = pd.DataFrame(results_list)
        st.dataframe(df, use_container_width=True)
