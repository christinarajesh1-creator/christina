import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(layout="wide")

@st.cache_data
def safe_audio_load(file_bytes, target_sr=16000):
    try:
        # Use io.BytesIO to keep the file in memory
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=target_sr, mono=True, duration=35)
        return y, sr
    except Exception as e:
        return None, None

def forensic_analysis(y, sr, filename="Audio"):
    if y is None or len(y) < sr:
        return default_human_result(filename), []
    
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)
    duration = len(y) / sr
    
    # 1. Improved Breath Detection (RMS Energy)
    hop = 512
    rms = librosa.feature.rms(y=y_norm, hop_length=hop)[0]
    times = librosa.frames_to_time(rms, sr=sr, hop_length=hop)
    
    # Adaptive thresholding for breath spikes
    rms_median = np.median(rms)
    rms_std = np.std(rms)
    peaks, _ = signal.find_peaks(rms, height=rms_median + (0.5 * rms_std), 
                               distance=int(sr * 1.0 / hop))
    
    events = times[peaks]
    events = events[(events > 0.5) & (events < duration - 0.5)]
    
    # Filter close events (Breaths usually have a refractory period)
    filtered_events = []
    for t in sorted(events):
        if not filtered_events or t - filtered_events[-1] > 0.8:
            filtered_events.append(t)
    events = filtered_events
    
    if len(events) < 2:
        # Lack of breath is a strong AI indicator in long clips
        return default_human_result(filename) if duration < 5 else default_ai_result(filename), events

    # --- THE 6 PARAMETERS ---
    # 1. Inter-Breath Interval (IBI) CV: Human rhythm is irregular
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if len(ibis) > 0 else 0
    
    # 2. Breath Duration CV: Humans vary breath length
    durs = []
    for t in events:
        start, end = max(0, int((t-0.2)*sr)), min(len(y), int((t+0.2)*sr))
        durs.append(np.sqrt(np.mean(y[start:end]**2)))
    dur_cv = np.std(durs) / np.mean(durs) if len(durs) > 0 else 0

    # 3. Peak Amplitude CV: Humans don't inhale with the same force
    amps = [np.max(np.abs(y_norm[max(0, int((t-0.1)*sr)):min(len(y), int((t+0.1)*sr))])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0

    # 4. Background Noise Consistency (Silence CV): AI often has "perfect" digital silence
    zcr = librosa.feature.zero_crossing_rate(y_norm)[0]
    zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-10)

    # 5. Spectral Flux (Timbre variation)
    flux = librosa.onset.onset_strength(y=y_norm, sr=sr)
    flux_cv = np.std(flux) / (np.mean(flux) + 1e-10)

    # 6. High-Frequency Decay (AI often cuts off at 8kHz or 16kHz)
    spec = np.abs(librosa.stft(y))
    high_freq_energy = np.mean(spec[int(spec.shape[0]*0.8):, :])
    low_freq_energy = np.mean(spec[:int(spec.shape[0]*0.2), :])
    hf_ratio = high_freq_energy / (low_freq_energy + 1e-10)

    # SCORING LOGIC: Low variability (CV) = AI; High variability = HUMAN
    # hf_ratio: Higher usually means more natural harmonic "air"
    human_var_score = (ibi_cv * 2.0) + (dur_cv * 1.5) + (amp_cv * 1.5) + (flux_cv * 0.5) + (zcr_cv * 0.2)
    
    # AI voices usually have CVs < 0.1 for timing
    ai_prob = 1.0 - (human_var_score / 2.5) 
    ai_prob = np.clip(ai_prob, 0.05, 0.98)
    
    status = "AI" if ai_prob > 0.55 else "HUMAN"
    
    return {
        "File": filename, "AI Probability": f"{ai_prob:.0%}", "Status": status,
        "Breaths Detected": len(events), "Timing_Var": f"{ibi_cv:.3f}", 
        "Amp_Var": f"{amp_cv:.3f}", "Spectral_Flux": f"{flux_cv:.3f}",
        "HF_Ratio": f"{hf_ratio:.4f}"
    }, events

def default_human_result(name):
    return {"File": name, "AI Probability": "15%", "Status": "HUMAN", "Breaths Detected": 0}, []

def default_ai_result(name):
    return {"File": name, "AI Probability": "85%", "Status": "AI (No Breaths)", "Breaths Detected": 0}, []

st.title("🔬 Forensic Voice Analysis")

uploaded_files = st.file_uploader("Upload Audio", type=['wav','mp3','m4a'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    
    for uploaded_file in uploaded_files:
        # Reset file pointer for every loop
        uploaded_file.seek(0)
        bytes_data = uploaded_file.read()
        y, sr = safe_audio_load(bytes_data)
        
        if y is not None:
            metrics, events = forensic_analysis(y, sr, uploaded_file.name)
            results_list.append(metrics)
            
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#1f77b4', alpha=0.7)
            for e in events:
                ax.axvline(x=e, color='red', linestyle='--', alpha=0.6)
            ax.set_title(f"{uploaded_file.name} - {metrics['Status']} ({metrics['AI Probability']} AI)")
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

    st.subheader("Deep Metric Analysis")
    st.dataframe(pd.DataFrame(results_list), use_container_width=True)
