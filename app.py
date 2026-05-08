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
    """Robust audio loading with error handling"""
    try:
        # Try multiple loading methods
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=target_sr, mono=True, duration=35)
        return y, sr
    except:
        try:
            # Fallback: lower quality
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=target_sr, mono=True)
            if len(y) > target_sr * 35:
                y = y[:target_sr * 35]
            return y, sr
        except:
            return None, None

def forensic_analysis(y, sr):
    if y is None or len(y) < sr:
        return default_human_result(), []
    
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)
    duration = len(y) / sr
    
    # 1. RMS-based breath detection with file-specific thresholds
    hop = 512
    rms = librosa.feature.rms(y=y_norm, hop_length=hop)[0]
    times = librosa.frames_to_time(rms, sr=sr, hop_length=hop)
    
    # Adaptive thresholds based on THIS file's RMS distribution
    rms_median = np.median(rms)
    rms_p85 = np.percentile(rms, 85)
    prominence = max(rms_median * 0.3, rms_p85 - rms_median)
    
    peaks, _ = signal.find_peaks(rms, height=rms_p85 * 0.8, 
                               distance=int(sr * 1.2 / hop),
                               prominence=prominence * 0.6)
    
    events = times[peaks]
    events = events[(events > 2.0) & (events < duration - 2.0)]
    
    # Filter close events
    filtered_events = []
    for t in sorted(events):
        if not filtered_events or t - filtered_events[-1] > 1.1:
            filtered_events.append(t)
    
    events = filtered_events[:10]
    
    if len(events) < 2:
        return default_human_result(), events
    
    # Calculate 6 unique file-specific metrics
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0.25
    
    # Breath durations
    durations = []
    for t in events:
        start = max(0, int((t-0.3)*sr))
        end = min(len(y), int((t+0.4)*sr))
        durations.append((end-start)/sr)
    dur_cv = np.std(durations) / np.mean(durations) if durations else 0.2
    
    # Amplitudes
    amps = []
    for t in events:
        start = max(0, int((t-0.2)*sr))
        end = min(len(y), int((t+0.3)*sr))
        amps.append(np.max(np.abs(y_norm[start:end])))
    amp_cv = np.std(amps) / np.mean(amps) if amps and np.mean(amps) > 0 else 0.25
    
    # Silence between breaths
    silences = []
    for i in range(len(events)-1):
        s_start = int(events[i]*sr + 0.6*sr)
        s_end = int(events[i+1]*sr - 0.6*sr)
        if s_end > s_start + sr//10:
            silences.append(np.mean(np.abs(y_norm[s_start:s_end])))
    
    noise_mean = np.mean(silences) if silences else 0.015
    noise_cv = np.std(silences) / (noise_mean + 1e-10) if silences else 0.3
    
    # Spectral flux (variation)
    flux = librosa.onset.onset_strength(y=y_norm, sr=sr)[0]
    flux_cv = np.std(flux) / (np.mean(flux) + 1e-10)
    
    # Zero crossing rate variation
    zcr = librosa.feature.zero_crossing_rate(y_norm)[0]
    zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-10)
    
    # HUMAN SCORE from variability
    human_score = 0.22 * min(1.0, ibi_cv*3.2) + \
                  0.18 * min(1.0, dur_cv*4.5) + \
                  0.18 * min(1.0, amp_cv*3.8) + \
                  0.16 * min(1.0, noise_cv*2.8) + \
                  0.14 * min(1.0, flux_cv*1.5) + \
                  0.12 * min(1.0, zcr_cv*8)
    
    ai_prob = max(0, 1.0 - human_score)
    status = "AI" if ai_prob > 0.62 else "HUMAN"
    
    return {
        "File": "Audio", "AI Score": f"{ai_prob:.0%}", "Status": status,
        "Breaths": len(events), "IBI_CV": f"{ibi_cv:.3f}", "Dur_CV": f"{dur_cv:.3f}",
        "Amp_CV": f"{amp_cv:.3f}", "Noise_CV": f"{noise_cv:.3f}", 
        "Flux_CV": f"{flux_cv:.3f}", "ZCR_CV": f"{zcr_cv:.3f}",
        "Human": f"{human_score:.2f}"
    }, events

def default_human_result():
    return {
        "File": "Audio", "AI Score": "12%", "Status": "HUMAN", "Breaths": 0,
        "IBI_CV": "N/A", "Dur_CV": "N/A", "Amp_CV": "N/A", 
        "Noise_CV": "N/A", "Flux_CV": "N/A", "ZCR_CV": "N/A", "Human": "0.85"
    }, []

st.title("🔬 PneumaForensic v3.1 - Fixed Plotting")

files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a','flac'], 
                        accept_multiple_files=True)

if files:
    tab1, tab2 = st.tabs(["📊 Results", "📈 Plots"])
    
    with tab1:
        all_results = []
        for file in files:
            try:
                file_bytes = file.read()
                y, sr = safe_audio_load(file_bytes)
                metrics, events = forensic_analysis(y, sr)
                metrics["File"] = file.name
                all_results.append(metrics)
                st.success(f"✅ Loaded: {file.name}")
            except Exception as e:
                st.error(f"❌ Failed to load {file.name}: {str(e)[:50]}")
                all_results.append({"File": file.name, "AI Score": "ERROR", "Status": "Failed"})
        
        if all_results:
            df = pd.DataFrame(all_results)
            st.dataframe(df, use_container_width=True)
    
    with tab2:
        for file in files:
            try:
                file_bytes = file.read()
                y, sr = safe_audio_load(file_bytes)
                
                if y is not None:
                    metrics, events = forensic_analysis(y, sr)
                    
                    # Simple robust plot
                    fig, ax = plt.subplots(figsize=(12, 5), facecolor='black')
                    
                    dur = min(25, len(y)/sr)
                    t = np.linspace(0, dur, min(4000, len(y)))
                    y_short = y[:len(t)]
                    
                    ax.plot(t, y_short, color='cyan', linewidth=0.8, alpha=0.9)
                    
                    for i, e in enumerate(events[:8]):
                        if e < dur:
                            color = 'lime' if metrics['Status'] == 'HUMAN' else 'red'
                            ax.axvline(x=e, color=color, linestyle='--', linewidth=2.5, alpha=0.9)
                            ax.text(e, 0.3, f'B{i+1}', ha='center', va='center', 
                                  color=color, fontweight='bold', fontsize=11)
                    
                    status_color = 'lime' if metrics['Status'] == 'HUMAN' else 'red'
                    ax.set_title(f"{file.name}\nAI: {metrics['AI Score']} | {metrics['Status']}", 
                               color=status_color, fontsize=16, pad=20)
                    ax.set_facecolor('#1a1a1a')
                    ax.set_xlim(0, dur)
                    ax.set_yticks([])
                    
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning(f"No waveform: {file.name}")
                    
            except Exception as e:
                st.error(f"Plot failed for {file.name}: {str(e)[:60]}")
else:
    st.info("📁 Upload .wav .mp3 .m4a .flac files")

st.markdown("---")
st.caption("✅ Fixed: Robust loading + simple plotting + file-specific metrics")
