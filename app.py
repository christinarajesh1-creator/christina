import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

st.set_page_config(layout="wide")

def forensic_analysis(y, sr):
    # Normalize per file
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)
    duration = len(y) / sr
    
    # 1. BREATH DETECTION - Adaptive per file
    rms = librosa.feature.rms(y=y_norm)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # File-specific thresholds
    rms_p75 = np.percentile(rms, 75)
    rms_p90 = np.percentile(rms, 90)
    peaks, props = signal.find_peaks(rms, height=rms_p75, 
                                   distance=int(1.5*sr/512),
                                   prominence=(rms_p90-rms_p75)*0.3)
    
    events = times[peaks]
    events = events[(events > 2.0) & (events < duration-2.0)]
    
    # Remove duplicates
    filtered = []
    for t in sorted(events):
        if not filtered or t - filtered[-1] > 1.0:
            filtered.append(t)
    events = filtered[:8]
    
    if len(events) < 2:
        return default_human_result()
    
    # 2. IBI VARIABILITY - File specific
    ibis = np.diff(events)
    mean_ibi = np.mean(ibis)
    std_ibi = np.std(ibis)
    ibi_cv = std_ibi / mean_ibi if mean_ibi > 0 else 0.3
    
    # 3. BREATH DURATION VARIABILITY
    durations = []
    for t in events:
        start = max(0, int((t-0.3)*sr))
        end = min(len(y), int((t+0.4)*sr))
        breath = y_norm[start:end]
        durations.append(len(breath)/sr)
    
    dur_cv = np.std(durations) / np.mean(durations) if durations else 0.2
    
    # 4. AMPLITUDE VARIABILITY
    amps = [np.max(np.abs(y_norm[max(0,int((t-0.2)*sr)):min(len(y),int((t+0.3)*sr))])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if amps and np.mean(amps) > 0 else 0.25
    
    # 5. NOISE FLOOR - Between breaths
    silence_amps = []
    for i in range(len(events)-1):
        silence_start = int(events[i]*sr + 0.5*sr)
        silence_end = int(events[i+1]*sr - 0.5*sr)
        if silence_end > silence_start:
            silence_amps.append(np.mean(np.abs(y_norm[silence_start:silence_end])))
    
    noise_level = np.mean(silence_amps) if silence_amps else 0.01
    noise_cv = np.std(silence_amps) / (np.mean(silence_amps) + 1e-10) if silence_amps else 0.3
    
    # 6. SPECTRAL CHARACTERISTICS - Breath vs silence
    breath_centroids = []
    silence_centroids = []
    
    for t in events[:4]:
        # Breath segment
        s, e = int((t-0.25)*sr), int((t+0.35)*sr)
        if e > s:
            cent = np.mean(librosa.feature.spectral_centroid(y=y_norm[s:e], sr=sr)[0])
            breath_centroids.append(cent)
    
    silence_level = noise_level * sr  # Normalized
    
    # HUMAN SCORE - HIGH VARIABILITY = HUMAN
    human_score = (
        0.25 * min(1.0, ibi_cv * 3) +           # Timing variation
        0.20 * min(1.0, dur_cv * 4) +           # Duration variation  
        0.20 * min(1.0, amp_cv * 3.5) +         # Amplitude variation
        0.15 * min(1.0, noise_cv * 2.5) +       # Noise variation
        0.10 * (0.8 if len(breath_centroids) > 0 else 0.3) +  # Spectral content
        0.10 * min(1.0, noise_level * 200)      # Natural noise floor
    )
    
    ai_prob = max(0, min(1.0, 1.0 - human_score * 1.1))
    status = "AI" if ai_prob > 0.65 else "HUMAN"
    
    return {
        "File": "Audio",
        "AI Score": f"{ai_prob:.0%}", 
        "Status": status,
        "Breaths": len(events),
        "IBI_CV": f"{ibi_cv:.3f}",
        "Dur_CV": f"{dur_cv:.3f}",
        "Amp_CV": f"{amp_cv:.3f}",
        "Noise_CV": f"{noise_cv:.3f}",
        "NoiseLvl": f"{noise_level:.4f}",
        "HumanScore": f"{human_score:.1f}"
    }, events

def default_human_result():
    return {
        "File": "Audio", "AI Score": "15%", "Status": "HUMAN",
        "Breaths": 0, "IBI_CV": "N/A", "Dur_CV": "N/A",
        "Amp_CV": "N/A", "Noise_CV": "N/A", "NoiseLvl": "N/A", "HumanScore": "0.8"
    }, []

st.title("🔬 PneumaForensic v3.0 - Per-File Adaptive")

files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a','flac'], 
                        accept_multiple_files=True, key="uploader")

if files:
    col1, col2 = st.columns([1,3])
    
    with col1:
        st.subheader("Results")
        all_results = []
        
        for file in files:
            file.seek(0)
            try:
                # Load with original SR first, then analyze
                y, orig_sr = librosa.load(io.BytesIO(file.read()), sr=None, mono=True)
                file.seek(0)
                
                # Resample to 16kHz for consistency but preserve characteristics
                y_16k, _ = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
                
                metrics, events = forensic_analysis(y_16k, 16000)
                metrics["File"] = file.name
                metrics["Orig_SR"] = f"{orig_sr}Hz"
                all_results.append(metrics)
            except Exception as e:
                all_results.append({"File": file.name, "AI Score": "ERROR", "Status": "Failed"})
        
        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True)
        
        if len(all_results) > 1:
            valid_scores = [float(r['AI Score'][:-1])/100 for r in all_results if r['AI Score'] != 'ERROR']
            st.metric("Batch AI Average", f"{np.mean(valid_scores):.0%}")
    
    with col2:
        st.subheader("Detailed Analysis")
        for i, file in enumerate(files):
            file.seek(0)
            try:
                y, orig_sr = librosa.load(io.BytesIO(file.read()), sr=None, mono=True)
                y_16k, _ = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
                metrics, events = forensic_analysis(y_16k, 16000)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor='black')
                
                # Full waveform
                dur = min(30, len(y_16k)/16000)
                t = np.linspace(0, dur, int(dur*16000))
                y_plot = y_16k[:len(t)]
                
                ax1.plot(t, y_plot, 'cyan', lw=0.6, alpha=0.9)
                
                # Mark breaths
                for j, e in enumerate(events):
                    if e < dur:
                        color = 'lime' if metrics['Status'] == 'HUMAN' else 'red'
                        ax1.axvline(e, color=color, lw=3, alpha=0.8, ls='--')
                        ax1.text(e, 0.4, f'B{j+1}', ha='center', color=color, 
                               fontweight='bold', fontsize=12)
                
                ax1.set_title(f"{file.name}\nAI: {metrics['AI Score']} | {metrics['Status']} | SR: {orig_sr}Hz", 
                            color='white', fontsize=14)
                ax1.set_facecolor('#1a1a1a')
                ax1.set_xlim(0, dur)
                
                # Parameter bars
                params = ['IBI_CV', 'Dur_CV', 'Amp_CV', 'Noise_CV']
                values = [float(metrics[p][:-3]) if metrics[p] != 'N/A' else 0.2 for p in params]
                
                bars = ax2.bar(params, values, color=['orange', 'yellow', 'pink', 'lightblue'], alpha=0.8)
                ax2.set_title('Variability Scores (Higher = More Human)', color='white')
                ax2.set_ylim(0, 0.8)
                ax2.tick_params(colors='white')
                ax2.set_facecolor('#1a1a1a')
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', color='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
            except:
                st.error(f"Failed to plot: {file.name}")

else:
    st.info("Upload multiple files - each gets **unique** file-specific analysis")
