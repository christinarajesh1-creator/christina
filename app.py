import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

st.set_page_config(layout="wide")

def forensic_analysis(y, sr):
    duration = len(y) / sr
    
    # 1. BREATH DETECTION - RMS peaks
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    events = []
    for i in range(200, len(rms)-200):
        if rms[i] > np.percentile(rms, 85) and rms[i] > rms[i-1] and rms[i] > rms[i+1]:
            t = times[i]
            if t > 1.5:
                events.append(t)
    
    # Filter spacing
    filtered_events = []
    last_t = 0
    for t in sorted(events):
        if t - last_t > 1.0:
            filtered_events.append(t)
            last_t = t
    
    events = filtered_events[:8]
    
    if len(events) < 2:
        return {
            "File": "N/A",
            "AI Score": "95%",
            "Status": "AI",
            "Breaths": 0,
            "Timing": "100%",
            "Purity": "100%",
            "Noise": "100%",
            "AmpVar": "100%"
        }, []
    
    # 1. TIMING REGULARITY - AI = LOW CV
    ibis = np.diff(events)
    timing_cv = np.std(ibis) / np.mean(ibis)
    timing_ai = min(1.0, max(0, (0.3 - timing_cv) * 5))  # Low CV = high AI
    
    # 2. BREATH SPECTRAL PURITY - AI = CLEAN
    purity_scores = []
    for t in events:
        start = max(0, int((t-0.15)*sr))
        end = min(len(y), int((t+0.25)*sr))
        breath = y[start:end]
        if len(breath) > sr//4:
            purity = np.mean(librosa.feature.spectral_flatness(breath))
            purity_scores.append(purity)
    
    avg_purity = np.mean(purity_scores)
    purity_ai = min(1.0, max(0, (avg_purity - 0.35) * 3))
    
    # 3. NOISE FLOOR - AI = SILENT
    silence_mask = np.abs(y) < np.percentile(np.abs(y), 15)
    noise_floor = np.std(y[silence_mask])
    noise_ai = min(1.0, max(0, 1.0 - noise_floor * 8000))
    
    # 4. AMPLITUDE VARIATION - AI = CONSISTENT
    amps = []
    for t in events:
        start = max(0, int((t-0.1)*sr))
        end = min(len(y), int((t+0.2)*sr))
        amp = np.max(np.abs(y[start:end]))
        amps.append(amp)
    
    amp_cv = np.std(amps) / np.mean(amps) if amps else 1
    amp_ai = min(1.0, max(0, (0.4 - amp_cv) * 3))
    
    # TOTAL SCORE
    score = 0.28*timing_ai + 0.25*purity_ai + 0.25*noise_ai + 0.22*amp_ai
    
    status = "AI" if score > 0.6 else "HUMAN"
    
    metrics = {
        "File": "Audio",
        "AI Score": f"{score:.0%}",
        "Status": status,
        "Breaths": len(events),
        "Timing": f"{timing_ai:.0%}",
        "Purity": f"{purity_ai:.0%}",
        "Noise": f"{noise_ai:.0%}",
        "AmpVar": f"{amp_ai:.0%}",
        "CV": f"{timing_cv:.2f}"
    }
    
    return metrics, events

st.title("PneumaForensic")

files = st.file_uploader("Upload", type=['wav','mp3','m4a'], accept_multiple_files=True)

if files:
    tab1, tab2 = st.tabs(["📊 Results", "📈 Graphs"])
    
    all_results = []
    
    with tab1:
        for file in files:
            file.seek(0)
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
                metrics, events = forensic_analysis(y, sr)
                metrics["File"] = file.name
                all_results.append(metrics)
            except:
                all_results.append({"File": file.name, "AI Score": "ERROR"})
        
        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        for file in files:
            file.seek(0)
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
                metrics, events = forensic_analysis(y, sr)
                
                fig, ax = plt.subplots(figsize=(12, 4))
                dur = min(25, len(y)/sr)
                t = np.linspace(0, dur, min(4000, len(y)))
                y_short = y[:len(t)]
                
                ax.plot(t, y_short, color='#00d4ff', lw=0.6, alpha=0.8)
                
                for i, e in enumerate(events):
                    if e < dur:
                        ax.axvline(e, color='red', ls='--', lw=2.5, alpha=0.9)
                        ax.text(e, 0.1, str(i+1), ha='center', color='red', fontweight='bold')
                
                color = 'red' if float(metrics['AI Score'][:-1])/100 > 0.6 else 'green'
                ax.set_title(f"{file.name} | {metrics['AI Score']} | {metrics['Status']}", 
                           color=color, fontsize=14)
                ax.set_facecolor('#111111')
                ax.set_xlim(0, dur)
                ax.set_ylim(-0.8, 0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            except:
                st.write(f"No graph: {file.name}")

if not files:
    st.info("Upload audio files to analyze breathing patterns")
