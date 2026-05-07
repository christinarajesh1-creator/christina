import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def detect_roger_ai(y, sr):
    """Detects ElevenLabs Roger AI - high score = AI"""
    
    # Find breath-like energy peaks
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Roger AI breaths: high energy, regular spacing
    peaks = []
    for i in range(500, len(rms)-500):
        if (rms[i] > np.percentile(rms, 92) and 
            rms[i] > rms[i-1]*1.2 and rms[i] > rms[i+1]*1.2):
            peaks.append(times[i])
    
    # Filter minimum spacing
    events = []
    last_t = 0
    for p in peaks:
        if p > 2 and p - last_t > 1.0:
            events.append(p)
            last_t = p
    
    if len(events) < 3:
        return 0.95, events  # Few breaths = AI
    
    # KEY AI SIGNALS
    ibis = np.diff(events)
    
    # 1. PERFECT REGULARITY (Roger CV < 0.25)
    cv = np.std(ibis) / np.mean(ibis)
    timing_ai = 1.0 if cv < 0.25 else 0.1
    
    # 2. BREATH CLEANLINESS
    clean_breaths = 0
    for t in events[:5]:
        start = max(0, int((t-0.12)*sr))
        end = min(len(y), int((t+0.18)*sr))
        breath = y[start:end]
        if len(breath) > 100:
            flatness = np.mean(librosa.feature.spectral_flatness(breath))
            if flatness > 0.65:  # Roger threshold
                clean_breaths += 1
    
    purity_ai = clean_breaths / max(1, len(events[:5]))
    
    # 3. SILENCE QUALITY
    silence = y[(np.abs(y) < 0.002)]
    silence_std = np.std(silence) if len(silence) > 1000 else 0
    silence_ai = 1.0 if silence_std < 0.0005 else 0.0
    
    # 4. BREATH AMPLITUDE CONSISTENCY
    amps = []
    for t in events:
        start = max(0, int((t-0.1)*sr))
        end = min(len(y), int((t+0.2)*sr))
        amps.append(np.max(np.abs(y[start:end])))
    
    amp_cv = np.std(amps) / np.mean(amps) if amps else 1
    amp_ai = 1.0 if amp_cv < 0.3 else 0.2  # Roger = consistent volume
    
    # ROGER AI SCORE
    score = (timing_ai * 0.35 + purity_ai * 0.25 + silence_ai * 0.2 + amp_ai * 0.2)
    
    return score, events

st.title("PneumaForensic")

files = st.file_uploader("Upload", type=['wav','mp3','m4a'], accept_multiple_files=True)

if files:
    col1, col2 = st.columns(2)
    
    results = []
    
    with col1:
        for file in files:
            file.seek(0)
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
                score, events = detect_roger_ai(y, sr)
                
                status = "AI" if score > 0.7 else "HUMAN"
                results.append({
                    "File": file.name,
                    "AI": f"{score:.1%}",
                    "Status": status,
                    "Breaths": len(events)
                })
            except:
                results.append({"File": file.name, "AI": "ERROR", "Status": "Failed"})
        
        df = pd.DataFrame(results)
        st.dataframe(df)
    
    with col2:
        for file in files:
            file.seek(0)
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
                score, events = detect_roger_ai(y, sr)
                
                fig, ax = plt.subplots(figsize=(8, 3))
                dur = min(20, len(y)/sr)
                t = np.linspace(0, dur, 2000)
                y_plot = y[:2000]
                
                ax.plot(t, y_plot, 'cyan', lw=0.7)
                for e in events:
                    if e < dur:
                        ax.axvline(e, 'red', ls='--', lw=3)
                
                color = 'red' if score > 0.7 else 'green'
                ax.set_title(f"{score:.1%}", color=color, fontsize=20)
                ax.set_facecolor('black')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            except:
                pass
