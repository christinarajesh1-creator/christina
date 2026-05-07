import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 2:
        return 0.85
    
    # 1. IBI Regularity (28%)
    ibis = np.diff(events)
    p1 = 1.0 - (np.std(ibis) / np.mean(ibis)) if np.mean(ibis) > 0 else 1.0
    
    # 2. Breath Amplitude (15%)
    amps = []
    for t in events:
        start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
        amps.append(np.max(np.abs(y[start:end])) if start < end else 0.01)
    p2 = 1.0 - (np.std(amps) / np.mean(amps)) if np.mean(amps) > 0 else 1.0
    
    # 3. Breath Duration (12%)
    p3 = 0.9 if len(set([round(x, 1) for x in ibis])) < len(ibis)/2 else 0.2
    
    # 4. Breath Presence (15%)
    p4 = np.clip(1.0 - (len(events) / (duration / 8)), 0, 1)
    
    # 5. Spectral Continuity (12%)
    zcr = librosa.feature.zero_crossing_rate(y)[0] # Added [0] to flatten
    p5 = np.clip(np.std(zcr) * 20, 0, 1)
    
    # 6. Breath Similarity (18%)
    mfccs = []
    for t in events:
        start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
        if (end - start) > 512:
            m = librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=13)
            mfccs.append(np.mean(m, axis=1))
    
    p6 = 0.5
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        p6 = np.clip(1 - (np.mean(dists) / 400), 0, 1)

    return (p1*0.28) + (p2*0.15) + (p3*0.12) + (p4*0.15) + (p5*0.12) + (p6*0.18)

st.title("🫁 PneumaForensic")

files = st.file_uploader("Batch Upload", type=['wav', 'mp3', 'm4a'], accept_multiple_files=True)

if files:
    results = []
    for f in files:
        try:
            # Use librosa to load from buffer
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
            y = librosa.util.normalize(y)
            duration = len(y) / sr
            
            # Use [0] to flatten the RMS output to 1D
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            threshold = np.percentile(rms, 15)
            
            events = []
            last_t = -5.0
            # Breath detection logic
            for i, val in enumerate(rms):
                if val < threshold * 0.7:
                    t = times[i]
                    if t - last_t > 1.8:
                        events.append(float(t))
                        last_t = t
            
            score = get_forensic_score(y, sr, events, duration)
            results.append({"name": f.name, "score": score, "y": y, "events": events, "dur": duration})
        except Exception as e:
            st.error(f"Error reading {f.name}: {e}")
            continue

    if results:
        # Batch Summary
        df = pd.DataFrame([{"File": r['name'], "AI Score": f"{r['score']:.0%}"} for r in results])
        st.table(df)

        # Visualizations
        for r in results:
            fig, ax = plt.subplots(figsize=(15, 2.5))
            
            # The Gray Waves (Audio)
            t_axis = np.linspace(0, r['dur'], len(r['y']))
            ax.plot(t_axis, r['y'], color='gray', alpha=0.3, lw=0.5)
            
            # The Red Dashed Lines (Breaths)
            for e in r['events']:
                ax.axvline(e, color='red', linestyle='--', lw=1.5, alpha=0.8)
                
            status_color = "red" if r['score'] > 0.6 else "green"
            ax.set_title(f"{r['name']} | AI Confidence: {r['score']:.0%}", color=status_color, loc='left', fontweight='bold')
            ax.set_xlim(0, r['dur'])
            ax.set_ylim(-1, 1)
            ax.axis('off')
            
            st.pyplot(fig)
            st.divider()
