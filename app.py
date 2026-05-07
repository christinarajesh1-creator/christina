import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.set_page_config(layout="wide", page_title="PneumaForensic")

def analyze_audio(y, sr, filename):
    try:
        y = librosa.util.normalize(y)
        duration = len(y) / sr
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        
        # Detection Logic
        threshold = np.percentile(rms, 15)
        breath_candidates = times[rms < threshold * 0.7]
        
        events = []
        last_t = -5.0
        for t in breath_candidates:
            if t - last_t > 1.5: 
                events.append(float(t))
                last_t = t
        
        # 6-Parameter Calculations
        # 1. IBI Regularity (28%)
        ibis = np.diff(events) if len(events) > 1 else [0]
        p1 = 1.0 - (np.std(ibis) / np.mean(ibis)) if np.mean(ibis) > 0 else 1.0
        
        # 2. Breath Amplitude (15%)
        amps = [np.max(np.abs(y[int(t*sr):int((t+0.4)*sr)])) for t in events]
        p2 = 1.0 - (np.std(amps) / np.mean(amps)) if np.mean(amps) > 0 else 1.0
        
        # 3. Breath Duration (12%)
        p3 = 0.8 if len(set([round(x, 2) for x in ibis])) < len(ibis)/2 else 0.2
        
        # 4. Breath Presence (15%)
        p4 = 1.0 if len(events) < (duration / 10) else 0.1
        
        # 5. Spectral Continuity (12%)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        p5 = np.clip(np.std(zcr) * 15, 0, 1)
        
        # 6. Breath Similarity (18%)
        mfccs = [np.mean(librosa.feature.mfcc(y=y[int(t*sr):int((t+0.4)*sr)], sr=sr, n_mfcc=13), axis=1) for t in events]
        p6 = 0.0
        if len(mfccs) > 1:
            dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
            p6 = np.clip(1 - (np.mean(dists) / 300), 0, 1)

        score = (p1*0.28) + (p2*0.15) + (p3*0.12) + (p4*0.15) + (p5*0.12) + (p6*0.18)
        
        return {
            "filename": filename,
            "score": round(np.clip(score, 0, 1), 2),
            "events": events,
            "y": y,
            "sr": sr,
            "duration": duration
        }
    except:
        return None

st.title("🫁 PneumaForensic")

files = st.file_uploader("Batch Upload", type=['wav', 'mp3', 'm4a'], accept_multiple_files=True)

if files:
    results = []
    for f in files:
        y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
        res = analyze_audio(y, sr, f.name)
        if res: results.append(res)

    # Summary Table
    df = pd.DataFrame([{"File": r['filename'], "AI Score": f"{r['score']:.0%}"} for r in results])
    st.table(df)

    # Detailed Visual Analysis
    for r in results:
        fig, ax = plt.subplots(figsize=(15, 3))
        
        # The Gray Waves
        time_axis = np.linspace(0, r['duration'], len(r['y']))
        ax.plot(time_axis, r['y'], color='gray', alpha=0.5, lw=0.5)
        
        # The Red Dashed Lines
        for e in r['events']:
            ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
        color = "red" if r['score'] > 0.5 else "green"
        ax.set_title(f"{r['filename']} | Detection Score: {r['score']:.0%}", color=color, loc='left', fontsize=12, fontweight='bold')
        ax.set_xlim(0, r['duration'])
        ax.set_ylim(-1, 1)
        ax.axis('off')
        
        st.pyplot(fig)
        st.divider()

    st.download_button("Export Data", df.to_csv(index=False).encode('utf-8'), "pneuma_report.csv")
