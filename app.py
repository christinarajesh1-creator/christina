import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.95, {"Error": "Insufficient Breaths"}

    # 1. IBI Regularity (28%) - PENALIZE LOW CV
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    # AI often has CV < 0.2. Humans are usually > 0.5.
    p1 = np.clip(1.0 - (ibi_cv / 0.4), 0, 1)

    # 2. Breath Amplitude (15%) - Uniform Volume
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p2 = np.clip(1.0 - (amp_cv / 0.3), 0, 1)

    # 3. Breath Duration (12%)
    p3 = 1.0 if (len(set(np.round(ibis, 1))) < len(ibis) * 0.4) else 0.1

    # 4. Breath Presence (15%)
    breaths_per_min = (len(events) / duration) * 60
    p4 = 1.0 if breaths_per_min > 22 or breaths_per_min < 6 else 0.2

    # 5. Spectral Continuity (12%)
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p5 = np.clip(np.std(zcr) * 35, 0, 1)

    # 6. Breath Similarity (18%)
    mfccs = []
    for t in events:
        start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
        if (end - start) > 512:
            mfccs.append(np.mean(librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=13), axis=1))
    
    p6 = 0.5
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        p6 = np.clip(1.0 - (np.mean(dists) / 180), 0, 1)

    score = (p1*0.28) + (p6*0.18) + (p4*0.15) + (p2*0.15) + (p5*0.12) + (p3*0.12)
    
    metrics = {
        "IBI Regularity (CV)": f"{ibi_cv:.2f}",
        "Timing Score (28%)": f"{p1:.0%}",
        "Similarity Score (18%)": f"{p6:.0%}",
        "Presence Score (15%)": f"{p4:.0%}",
        "Amplitude Score (15%)": f"{p2:.0%}",
        "Splice Score (12%)": f"{p5:.0%}",
        "Duration Score (12%)": f"{p3:.0%}"
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

st.title("🫁 PneumaForensic")

files = st.file_uploader("Upload Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    results = []
    for f in files:
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
            y = librosa.util.normalize(y)
            rms = librosa.feature.rms(y=y).flatten()
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            threshold = np.percentile(rms, 15)
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold * 0.7:
                    t = times[i]
                    if t - last_t > 1.7: # Harder gate for ElevenLabs
                        events.append(float(t))
                        last_t = t
            
            score, metrics = get_forensic_score(y, sr, events, len(y)/sr)
            results.append({"name": f.name, "score": score, "metrics": metrics, "y": y, "events": events, "dur": len(y)/sr})
        except: continue

    for r in results:
        # Graph
        fig, ax = plt.subplots(figsize=(15, 2))
        ax.plot(np.linspace(0, r['dur'], len(r['y'])), r['y'], color='gray', alpha=0.3, lw=0.5)
        for e in r['events']:
            ax.axvline(e, color='red', linestyle='--', lw=1.5)
        
        status = "red" if r['score'] > 0.6 else "green"
        ax.set_title(f"{r['name']} | AI Confidence: {r['score']:.0%}", color=status, loc='left', fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)
        
        # New Forensic Breakdown Table
        with st.expander(f"📊 Forensic Breakdown: {r['name']}"):
            st.table(pd.DataFrame([r['metrics']]).T.rename(columns={0: "Value"}))
        st.divider()
