import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    """Hardened 6-Parameter Forensic Model"""
    if len(events) < 2:
        return 0.90 # High suspicion for zero/single breath
    
    # 1. IBI Regularity (28%) - Detecting the 'Perfect Grid'
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    # If CV is low (under 0.2), it's a robotic grid.
    p1 = np.clip(1.0 - (ibi_cv / 0.4), 0, 1)

    # 2. Breath Amplitude (15%) - AI volume is too consistent
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p2 = np.clip(1.0 - (amp_cv / 0.3), 0, 1)

    # 3. Breath Duration (12%) - Identical length breaths
    # We check if the durations of the detected 'silence' are copy-pasted
    p3 = 1.0 if (len(set(np.round(ibis, 1))) < len(ibis) * 0.5) else 0.1

    # 4. Breath Presence (15%) - Breath-to-Speech ratio
    # AI often 'over-breathes' or 'under-breathes' compared to 1 breath per ~6-8s
    expected_ratio = duration / 7.0
    p4 = np.clip(abs(1.0 - (len(events) / expected_ratio)), 0, 1)

    # 5. Spectral Continuity (12%) - Looking for 'Splice' artifacts
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    p5 = np.clip(np.std(zcr) * 25, 0, 1) # High ZCR flux suggests digital stitching

    # 6. Breath Similarity (18%) - The 'Copy-Paste' Audio check
    mfccs = []
    for t in events:
        start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
        if (end - start) > 512:
            mfccs.append(np.mean(librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=13), axis=1))
    
    p6 = 0.5
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        # Low distance = identical texture = AI
        p6 = np.clip(1.0 - (np.mean(dists) / 250), 0, 1)

    # Final Weighted Calculation
    return (p1*0.28) + (p2*0.15) + (p3*0.12) + (p4*0.15) + (p5*0.12) + (p6*0.18)

st.title("🫁 PneumaForensic")

files = st.file_uploader("Batch Upload", type=['wav', 'mp3', 'm4a'], accept_multiple_files=True)

if files:
    results = []
    for f in files:
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
            y = librosa.util.normalize(y)
            
            # Use RMS to find gaps (breaths)
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            threshold = np.percentile(rms, 15)
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold * 0.7:
                    t = times[i]
                    if t - last_t > 1.8:
                        events.append(float(t))
                        last_t = t
            
            score = get_forensic_score(y, sr, events, len(y)/sr)
            results.append({"name": f.name, "score": score, "y": y, "events": events, "dur": len(y)/sr})
        except: continue

    if results:
        # Results Table
        st.table(pd.DataFrame([{"File": r['name'], "AI Score": f"{r['score']:.0%}"} for r in results]))

        # Graph Visuals
        for r in results:
            fig, ax = plt.subplots(figsize=(15, 2))
            ax.plot(np.linspace(0, r['dur'], len(r['y'])), r['y'], color='gray', alpha=0.3, lw=0.5)
            for e in r['events']:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
            color = "red" if r['score'] > 0.55 else "green"
            ax.set_title(f"{r['name']} | Forensic Score: {r['score']:.0%}", color=color, loc='left', fontweight='bold')
            ax.set_xlim(0, r['dur'])
            ax.axis('off')
            st.pyplot(fig)
            st.divider()
