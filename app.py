import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.set_page_config(layout="wide", page_title="PneumaForensic")

def analyze_forensic_audio(y, sr, filename):
    try:
        # Pre-processing & Normalization
        y = librosa.util.normalize(y)
        duration = len(y) / sr
        
        # Flattening 2D arrays to 1D prevents blank-screen crashes
        rms = librosa.feature.rms(y=y).flatten() 
        zcr = librosa.feature.zero_crossing_rate(y).flatten()
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        
        # Hardened Breath Detection
        threshold = np.percentile(rms, 15)
        events, last_t = [], -5.0
        for i, val in enumerate(rms):
            if val < threshold * 0.7:
                t = times[i]
                if t - last_t > 1.8:
                    events.append(float(t))
                    last_t = t
        
        if len(events) < 2:
            return {"name": filename, "score": 0.85, "events": [], "y": y, "dur": duration}

        # 1. ElevenLabs Hardened Timing (28%) - Catching the 'Perfect Grid'
        ibis = np.diff(events)
        ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
        p1 = np.clip(1.0 - (ibi_cv / 0.3), 0, 1)

        # 2. Spectral Purity Check (NEW - 25%) 
        # Human breaths are 'noisier'; AI breaths are 'cleaner' artifacts.
        flatness_scores = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
        p_spectral = np.clip(1.0 - (np.mean(flatness_scores) * 15), 0, 1)

        # 3. Breath Similarity (18%) - Copy-Paste Texture Check
        mfccs = [np.mean(librosa.feature.mfcc(y=y[int(t*sr):int((t+0.4)*sr)], sr=sr, n_mfcc=20), axis=1) for t in events]
        p6_sim = 0.5
        if len(mfccs) > 1:
            dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
            p6_sim = np.clip(1.0 - (np.mean(dists) / 220), 0, 1)

        # Weighted Final Result
        total_score = (p1 * 0.35) + (p_spectral * 0.35) + (p6_sim * 0.30)
        return {"name": filename, "score": round(total_score, 2), "events": events, "y": y, "dur": duration}

    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None

st.title("🫁 PneumaForensic")
uploaded_files = st.file_uploader("Upload Audio", type=['wav', 'mp3'], accept_multiple_files=True)

if uploaded_files:
    results = []
    for f in uploaded_files:
        data, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
        res = analyze_forensic_audio(data, sr, f.name)
        if res: results.append(res)

    if results:
        # Visual Timeline: Gray Waves & Red Dashed Lines
        for r in results:
            fig, ax = plt.subplots(figsize=(15, 2))
            ax.plot(np.linspace(0, r['dur'], len(r['y'])), r['y'], color='gray', alpha=0.3)
            for e in r['events']:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
            color = "red" if r['score'] > 0.6 else "green"
            ax.set_title(f"{r['name']} | AI Confidence: {r['score']:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
