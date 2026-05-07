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
        return 0.98, {"Status": "Abnormal (Insufficient Breaths)"}

    # 1. IBI Regularity (28%)
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1 = np.clip(1.0 - (ibi_cv / 0.55), 0, 1)

    # 2. Purity/Texture (18%)
    flatness_scores = []
    for t in events:
        start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
        segment = y[start:end]
        if len(segment) > 256:
            flatness_scores.append(np.mean(librosa.feature.spectral_flatness(y=segment)))
    p2 = np.clip(1.0 - (np.mean(flatness_scores) * 20), 0, 1) if flatness_scores else 0.5

    # 3. Presence/Density (15%)
    breaths_per_min = (len(events) / duration) * 60
    p3 = 1.0 if breaths_per_min > 22 or breaths_per_min < 7 else 0.2

    # 4. Amplitude Var (15%)
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p4 = np.clip(1.0 - (amp_cv / 0.4), 0, 1)

    # 5. Digital Splice (12%)
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p5 = np.clip(np.std(zcr) * 45, 0, 1) 

    # 6. Similarity Score (12%)
    mfccs = [np.mean(librosa.feature.mfcc(y=y[int(t*sr):int((t+0.4)*sr)], sr=sr, n_mfcc=20), axis=1) for t in events]
    p6 = 0.5
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        p6 = np.clip(1.0 - (np.mean(dists) / 150), 0, 1)

    score = (p1*0.28) + (p2*0.18) + (p3*0.15) + (p4*0.15) + (p5*0.12) + (p6*0.12)
    
    metrics = {
        "IBI Regularity (CV)": f"{ibi_cv:.2f}",
        "Regularity Score (28%)": f"{p1:.0%}",
        "Purity/Texture (18%)": f"{p2:.0%}",
        "Presence/Density (15%)": f"{p3:.0%}",
        "Amplitude Var (15%)": f"{p4:.0%}",
        "Digital Splice (12%)": f"{p5:.0%}",
        "Similarity Score (12%)": f"{p6:.0%}"
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

st.title("🫁 PneumaForensic")

files = st.file_uploader("Upload Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
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
                    if t - last_t > 1.7:
                        events.append(float(t))
                        last_t = t
            
            score, metrics = get_forensic_score(y, sr, events, len(y)/sr)
            
            fig, ax = plt.subplots(figsize=(15, 2))
            ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='gray', alpha=0.3, lw=0.5)
            for e in events:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
            color = "red" if score > 0.6 else "green"
            ax.set_title(f"{f.name} | AI Confidence: {score:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            
            with st.expander(f"📊 Forensic Breakdown: {f.name}"):
                st.table(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}))
            st.divider()
        except Exception as e:
            st.error(f"Error processing {f.name}: {e}")
