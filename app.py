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
        return 0.98, {k: "100%" for k in ["Timing", "Purity", "Presence", "Amplitude", "Splice", "Similarity"]}

    # 1. Timing Regularity (28%) - Penalize robotic CV
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1 = np.clip(1.3 - (ibi_cv * 2), 0, 1)

    # 2. Spectral Purity (18%) - Catching 'Clean' AI Breaths
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    p2 = np.clip(1.0 - (np.mean(flatness) * 80), 0, 1) if flatness else 0.5

    # 3. Breath Presence (15%) - Density Check
    bpm = (len(events) / duration) * 60
    p3 = 1.0 if bpm > 22 or bpm < 6 else 0.1

    # 4. Breath Amplitude (15%) - Uniform Volume
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p4 = np.clip(1.0 - (amp_cv / 0.4), 0, 1)

    # 5. Spectral Continuity (12%) - Digital Splice/ZCR flux
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p5 = np.clip(np.std(zcr) * 15, 0, 1) 

    # 6. Breath Similarity (12%) - Texture Reuse
    mfccs = [np.mean(librosa.feature.mfcc(y=y[int(t*sr):int((t+0.4)*sr)], sr=sr, n_mfcc=13), axis=1) for t in events]
    p6 = 0.0
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        p6 = np.clip(1.0 - (np.mean(dists) / 250), 0, 1)

    score = (p1*0.28) + (p2*0.18) + (p3*0.15) + (p4*0.15) + (p5*0.12) + (p6*0.12)
    
    metrics = {
        "AI Confidence": f"{score:.0%}",
        "Timing (28%)": f"{p1:.0%}",
        "Purity (18%)": f"{p2:.0%}",
        "Presence (15%)": f"{p3:.0%}",
        "Amplitude (15%)": f"{p4:.0%}",
        "Splice (12%)": f"{p5:.0%}",
        "Similarity (12%)": f"{p6:.0%}",
        "CV": round(ibi_cv, 2)
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

st.title("🫁 PneumaForensic")

files = st.file_uploader("Upload Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    all_data = []
    
    for f in files:
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
            y = librosa.util.normalize(y)
            rms = librosa.feature.rms(y=y).flatten()
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            threshold = np.percentile(rms, 12)
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold:
                    t = times[i]
                    if t - last_t > 1.7:
                        events.append(float(t))
                        last_t = t
            
            score, metrics = get_forensic_score(y, sr, events, len(y)/sr)
            row = {"Filename": f.name}
            row.update(metrics)
            all_data.append(row)
        except: continue

    if all_data:
        st.subheader("📋 6-Parameter Batch Report")
        df = pd.DataFrame(all_data)
        st.dataframe(df, use_container_width=True)
        st.download_button("Export CSV", df.to_csv(index=False).encode('utf-8'), "pneuma_results.csv")
