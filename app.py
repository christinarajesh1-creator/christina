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
        return 0.98, {"Status": "Abnormal (No Breaths)"}

    # 1. Digital Splice (30%) - ROGER'S WEAKEST POINT
    # AI breaths have metronomic 'seams'. Humans have 'messy' airflow.
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p1_splice = np.clip(np.std(zcr) * 60, 0, 1) # Extreme sensitivity

    # 2. Spectral Flatness / 'Purity' (20%)
    # Roger breaths are 'cleaner' (lower flatness) than chaotic human noise.
    flatness = []
    for t in events:
        start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
        if (end - start) > 256:
            flatness.append(np.mean(librosa.feature.spectral_flatness(y=y[start:end])))
    p2_texture = np.clip(1.0 - (np.mean(flatness) * 35), 0, 1) if flatness else 0.8

    # 3. ElevenLabs 'Situational Awareness' Grid (20%)
    # Roger mimics variety (CV ~0.35). We penalize anything below CV 0.5.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p3_reg = np.clip(1.1 - (ibi_cv * 1.5), 0, 1)

    # 4. Inhale Amplitude Consistency (10%)
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p4_amp = np.clip(1.0 - (amp_cv * 2.5), 0, 1)

    # 5. Breath Density (10%)
    # Roger 'over-breathes'. Humans breathe every 6-10s.
    breaths_per_min = (len(events) / duration) * 60
    p5_density = 1.0 if breaths_per_min > 25 or breaths_per_min < 6 else 0.2

    # 6. MFCC Similarity (10%)
    # Checks if Roger is recycling the same 'sigh' artifact.
    mfccs = [np.mean(librosa.feature.mfcc(y=y[int(t*sr):int((t+0.4)*sr)], sr=sr, n_mfcc=20), axis=1) for t in events]
    p6_sim = 0.5
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        p6_sim = np.clip(1.0 - (np.mean(dists) / 180), 0, 1)

    # FINAL TOTAL
    score = (p1_splice*0.30) + (p2_texture*0.20) + (p3_reg*0.20) + (p4_amp*0.10) + (p5_density*0.10) + (p6_sim*0.10)
    
    metrics = {
        "Digital Splice (30%)": f"{p1_splice:.0%}",
        "Spectral Purity (20%)": f"{p2_texture:.0%}",
        "Regularity (20%)": f"{p3_reg:.0%}",
        "Amplitude Var (10%)": f"{p4_amp:.0%}",
        "Density (10%)": f"{p5_density:.0%}",
        "Similarity (10%)": f"{p6_sim:.0%}",
        "IBI CV": f"{ibi_cv:.2f}"
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

# UI (Batch Analysis)
st.title("🫁 PneumaForensic: Roger/ElevenLabs Deep-Scan")
files = st.file_uploader("Upload Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    for f in files:
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
            y = librosa.util.normalize(y)
            rms = librosa.feature.rms(y=y).flatten()
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            threshold = np.percentile(rms, 10) # Aggressive detection
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold:
                    t = times[i]
                    if t - last_t > 1.4: # Catch Roger's rapid breathing
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
            
            with st.expander(f"📊 Forensic Report: {f.name}"):
                st.table(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}))
            st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
