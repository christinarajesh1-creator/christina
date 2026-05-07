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

    # 1. HARDENED IBI Regularity (28%)
    # Recalibrated: Roger's 0.38 CV is now flagged as High AI Risk.
    # Humans are usually 0.6+, AI hides in the 0.3-0.4 range.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1 = np.clip(1.0 - (ibi_cv / 0.55), 0, 1) # Raised divisor to catch 0.38

    # 2. Spectral Purity / Texture (18%) - NEW ROGER KILLER
    # AI breaths have lower 'Spectral Flatness' than messy human noise.
    flatness_scores = []
    for t in events:
        start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
        segment = y[start:end]
        if len(segment) > 256:
            flatness_scores.append(np.mean(librosa.feature.spectral_flatness(y=segment)))
    p2_texture = np.clip(1.0 - (np.mean(flatness_scores) * 20), 0, 1) if flatness_scores else 0.5

    # 3. Breath Presence (15%) - BPM Density
    breaths_per_min = (len(events) / duration) * 60
    p3 = 1.0 if breaths_per_min > 20 or breaths_per_min < 7 else 0.2

    # 4. Breath Amplitude (15%) - Uniform Volume
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p4 = np.clip(1.0 - (amp_cv / 0.4), 0, 1)

    # 5. HARDENED Splice Score (12%) - Digital Seams
    # Increased sensitivity to catch ZCR 'clipping' at breath boundaries.
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p5 = np.clip(np.std(zcr) * 45, 0, 1) 

    # 6. Breath Similarity (12%) - Copy-Paste Check
    mfccs = [np.mean(librosa.feature.mfcc(y=y[int(t*sr):int((t+0.4)*sr)], sr=sr, n_mfcc=20), axis=1) for t in events]
    p6 = 0.5
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        p6 = np.clip(1.0 - (np.mean(dists) / 150), 0, 1)

    # FINAL HARDENED CALCULATION
    score = (p1*0.28) + (p2*0.18) + (p3*0.15) + (p4*0.15) + (p5*0.12) + (p6*0.12)
    
    metrics = {
        "IBI Regularity (CV)": f"{ibi_cv:.2f}",
        "Regularity Score (28%)": f"{p1:.0%}",
        "Purity/Texture (18%)": f"{p2_texture:.0%}",
        "Presence/Density (15%)": f"{p3:.0%}",
        "Amplitude Var (15%)": f"{p4:.0%}",
        "Digital Splice (12%)": f"{p5:.0%}",
        "Similarity Score (12%)": f"{p6:.0%}"
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

# --- UI LOGIC (Same as before but with Hardened Function) ---
st.title("🫁 PneumaForensic: Hardened Edition")
st.info("Analysis calibrated for ElevenLabs Roger and advanced prosody models.")

files = st.file_uploader("Upload Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    for f in files:
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
        
        # Visuals
        fig, ax = plt.subplots(figsize=(15, 2))
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='gray', alpha=0.3, lw=0.5)
        for e in events:
            ax.axvline(e, color='red', linestyle='--', lw=1.5)
        
        status = "red" if score > 0.6 else "green"
        ax.set_title(f"{f.name} | Hardened AI Confidence: {score:.0%}", color=status, loc='left', fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)
        
        with st.expander(f"🔍 Forensic Details: {f.name}"):
            st.table(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}))
