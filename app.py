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
        return 0.95, {"Status": "Suspiciously Low Breath Count"}

    # 1. Robotic Timing (30%) - Focus on "Perfect Spacing"
    # Humans are chaotic (CV > 0.6). AI Roger mimics variety (CV 0.3-0.4).
    # We penalize the 'Roger Zone' (0.2 - 0.45) heavily.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1_reg = np.clip(1.3 - (ibi_cv * 2), 0, 1) 

    # 2. Spectral Flatness / "Empty Silence" (25%)
    # AI breaths are "digitally flat". Human breaths have "noise/friction".
    flatness_scores = []
    for t in events:
        start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
        segment = y[start:end]
        if len(segment) > 128:
            flatness_scores.append(np.mean(librosa.feature.spectral_flatness(y=segment)))
    
    # If the breath is too 'flat' or 'clean' (low energy flux), it's AI.
    avg_flatness = np.mean(flatness_scores) if flatness_scores else 0
    p2_texture = np.clip(1.0 - (avg_flatness * 100), 0, 1)

    # 3. Dynamic Range Purity (20%)
    # Humans have microphone 'hiss'. AI Roger has absolute zero noise floor.
    # We penalize too much perfection in the quietest parts.
    noise_floor = np.std(y[y < np.percentile(y, 5)])
    p3_silence = np.clip(1.0 - (noise_floor * 500), 0, 1)

    # 4. Digital Splice (25%) - Anchor for Stitching
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p4_splice = np.clip(np.std(zcr) * 50, 0, 1)

    score = (p1_reg * 0.30) + (p2_texture * 0.25) + (p3_silence * 0.20) + (p4_splice * 0.25)
    
    metrics = {
        "Timing Regularity (30%)": f"{p1_reg:.0%}",
        "Spectral Purity (25%)": f"{p2_texture:.0%}",
        "Digital Silence (20%)": f"{p3_silence:.0%}",
        "Digital Splice (25%)": f"{p4_splice:.0%}",
        "IBI CV (Human > 0.6)": f"{ibi_cv:.2f}"
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

st.title("🫁 PneumaForensic")

files = st.file_uploader("Upload", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    for f in files:
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
            y = librosa.util.normalize(y)
            rms = librosa.feature.rms(y=y).flatten()
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            threshold = np.percentile(rms, 10) # Aggressive gate
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold:
                    t = times[i]
                    if t - last_t > 1.4:
                        events.append(float(t))
                        last_t = t
            
            score, metrics = get_forensic_score(y, sr, events, len(y)/sr)
            
            fig, ax = plt.subplots(figsize=(15, 2))
            ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='gray', alpha=0.3, lw=0.5)
            for e in events:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
            color = "red" if score > 0.55 else "green"
            ax.set_title(f"{f.name} | AI Confidence: {score:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            
            with st.expander(f"📊 Forensic Report: {f.name}"):
                st.table(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}))
            st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
