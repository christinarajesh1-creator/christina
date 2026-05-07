import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.98, {"Status": "Suspiciously Clean/Silent"}

    # 1. Biological Chaos (35%) - THE ROGER KILLER
    # AI breaths are 'flat' noise. Human breaths are 'chaotic' noise.
    # We measure Spectral Flatness: Low Flatness = High Chaos = Human.
    flatness_scores = []
    for t in events:
        start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
        segment = y[start:end]
        if len(segment) > 256:
            flatness_scores.append(np.mean(librosa.feature.spectral_flatness(y=segment)))
    
    avg_flatness = np.mean(flatness_scores) if flatness_scores else 0.5
    # If the sound is too 'perfectly flat' (AI), the score goes UP.
    p1_chaos = np.clip(avg_flatness * 10, 0, 1)

    # 2. Timing Entropy (25%)
    # Roger mimics variety (CV 0.47). Humans are usually > 0.65.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p2_timing = np.clip(1.1 - ibi_cv, 0, 1)

    # 3. Dynamic Range Flux (20%)
    # Humans have 'tail noise' and mic hiss. AI Roger has near-zero noise floor.
    # We penalize too much silence between phrases.
    noise_floor = np.std(y[y < np.percentile(y, 10)])
    p3_silence = np.clip(1.0 - (noise_floor * 200), 0, 1)

    # 4. Rhythm Grid (20%)
    # Checks if the breaths appear in a predictable 'meter'
    p4_grid = 1.0 if (len(set(np.round(ibis, 1))) < len(ibis) * 0.5) else 0.2

    # FINAL WEIGHTED SCORE
    score = (p1_chaos * 0.35) + (p2_timing * 0.25) + (p3_silence * 0.20) + (p4_grid * 0.20)
    
    metrics = {
        "Spectral Purity (35%)": f"{p1_chaos:.0%}",
        "Timing Regularity (25%)": f"{p2_timing:.0%}",
        "Digital Silence (20%)": f"{p3_silence:.0%}",
        "Rhythm Grid (20%)": f"{p4_grid:.0%}",
        "IBI CV (Human > 0.6)": f"{ibi_cv:.2f}"
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
            threshold = np.percentile(rms, 12)
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold:
                    t = times[i]
                    if t - last_t > 1.6:
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
