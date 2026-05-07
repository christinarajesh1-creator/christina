import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.95, {"Status": "Suspiciously Low Breath Count"}

    # 1. Pattern Fatigue (35%) - THE AI KILLER
    # AI (Roger) maintains the same 'energy signature' throughout. 
    # Humans get 'tired' or change intensity. We measure Spectral Flux Variance.
    S = np.abs(librosa.stft(y))
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    flux_var = np.std(flux) / np.mean(flux) if np.mean(flux) > 0 else 0
    # AI is too 'steady' (Low Flux Variance). Humans are chaotic (High Flux Variance).
    p1_fatigue = np.clip(1.0 - (flux_var * 0.8), 0, 1)

    # 2. Hardened Timing CV (25%)
    # Roger mimics variety (0.47 CV). We now penalize anything below 0.60.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p2_timing = np.clip(1.1 - ibi_cv, 0, 1)

    # 3. Breath Complexity (20%) - NEW
    # AI breaths have repeating internal patterns. Humans are 100% unique noise.
    # We use Zero-Crossing flux to detect 'texture repetition'.
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    zcr_consistency = np.std(zcr) / np.mean(zcr) if np.mean(zcr) > 0 else 0
    p3_complexity = np.clip(1.0 - zcr_consistency, 0, 1)

    # 4. Digital Splice (20%) - Reduced Weight
    # Lowered weight so noisy human mics don't trigger 100% AI flags.
    p4_splice = np.clip(np.std(zcr) * 35, 0, 1)

    score = (p1_fatigue * 0.35) + (p2_timing * 0.25) + (p3_complexity * 0.20) + (p4_splice * 0.20)
    
    metrics = {
        "Pattern Fatigue (35%)": f"{p1_fatigue:.0%}",
        "Timing Regularity (25%)": f"{p2_timing:.0%}",
        "Texture Consistency (20%)": f"{p3_complexity:.0%}",
        "Digital Splice (20%)": f"{p4_splice:.0%}",
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
            threshold = np.percentile(rms, 10)
            
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
            
            color = "red" if score > 0.5 else "green"
            ax.set_title(f"{f.name} | AI Confidence: {score:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            
            with st.expander(f"📊 Forensic Report: {f.name}"):
                st.table(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}))
            st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
