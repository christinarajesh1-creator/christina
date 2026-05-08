import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

st.set_page_config(layout="wide")

def forensic_breath_detect(file):
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        
        # 1. CLEAN: Remove room rumble and flatten noise
        y_filt = librosa.effects.preemphasis(y)
        rms = librosa.feature.rms(y=y_filt, frame_length=1024, hop_length=256)[0]
        times = librosa.frames_to_time(rms, sr=sr)
        
        # 2. SENSITIVE SEARCH: Find drops in energy (breaths)
        # We use a rolling mean to find local energy dips
        rolling_noise = pd.Series(rms).rolling(window=15, center=True).mean().fillna(np.mean(rms))
        breath_candidates = (rms < rolling_noise * 0.8).astype(float)
        
        peaks, _ = find_peaks(breath_candidates, height=0.3, distance=sr//4, prominence=0.1)
        events = [times[p] for p in peaks if 1.0 < times[p] < (len(y)/sr - 1.0)]

        # 3. RESEARCH METRICS
        breath_count = len(events)
        if breath_count >= 2:
            ibis = np.diff(events)
            reg_score = np.std(ibis) / np.mean(ibis)
            # Human = Higher regularity variation (> 0.25)
            ai_score = 0.2 + (0.5 if reg_score < 0.25 else 0.05)
        else:
            reg_score = 0
            ai_score = 0.85 # Flag as AI if 0-1 breaths found

        return {
            "Filename": file.name, "AI_Score": ai_score, "Status": "🤖 AI" if ai_score > 0.5 else "👤 HUMAN",
            "Breaths": breath_count, "IBI_Var": round(reg_score, 3), "y": y, "ev": events
        }
    except: return None

st.title("Forensic Breath Analyzer v9.0")
uploaded = st.file_uploader("Upload Batch", accept_multiple_files=True)

if uploaded:
    data = [forensic_breath_detect(f) for f in uploaded if f]
    df = pd.DataFrame([r for r in data if r])
    
    if not df.empty:
        # 1. Parameter Table
        st.dataframe(df[["Filename", "AI_Score", "Status", "Breaths", "IBI_Var"]], use_container_width=True)
        
        # 2. Visual Proof (Graphs)
        st.subheader("Temporal Breath Markers")
        cols = st.columns(2)
        for i, res in enumerate(data[:4]):
            with cols[i%2]:
                fig, ax = plt.subplots(figsize=(10, 3), facecolor='#111')
                ax.plot(res['y'], color='white', alpha=0.3, lw=0.5)
                for p in res['ev']:
                    ax.axvline(p*16000, color='#00FF00', lw=2, label='Breath')
                ax.set_facecolor('black')
                ax.set_title(f"{res['Filename']}: {res['Breaths']} Breaths Found", color='white')
                st.pyplot(fig)
