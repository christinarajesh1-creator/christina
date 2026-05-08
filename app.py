import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

st.set_page_config(layout="wide")

def process_forensic(file):
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        
        # 1. Noise Gate: Strip background hum to find the 'real' breaths
        S_full, phase = librosa.magphase(librosa.stft(y))
        noise_floor = np.median(S_full, axis=1, keepdims=True)
        S_clean = S_full - (noise_floor * 1.5) # Spectral subtraction
        y_clean = librosa.istft(S_clean * phase)
        
        # 2. Advanced Breath Search (Using cleaned signal)
        rms = librosa.feature.rms(y=y_clean)[0]
        # Look for the 'hiss' frequency where breaths live (2kHz - 5kHz)
        y_filt = librosa.effects.preemphasis(y_clean)
        peaks, _ = find_peaks(rms, height=np.mean(rms)*0.5, distance=sr//3)
        events = librosa.frames_to_time(peaks, sr=sr)

        # 3. Research Parameters
        # AI often has 'Identical' breaths. Humans have unique ones.
        if len(events) >= 2:
            ibis = np.diff(events)
            regularity = np.std(ibis) / np.mean(ibis)
            # High regularity (low score) = AI. High variation = Human.
            ai_score = 0.8 if regularity < 0.2 else 0.3
        else:
            regularity = 0
            ai_score = 0.85 # Default if no breaths found

        return {
            "Filename": file.name,
            "AI_Prob": ai_score,
            "Breaths_Found": len(events),
            "Reg_Score": round(regularity, 3),
            "Status": "🤖 AI" if ai_score > 0.5 else "👤 HUMAN",
            "y": y, "ev": events
        }
    except: return None

st.title("Forensic Analysis v7.0 (Noise-Filtered)")
uploaded = st.file_uploader("Upload", accept_multiple_files=True)

if uploaded:
    data = [process_forensic(f) for f in uploaded if f]
    df = pd.DataFrame([r for r in data if r])
    
    if not df.empty:
        st.dataframe(df[["Filename", "AI_Prob", "Status", "Breaths_Found", "Reg_Score"]], use_container_width=True)
        
        # Graphs to prove it's finding them
        cols = st.columns(2)
        for i, res in enumerate(data[:4]):
            with cols[i%2]:
                fig, ax = plt.subplots(figsize=(10,3))
                ax.plot(res['y'], color='gray', alpha=0.4)
                for p in res['ev']:
                    ax.axvline(p*16000, color='lime', lw=2, label='Breath')
                ax.set_title(f"{res['Filename']}: {res['Breaths_Found']} Breaths")
                st.pyplot(fig)
