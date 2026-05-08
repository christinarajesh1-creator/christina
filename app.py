import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

st.set_page_config(layout="wide")

def process_audio(file):
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        # 1. Sensitive Breath Detection (Adjusted for second-device recordings)
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        # Lowered threshold to find breaths in noisy/re-recorded audio
        threshold = np.percentile(rms, 25) 
        peaks, _ = find_peaks((rms < threshold).astype(float), distance=sr//4, prominence=0.1)
        events = librosa.frames_to_time(peaks, sr=sr)
        
        # 2. Parameter Calculation
        # IBI Regularity (AI is often too perfect)
        ibis = np.diff(events) if len(events) > 1 else [0]
        reg_score = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
        
        # Texture (High frequency hiss check)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        tex_score = np.mean(sc) / 3000 # Normalized
        
        # 3. Final AI Probability
        # If no breaths found, high AI probability. If found, check regularity.
        if len(events) < 2:
            ai_prob = 0.85
        else:
            # Human = higher variation (reg_score > 0.3)
            ai_prob = 0.2 + (0.6 if reg_score < 0.3 else 0.1)
            
        return {
            "Filename": file.name,
            "AI_Prob": ai_prob,
            "Breaths": len(events),
            "Regularity": round(reg_score, 3),
            "Texture": round(tex_score, 3),
            "Status": "🤖 AI" if ai_prob > 0.5 else "👤 HUMAN",
            "y": y, "events": events
        }
    except: return None

st.title("Forensic Analysis v5.2")
uploaded = st.file_uploader("Upload", accept_multiple_files=True)

if uploaded:
    data = []
    for f in uploaded:
        res = process_audio(f)
        if res: data.append(res)
    
    if data:
        df = pd.DataFrame(data)
        # Show all parameters
        st.subheader("Analysis Parameters")
        st.dataframe(df[["Filename", "AI_Prob", "Status", "Breaths", "Regularity", "Texture"]], use_container_width=True)
        
        # Show Graphs
        st.subheader("Visual Breath Patterns")
        cols = st.columns(2)
        for i, res in enumerate(data[:4]): # Plot first 4 files
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(res['y'][:16000*5], color='gray', alpha=0.5) # First 5 seconds
                for e in res['events']:
                    if e < 5: ax.axvline(e*16000, color='red', linestyle='--')
                ax.set_title(f"{res['Filename']} (Red = Detected Breaths)")
                st.pyplot(fig)
