import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
from scipy.signal import find_peaks

st.set_page_config(layout="wide")

def process_forensic_audio(file):
    try:
        # Load audio (16k mono is research standard)
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        if len(y) < 16000: return None
        
        # 1. Detect Breath Candidates
        rms = librosa.feature.rms(y=y)[0]
        low_energy = (rms < np.percentile(rms, 30)).astype(float)
        peaks, _ = find_peaks(low_energy, height=0.4, distance=sr//3)
        events = librosa.frames_to_time(peaks, sr=sr)
        
        # Default AI score if no breaths are found (common in AI)
        if len(events) < 2:
            return {"Filename": file.name, "AI_Prob": 0.85, "Breaths": 0, "Status": "🤖 AI"}

        # 2. Token Similarity (Catching repeated AI breath samples)
        breath_samples = []
        for t in events[:5]:
            s, e = int(t*sr), int((t+0.2)*sr)
            m = librosa.feature.mfcc(y=y[s:e], sr=sr, n_mfcc=13).mean(1)
            breath_samples.append(m)
        
        sims = [np.corrcoef(breath_samples[i], breath_samples[i+1])[0,1] for i in range(len(breath_samples)-1)]
        avg_sim = np.mean(sims) if sims else 0
        
        # 3. Decision Logic
        # Penalty for too much similarity (AI) and perfect rhythm
        token_penalty = 0.4 if avg_sim > 0.8 else 0.0
        rhythm_penalty = 0.3 if (np.std(np.diff(events)) < 0.15) else 0.0
        ai_score = min(0.2 + token_penalty + rhythm_penalty, 1.0)
        
        return {
            "Filename": file.name,
            "AI_Prob": ai_score,
            "Breaths": len(events),
            "Status": "🤖 AI" if ai_score > 0.50 else "👤 HUMAN"
        }
    except:
        return None

# --- UI ---
st.title("Forensic Analysis")

uploaded = st.file_uploader("Upload", accept_multiple_files=True, label_visibility="collapsed")

if uploaded:
    data = []
    for f in uploaded:
        res = process_forensic_audio(f)
        if res: 
            data.append(res)
    
    if data:
        df = pd.DataFrame(data)
        # Display results
        st.dataframe(df, use_container_width=True)
        
        # Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Export CSV", data=csv, file_name="results.csv")
    else:
        st.error("Could not process audio. Ensure files are > 1 second.")
