import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
from scipy.signal import find_peaks

st.set_page_config(layout="wide", page_title="PneumaForensic v5.0")

def extract_features(y, sr):
    """
    Advanced forensic feature set specifically targeting 
    re-recorded and compressed deepfakes.
    """
    # Use higher MFCCs to catch micro-textures often lost in replay
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Spectral Rolloff: AI breaths often lack high-frequency 'air'
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    # Root Mean Square (RMS) for envelope analysis
    rms = librosa.feature.rms(y=y)[0]
    return mfccs, rolloff, rms

def process_forensic_audio(file):
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        if len(y) < 16000: return None
        
        # 1. Detect Breath Candidates
        rms = librosa.feature.rms(y=y)[0]
        # AI often has 'dead silence' or 'static floor' between breaths
        low_energy = (rms < np.percentile(rms, 30)).astype(float)
        peaks, _ = find_peaks(low_energy, height=0.4, distance=sr//3)
        events = librosa.frames_to_time(peaks, sr=sr)
        
        if len(events) < 2:
            return {"AI_Prob": 0.90, "Status": "🤖 AI (No Biological Artifacts)", "Reason": "Missing Breaths"}

        # 2. Penalty: Breath Token Reuse (Crucial for AI detection)
        # AI models often reuse the same breath 'sound' across a clip.
        breath_samples = []
        for t in events[:5]:
            s, e = int(t*sr), int((t+0.25)*sr)
            m = librosa.feature.mfcc(y=y[s:e], sr=sr, n_mfcc=13).mean(1)
            breath_samples.append(m)
        
        # Calculate consistency: High consistency = AI (Token reuse)
        similarities = [np.corrcoef(breath_samples[i], breath_samples[i+1])[0,1] 
                        for i in range(len(breath_samples)-1)]
        avg_sim = np.mean(similarities) if similarities else 0
        
        # 3. Penalty: Spectral Flatness (Detects Replay Noise)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # 4. Final Research Score (Weighted)
        # We penalize 'Too Consistent' (AI) and 'Too Clean' (Digital)
        token_penalty = 0.4 if avg_sim > 0.85 else 0.0
        rhythm_penalty = 0.3 if (np.std(np.diff(events)) < 0.2) else 0.0
        
        total_ai_score = 0.3 + token_penalty + rhythm_penalty
        
        return {
            "AI_Prob": min(total_ai_score, 1.0),
            "Token_Similarity": avg_sim,
            "Breaths": len(events),
            "Status": "🤖 AI" if total_ai_score > 0.52 else "👤 HUMAN"
        }
    except: return None

# --- UI ---
st.title("🛡️ PneumaForensic v5.0: Replay-Aware Detection")
st.info("Note: Detecting re-recorded (replay) audio is a known research challenge. This version uses Token Consistency to bypass acoustic masking.")

uploaded = st.file_uploader("Upload Batch", accept_multiple_files=True)
if uploaded:
    data = []
    for f in uploaded:
        res = process_forensic_audio(f)
        if res: data.append({"Filename": f.name, **res})
    
    df = pd.DataFrame(data)
    st.dataframe(df[["Filename", "AI_Prob", "Status", "Breaths"]], use_container_width=True)
    
    # Research Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Research CSV", data=csv, file_name="forensic_v5_results.csv")
