import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import gc

st.set_page_config(page_title="PneumaForensic v15.0", layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True, duration=30)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # 1. High-Sensitivity Breath Tracking
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.1, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        is_ai = duration > 5
        return {"File": name, "Status": "AI (No Bio)" if is_ai else "HUMAN", "AI Prob": "95%" if is_ai else "10%"}, []

    # --- THE 6 PARAMETERS (STRICT MEASUREMENT) ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi)
    
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps)
    
    presence = (len(breaths) * 0.45) / duration

    # Similarity Fingerprint (Detects ElevenLabs Clone Breaths)
    textures = []
    for b in breaths[:5]:
        start, end = int(b * sr), int((b + 0.2) * sr)
        seg = y_norm[start:end]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    sim_val = np.mean(np.std(textures, axis=0)) if len(textures) > 1 else 5.0

    zcr = librosa.feature.zero_crossing_rate(y_norm).flatten()
    zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-10)

    # --- THE DECISION MATRIX ---
    # We define "Human Territory" and "AI Territory"
    
    score = 0
    # AI SIGNATURE 1: The Inhale Clone (Strongest ElevenLabs Indicator)
    if sim_val < 1.3: score += 50 
    
    # AI SIGNATURE 2: Robotic Rhythm (Tuned to exclude human conversation)
    if 0.16 < ibi_cv < 0.26: score += 30
    elif ibi_cv < 0.14: score += 50
    
    # AI SIGNATURE 3: Over-Breathing
    if presence > 0.31: score += 25

    # HUMAN SHIELD: If it's messy, it's a person.
    is_messy = (ibi_cv > 0.36) or (amp_cv > 0.40) or (sim_val > 2.8)

    if is_messy:
        final_score = min(30, score) # Biological variance overrides AI flags
    else:
        final_score = score

    status = "AI" if final_score >= 45 else "HUMAN"
    
    return {
        "File": name, "Status": status, "AI Prob": f"{min(99, final_score)}%",
        "IBI Reg": round(ibi_cv, 3), "Amp Var": round(amp_cv, 3), 
        "Presence": f"{presence:.1%}", "ZCR Var": round(zcr_cv, 3), "Sim Val": round(sim_val, 3)
    }, breaths

# --- UI SECTION ---
st.title("🔬 PneumaForensic v15.0")
uploaded_files = st.file_uploader("", type=['wav', 'mp3'], accept_multiple_files=True)

if uploaded_files:
    results = []
    for f in uploaded_files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            m, b_times = forensic_analysis(y, sr, f.name)
            results.append(m)
            with st.expander(f"{f.name} - {m['Status']}"):
                fig, ax = plt.subplots(figsize=(12, 1.2))
                ax.plot(y, color='gray', alpha=0.4) 
                for bt in b_times:
                    ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            gc.collect()

    df = pd.DataFrame(results)
    if not df.empty:
        def color_status(v): return f'color: {"#ff4b4b" if "AI" in v else "#00f900"}; font-weight: bold'
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
