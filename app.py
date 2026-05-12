import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import gc

# 1. Page Config must be the first Streamlit command
st.set_page_config(page_title="Deepfake Voice Detection", layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        # Load mono at 16kHz for consistent forensic analysis
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True, duration=30)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # Precise Breath Tracking
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Identify breath candidates
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.12, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {"File": name, "Status": "AI (No Bio)", "AI Prob": "98%", "IBI Reg": 0, "Amp Var": 0, "Presence": "0%", "Sim Val": 0}, []

    # --- PARAMETER EXTRACTION ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    presence = (len(breaths) * 0.45) / duration

    # Similarity Fingerprint (Detects AI Clone Breaths)
    textures = []
    for b in breaths[:6]:
        start, end = int(b * sr), int((b + 0.25) * sr)
        seg = y_norm[start:end]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    
    # Measures 'Biological entropy' (How different each breath is)
    # Scaled by 10 to match your 12.0 - 45.0 range
    sim_val = np.mean(np.std(textures, axis=0)) * 10 if len(textures) > 1 else 50.0

    # --- DECISION ENGINE v17.1 ---
    ai_points = 15
    if sim_val < 25.0: ai_points += 60
    elif sim_val < 32.0: ai_points += 35
    if presence > 0.26: ai_points += 25
    if 0.17 < ibi_cv < 0.31: ai_points += 20

    # Humanity Shield Override
    is_true_human = (sim_val > 45.0) or (ibi_cv > 0.45)
    final_prob = 15 if is_true_human else min(99, ai_points)
    status = "AI" if final_prob >= 50 else "HUMAN"

    return {
        "File": name, 
        "Status": status, 
        "AI Prob": f"{final_prob}%",
        "IBI Reg": round(ibi_cv, 3), 
        "Amp Var": round(amp_cv, 3), 
        "Presence": f"{presence:.1%}", 
        "Sim Val": round(sim_val, 3)
    }, breaths

# --- UI SECTION ---
st.title("🔬 PneumaForensic v17.1")

files = st.file_uploader("Upload audio batch", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    results = []
    for f in files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            try:
                m, b_times = forensic_analysis(y, sr, f.name)
                results.append(m)
                
                with st.expander(f"Visual Scan: {f.name}"):
                    fig, ax = plt.subplots(figsize=(12, 1.2))
                    t = np.linspace(0, len(y)/sr, len(y))
                    ax.plot(t, y, color='gray', alpha=0.4) 
                    for bt in b_times:
                        ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2)
                    ax.set_title(f"{m['Status']} ({m['AI Prob']})", color='white', loc='right')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"Error on {f.name}: {e}")
            
            del y
            gc.collect()

    st.divider()
    if results:
        df = pd.DataFrame(results)
        def color_status(v): return f'color: {"#ff4b4b" if "AI" in v else "#00f900"}; font-weight: bold'
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
