import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import gc

st.set_page_config(page_title="Deepfake Voice Detection", layout="wide")

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
    
    # Breath Tracking
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.12, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        status = "AI (No Breaths)" if duration > 5 else "HUMAN (Short)"
        return {"File": name, "Status": status, "AI Prob": "90%" if "AI" in status else "10%"}, []

    # --- PARAMETERS ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    presence_ratio = (len(breaths) * 0.48) / duration

    # The "Clone Check" (Neural Inhale Similarity)
    textures = []
    for b in breaths[:6]:
        start, end = int(b * sr), int((b + 0.2) * sr)
        seg = y_norm[start:end]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    
    # Sim_val: Low (< 1.3) = Copy-pasted breaths (AI); High (> 2.0) = Unique human breaths
    sim_val = np.mean(np.std(textures, axis=0)) if len(textures) > 1 else 5.0

    # --- SIGNATURE DECISION ENGINE ---
    # We use a priority system instead of adding points
    
    # 1. Check for "Cloned" AI signatures (Highest Accuracy)
    if sim_val < 1.35:
        status = "AI (Cloned Breaths)"
        prob = 98
    # 2. Check for "Robot" Rhythm
    elif ibi_cv < 0.14:
        status = "AI (Robotic Rhythm)"
        prob = 95
    # 3. Check for "Over-Engineered" Realism (High breath count + specific timing)
    elif (0.17 < ibi_cv < 0.28) and (presence_ratio > 0.27):
        status = "AI (Neural Pattern)"
        prob = 85
    # 4. Check for Biological Irregularity (The Human Shield)
    elif (ibi_cv > 0.35) or (amp_cv > 0.38) or (sim_val > 2.5):
        status = "HUMAN"
        prob = 15
    else:
        status = "HUMAN"
        prob = 35

    return {
        "File": name, "Status": status, "AI Prob": f"{prob}%",
        "IBI Reg": round(ibi_cv, 3), "Amp Var": round(amp_cv, 3), 
        "Presence": f"{presence_ratio:.1%}", "Sim Val": round(sim_val, 3)
    }, breaths

# --- UI SECTION ---
st.title("🔬 PneumaForensic v14.0")
files = st.file_uploader("", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    results = []
    for f in files:
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
