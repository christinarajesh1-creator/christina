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
        # High-res load to catch neural texture clones
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True, duration=30)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # 1. Advanced Breath Tracking
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.15, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {"File": name, "Status": "AI (No Bio)", "AI Prob": "95%", "IBI Reg": 0, "Amp Var": 0, "Presence": "0%", "Sim Val": 0}, []

    # --- PARAMETER EXTRACTION ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    presence = (len(breaths) * 0.45) / duration

    # The "Neural Clone" Fingerprint (Detects copy-pasted breaths)
    textures = []
    for b in breaths[:6]:
        start, end = int(b * sr), int((b + 0.22) * sr)
        seg = y_norm[start:end]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    
    # Sim_val: AI usually < 1.45 (Cloned/Similar). Humans usually > 2.2 (Unique).
    sim_val = np.mean(np.std(textures, axis=0)) if len(textures) > 1 else 10.0

    # --- THE DETECTION ENGINE ---
    # We switch to a "Red Flag" system
    red_flags = 0
    
    # Flag 1: The "Uncanny" Timing (ElevenLabs hides in the 0.18-0.28 range)
    if 0.16 < ibi_cv < 0.31: red_flags += 2
    elif ibi_cv < 0.15: red_flags += 3 # Robotic
    
    # Flag 2: Spectral Clones (The strongest AI indicator)
    if sim_val < 1.6: red_flags += 3
    
    # Flag 3: Synthetic Presence
    if presence > 0.28: red_flags += 1
    
    # Flag 4: Flat Amplitude (No lung depletion)
    if amp_cv < 0.19: red_flags += 1

    # --- HUMAN PROTECTION ---
    # If the voice is naturally chaotic, ignore the red flags
    if (ibi_cv > 0.38) or (amp_cv > 0.40) or (sim_val > 2.8):
        status = "HUMAN"
        prob = 15
    elif red_flags >= 3:
        status = "AI"
        prob = min(99, 40 + (red_flags * 15))
    else:
        status = "HUMAN"
        prob = 35

    return {
        "File": name, "Status": status, "AI Prob": f"{prob}%",
        "IBI Reg": round(ibi_cv, 3), "Amp Var": round(amp_cv, 3), 
        "Presence": f"{presence:.1%}", "Sim Val": round(sim_val, 3)
    }, breaths

# --- UI SECTION ---
st.title("🔬 PneumaForensic v16.0")
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
