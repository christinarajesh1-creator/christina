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
    
    # 1. Advanced Breath Tracking
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.15, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {"File": name, "Status": "AI", "AI Prob": "98%", "IBI Reg": 0, "Amp Var": 0, "Presence": "0%", "ZCR Var": 0, "Sim Val": 0}, []

    # --- PARAMETERS ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    presence_ratio = (len(breaths) * 0.45) / duration
    
    zcr = librosa.feature.zero_crossing_rate(y_norm).flatten()
    zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-10)

    # Similarity Fingerprint (MFCC)
    textures = []
    for b in breaths[:4]:
        start, end = int(b * sr), int((b + 0.25) * sr)
        seg = y_norm[start:end]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    sim_val = np.mean(np.std(textures, axis=0)) if len(textures) > 1 else 10.0

    # --- FORENSIC LOGIC: HUMANITY SHIELD ---
    ai_points = 0
    
    # Flag 1: The "Uncanny" Timing (0.16 - 0.28)
    if 0.16 < ibi_cv < 0.28: 
        ai_points += 40
    elif ibi_cv < 0.14: 
        ai_points += 50
    
    # Flag 2: Over-Breathing (Tuned to exclude heavy-breathing humans)
    if presence_ratio > 0.32: 
        ai_points += 20
    
    # Flag 3: Clone Check (Most accurate AI indicator)
    if sim_val < 1.25: 
        ai_points += 45

    # --- THE OVERRIDE: IRREGULARITY = HUMAN ---
    # Humans have high 'biological messiness'. 
    # If timing OR volume is very irregular, it's almost certainly human.
    is_irregular = (ibi_cv > 0.32) or (amp_cv > 0.35) or (sim_val > 2.5)

    if is_irregular:
        final_prob = min(35, ai_points) # Force score below 50%
    else:
        final_prob = min(99, ai_points + 10)

    status = "AI" if final_prob >= 50 else "HUMAN"
    
    return {
        "File": name, "Status": status, "AI Prob": f"{final_prob}%",
        "IBI Reg": round(ibi_cv, 3), "Amp Var": round(amp_cv, 3), 
        "Presence": f"{presence_ratio:.1%}", "ZCR Var": round(zcr_cv, 3), "Sim Val": round(sim_val, 3)
    }, breaths

# --- UI SECTION ---
st.title("🔬 PneumaForensic v13.2")

files = st.file_uploader("", type=['wav', 'mp3'], accept_multiple_files=True)

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
                st.error(f"Error analyzing {f.name}: {e}")
            del y
            gc.collect()

    st.divider()
    df = pd.DataFrame(results)
    if not df.empty:
        def color_status(v): return f'color: {"#ff4b4b" if "AI" in v else "#00f900"}; font-weight: bold'
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
