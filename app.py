import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import gc

st.set_page_config(page_title="PneumaForensic v12.0", layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True, duration=30)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # 1. Precise Breath Tracking
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.1, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        # Long clips without breaths are almost always AI
        status = "AI (No Breaths)" if duration > 7 else "HUMAN (Short)"
        prob = "90%" if duration > 7 else "15%"
        return {"File": name, "Status": status, "AI Prob": prob, "IBI CV": 0, "Amp CV": 0, "Presence": "0%", "B-Sim": 0}, []

    # --- PARAMETER EXTRACTION ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    presence_ratio = (len(breaths) * 0.45) / duration

    # Breath Clone Detection (MFCC Texture)
    textures = []
    for b in breaths[:5]:
        start, end = int(b * sr), int((b + 0.2) * sr)
        seg = y_norm[start:end]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    
    sim_val = np.mean(np.std(textures, axis=0)) if len(textures) > 1 else 3.0

    # --- BIOLOGICAL ANCHOR LOGIC ---
    # Humans have high organic messiness. If these values are high, it's a Human.
    is_human_biology = (ibi_cv > 0.38) or (amp_cv > 0.38) or (sim_val > 2.8)

    # AI Detection Flags
    ai_points = 0
    if 0.17 < ibi_cv < 0.26: ai_points += 40  # Uncanny Rhythm
    if ibi_cv < 0.13: ai_points += 50         # Robotic Rhythm
    if sim_val < 1.2: ai_score += 45          # Clone Breaths
    if presence_ratio > 0.33: ai_points += 30 # Over-breathing

    # Decision Engine
    if is_human_biology:
        # The Anchor: Biological chaos overrides AI flags
        final_prob = min(30, ai_points)
    else:
        final_prob = min(99, ai_points + 10)

    status = "AI" if final_prob >= 50 else "HUMAN"
    
    return {
        "File": name, "Status": status, "AI Prob": f"{final_prob}%",
        "IBI CV": round(ibi_cv, 3), "Amp CV": round(amp_cv, 3), 
        "Presence": f"{presence_ratio:.1%}", "B-Sim": round(sim_val, 3)
    }, breaths

# --- STREAMLIT UI ---
st.title("🔬 PneumaForensic v12.0")

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
                with st.expander(f"{f.name} - {m['Status']}"):
                    fig, ax = plt.subplots(figsize=(12, 1.2))
                    ax.plot(y, color='gray', alpha=0.4) 
                    for bt in b_times:
                        ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"Error: {e}")
            del y
            gc.collect()

    st.divider()
    df = pd.DataFrame(results)
    if not df.empty:
        def color_status(v): return f'color: {"#ff4b4b" if "AI" in v else "#00f900"}; font-weight: bold'
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
