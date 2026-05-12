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
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True, duration=30)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.08, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {"File": name, "Status": "AI", "AI Prob": "99%", "IBI Reg": 0, "Amp Var": 0, "Presence": "0%", "Sim Val": 0}, []

    # PARAMETERS
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    presence = (len(breaths) * 0.45) / duration

    # Sim Val: Detecting Biological Entropy
    textures = []
    for b in breaths[:6]:
        start, end = int(b * sr), int((b + 0.25) * sr)
        seg = y_norm[start:end]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    
    sim_val = np.mean(np.std(textures, axis=0)) * 10 if len(textures) > 1 else 50.0

    # DECISION ENGINE: AI UNTIL PROVEN HUMAN
    # If the voice is messy/irregular (High Sim Val & high CV), it's human.
    is_human = (sim_val > 35.0) and (ibi_cv > 0.31 or amp_cv > 0.31)

    if is_human:
        final_prob = 15
        status = "HUMAN"
    else:
        # Penalize low entropy (AI traits)
        if sim_val < 22.0: final_prob = 99
        elif sim_val < 32.0: final_prob = 85
        else: final_prob = 65
        status = "AI"

    return {
        "File": name, "Status": status, "AI Prob": f"{final_prob}%",
        "IBI Reg": round(ibi_cv, 3), "Amp Var": round(amp_cv, 3), 
        "Presence": f"{presence:.1%}", "Sim Val": round(sim_val, 3)
    }, breaths

# --- UI ---
st.title("🔬 PneumaForensic v18.1")

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
                    t = np.linspace(0, len(y)/sr, len(y))
                    ax.plot(t, y, color='gray', alpha=0.4) 
                    for bt in b_times:
                        ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"Error: {e}")
            del y
            gc.collect()

    df = pd.DataFrame(results)
    if not df.empty:
        def color_status(v):
            return f'color: {"#ff4b4b" if "AI" in v else "#00f900"}; font-weight: bold'
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
