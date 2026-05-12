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
    
    # 1. High-Sensitivity Breath Tracking
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop)
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.1, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    # If no breaths, we suspect AI but remain cautious
    if len(breaths) < 2:
        status = "AI (No Breaths)" if duration > 6 else "HUMAN (Short Clip)"
        prob = "85%" if duration > 6 else "20%"
        return {"File": name, "Status": status, "AI Prob": prob}, []

    # --- PARAMETER CALCULATION ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi)
    
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps)
    
    presence_ratio = (len(breaths) * 0.45) / duration

    textures = []
    for b in breaths[:4]:
        start = int(b * sr)
        seg = y_norm[start:start+int(0.25*sr)]
        if len(seg) >= int(0.15*sr):
            m = np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1)
            textures.append(m)
    sim_val = np.mean(np.std(textures, axis=0)) if len(textures) > 1 else 2.0

    # --- THE HUMANITY SHIELD LOGIC ---
    ai_score = 0
    
    # 1. Biological Chaos Check: Humans are usually very irregular (>0.3 CV)
    is_biological = (ibi_cv > 0.35) or (amp_cv > 0.35) or (sim_val > 2.5)

    # 2. AI Red Flags
    if 0.18 < ibi_cv < 0.25: ai_score += 30 # The "Uncanny" Rhythm
    if ibi_cv < 0.12: ai_score += 40        # Too robotic
    if sim_val < 1.1: ai_score += 35        # Clone Breaths (ElevenLabs)
    if presence_ratio > 0.32: ai_score += 20 # Over-breathing

    # 3. Final Protection: If it shows biological chaos, cap the AI probability
    if is_biological:
        final_prob = min(35, ai_score) # Force it to stay below the AI threshold
    else:
        final_prob = min(99, ai_score + 10)

    status = "AI" if final_prob > 50 else "HUMAN"
    
    return {
        "File": name, "Status": status, "AI Prob": f"{final_prob}%",
        "IBI CV": round(ibi_cv, 3), "Amp CV": round(amp_cv, 3), 
        "Presence": f"{presence_ratio:.1%}", "B-Sim": round(sim_val, 3)
    }, breaths

# --- UI SECTION ---
st.title("🔬 PneumaForensic v10.2")

files = st.file_uploader("Upload Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    results = []
    for f in files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            metrics, b_times = forensic_analysis(y, sr, f.name)
            results.append(metrics)
            
            with st.expander(f"Visual Scan: {f.name}"):
                fig, ax = plt.subplots(figsize=(12, 1.2))
                t = np.linspace(0, len(y)/sr, len(y))
                ax.plot(t, y, color='gray', alpha=0.4) 
                for bt in b_times:
                    ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2)
                ax.set_title(f"{metrics['Status']} ({metrics['AI Prob']})", color='white', loc='right')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            del y
            gc.collect()

    st.divider()
    df = pd.DataFrame(results)
    def color_status(val):
        return f'color: {"#ff4b4b" if "AI" in val else "#00f900"}; font-weight: bold'
    if not df.empty:
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
