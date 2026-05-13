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
        # High resolution to capture micro-friction inside vocal air streams
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True, duration=30)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # 1. Micro-Segment Windowing
    hop = 128  
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.12, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {
            "File": name, "Status": "AI", "AI Prob": "98%", 
            "IBI Reg": 0.0, "Amp Var": 0.0, "Presence": "0%", "Sim Val": 0.0
        }, []

    # --- PARAMETER EXTRACTION ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    presence_ratio = (len(breaths) * 0.45) / duration

    # 4. Multi-Frame Texture
    textures = []
    for b in breaths[:5]:
        start = int(b * sr)
        seg = y_norm[start:start+int(0.2*sr)]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    sim_val = np.mean(np.std(textures, axis=0)) * 100 if len(textures) > 1 else 150.0

    # --- EXTRACTION OF THE MICRO-DYNAMIC AI PATHWAY ---
    # Tracking the correlation between spectral centroid movement and flux velocity
    centroid = librosa.feature.spectral_centroid(y=y_norm, sr=sr, hop_length=hop).flatten()
    flux = librosa.onset.onset_strength(y=y_norm, sr=sr, hop_length=hop)
    
    # Truncate arrays to identical length for correlation checking
    min_len = min(len(centroid), len(flux))
    c_slice = centroid[:min_len]
    f_slice = flux[:min_len]
    
    # Calculate Cross-Feature Structural Consistency
    # AI vocoders exhibit a highly stable, repeating structural trace here
    cross_corr = np.corrcoef(c_slice, f_slice)[0, 1] if min_len > 2 else 0.0
    cross_cv = np.std(c_slice * f_slice) / (np.mean(c_slice * f_slice) + 1e-10)

    # --- FORENSIC DECISION MATRIX ---
    # Unbiased calculation based on mathematical micro-features instead of names
    ai_score = 15.0
    
    # Check 1: Macro Mimicry Trap 
    # Flags if timing/volume is pretending to be human but falls in the generation zone
    if 0.16 < ibi_cv < 0.32 and amp_cv < 0.28:
        ai_score += 25

    # Check 2: Cloned Breath Block Texture Check
    if sim_val < 140.0:
        ai_score += 25

    # Check 3: Micro-Phase Modulation Analysis (The Ultimate AI Identifier)
    # Human air friction has high variance; AI vocoders are highly linear (cross_corr sits near stable bins)
    if 0.45 < abs(cross_corr) < 0.78:
        ai_score += 35
    if cross_cv < 1.15:
        ai_score += 20

    final_prob = min(99, int((ai_score / 120.0) * 100))
    
    # Biological Chaos Override
    if ibi_cv > 0.21 and amp_cv > 0.24 and cross_cv > 1.30:
        status = "HUMAN"
        final_prob = max(8, int(final_prob * 0.3))
    else:
        status = "AI" if final_prob >= 50 else "HUMAN"

    return {
        "File": name, 
        "Status": status, 
        "AI Prob": f"{final_prob}%",
        "IBI Reg": round(ibi_cv, 6), 
        "Amp Var": round(amp_cv, 6), 
        "Presence": f"{presence_ratio:.1%}", 
        "Sim Val": round(sim_val, 6)
    }, breaths

# --- UI LAYER ---
st.title("🔬 Deepfake Voice Detection Engine")
st.caption("Title: Deepfake Voice Detection Based on Breath Pattern Analysis")

uploaded_files = st.file_uploader("Upload Audio Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    for f in uploaded_files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            try:
                metrics, b_times = forensic_analysis(y, sr, f.name)
                results_list.append(metrics)
                
                with st.expander(f"Visualizing Structural Pauses: {f.name}"):
                    fig, ax = plt.subplots(figsize=(12, 1.2))
                    t = np.linspace(0, len(y)/sr, len(y))
                    ax.plot(t, y, color='gray', alpha=0.4, linewidth=0.5) 
                    for bt in b_times:
                        ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"Processing error: {e}")
            del y
            gc.collect()

    df = pd.DataFrame(results_list)
    if not df.empty:
        def color_status(v):
            return f'color: {"#ff4b4b" if "AI" in v else "#00f900"}; font-weight: bold'
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
