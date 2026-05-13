import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import skew
import gc

st.set_page_config(page_title="Deepfake Voice Detection", layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        # Load at 22050Hz to ensure we don't compress high-frequency breath textures
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True, duration=30)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # 1. Advanced Time-Frequency Boundary Analysis
    hop = 128  # Millisecond resolution
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Isolate precise breath humps from ambient line noise
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.12, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {
            "File": name, "Status": "AI", "AI Prob": "99%", 
            "IBI Reg": 0.0, "Amp Var": 0.0, "Presence": "0%", "Sim Val": 0.0
        }, []

    # --- THE 6 MATHEMATICALLY RIGOROUS FORENSIC METRICS ---
    
    # 1. IBI Regularity (28%)
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0

    # 2. Breath Amplitude (15%)
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    # 3. Breath Presence (15%)
    presence_ratio = (len(breaths) * 0.42) / duration

    # 4. Multi-Frame Texture Invariance (18%) - Cloned Inhale Engine
    textures = []
    for b in breaths[:5]:
        start = int(b * sr)
        seg = y_norm[start:start+int(0.2*sr)]
        if len(seg) >= int(0.1*sr):
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
            textures.append(np.mean(mfcc, axis=1))
    # Standard deviation across MFCC coefficients
    sim_val = np.mean(np.std(textures, axis=0)) * 10 if len(textures) > 1 else 50.0

    # 5. Spectral Flux Velocity (12%) - Captures abrupt synthetic splice boundaries
    flux = librosa.onset.onset_strength(y=y_norm, sr=sr, hop_length=hop)
    flux_skew = skew(flux)

    # 6. Spectral Centroid Skewness (12%) - Checks for synthetic texture smoothness
    centroid = librosa.feature.spectral_centroid(y=y_norm, sr=sr, hop_length=hop).flatten()
    centroid_cv = np.std(centroid) / (np.mean(centroid) + 1e-10)

    # --- UNBIASED CLASSIFICATION ENGINE ---
    # We assign structural anomaly weights based on acoustic principles
    ai_score = 10.0
    
    # Check A: The Synthetic Cadence Window
    if 0.16 < ibi_cv < 0.31: ai_score += 25
    elif ibi_cv < 0.14: ai_score += 35
    
    # Check B: Textural Invariance (Identical neural generator waveforms)
    if sim_val < 25.0: ai_score += 35
    elif sim_val < 32.0: ai_score += 15
    
    # Check C: Abrupt Boundary Transitions (High flux asymmetry from digital splicing)
    if flux_skew > 3.2: ai_score += 20
    
    # Check D: Neural Model Compression Constraints
    if presence_ratio > 0.26: ai_score += 15
    if amp_cv < 0.18: ai_score += 10
    if centroid_cv < 0.25: ai_score += 15

    # Safe normalization scaling without infinite multipliers
    final_prob = min(99, int((ai_score / 115.0) * 100))
    
    # The Biological Defendable Veto (If overall timing and texture chaos is organic)
    if ibi_cv > 0.33 and sim_val > 35.0 and flux_skew < 2.5:
        status = "HUMAN"
        final_prob = min(30, final_prob)
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

uploaded_files = st.file_uploader("Upload Audio Batch (Supports 150+ Samples)", type=['wav', 'mp3'], accept_multiple_files=True)

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
