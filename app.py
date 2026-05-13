import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
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
    
    hop = 128  
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(7)/7, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.15, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {"File": name, "Status": "AI", "AI Prob": "99%", "IBI Reg": 0, "Amp Var": 0, "Presence": "0%", "Sim Val": 0}, []

    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    presence_ratio = (len(breaths) * 0.35) / duration

    textures = []
    for b in breaths[:5]:
        start = int(b * sr)
        seg = y_norm[start:start+int(0.2*sr)]
        if len(seg) >= int(0.1*sr):
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
            textures.append(np.mean(mfcc, axis=1))
    
    sim_val = np.mean(np.std(textures, axis=0)) * 10 if len(textures) > 1 else 50.0

    flux = librosa.onset.onset_strength(y=y_norm, sr=sr, hop_length=hop)
    flux_cv = np.std(flux) / (np.mean(flux) + 1e-10)

    spec_flat = librosa.feature.spectral_flatness(y=y_norm, hop_length=hop).flatten()
    entropy_val = entropy(np.histogram(spec_flat, bins=10)[0])

    ai_indicators = 0
    
    if 0.17 < ibi_cv < 0.30: ai_indicators += 1.5
    elif ibi_cv < 0.13: ai_indicators += 2.0
    
    if sim_val < 25.0: ai_indicators += 2.0
    elif sim_val < 32.0: ai_indicators += 1.0
    
    if presence_ratio > 0.27: ai_indicators += 1.0
    
    if entropy_val < 1.85: ai_indicators += 1.5
    if amp_cv < 0.18: ai_indicators += 1.0

    raw_score = (ai_indicators / 6.5) * 100
    
    if ibi_cv > 0.33 and entropy_val > 2.0:
        final_prob = max(5, int(raw_score * 0.4))  
    else:
        final_prob = min(99, int(raw_score))

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

st.title("🔬 Deepfake Voice Detection")
st.caption("Forensic Analysis Engine based on Breath Patterns")

uploaded_files = st.file_uploader("Upload Forensic Audio Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    for f in uploaded_files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            try:
                metrics, b_times = forensic_analysis(y, sr, f.name)
                results_list.append(metrics)
                
                with st.expander(f"Waveform Analysis: {f.name}"):
                    fig, ax = plt.subplots(figsize=(12, 1.2))
                    t = np.linspace(0, len(y)/sr, len(y))
                    ax.plot(t, y, color='gray', alpha=0.4, linewidth=0.5) 
                    for bt in b_times:
                        ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"Runtime Loop Interrupted: {e}")
            del y
            gc.collect()

    df = pd.DataFrame(results_list)
    if not df.empty:
        def color_status(v):
            return f'color: {"#ff4b4b" if "AI" in v else "#00f900"}; font-weight: bold'
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
