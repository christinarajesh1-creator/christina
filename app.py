import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        return y, sr
    except:
        return None, None

def analyze_voice(y, sr, name):
    y = librosa.util.normalize(y)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Breath Detection
    hop_length = 256
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    rms_smooth = np.convolve(rms[0], np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.15, distance=sr//hop_length)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]

    # Parameter Extraction
    ibi = np.diff(breaths) if len(breaths) > 1 else np.array([])
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    
    amps = [rms_smooth[p] for p in peaks] if len(peaks) > 0 else []
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-10)

    # Entropy/Flatness (Detects artificial noise injection)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # RE-CALIBRATED AI DETECTION TIGHTENING
    ai_score = 0
    
    # 1. Timing Tightening (AI often hits exactly 0.18-0.22 CV to "sound human")
    if 0.17 < ibi_cv < 0.23: ai_score += 35  # This is the "Suspect Simulation" zone
    elif ibi_cv < 0.14 or ibi_cv > 0.48: ai_score += 40
    
    # 2. Amplitude Consistency
    if 0.10 < amp_cv < 0.14: ai_score += 30  # Common AI target range
    elif amp_cv < 0.08: ai_score += 40
    
    # 3. Spectral Fingerprint
    if flatness > 0.05: ai_score += 20  # AI noise is often too "white" or "flat"
    if zcr_cv > 0.45: ai_score += 25    # Unnatural texture jitter
    
    # 4. Quantity Check
    if len(breaths) > 30: ai_score += 20  # Synthetic "over-breathing"
    if len(breaths) < 3 and duration > 10: ai_score += 50 # Unnatural lack of breath

    status = "AI" if ai_score > 55 else "HUMAN"
    prob = min(99, ai_score + 15) if status == "AI" else max(1, ai_score // 2)

    return {
        "File": name,
        "Status": status,
        "AI Probability": f"{prob}%",
        "Breaths": len(breaths),
        "Timing_CV": round(ibi_cv, 3),
        "Amp_CV": round(amp_cv, 3),
        "Texture_CV": round(zcr_cv, 3),
        "Flatness": round(flatness, 4)
    }, breaths

st.title("PneumaForensic v7.0 (Tightened)")

files = st.file_uploader("", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    results = []
    for f in files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            m, p = analyze_voice(y, sr, f.name)
            results.append(m)
            with st.expander(f.name):
                fig, ax = plt.subplots(figsize=(10, 1.2))
                ax.plot(y, color='red' if m['Status'] == "AI" else 'lime', alpha=0.5)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

    df = pd.DataFrame(results)
    def color_status(s): return "color: #ff4b4b" if s == "AI" else "color: #00f900"
    
    try:
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
    except:
        st.dataframe(df, use_container_width=True)
