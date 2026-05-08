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
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.1, distance=sr//hop_length)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]

    # Parameter Extraction
    ibi = np.diff(breaths) if len(breaths) > 1 else [0]
    ibi_cv = np.std(ibi) / np.mean(ibi) if np.mean(ibi) > 0 else 0
    
    amps = [rms_smooth[p] for p in peaks] if len(peaks) > 0 else [0]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-10)

    spec = np.abs(librosa.stft(y))
    hf_ratio = np.mean(spec[int(len(spec)*0.7):]) / (np.mean(spec[:int(len(spec)*0.2)]) + 1e-10)
    
    # Forensic Logic: Humans have mid-range variation. 
    # AI has either "perfect" low CV or "unnatural" high CV from noise injection.
    ai_score = 0
    if ibi_cv < 0.15 or ibi_cv > 0.45: ai_score += 30
    if amp_cv < 0.12 or amp_cv > 0.48: ai_score += 25
    if zcr_cv > 0.42: ai_score += 20
    if hf_ratio < 0.002: ai_score += 15
    if len(breaths) > 35: ai_score += 10

    status = "AI" if ai_score > 45 else "HUMAN"
    prob = min(99, ai_score + 10) if status == "AI" else max(5, ai_score)

    return {
        "File": name,
        "Status": status,
        "AI Probability": f"{prob}%",
        "Breaths": len(breaths),
        "Timing_CV": round(ibi_cv, 3),
        "Amp_CV": round(amp_cv, 3),
        "Texture_CV": round(zcr_cv, 3)
    }, breaths

st.title("PneumaForensic v6.0")

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
                fig, ax = plt.subplots(figsize=(10, 1.5))
                ax.plot(y, color='red' if m['Status'] == "AI" else 'lime', alpha=0.6)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

    df = pd.DataFrame(results)
    
    def color_status(s):
        return "color: #ff4b4b" if s == "AI" else "color: #00f900"

    try:
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
    except:
        st.dataframe(df, use_container_width=True)
