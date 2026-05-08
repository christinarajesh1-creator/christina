import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(page_title="PneumaForensic v5.0", layout="wide")

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
    
    # 1. High-Sensitivity Breath Detection
    hop_length = 256
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Adaptive Threshold: AI breaths often sit in a specific energy band
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.2, distance=sr//hop_length)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]

    # --- THE 6 FORENSIC PARAMETERS ---
    
    # 1. Timing Irregularity (IBI CV)
    ibi = np.diff(breaths) if len(breaths) > 1 else [0]
    ibi_cv = np.std(ibi) / np.mean(ibi) if np.mean(ibi) > 0 else 0
    
    # 2. Amplitude Variance (Breath Force)
    amps = [rms_smooth[p] for p in peaks] if len(peaks) > 0 else [0]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    
    # 3. Spectral Flux (Timbre Consistency)
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    flux_cv = np.std(flux) / np.mean(flux)

    # 4. Zero Crossing Rate (Breath Texture)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_cv = np.std(zcr) / np.mean(zcr)

    # 5. High-Frequency Air (HF Ratio)
    # AI often has a "cut-off" or "hiss" at high frequencies
    spec = np.abs(librosa.stft(y))
    hf_energy = np.mean(spec[int(len(spec)*0.7):])
    lf_energy = np.mean(spec[:int(len(spec)*0.2)])
    hf_ratio = hf_energy / (lf_energy + 1e-10)

    # 6. Silence Floor Floor (Digital Silence Check)
    silence_noise = np.percentile(rms, 5)

    # --- DETECTOR LOGIC: THE GOLDILOCKS CHECK ---
    # Humans usually stay between 0.15 and 0.40 for CVs.
    # AI is either < 0.1 (Robotic) or > 0.45 (Chaotic/Noisy WhatsApp artifacts)
    
    ai_indicators = 0
    if ibi_cv < 0.14 or ibi_cv > 0.45: ai_indicators += 1  # Irregular rhythm
    if amp_cv < 0.10 or amp_cv > 0.50: ai_indicators += 1  # Static volume
    if zcr_cv > 0.40: ai_indicators += 1                  # Digital texture hiss
    if hf_ratio < 0.002: ai_indicators += 1               # Missing "air" (Typical AI)
    if silence_noise < 0.0001: ai_indicators += 1         # Perfect digital silence

    # Probability Calculation
    # We baseline at 50% and move based on indicators
    base_prob = 0.45 
    if len(breaths) > 35: base_prob += 0.2 # Excessive breathing is a common AI trait
    
    final_prob = np.clip(base_prob + (ai_indicators * 0.15), 0.1, 0.99)
    
    # Force AI status if indicators are high
    status = "AI" if final_prob > 0.60 or ai_indicators >= 3 else "HUMAN"

    return {
        "File": name,
        "Status": status,
        "AI Probability": f"{final_prob:.0%}",
        "Breaths": len(breaths),
        "Timing_CV": f"{ibi_cv:.3f}",
        "Amp_CV": f"{amp_cv:.3f}",
        "Texture_CV": f"{zcr_cv:.3f}",
        "HF_Ratio": f"{hf_ratio:.5f}",
        "AI_Flags": ai_indicators
    }, breaths

# --- STREAMLIT UI ---
st.title("🔬 PneumaForensic v5.0")
st.markdown("### Specialized for WhatsApp & Synthetic Voice Detection")

uploaded_files = st.file_uploader("Upload .wav or .mp3", type=['wav', 'mp3'], accept_multiple_files=True)

if uploaded_files:
    report_data = []
    for f in uploaded_files:
        f.seek(0)
        y, sr = load_audio(f.read())
        
        if y is not None:
            metrics, peaks = analyze_voice(y, sr, f.name)
            report_data.append(metrics)
            
            # Simplified Visualizer
            with st.expander(f"Analysis: {f.name}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.plot(y, color='cyan' if metrics['Status'] == "HUMAN" else 'red', alpha=0.6)
                    for p in peaks:
                        ax.axvline(x=p*sr, color='white', linestyle='--', alpha=0.3)
                    ax.set_axis_off()
                    fig.patch.set_facecolor('#0e1117')
                    st.pyplot(fig)
                with col2:
                    st.metric("AI Flags", metrics['AI_Flags'])
                    st.metric("Probability", metrics['AI Probability'])

    st.divider()
    st.subheader("📊 Forensic Batch Results")
    df = pd.DataFrame(report_data)
    
    # Styling the dataframe
    def color_status(val):
        color = '#ff4b4b' if val == "AI" else '#00f900'
        return f'color: {color}; font-weight: bold'
    
    st.dataframe(df.style.applymap(color_status, subset=['Status']), use_container_width=True)
else:
    st.info("Awaiting audio files for forensic scan...")
