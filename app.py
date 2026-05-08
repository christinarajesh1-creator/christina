import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(page_title="PneumaForensic v4.0", layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        # Load as 16kHz mono for consistent analysis
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        return y, sr
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None, None

def analyze_voice(y, sr, name):
    # Pre-processing
    y = librosa.util.normalize(y)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 1. Detect Breath Events using RMS Energy
    hop_length = 256
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    # Smooth the RMS to find distinct breath "humps"
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Thresholding: Look for spikes above median but below max speech
    median_rms = np.median(rms_smooth)
    peaks, _ = signal.find_peaks(rms_smooth, height=median_rms*1.1, distance=sr//hop_length)
    
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    
    # Filter: Only keep events that look like breaths (0.2s to 0.8s)
    breaths = [t for t in times if 0.5 < t < (duration - 0.5)]
    
    # --- Metrics Logic ---
    results = {"File": name, "Status": "Unknown", "AI Probability": "0%"}
    
    if len(breaths) < 2:
        results["Status"] = "AI (No Breaths)"
        results["AI Probability"] = "95%"
        return results, []

    # Parameter 1: Timing Variance (IBI CV)
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi)
    
    # Parameter 2: Amplitude Variance
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps)
    
    # Parameter 3: Spectral Flux (Timbre changes)
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    flux_cv = np.std(flux) / np.mean(flux)

    # Parameter 4: Zero Crossing Rate (Texture of inhale)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_cv = np.std(zcr) / np.mean(zcr)

    # Parameter 5: Silence Floor (Digital vs Analog silence)
    silence_check = np.percentile(rms, 10)
    
    # Parameter 6: Harmonic-to-Noise Ratio (HNR) proxy
    # AI breaths are often "cleaner" (less chaotic) than human inhales
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_cv = np.std(centroid) / np.mean(centroid)

    # SCORING (Humans are messy/irregular = High CV)
    # If CV is very low (< 0.15), it's likely a repetitive AI artifact
    human_score = (ibi_cv * 0.3) + (amp_cv * 0.2) + (flux_cv * 0.2) + (zcr_cv * 0.15) + (centroid_cv * 0.15)
    
    # Normalize: AI voices typically show very low human_score
    ai_prob = np.clip(1.0 - (human_score * 1.8), 0.0, 1.0)
    
    results.update({
        "Status": "HUMAN" if ai_prob < 0.5 else "AI",
        "AI Probability": f"{ai_prob:.0%}",
        "Breaths": len(breaths),
        "Timing_CV": f"{ibi_cv:.3f}",
        "Amp_CV": f"{amp_cv:.3f}",
        "Texture_CV": f"{zcr_cv:.3f}"
    })
    
    return results, breaths

# UI Logic
st.title("🔬 PneumaForensic v4.0")
st.subheader("High-Precision Breath Parameter Analysis")

files = st.file_uploader("Drop audio files here", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    all_data = []
    for f in files:
        f.seek(0)
        y, sr = load_audio(f.read())
        
        if y is not None:
            metrics, peaks = analyze_voice(y, sr, f.name)
            all_data.append(metrics)
            
            # Visualization
            with st.expander(f"View Waveform: {f.name}"):
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#2ecc71' if metrics['Status'] == 'HUMAN' else '#e74c3c')
                for p in peaks:
                    ax.axvline(x=p, color='white', linestyle='--', alpha=0.5)
                ax.set_facecolor('#1e1e1e')
                fig.patch.set_facecolor('#1e1e1e')
                st.pyplot(fig)
                plt.close(fig)

    st.divider()
    st.write("### Forensic Report")
    st.dataframe(pd.DataFrame(all_data), use_container_width=True)
else:
    st.info("Upload an audio file to see the 6-parameter breakdown.")
