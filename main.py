import streamlit as st

# 1. MUST BE THE ABSOLUTE FIRST LINE OF CODE
st.set_page_config(page_title="Deepfake Voice Detection", layout="wide")

import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.stats import skew
import gc

def load_audio_scipy(file_bytes):
    try:
        sr, y = wavfile.read(io.BytesIO(file_bytes))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if y.dtype != np.float32:
            y = y.astype(np.float32) / (np.max(np.abs(y)) + 1e-10)
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        if len(y) > sr * 30:
            y = y[:sr * 30]
        return y, sr
    except:
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
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.10, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {
            "File": name, "Status": "AI (No Bio)", "AI Prob": "98%", 
            "IBI Reg": 0.0, "Amp Var": 0.0, "Presence": "0%", "SFPA Asymmetry": 0.0
        }, []

    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    presence_ratio = (len(breaths) * 0.45) / duration

    sfpa_metrics = []
    stft_matrix = librosa.stft(y_norm, n_fft=512, hop_length=hop)
    magnitude = np.abs(stft_matrix)
    
    total_rows, total_cols = magnitude.shape
    hf_start_idx = int(total_rows * 0.75)
    hf_band = magnitude[hf_start_idx:, :]
    hf_flux = np.diff(hf_band, axis=1)
    
    for b in breaths[:5]:
        frame_idx = int((b * sr) / hop)
        start_frame = max(0, frame_idx - 5)
        end_frame = min(total_cols - 1, frame_idx + 15)
        frame_slice = hf_flux[:, start_frame:end_frame]
        if frame_slice.size > 0:
            sfpa_metrics.append(skew(frame_slice.flatten()))

    sfpa_final = np.mean(np.abs(sfpa_metrics)) if len(sfpa_metrics) > 0 else 0.0

    ai_weight = 10.0
    if sfpa_final < 1.85:
        ai_weight += 45.0
    elif sfpa_final < 1.98:
        ai_weight += 20.0
        
    if 0.16 < ibi_cv < 0.31:
        ai_weight += 20.0
    if presence_ratio > 0.26:
        ai_weight += 15.0
    if amp_cv < 0.19:
        ai_weight += 10.0

    final_prob = min(99, int((ai_weight / 100.0) * 100))
    if ibi_cv > 0.32 and sfpa_final > 2.10:
        status = "HUMAN"
        final_prob = min(25, final_prob)
    else:
        status = "AI" if final_prob >= 55 else "HUMAN"

    return {
        "File": name, 
        "Status": status, 
        "AI Prob": f"{final_prob}%",
        "IBI Reg": round(ibi_cv, 6), 
        "Amp Var": round(amp_cv, 6), 
        "Presence": f"{presence_ratio:.1%}", 
        "SFPA Asymmetry": round(sfpa_final, 6)
    }, breaths

st.title("🔬 Deepfake Voice Detection Engine")
st.caption("Forensic Analysis Pipeline Based on Micro-Phase Acoustic Friction")

uploaded_files = st.file_uploader("Upload Forensic Audio Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    for f in uploaded_files:
        f.seek(0)
        y, sr = load_audio_scipy(f.read())
        if y is not None:
            try:
                metrics, b_times = forensic_analysis(y, sr, f.name)
                results_list.append(metrics)
                
                with st.expander(f"Waveform Analysis: {f.name} -> {metrics['Status']}"):
                    fig, ax = plt.subplots(figsize=(12, 1.2))
                    t = np.linspace(0, len(y)/sr, len(y))
                    ax.plot(t, y, color='gray', alpha=0.4, linewidth=0.5) 
                    for bt in b_times:
                        ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"Error executing file {f.name}: {e}")
            del y
            gc.collect()

    if results_list:
        df = pd.DataFrame(results_list)
        st.dataframe(df, use_container_width=True)
