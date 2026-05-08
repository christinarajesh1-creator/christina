import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

@st.cache_data
def fast_breath_analyze(file_bytes, filename):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True, duration=15.0)
        duration = len(y) / sr
        
        if duration < 2:
            return None, {"Filename": filename, "AI_Score": 0.95, "Status": "🤖 AI", 
                         "P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 0, "P6": 0, "y": y, "ev": []}
        
        hop_length = 512
        frame_length = 1024
        
        # Fast RMS envelope
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(rms, sr=sr, hop_length=hop_length)
        
        # Fast smoothing + detection
        smooth_rms = np.convolve(rms, np.ones(20)/20, mode='same')
        breath_mask = (rms < smooth_rms * 0.68) & (rms < np.percentile(rms, 18))
        
        peaks, _ = find_peaks(breath_mask.astype(float), distance=15, height=0.25)
        breath_times = times[peaks]
        breath_times = breath_times[(breath_times > 0.2) & (breath_times < duration - 0.3)]
        
        breath_count = len(breath_times)
        breaths_per_10s = breath_count * 10 / duration
        
        if breath_count < 2:
            ai_score = 0.92
            status = "🤖 AI"
            params = {"P1": int(breaths_per_10s), "P2": 0, "P3": 0, "P4": 0, "P5": 0, "P6": 0}
        else:
            # Fast parameter calc
            ibis = np.diff(breath_times)
            p2_mean = np.mean(ibis)
            p3_cv = np.std(ibis) / p2_mean if p2_mean > 0 else 0
            
            breath_durs = []
            for t in breath_times[:6]:  # Max 6 breaths
                start = max(0, int((t-0.3)*sr))
                end = min(len(y), int((t+0.7)*sr))
                if end > start:
                    seg_rms = librosa.feature.rms(y=y[start:end], hop_length=hop_length)[0]
                    dur = np.sum(seg_rms > np.mean(seg_rms)*0.4) * hop_length / sr
                    if 0.2 < dur < 1.5:
                        breath_durs.append(dur)
            
            p4_mean = np.mean(breath_durs) if breath_durs else 0
            p5_cv = np.std(breath_durs) / p4_mean if breath_durs and p4_mean > 0 else 0
            
            ibis_norm = np.clip((ibis - np.min(ibis)) / (np.max(ibis) - np.min(ibis) + 1e-6), 0, 1)
            p6_entropy = entropy(ibis_norm + 1e-8)
            
            # Scores
            p1_score = 0.9 if breaths_per_10s < 12 or breaths_per_10s > 25 else 0.1
            p2_score = 0.85 if not (2.8 <= p2_mean <= 4.5) else 0.05
            p3_score = 0.75 if not (0.20 <= p3_cv <= 0.45) else 0.1
            p4_score = 0.8 if not (0.4 <= p4_mean <= 1.0) else 0.05
            p5_score = 0.7 if not (0.15 <= p5_cv <= 0.50) else 0.1
            p6_score = 0.65 if p6_entropy < 0.6 or p6_entropy > 1.1 else 0.15
            
            ai_score = (p1_score * 0.25 + p2_score * 0.20 + p3_score * 0.20 + 
                       p4_score * 0.15 + p5_score * 0.10 + p6_score * 0.10)
            status = "🤖 AI" if ai_score > 0.55 else "👤 HUMAN"
            
            params = {
                "P1": int(breaths_per_10s),
                "P2": round(p2_mean, 2),
                "P3": round(p3_cv, 3),
                "P4": round(p4_mean, 3),
                "P5": round(p5_cv, 3),
                "P6": round(p6_entropy, 3)
            }
        
        result = {
            "Filename": filename,
            "AI_Score": round(ai_score, 3),
            "Status": status,
            **params,
            "y": y,
            "ev": breath_times.tolist()
        }
        
        return rms, times, result
        
    except:
        return None, None, {"Filename": filename, "AI_Score": 1.0, "Status": "🤖 AI", 
                           "P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 0, "P6": 0, "y": [], "ev": []}

st.title("Fast Forensic Breath Analyzer v3.0")

progress_bar = st.progress(0)
status_text = st.empty()

uploaded = st.file_uploader("Upload Audio", type=['wav','mp3','m4a','flac'], accept_multiple_files=True)

if uploaded:
    results = []
    viz_data = []
    
    for i, file in enumerate(uploaded):
        progress_bar.progress((i+1) / len(uploaded))
        status_text.text(f'Analyzing {file.name}...')
        
        # Reset file pointer
        file.seek(0)
        rms, times, result = fast_breath_analyze(file.read(), file.name)
        
        results.append(result)
        if rms is not None:
            viz_data.append((rms, times, result))
    
    df = pd.DataFrame(results)
    st.subheader("Results")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.subheader("Breath Visualizations")
    cols = st.columns(3)
    
    for i, (rms, times, res) in enumerate(viz_data[:9]):
        with cols[i%3]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor='black')
            
            # Waveform + breaths
            ax1.plot(res['y'], color='gray', linewidth=0.4, alpha=0.6)
            for t in res['ev']:
                ax1.axvline(t*16000, color='lime', linewidth=2, alpha=0.9)
            ax1.set_facecolor('#0a0a0a')
            ax1.set_title(f"{res['Filename']}\n{res['Status']} (AI: {res['AI_Score']})", 
                         color='white', fontsize=11)
            
            # RMS envelope
            ax2.plot(times, rms, color='cyan', linewidth=1, alpha=0.8)
            ax2.fill_between(times, rms, alpha=0.3, color='cyan')
            for t in res['ev']:
                ax2.axvline(t, color='red', linewidth=2, alpha=0.9)
            ax2.set_facecolor('#0a0a0a')
            ax2.set_title("Breath Energy Envelope", color='white', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    progress_bar.progress(1.0)
    status_text.text(f'✅ Analysis complete! {len(results)} files processed')
