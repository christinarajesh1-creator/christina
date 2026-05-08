import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

def forensic_breath_detect(file):
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        duration = len(y) / sr
        
        if duration < 3:
            return {"Filename": file.name, "AI_Score": 0.95, "Status": "🤖 AI", 
                   "P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 0, "P6": 0, "y": y, "ev": []}
        
        # 1. Advanced signal cleaning
        y_filt = librosa.effects.preemphasis(y)
        S = np.abs(librosa.stft(y_filt))
        S_denoise = librosa.decompose.nn_filter(S, aggregate=np.median)
        y_denoise = librosa.istft(S_denoise)
        
        # 2. Multi-band RMS envelope (focus on breath frequencies 100-800Hz)
        hop_length = 512
        fmin, fmax = 100, 800
        rms_bands = []
        for band in [(fmin, 200), (200, 400), (400, fmax)]:
            y_band = librosa.effects.harmonic(y_denoise)
            y_band = butter_bandpass_filter(y_band, band[0], band[1], sr, order=4)
            rms = librosa.feature.rms(y=y_band, frame_length=2048, hop_length=hop_length)[0]
            rms_bands.append(rms)
        rms_combined = np.mean(rms_bands, axis=0)
        times = librosa.frames_to_time(rms_combined, sr=sr, hop_length=hop_length)
        
        # 3. Adaptive breath detection
        smooth_rms = pd.Series(rms_combined).rolling(40, center=True, min_periods=10).mean()
        smooth_rms = smooth_rms.fillna(method='bfill').fillna(method='ffill').values
        
        # Local minima detection with physiological constraints
        breath_candidates = []
        for i in range(10, len(rms_combined)-10):
            if (rms_combined[i] < smooth_rms[i] * 0.6 and 
                rms_combined[i] < np.percentile(rms_combined[max(0,i-50):min(len(rms_combined),i+50)], 15)):
                breath_candidates.append(i)
        
        # Refine with peak detection on inverted signal
        if breath_candidates:
            candidate_times = times[breath_candidates]
            inv_rms = 1.0 / (rms_combined[breath_candidates] + 1e-8)
            peaks, _ = find_peaks(inv_rms, height=np.percentile(inv_rms, 70), 
                                distance=20, prominence=np.std(inv_rms)*0.3)
            breath_frames = np.array(breath_candidates)[peaks]
            breath_times = times[breath_frames]
            
            # Filter by duration and position
            breath_times = breath_times[(breath_times > 0.3) & (breath_times < duration - 0.5)]
        else:
            breath_times = np.array([])
        
        breath_count = len(breath_times)
        
        if breath_count < 2:
            return {"Filename": file.name, "AI_Score": 0.92, "Status": "🤖 AI", 
                   "P1": breath_count, "P2": 0, "P3": 0, "P4": 0, "P5": 0, "P6": 0, "y": y, "ev": []}
        
        # 6 Research Parameters (optimized thresholds)
        
        # P1: Breath count per 10s (Human: 12-25, AI: <10 or >30)
        breaths_per_10s = breath_count * 10 / duration
        p1_score = 0.9 if breaths_per_10s < 12 or breaths_per_10s > 25 else 0.1
        
        # P2: IBI Mean (Human: 2.8-4.5s)
        ibis = np.diff(breath_times)
        p2_mean = np.mean(ibis)
        p2_score = 0.85 if not (2.8 <= p2_mean <= 4.5) else 0.05
        
        # P3: IBI CV (Human: 0.20-0.45)
        p3_cv = np.std(ibis) / p2_mean if p2_mean > 0 else 0
        p3_score = 0.75 if not (0.20 <= p3_cv <= 0.45) else 0.1
        
        # P4: Breath duration (Human: 0.4-1.0s)
        breath_durations = []
        for t in breath_times:
            start = max(0, int((t-0.4)*sr))
            end = min(len(y_denoise), int((t+1.0)*sr))
            if end > start:
                breath_seg = y_denoise[start:end]
                breath_rms = librosa.feature.rms(y=breath_seg, hop_length=hop_length)[0]
                dur = np.sum(breath_rms > np.mean(breath_rms)*0.4) * hop_length / sr
                if 0.2 < dur < 1.5:
                    breath_durations.append(dur)
        
        p4_mean = np.mean(breath_durations) if breath_durations else 0
        p4_score = 0.8 if not (0.4 <= p4_mean <= 1.0) else 0.05
        
        # P5: Breath duration CV (Human: 0.15-0.50)
        p5_cv = np.std(breath_durations) / p4_mean if breath_durations and p4_mean > 0 else 0
        p5_score = 0.7 if not (0.15 <= p5_cv <= 0.50) else 0.1
        
        # P6: Rhythm entropy (Human: 0.6-1.1)
        if len(ibis) > 1:
            ibis_norm = (ibis - np.min(ibis)) / (np.max(ibis) - np.min(ibis) + 1e-8)
            p6_entropy = entropy(ibis_norm + 1e-8)
        else:
            p6_entropy = 0
        p6_score = 0.65 if p6_entropy < 0.6 or p6_entropy > 1.1 else 0.15
        
        # Weighted AI score (higher = more AI-like)
        ai_score = (p1_score * 0.25 + p2_score * 0.20 + p3_score * 0.20 + 
                   p4_score * 0.15 + p5_score * 0.10 + p6_score * 0.10)
        
        status = "🤖 AI" if ai_score > 0.55 else "👤 HUMAN"
        
        return {
            "Filename": file.name,
            "AI_Score": round(ai_score, 3),
            "Status": status,
            "P1": int(breaths_per_10s),
            "P2": round(p2_mean, 2),
            "P3": round(p3_cv, 3),
            "P4": round(p4_mean, 3),
            "P5": round(p5_cv, 3),
            "P6": round(p6_entropy, 3),
            "y": y, "ev": breath_times.tolist()
        }
    except:
        return None

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

st.title("Forensic Breath Analyzer v2.0")

uploaded = st.file_uploader("Upload Audio", type=['wav','mp3','m4a','flac'], accept_multiple_files=True)

if uploaded:
    results = [forensic_breath_detect(f) for f in uploaded]
    df = pd.DataFrame([r for r in results if r])
    
    if not df.empty:
        st.dataframe(df[["Filename", "Status", "AI_Score", "P1", "P2", "P3", "P4", "P5", "P6"]], 
                    use_container_width=True, hide_index=True)
        
        cols = st.columns(3)
        for i, res in enumerate(results[:6]):
            if res is None or not res.get('ev'):
                continue
                
            with cols[i%3]:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='black')
                
                ax1.plot(res['y'], color='gray', linewidth=0.5, alpha=0.7)
                for t in res['ev']:
                    ax1.axvline(t*16000, color='lime', alpha=0.9, linewidth=2)
                ax1.set_facecolor('black')
                ax1.set_title(f"{res['Filename']} | {res['Status']} | Score: {res['AI_Score']}", 
                             color='white', fontsize=10)
                ax1.margins(x=0)
                
                hop = 512
                rms = librosa.feature.rms(y=res['y'], frame_length=2048, hop_length=hop)[0]
                times = np.arange(len(rms)) * hop / 16000
                ax2.plot(times, rms, color='cyan', linewidth=1, alpha=0.8)
                for t in res['ev']:
                    ax2.axvline(t, color='red', linewidth=2, alpha=0.9)
                ax2.fill_between(times, rms, alpha=0.2, color='cyan')
                ax2.set_facecolor('black')
                ax2.set_title("Breath Energy", color='white', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
