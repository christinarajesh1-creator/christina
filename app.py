import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import entropy

st.set_page_config(layout="wide")

def forensic_breath_detect(file):
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        duration = len(y) / sr
        
        # 1. CLEAN SIGNAL
        y_filt = librosa.effects.preemphasis(y)
        y_denoise = librosa.decompose.nn_filter(y_filt, aggregate=np.median)
        
        # 2. BREATH ENERGY ENVELOPE
        rms = librosa.feature.rms(y=y_denoise, frame_length=2048, hop_length=512)[0]
        times = librosa.frames_to_time(rms, sr=sr)
        
        # 3. BREATH DETECTION - Multi-scale energy drops
        smooth_rms = pd.Series(rms).rolling(25, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        breath_mask = (rms < smooth_rms * 0.65) & (rms < np.percentile(rms, 20))
        
        # Peak detection with human-like constraints
        peaks, props = find_peaks(breath_mask.astype(float), 
                                height=0.4, 
                                distance=sr*0.8,  # Min 0.8s between breaths
                                prominence=0.15,
                                width=3)
        
        breath_times = times[peaks]
        breath_times = breath_times[(breath_times > 0.5) & (breath_times < duration - 1.0)]
        
        # 4. 6 RESEARCH BREATH PARAMETERS
        breath_count = len(breath_times)
        
        if breath_count < 2:
            return {
                "Filename": file.name,
                "AI_Score": 0.9,
                "Status": "🤖 AI",
                "P1_Breaths": breath_count,
                "P2_IBI_Mean": 0, "P3_IBI_CV": 0,
                "P4_Breath_Dur_Mean": 0, "P5_Breath_Dur_CV": 0,
                "P6_Rhythm_Entropy": 0,
                "y": y, "ev": []
            }
        
        # P1: Total breaths (AI: 0-4, Human: 5+ in 10s+)
        p1_score = max(0, min(1, (8 - breath_count) / 8)) if breath_count < 8 else 0
        
        # P2: IBI Mean (s) - Human: 2.5-5s, AI: too regular/short
        ibis = np.diff(breath_times)
        p2_mean = np.mean(ibis)
        p2_score = 0.3 if 2.0 < p2_mean < 6.0 else 0.8
        
        # P3: IBI Coefficient of Variation - Human: 0.25-0.6
        p3_cv = np.std(ibis) / p2_mean
        p3_score = 0.1 if 0.25 < p3_cv < 0.6 else 0.7
        
        # P4: Breath Duration Mean (s) - Human: 0.3-1.2s
        breath_durations = []
        for i, t in enumerate(breath_times):
            start = max(0, int((t-0.3)*sr))
            end = min(len(y), int((t+0.8)*sr))
            if end > start:
                breath_rms = librosa.feature.rms(y=y_denoise[start:end])[0]
                dur = np.sum(breath_rms > np.mean(rms)*0.3) * 512 / sr
                breath_durations.append(dur)
        
        p4_mean = np.mean(breath_durations) if breath_durations else 0
        p4_score = 0.2 if 0.3 < p4_mean < 1.2 else 0.6
        
        # P5: Breath Duration CV - Human variation
        p5_cv = np.std(breath_durations) / p4_mean if breath_durations and p4_mean > 0 else 0
        p5_score = 0.15 if 0.2 < p5_cv < 0.8 else 0.5
        
        # P6: Rhythm Entropy - Human: higher irregularity
        ibis_norm = (ibis - np.min(ibis)) / (np.max(ibis) - np.min(ibis) + 1e-8)
        p6_entropy = entropy(ibis_norm + 1e-8)
        p6_score = 0.1 if p6_entropy > 0.8 else 0.6
        
        # FINAL AI SCORE (weighted)
        ai_score = (p1_score * 0.25 + p2_score * 0.15 + p3_score * 0.20 + 
                   p4_score * 0.15 + p5_score * 0.15 + p6_score * 0.10)
        
        return {
            "Filename": file.name,
            "AI_Score": round(ai_score, 3),
            "Status": "🤖 AI" if ai_score > 0.5 else "👤 HUMAN",
            "P1_Breaths": breath_count,
            "P2_IBI_Mean": round(p2_mean, 2),
            "P3_IBI_CV": round(p3_cv, 3),
            "P4_Breath_Dur_Mean": round(p4_mean, 3),
            "P5_Breath_Dur_CV": round(p5_cv, 3),
            "P6_Rhythm_Entropy": round(p6_entropy, 3),
            "y": y, "ev": breath_times.tolist()
        }
    except:
        return None

st.title("Forensic Breath Analyzer v10.0 - 6 Parameter Detection")

uploaded = st.file_uploader("Upload Audio Files", type=['wav','mp3','m4a','flac'], accept_multiple_files=True)

if uploaded:
    results = [forensic_breath_detect(f) for f in uploaded]
    df = pd.DataFrame([r for r in results if r])
    
    if not df.empty:
        # Results Table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visualizations
        st.subheader("Breath Event Detection")
        cols = st.columns(3)
        
        for i, res in enumerate(results[:6]):
            if res is None or not res['ev']:
                continue
                
            with cols[i%3]:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor='black')
                fig.patch.set_facecolor('black')
                
                # Audio waveform + breaths
                ax1.plot(res['y'], color='#1a1a1a', linewidth=0.3)
                for t in res['ev']:
                    ax1.axvline(t*16000, color='#00FF41', linewidth=2)
                ax1.set_facecolor('#0a0a0a')
                ax1.set_title(f"{res['Filename']}\n{res['Status']} (Score: {res['AI_Score']})", 
                             color='white', fontsize=10)
                
                # Breath energy envelope
                hop = 512
                rms = librosa.feature.rms(y=res['y'], frame_length=2048, hop_length=hop)[0]
                times = np.arange(len(rms)) * hop / 16000
                ax2.plot(times, rms, color='#666', linewidth=0.8)
                ax2.fill_between(times, rms, alpha=0.3, color='#00FF41')
                for t in res['ev']:
                    ax2.axvline(t, color='#FF0040', linewidth=1.5, alpha=0.8)
                ax2.set_facecolor('#0a0a0a')
                ax2.set_title("Breath Energy Envelope", color='white', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
