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

def analyze_breath_audio(file_bytes, filename):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True, duration=20.0)
        duration = len(y) / sr
        
        if duration < 2:
            return {
                "Filename": filename, "Duration": f"{duration:.1f}s",
                "AI_Score": 0.8, "Status": "🤖 AI",
                "P1_Count": 0, "P1_Norm": 1.0,
                "P2_IBI": 0, "P2_Norm": 1.0,
                "P3_IBI_CV": 0, "P3_Norm": 1.0,
                "P4_Dur": 0, "P4_Norm": 1.0,
                "P5_Dur_CV": 0, "P5_Norm": 1.0,
                "P6_Entropy": 0, "P6_Norm": 1.0,
                "y": y[:40000], "rms": np.zeros(100), "times": np.linspace(0,2,100), "ev": []
            }
        
        hop_length = 512
        y_hp = librosa.effects.preemphasis(y)
        
        # Safe FFT breath filtering
        fft = np.fft.rfft(y_hp)
        freqs = np.fft.rfftfreq(len(y_hp), 1/sr)
        breath_mask = (freqs > 200) & (freqs < 1000)
        fft_breath = fft.copy()
        fft_breath[~breath_mask] = 0
        y_breath = np.fft.irfft(fft_breath, n=len(y_hp))
        rms_breath = librosa.feature.rms(y=y_breath, frame_length=1024, hop_length=hop_length)[0]
        times = librosa.frames_to_time(rms_breath, sr=sr, hop_length=hop_length)
        
        # Safe smoothing
        smooth_rms = np.convolve(rms_breath, np.ones(25)/25, mode='same')
        
        # FIXED: Safe local minima detection
        local_min = np.zeros_like(rms_breath, dtype=bool)
        for i in range(30, len(rms_breath)-30):
            window = rms_breath[max(0, i-30):min(len(rms_breath), i+31)]
            if len(window) > 10:  # Ensure valid window
                local_min[i] = (rms_breath[i] < 0.72 * smooth_rms[i] and 
                               rms_breath[i] < np.percentile(window, 22))
        
        peaks, _ = find_peaks(local_min.astype(float), height=0.35, distance=20, prominence=0.18)
        breath_times = times[peaks]
        breath_times = breath_times[(breath_times > 0.4) & (breath_times < duration - 0.8)]
        
        breath_count = len(breath_times)
        breaths_per_10s = breath_count * 10 / duration
        
        # 6 PARAMETER ANALYSIS WITH PEAK HUMAN VALUES
        if breath_count < 2:
            return {
                "Filename": filename, "Duration": f"{duration:.1f}s",
                "AI_Score": 0.82, "Status": "🤖 AI",
                "P1_Count": breath_count, "P1_Norm": 1.0,
                "P2_IBI": 0, "P2_Norm": 0.9,
                "P3_IBI_CV": 0, "P3_Norm": 0.9,
                "P4_Dur": 0, "P4_Norm": 0.85,
                "P5_Dur_CV": 0, "P5_Norm": 0.8,
                "P6_Entropy": 0, "P6_Norm": 0.75,
                "y": y[:40000], "rms": rms_breath[:200], "times": times[:200], "ev": []
            }
        
        # P1: Breath rate deviation from 18/10min
        p1_norm = max(0, min(1, abs(breaths_per_10s - 18) / 12))
        p1_score = 0.85 * p1_norm
        
        # P2: IBI deviation from 3.75s
        ibis = np.diff(breath_times)
        p2_ibi = np.mean(ibis)
        p2_norm = max(0, min(1, abs(p2_ibi - 3.75) / 1.25))
        p2_score = 0.75 * p2_norm
        
        # P3: IBI CV deviation from 0.35
        p3_cv = np.std(ibis) / p2_ibi
        p3_norm = max(0, min(1, abs(p3_cv - 0.35) / 0.15))
        p3_score = 0.65 * p3_norm
        
        # P4: Breath duration deviation from 1.0s
        breath_durs = []
        for t in breath_times[:8]:
            t_frame = np.argmin(np.abs(times - t))
            start = max(0, t_frame - 12)
            end = min(len(rms_breath), t_frame + 25)
            seg = rms_breath[start:end]
            if len(seg) > 5:
                local_mean = np.mean(seg)
                dur_frames = np.sum(seg < 0.45 * local_mean)
                dur_sec = dur_frames * hop_length / sr
                if 0.3 < dur_sec < 2.5:
                    breath_durs.append(dur_sec)
        
        p4_dur = np.mean(breath_durs) if breath_durs else 0
        p4_norm = max(0, min(1, abs(p4_dur - 1.0) / 0.5))
        p4_score = 0.7 * p4_norm
        
        p5_cv = np.std(breath_durs) / p4_dur if breath_durs and p4_dur > 0 else 0
        p5_norm = max(0, min(1, abs(p5_cv - 0.375) / 0.225))
        p5_score = 0.6 * p5_norm
        
        # P6: Entropy (low entropy = regular/AI)
        ibis_hist, _ = np.histogram(ibis, bins=8, range=(1,6), density=True)
        p6_entropy = entropy(ibis_hist + 1e-8)
        p6_norm = max(0, 1 - p6_entropy / 1.4)
        p6_score = 0.55 * p6_norm
        
        ai_score = (p1_score + p2_score + p3_score + p4_score + p5_score + p6_score) / 6
        status = "🤖 AI" if ai_score > 0.5 else "👤 HUMAN"
        
        return {
            "Filename": filename, "Duration": f"{duration:.1f}s",
            "AI_Score": round(ai_score, 3), "Status": status,
            "P1_Count": breath_count, "P1_Norm": round(p1_norm, 3),
            "P2_IBI": round(p2_ibi, 2), "P2_Norm": round(p2_norm, 3),
            "P3_IBI_CV": round(p3_cv, 3), "P3_Norm": round(p3_norm, 3),
            "P4_Dur": round(p4_dur, 3), "P4_Norm": round(p4_norm, 3),
            "P5_Dur_CV": round(p5_cv, 3), "P5_Norm": round(p5_norm, 3),
            "P6_Entropy": round(p6_entropy, 3), "P6_Norm": round(p6_norm, 3),
            "y": y[:40000], "rms": rms_breath[:200], "times": times[:200], "ev": breath_times.tolist()
        }
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return None

st.title("🔬 Forensic Breath Analysis")

progress_bar = st.progress(0)

uploaded = st.file_uploader("Upload audio files", type=['wav','mp3','m4a','flac'], accept_multiple_files=True)

if uploaded:
    results = []
    for i, file in enumerate(uploaded):
        progress_bar.progress((i+1) / len(uploaded))
        file.seek(0)
        result = analyze_breath_audio(file.read(), file.name)
        if result:
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Raw Parameters")
            raw_cols = ["Filename", "Duration", "Status", "AI_Score", 
                       "P1_Count", "P2_IBI", "P3_IBI_CV", "P4_Dur", "P5_Dur_CV", "P6_Entropy"]
            st.dataframe(df[raw_cols], use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📈 AI Deviation Scores")
            norm_cols = ["Filename", "Status", "AI_Score",
                        "P1_Norm", "P2_Norm", "P3_Norm", "P4_Norm", "P5_Norm", "P6_Norm"]
            norm_df = df[norm_cols].copy()
            norm_df.columns = ["File", "Status", "AI Score", "Rate", "IBI", "IBI-CV", "Dur", "Dur-CV", "Entropy"]
            st.dataframe(norm_df, use_container_width=True, hide_index=True)
        
        st.subheader("🎵 Audio Analysis")
        cols = st.columns(3)
        for i, res in enumerate(results[:9]):
            if len(res['ev']) == 0 or len(res['rms']) < 50:
                continue
                
            with cols[i%3]:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), facecolor='black')
                
                color = 'lime' if 'HUMAN' in res['Status'] else 'red'
                
                ax1.plot(res['y'], color='gray', linewidth=0.5, alpha=0.5)
                for t in res['ev']:
                    if t * 16000 < len(res['y']):
                        ax1.axvline(t*16000, color=color, linewidth=3, alpha=0.9)
                ax1.set_facecolor('#0a0a0a')
                ax1.set_title(f"{res['Filename'][:25]}", color='white', fontsize=11)
                ax1.set_ylabel("Amplitude", color='white')
                
                ax2.plot(res['times'], res['rms'], color='cyan', linewidth=1.2)
                ax2.fill_between(res['times'], res['rms'], alpha=0.3, color='cyan')
                for t in res['ev']:
                    if t < res['times'][-1]:
                        ax2.axvline(t, color=color, linewidth=2.5, alpha=1.0)
                ax2.set_facecolor('#0a0a0a')
                ax2.set_xlabel("Time (s)", color='white')
                ax2.set_ylabel("Breath Energy", color='white')
                ax2.set_title(f"{res['Status']} | Score: {res['AI_Score']}", color='white', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        progress_bar.progress(1.0)
        st.success(f"✅ Analyzed {len(results)} files")
        
        with st.expander("📋 Parameters Explained"):
            st.markdown("""
            
