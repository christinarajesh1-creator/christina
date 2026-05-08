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
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True, duration=20.0)
    duration = len(y) / sr
    
    hop_length = 512
    y_hp = librosa.effects.preemphasis(y)
    
    # Breath-focused RMS (200-1000Hz)
    fft = np.fft.rfft(y_hp)
    freqs = np.fft.rfftfreq(len(y_hp), 1/sr)
    breath_mask = (freqs > 200) & (freqs < 1000)
    fft_breath = fft.copy()
    fft_breath[~breath_mask] = 0
    y_breath = np.fft.irfft(fft_breath, n=len(y_hp))
    rms_breath = librosa.feature.rms(y=y_breath, frame_length=1024, hop_length=hop_length)[0]
    times = librosa.frames_to_time(rms_breath, sr=sr, hop_length=hop_length)
    
    smooth_rms = np.convolve(rms_breath, np.ones(25)/25, mode='same')
    local_min = np.zeros_like(rms_breath, dtype=bool)
    
    for i in range(25, len(rms_breath)-25):
        local_min[i] = (rms_breath[i] < 0.72 * smooth_rms[i] and 
                       rms_breath[i] < np.percentile(rms_breath[i-40:i+40], 22))
    
    peaks, _ = find_peaks(local_min.astype(float), height=0.35, distance=20, prominence=0.18)
    breath_times = times[peaks]
    breath_times = breath_times[(breath_times > 0.4) & (breath_times < duration - 0.8)]
    
    # DETAILED 6 PARAMETER ANALYSIS
    breath_count = len(breath_times)
    breaths_per_10s = breath_count * 10 / duration
    
    if breath_count < 2:
        return {
            "Filename": filename, "Duration": f"{duration:.1f}s",
            "AI_Score": 0.82, "Status": "🤖 AI",
            "P1_Count": breath_count, "P1_Norm": 0,
            "P2_IBI": 0, "P2_Norm": 0.9,
            "P3_IBI_CV": 0, "P3_Norm": 0.9,
            "P4_Dur": 0, "P4_Norm": 0.85,
            "P5_Dur_CV": 0, "P5_Norm": 0.8,
            "P6_Entropy": 0, "P6_Norm": 0.75,
            "y": y, "rms": rms_breath, "times": times, "ev": []
        }
    
    # P1: Breath rate (Human: 12-24 breaths/10min)
    p1_norm = max(0, min(1, abs(breaths_per_10s - 18) / 12))  # Peak at 18
    p1_score = 0.85 * p1_norm
    
    # P2: Inter-breath interval (Human: 2.5-5s)
    ibis = np.diff(breath_times)
    p2_ibi = np.mean(ibis)
    p2_norm = max(0, min(1, abs(p2_ibi - 3.75) / 1.25))  # Peak at 3.75s
    p2_score = 0.75 * p2_norm
    
    # P3: IBI variability (Human CV: 0.2-0.5)
    p3_cv = np.std(ibis) / p2_ibi
    p3_norm = max(0, min(1, abs(p3_cv - 0.35) / 0.15))  # Peak at 0.35
    p3_score = 0.65 * p3_norm
    
    # P4: Breath duration (Human: 0.5-1.5s)
    breath_durs = []
    for t in breath_times:
        t_frame = np.argmin(np.abs(times - t))
        start = max(0, t_frame - 10)
        end = min(len(rms_breath), t_frame + 20)
        seg = rms_breath[start:end]
        local_mean = np.mean(seg)
        dur_frames = np.sum(seg < 0.45 * local_mean)
        dur_sec = dur_frames * hop_length / sr
        if 0.3 < dur_sec < 2.5:
            breath_durs.append(dur_sec)
    
    p4_dur = np.mean(breath_durs) if breath_durs else 0
    p4_norm = max(0, min(1, abs(p4_dur - 1.0) / 0.5))  # Peak at 1.0s
    p4_score = 0.7 * p4_norm
    
    # P5: Duration variability (Human CV: 0.15-0.6)
    p5_cv = np.std(breath_durs) / p4_dur if breath_durs and p4_dur > 0 else 0
    p5_norm = max(0, min(1, abs(p5_cv - 0.375) / 0.225))  # Peak at 0.375
    p5_score = 0.6 * p5_norm
    
    # P6: Rhythm entropy (Human: irregular)
    ibis_hist, _ = np.histogram(ibis, bins=10, range=(1,6), density=True)
    p6_entropy = entropy(ibis_hist + 1e-8)
    p6_norm = max(0, 1 - p6_entropy / 1.4)  # Higher entropy = more human
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
        "y": y, "rms": rms_breath, "times": times, "ev": breath_times.tolist()
    }

st.title("🔬 Forensic Breath Analysis - 6 Parameters")

progress_bar = st.progress(0)

uploaded = st.file_uploader("Upload audio files", type=['wav','mp3','m4a','flac'], accept_multiple_files=True)

if uploaded:
    results = []
    for i, file in enumerate(uploaded):
        progress_bar.progress((i+1) / len(uploaded))
        file.seek(0)
        result = analyze_breath_audio(file.read(), file.name)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # MAIN RESULTS TABLE
    st.subheader("📊 Analysis Results")
    st.dataframe(df[["Filename", "Duration", "Status", "AI_Score", 
                    "P1_Count", "P2_IBI", "P3_IBI_CV", "P4_Dur", "P5_Dur_CV", "P6_Entropy"]], 
                use_container_width=True, hide_index=True)
    
    # PARAMETER DETAILS
    st.subheader("📈 Parameter Breakdown")
    param_df = df[["Filename", "Status", "AI_Score",
                  "P1_Norm", "P2_Norm", "P3_Norm", "P4_Norm", "P5_Norm", "P6_Norm"]].copy()
    param_df.columns = ["File", "Status", "AI Score", 
                       "P1 Rate", "P2 IBI", "P3 Var", "P4 Dur", "P5 DurVar", "P6 Entropy"]
    st.dataframe(param_df, use_container_width=True, hide_index=True)
    
    # GRAPHS
    st.subheader("🎵 Breath Waveforms & Energy")
    cols = st.columns(3)
    
    for i, res in enumerate(results[:9]):
        if not res['ev']:
            continue
            
        with cols[i%3]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), facecolor='black')
            
            # Waveform
            color = 'lime' if 'HUMAN' in res['Status'] else 'red'
            ax1.plot(res['y'][:40000], color='gray', linewidth=0.5, alpha=0.5)  # First 2.5s
            for t in res['ev'][:3]:  # First 3 breaths
                if t*16000 < 40000:
                    ax1.axvline(t*16000, color=color, linewidth=3, alpha=0.9)
            ax1.set_facecolor('#0a0a0a')
            ax1.set_title(f"{res['Filename'][:20]}...\n{res['Status']} | Score: {res['AI_Score']}", 
                         color='white', fontsize=11, pad=10)
            ax1.set_ylabel("Amplitude", color='white')
            
            # Energy envelope
            ax2.plot(res['times'][:200], res['rms'][:200], color='cyan', linewidth=1.2)
            ax2.fill_between(res['times'][:200], res['rms'][:200], alpha=0.3, color='cyan')
            for t in res['ev'][:3]:
                if t < res['times'][199]:
                    ax2.axvline(t, color=color, linewidth=2.5, alpha=1.0)
            ax2.set_facecolor('#0a0a0a')
            ax2.set_title("Breath Energy Envelope", color='white', fontsize=10)
            ax2.set_xlabel("Time (s)", color='white')
            ax2.set_ylabel("RMS Energy", color='white')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    progress_bar.progress(1.0)
    st.success(f"✅ Analyzed {len(results)} files")
    
    # PARAMETER LEGEND
    with st.expander("📋 Parameter Reference"):
        st.markdown("""
        **P1 Rate**: Breaths/10s (Human peak: 18, range 6-30)  
        **P2 IBI**: Mean inter-breath interval (Human peak: 3.75s, range 2-6s)  
        **P3 Var**: IBI coefficient of variation (Human peak: 0.35, range 0.15-0.7)  
        **P4 Dur**: Mean breath duration (Human peak: 1.0s, range 0.4-1.8s)  
        **P5 DurVar**: Breath duration CV (Human peak: 0.375, range 0.1-0.9)  
        **P6 Entropy**: Rhythm complexity (Human: high irregularity)
        
        **Higher normalized scores = more AI-like deviation from human norms**
        """)
