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

def breath_analysis(file_bytes, filename):
    """Robust 6-parameter analysis with error handling"""
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        duration = len(y) / sr
        
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop_length)[0]
        times = librosa.frames_to_time(rms, sr=sr, hop_length=hop_length)
        
        # Simple smoothing
        smooth_rms = np.convolve(rms, np.ones(21)/21, mode='same')
        
        # Breath detection
        breath_mask = rms < smooth_rms * 0.68
        peaks, _ = find_peaks(breath_mask.astype(float), height=0.3, distance=20)
        breath_times = times[peaks]
        breath_times = breath_times[(breath_times > 0.3) & (breath_times < duration - 0.7)]
        breaths = len(breath_times)
        
        # Default values
        p1_rate = 0
        p2_ibi = 0
        p3_cv = 0
        p4_dur = 0
        p5_dur_cv = 0
        p6_entropy = 0
        
        if breaths > 1:
            ibis = np.diff(breath_times)
            p2_ibi = np.mean(ibis)
            p3_cv = np.std(ibis) / p2_ibi if p2_ibi > 0 else 0
            p1_rate = breaths * 60 / duration
            
            # Duration analysis
            durations = []
            for t in breath_times:
                t_idx = np.argmin(np.abs(times - t))
                start = max(0, t_idx-15)
                end = min(len(rms), t_idx+30)
                seg = rms[start:end]
                if len(seg) > 10:
                    thresh = np.percentile(seg, 35)
                    dur_frames = np.sum(seg < thresh)
                    dur = dur_frames * hop_length / sr
                    if 0.2 < dur < 3.0:
                        durations.append(dur)
            
            p4_dur = np.mean(durations) if durations else 0
            p5_dur_cv = np.std(durations) / p4_dur if durations and p4_dur > 0 else 0
            
            # Entropy
            if len(ibis) > 1:
                ibis_norm = np.clip((ibis - np.min(ibis)) / (np.max(ibis) - np.min(ibis) + 1e-8), 0, 1)
                p6_entropy = entropy(ibis_norm + 1e-8)
        
        # AI scoring
        ai_score = 0.9 if breaths < 2 else 0.3 + 0.1 * p3_cv + 0.1 * (1 - p6_entropy)
        status = "🤖 AI" if ai_score > 0.6 else "👤 HUMAN"
        
        return {
            'Filename': filename,
            'Duration_s': round(duration, 1),
            'Breaths': breaths,
            'P1_Rate': round(p1_rate, 1),
            'P2_IBI': round(p2_ibi, 2),
            'P3_CV': round(p3_cv, 3),
            'P4_Dur': round(p4_dur, 2),
            'P5_DurCV': round(p5_dur_cv, 3),
            'P6_Ent': round(p6_entropy, 3),
            'AI_Score': round(ai_score, 3),
            'Status': status,
            'times': times[:300],
            'rms': rms[:300],
            'breath_times': breath_times.tolist()
        }
    except:
        return {
            'Filename': filename,
            'Duration_s': 0,
            'Breaths': 0,
            'P1_Rate': 0,
            'P2_IBI': 0,
            'P3_CV': 0,
            'P4_Dur': 0,
            'P5_DurCV': 0,
            'P6_Ent': 0,
            'AI_Score': 1.0,
            'Status': 'ERROR',
            'times': np.array([]),
            'rms': np.array([]),
            'breath_times': []
        }

st.title("🔬 6-Parameter Breath Analyzer")

uploaded_files = st.file_uploader(
    "Upload audio files", 
    type=['wav','mp3','m4a','flac'], 
    accept_multiple_files=True
)

if uploaded_files:
    progress_bar = st.progress(0)
    results = []
    
    for i, file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))
        file.seek(0)
        result = breath_analysis(file.read(), file.name)
        results.append(result)
    
    # SAFE DATAFRAME
    df = pd.DataFrame(results)
    
    # MAIN RESULTS
    st.subheader("Results")
    cols1 = ['Filename', 'Status', 'AI_Score', 'Breaths', 'Duration_s']
    st.dataframe(df[cols1], use_container_width=True)
    
    # PARAMETERS
    st.subheader("6 Parameters")
    cols2 = ['Filename', 'P1_Rate', 'P2_IBI', 'P3_CV', 'P4_Dur', 'P5_DurCV', 'P6_Ent']
    st.dataframe(df[cols2], use_container_width=True)
    
    # SUMMARY
    col1, col2 = st.columns(2)
    ai_count = len(df[df['Status'] == '🤖 AI'])
    col1.metric("🤖 AI", ai_count)
    col2.metric("👤 Human", len(df) - ai_count)
    
    # GRAPHS
    st.subheader("Breath Detection")
    cols = st.columns(3)
    for i, result in enumerate(results[:9]):
        if len(result['breath_times']) == 0:
            continue
            
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(12, 6))
            color = 'green' if result['Status'] == '👤 HUMAN' else 'red'
            
            ax.plot(result['times'], result['rms'], 'cyan', linewidth=1.5)
            ax.fill_between(result['times'], result['rms'], alpha=0.3, color='cyan')
            for t in result['breath_times']:
                ax.axvline(t, color=color, linewidth=3, alpha=0.9)
            
            ax.set_title(f"{result['Filename'][:50]}\n"
                        f"{result['Status']} | Score: {result['AI_Score']}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("RMS Energy")
            plt.tight_layout()
            st.pyplot(fig)
    
    st.success(f"Analyzed {len(results)} files")

with st.expander("Parameters"):
    st.write("""
    **P1 Rate**: Breaths per minute
    **P2 IBI**: Inter-breath interval (s)
    **P3 CV**: Timing coefficient of variation
    **P4 Dur**: Mean breath duration (s)
    **P5 DurCV**: Duration coefficient of variation
    **P6 Ent**: Rhythm entropy (0-2)
    
    **Low breaths + low variation = AI**
    """)
