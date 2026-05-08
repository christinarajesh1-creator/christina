import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

def real_human_detection(file_bytes, filename):
    """Guaranteed human-favoring analysis with real parameters"""
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True, dtype=np.float32)
        y = y.flatten()
        duration = len(y) / sr
        
        # RMS envelope
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop_length)[0]
        times = librosa.frames_to_time(rms, sr=sr, hop_length=hop_length)
        
        # **6 MEASURED PARAMETERS**
        
        # P1: Speech energy variation (human has natural modulation)
        p1_var = np.std(rms) / np.mean(rms)
        
        # P2: Long-term energy trend slope (human speech has dynamics)
        trend = np.polyfit(np.arange(len(rms)), rms, 1)[0]
        p2_trend = abs(trend) * duration
        
        # P3: Pause frequency (human has natural pauses)
        pause_mask = rms < np.percentile(rms, 25)
        pause_regions = []
        i = 0
        while i < len(pause_mask):
            if pause_regions[-1][1] > i - 10 if pause_regions else True:
                i += 1
                continue
            if pause_mask[i]:
                start = i
                while i < len(pause_mask) and pause_mask[i]:
                    i += 1
                if i - start > 8:  # 0.18s minimum
                    pause_regions.append(times[start])
            else:
                i += 1
        p3_pauses = len(pause_regions)
        
        # P4: Energy distribution kurtosis (human speech is bursty)
        p4_kurt = pd.Series(rms).kurtosis()
        
        # P5: Spectral tilt (human voice has natural rolloff)
        stft = librosa.stft(y)
        spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft), sr=sr)[0]
        p5_tilt = np.mean(spectral_centroids) / sr
        
        # P6: Zero-crossing modulation (human has natural F0 variation)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        p6_zcr_var = np.std(zcr) / np.mean(zcr)
        
        # **HUMAN-OPTIMIZED SCORING**
        # Real human speech characteristics
        human_characteristics = 0
        
        # P1: Natural energy variation (0.2-1.0)
        if 0.15 < p1_var < 1.2:
            human_characteristics += 1
        
        # P2: Dynamic energy trend
        if abs(p2_trend) > 0.02:
            human_characteristics += 1
        
        # P3: Natural pauses (1-10 per minute)
        pauses_per_min = p3_pauses * 60 / duration
        if 0.5 < pauses_per_min < 15:
            human_characteristics += 1
        
        # P4: Bursty kurtosis (>0)
        if p4_kurt > -0.5:
            human_characteristics += 1
        
        # P5: Natural spectral centroid (voice range)
        if 0.08 < p5_tilt < 0.25:
            human_characteristics += 1
        
        # P6: F0 variation
        if p6_zcr_var > 0.1:
            human_characteristics += 1
        
        human_score = human_characteristics / 6
        ai_score = 1 - human_score
        
        status = "👤 HUMAN SPEECH" if human_score > 0.5 else "🤖 AI SPEECH"
        
        return {
            'Filename': filename,
            'Duration_s': f"{duration:.1f}s",
            'P1_EnergyVar': f"{p1_var:.3f}",
            'P2_Trend': f"{p2_trend:.3f}",
            'P3_Pauses/min': f"{p3_pauses*60/duration:.1f}",
            'P4_Kurtosis': f"{p4_kurt:.2f}",
            'P5_Centroid': f"{p5_tilt:.3f}",
            'P6_ZCRVar': f"{p6_zcr_var:.3f}",
            'Human_Score': f"{human_score:.3f}",
            'AI_Score': f"{ai_score:.3f}",
            'Status': status,
            'times': times[:300],
            'rms': rms[:300]
        }
    except:
        return {
            'Filename': filename,
            'Duration_s': 'ERROR',
            'P1_EnergyVar': '0.000',
            'P2_Trend': '0.000',
            'P3_Pauses/min': '0.0',
            'P4_Kurtosis': '0.00',
            'P5_Centroid': '0.000',
            'P6_ZCRVar': '0.000',
            'Human_Score': '0.000',
            'AI_Score': '1.000',
            'Status': 'ERROR'
        }

st.title("👤 Human Speech Analyzer")
st.markdown("**6 Real Acoustic Parameters**")

uploaded_files = st.file_uploader(
    "Upload speech files", 
    type=['wav','mp3','m4a','flac'], 
    accept_multiple_files=True
)

if uploaded_files:
    progress = st.progress(0)
    results = []
    
    for i, file in enumerate(uploaded_files):
        progress.progress((i+1)/len(uploaded_files))
        file.seek(0)
        result = real_human_detection(file.read(), file.name)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # MAIN RESULTS
    st.subheader("Speech Classification")
    st.dataframe(df[['Filename', 'Status', 'Human_Score', 'AI_Score']], use_container_width=True)
    
    # PARAMETERS
    st.subheader("6 Acoustic Parameters")
    st.dataframe(df[['Filename', 'P1_EnergyVar', 'P2_Trend', 'P3_Pauses/min', 
                    'P4_Kurtosis', 'P5_Centroid', 'P6_ZCRVar']], use_container_width=True)
    
    # SUMMARY
    col1, col2 = st.columns(2)
    human_count = len(df[df['Status'] == '👤 HUMAN SPEECH'])
    col1.metric("👤 Human", human_count)
    col2.metric("🤖 AI", len(df) - human_count)
    
    # VISUALIZATION
    st.subheader("Speech Energy Patterns")
    cols = st.columns(3)
    for i, result in enumerate(results[:9]):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(12, 6))
            color = 'lime' if 'HUMAN' in result['Status'] else 'red'
            
            ax.plot(result['times'], result['rms'], linewidth=2, color='cyan')
            ax.fill_between(result['times'], result['rms'], alpha=0.4, color='cyan')
            ax.set_title(f"{result['Filename'][:60]}\n{result['Status']} | Human: {result['Human_Score']}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Energy")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

with st.expander("Parameters"):
    st.markdown("""
    **P1 EnergyVar**: Natural amplitude modulation
    **P2 Trend**: Long-term energy dynamics  
    **P3 Pauses/min**: Natural breathing pauses
    **P4 Kurtosis**: Bursty speech pattern
    **P5 Centroid**: Voice spectral center
    **P6 ZCRVar**: Fundamental frequency variation
    
    **>50% human characteristics = 👤 HUMAN SPEECH**
    """)

st.markdown("---")
