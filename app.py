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

def precise_breath_analysis(file_bytes, filename):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        duration = len(y) / sr
        
        if duration < 2:
            return None
        
        # Breath bandpass 50-1500Hz
        sos = butter(4, [50, 1500], btype='band', fs=sr, output='sos')
        y_breath = filtfilt(sos, 1, y)
        
        hop_length = 256
        rms_breath = librosa.feature.rms(y=y_breath, frame_length=512, hop_length=hop_length)[0]
        times = librosa.frames_to_time(rms_breath, sr=sr, hop_length=hop_length)
        
        # Adaptive smoothing
        smooth_rms = np.zeros_like(rms_breath)
        for i in range(30, len(rms_breath)-30):
            window = rms_breath[max(0,i-30):min(len(rms_breath),i+31)]
            smooth_rms[i] = np.mean(window)
        
        # Precise breath detection
        breath_events = []
        i = 40
        while i < len(rms_breath) - 40:
            local_min = rms_breath[i] < np.percentile(rms_breath[i-40:i+41], 28)
            if local_min:
                before = np.mean(rms_breath[max(0,i-25):i])
                after = np.mean(rms_breath[i+1:min(len(rms_breath),i+26)])
                if before > rms_breath[i]*1.4 and after > rms_breath[i]*1.4:
                    start = i
                    while start > 5 and rms_breath[start] > rms_breath[i]*1.3:
                        start -= 1
                    end = i
                    while end < len(rms_breath)-5 and rms_breath[end] > rms_breath[i]*1.3:
                        end += 1
                    
                    breath_events.append({
                        'center_time': times[i],
                        'start_time': times[start],
                        'end_time': times[end],
                        'duration': (end - start) * hop_length / sr
                    })
                    i = end + 10
                else:
                    i += 1
            else:
                i += 1
        
        breaths = len(breath_events)
        
        if breaths == 0:
            return {
                'filename': filename, 'breaths': 0, 'ai_score': 0.90,
                'status': '🤖 AI', 'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0, 'p5': 0, 'p6': 0
            }
        
        # 6 MEDICAL PARAMETERS
        p1_rate = breaths * 60 / duration  # BPM
        
        ibis = np.diff([e['center_time'] for e in breath_events])
        p2_ibi = np.mean(ibis)
        p3_cv = np.std(ibis) / p2_ibi if len(ibis) > 0 else 0
        
        durations = [e['duration'] for e in breath_events]
        p4_dur = np.mean(durations)
        p5_dur_cv = np.std(durations) / p4_dur if p4_dur > 0 else 0
        
        ibis_hist, _ = np.histogram(ibis, bins=10, range=(1,8))
        p6_entropy = entropy(ibis_hist + 1e-8)
        
        # AI SCORING
        deviations = 0
        if not (12 <= p1_rate <= 25): deviations += 1
        if not (0.8 <= p2_ibi <= 5.0): deviations += 1
        if not (0.15 <= p3_cv <= 0.6): deviations += 1
        if not (0.5 <= p4_dur <= 2.5): deviations += 1
        if not (0.1 <= p5_dur_cv <= 0.8): deviations += 1
        if p6_entropy < 0.5: deviations += 1
        
        ai_score = min(1.0, deviations * 0.167)
        status = "🤖 AI" if ai_score > 0.67 else "👤 HUMAN"
        
        return {
            'filename': filename,
            'breaths': breaths,
            'p1_rate': round(p1_rate, 1),
            'p2_ibi': round(p2_ibi, 2),
            'p3_cv': round(p3_cv, 3),
            'p4_dur': round(p4_dur, 2),
            'p5_dur_cv': round(p5_dur_cv, 3),
            'p6_entropy': round(p6_entropy, 3),
            'ai_score': round(ai_score, 3),
            'status': status,
            'times': times[:400],
            'rms': rms_breath[:400],
            'breath_times': [e['center_time'] for e in breath_events]
        }
    except:
        return None

st.title("🏥 Precise Breath Analyzer")
st.markdown("**6 Medically-Accurate Parameters**")

uploaded_files = st.file_uploader(
    "Upload files", type=['wav','mp3','m4a','flac'], 
    accept_multiple_files=True
)

if uploaded_files:
    progress = st.progress(0)
    results = []
    
    for i, file in enumerate(uploaded_files):
        progress.progress((i+1)/len(uploaded_files))
        file.seek(0)
        result = precise_breath_analysis(file.read(), file.name)
        if result:
            results.append(result)
    
    df = pd.DataFrame(results)
    
    # RESULTS
    st.subheader("📊 Analysis")
    st.dataframe(df[['filename', 'status', 'ai_score', 'breaths', 
                    'p1_rate', 'p2_ibi', 'p3_cv']], use_container_width=True)
    
    st.subheader("🔬 Parameters")
    st.dataframe(df[['filename', 'p4_dur', 'p5_dur_cv', 'p6_entropy']], 
                use_container_width=True)
    
    # METRICS
    col1, col2 = st.columns(2)
    col1.metric("🤖 AI", len(df[df['status'] == '🤖 AI']))
    col2.metric("👤 Human", len(df[df['status'] == '👤 HUMAN']))
    
    # GRAPHS
    st.subheader("📈 Visualization")
    cols = st.columns(3)
    for i, result in enumerate(results[:9]):
        if result['breaths'] == 0: continue
            
        with cols[i%3]:
            fig, ax = plt.subplots(figsize=(12, 6))
            color = 'lime' if result['status'] == '👤 HUMAN' else 'red'
            
            ax.plot(result['times'], result['rms'], 'cyan', lw=1.5)
            for t in result['breath_times'][:12]:
                ax.axvline(t, color=color, lw=3, alpha=0.9)
            ax.set_title(f"{result['filename'][:50]}\n"
                        f"Score: {result['ai_score']} | {result['breaths']} breaths")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Breath Energy")
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with st.expander("Reference Ranges"):
        st.write("""
        P1 Rate: 12-25 BPM
        P2 IBI: 0.8-5.0s
        P3 CV: 0.15-0.6
        P4 Duration: 0.5-2.5s
        P5 Dur CV: 0.1-0.8
        P6 Entropy: >0.5 (complex)
        """)

st.markdown("---")
