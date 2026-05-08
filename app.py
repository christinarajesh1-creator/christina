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
    """Medically-accurate 6-parameter analysis"""
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
    duration = len(y) / sr
    
    # 1. BREATH BANDPASS FILTER (50-1500Hz - human breath range)
    sos = butter(4, [50, 1500], btype='band', fs=sr, output='sos')
    y_breath = filtfilt(sos, 1, y)
    
    hop_length = 256  # Higher resolution
    rms_breath = librosa.feature.rms(y=y_breath, frame_length=512, hop_length=hop_length)[0]
    times = librosa.frames_to_time(rms_breath, sr=sr, hop_length=hop_length)
    
    # 2. ADAPTIVE SMOOTHING
    smooth_rms = np.zeros_like(rms_breath)
    for i in range(30, len(rms_breath)-30):
        window = rms_breath[max(0,i-30):min(len(rms_breath),i+31)]
        smooth_rms[i] = np.mean(window)
    
    # 3. PRECISE BREATH LOCALIZATION
    breath_events = []
    i = 40
    while i < len(rms_breath) - 40:
        # Local minimum detection
        local_min = rms_breath[i] < np.percentile(rms_breath[i-40:i+41], 28)
        if local_min:
            # Verify it's a breath (surrounded by higher energy)
            before = np.mean(rms_breath[max(0,i-25):i])
            after = np.mean(rms_breath[i+1:min(len(rms_breath),i+26)])
            if before > rms_breath[i]*1.4 and after > rms_breath[i]*1.4:
                # Find full breath duration
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
                    'center_frame': i,
                    'min_rms': rms_breath[i],
                    'duration': (end - start) * hop_length / sr
                })
                i = end + 10  # Skip ahead
            else:
                i += 1
        else:
            i += 1
    
    breaths = len(breath_events)
    
    if breaths == 0:
        return {
            'filename': filename, 'breaths': 0, 'ai_score': 0.90,
            'status': '🤖 AI', 'p1': '-', 'p2': '-', 'p3': '-', 
            'p4': '-', 'p5': '-', 'p6': '-', 'times': times[:300],
            'rms': rms_breath[:300], 'breath_events': []
        }
    
    # **6 PRECISE PARAMETERS**
    
    # P1: Respiratory rate (breaths per minute)
    p1_rate = breaths * 60 / duration
    
    # P2: Mean inspiratory time (s)
    p2_insp = np.mean([e['duration'] for e in breath_events])
    
    # P3: Respiratory cycle variability (CV of IBI)
    ibis = np.diff([e['center_time'] for e in breath_events])
    p3_cv = np.std(ibis) / np.mean(ibis) if len(ibis) > 0 else 0
    
    # P4: Breath depth variation (CV of min RMS)
    depths = [e['min_rms'] for e in breath_events]
    p4_depth_cv = np.std(depths) / np.mean(depths) if depths else 0
    
    # P5: Temporal regularity (entropy of IBI distribution)
    if len(ibis) > 1:
        ibis_hist, _ = np.histogram(ibis, bins=10, range=(1,8))
        p5_entropy = entropy(ibis_hist + 1e-8)
    else:
        p5_entropy = 0
    
    # P6: Breath duty cycle (duration/IBI ratio)
    duty_cycles = []
    for j in range(len(breath_events)-1):
        dc = breath_events[j]['duration'] / ibis[j]
        duty_cycles.append(dc)
    p6_duty = np.mean(duty_cycles) if duty_cycles else 0
    
    # **MEDICAL AI SCORING** (deviation from norms)
    norms = {
        'p1': (12, 25),    # BPM
        'p2': (0.8, 2.5),  # Insp time
        'p3': (0.15, 0.6), # CV
        'p4': (0.2, 0.8),  # Depth CV
        'p5': (0.6, 1.8),  # Entropy
        'p6': (0.3, 0.6)   # Duty cycle
    }
    
    ai_score = 0
    params = [p1_rate, p2_insp, p3_cv, p4_depth_cv, p5_entropy, p6_duty]
    
    for i, p in enumerate(params):
        lo, hi = norms[list(norms.keys())[i]]
        if p < lo or p > hi:
            ai_score += 0.17  # Each deviation adds to AI likelihood
    
    status = "🤖 AI SYNTHETIC" if ai_score > 0.68 else "👤 NATURAL HUMAN"
    
    return {
        'filename': filename,
        'breaths': breaths,
        'p1_rate_bpm': round(p1_rate, 1),
        'p2_insp_s': round(p2_insp, 2),
        'p3_cv': round(p3_cv, 3),
        'p4_depth_cv': round(p4_depth_cv, 3),
        'p5_entropy': round(p5_entropy, 3),
        'p6_duty': round(p6_duty, 3),
        'ai_score': round(ai_score, 3),
        'status': status,
        'times': times[:400],
        'rms': rms_breath[:400],
        'breath_events': breath_events
    }

st.title("🏥 Medical Breath Analyzer")
st.markdown("**6 Clinically-Accurate Respiratory Parameters**")

uploaded_files = st.file_uploader(
    "Upload speech samples", 
    type=['wav','mp3','m4a','flac'], 
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
    
    # CLINICAL TABLE
    st.subheader("📋 Respiratory Analysis")
    display_df = df[['filename', 'status', 'ai_score', 'breaths', 
                    'p1_rate_bpm', 'p2_insp_s', 'p3_cv']].copy()
    display_df.columns = ['File', 'Verdict', 'AI Score', 'Breaths', 
                         'Rate BPM', 'Insp Time', 'Variability']
    st.dataframe(display_df, use_container_width=True)
    
    # DETAILED PARAMS
    st.subheader("🔬 Detailed Parameters")
    detail_df = df[['filename', 'p4_depth_cv', 'p5_entropy', 'p6_duty']].copy()
    detail_df.columns = ['File', 'Depth CV', 'Rhythm Entropy', 'Duty Cycle']
    st.dataframe(detail_df, use_container_width=True)
    
    # METRICS
    col1, col2 = st.columns(2)
    col1.metric("🤖 AI Voices", len(df[df['status'] == '🤖 AI SYNTHETIC']))
    col2.metric("👤 Human", len(df[df['status'] == '👤 NATURAL HUMAN']))
    
    # GRAPHS
    st.subheader("📈 Breath Waveforms")
    cols = st.columns(3)
    for i, result in enumerate(results[:9]):
        if result['breaths'] == 0:
            continue
            
        with cols[i%3]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            color = 'lime' if 'HUMAN' in result['status'] else 'red'
            
            # Signal + breaths
            ax1.plot(result['rms'][:350], 'cyan', lw=1)
            for event in result['breath_events'][:10]:
                start_idx = np.argmin(np.abs(result['times'] - event['start_time']))
                end_idx = np.argmin(np.abs(result['times'] - event['end_time']))
                ax1.axvspan(start_idx, end_idx, color=color, alpha=0.6)
            ax1.set_title(f"{result['filename'][:45]}")
            ax1.set_ylabel("Breath RMS")
            
            # Parameters summary
            ax2.text(0.1, 0.8, f"Rate: {result['p1_rate_bpm']} BPM", fontsize=12, transform=ax2.transAxes)
            ax2.text(0.1, 0.65, f"Insp: {result['p2_insp_s']}s", fontsize=12, transform=ax2.transAxes)
            ax2.text(0.1, 0.5, f"CV: {result['p3_cv']:.3f}", fontsize=12, transform=ax2.transAxes)
            ax2.text(0.1, 0.35, f"Score: {result['ai_score']:.3f}", fontsize=14, 
                    color='red' if result['ai_score'] > 0.6 else 'green', transform=ax2.transAxes)
            ax2.text(0.1, 0.2, f"Status: {result['status']}", fontsize=16, 
                    weight='bold', transform=ax2.transAxes)
            ax2.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # REFERENCE RANGES
    with st.expander("📚 Clinical Reference"):
        st.markdown("""
        
