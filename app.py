import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="AI Breath Detector")

def ai_breath_detector(file_bytes, filename):
    """Exact 6-parameter forensic analysis"""
    try:
        # Load and preprocess
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        duration = len(y) / sr
        
        # Breath detection (50-2000Hz + energy drop)
        sos = butter(3, [50, 2000], btype='band', fs=sr, output='sos')
        y_breath = filtfilt(sos, 1, y)
        
        hop_length = 512
        rms_breath = librosa.feature.rms(y=y_breath, frame_length=1024, hop_length=hop_length)[0]
        times = librosa.frames_to_time(rms_breath, sr=sr, hop_length=hop_length)
        
        smooth_rms = np.convolve(rms_breath, np.ones(25)/25, mode='same')
        
        # Find breath events (energy valleys)
        valley_mask = (rms_breath < smooth_rms * 0.7) & (rms_breath < np.percentile(rms_breath, 28))
        breath_peaks, props = find_peaks(valley_mask.astype(float), 
                                       height=0.35, distance=25, prominence=0.2)
        
        breath_times = times[breath_peaks]
        breath_frames = breath_peaks
        
        breaths = len(breath_times)
        
        if breaths < 2:
            return {
                'Filename': filename,
                'Status': '🤖 AI',
                'AI_Score': 0.92,
                'P1_IBI_Reg': '100%',
                'P2_Amp': '100%',
                'P3_Dur': '100%',
                'P4_Presence': '0%',
                'P5_SpecCont': '100%',
                'P6_Sim': '100%',
                'breath_times': [],
                'times': times[:300],
                'rms': rms_breath[:300],
                'y': y[:48000]
            }
        
        # **PARAMETER 1: IBI REGULARITY (28%)**
        ibis = np.diff(breath_times)
        ibi_mean = np.mean(ibis)
        ibi_std = np.std(ibis)
        ibi_cv = ibi_std / ibi_mean if ibi_mean > 0 else 0
        p1_reg = min(1.0, max(0, (0.3 - ibi_cv) / 0.3))  # Low CV = regular/AI
        
        # **PARAMETER 2: BREATH AMPLITUDE (15%)**
        breath_amplitudes = rms_breath[breath_frames]
        amp_mean = np.mean(breath_amplitudes)
        amp_std = np.std(breath_amplitudes)
        amp_cv = amp_std / amp_mean if amp_mean > 0 else 0
        p2_amp = min(1.0, max(0, (0.4 - amp_cv) / 0.4))  # Low variation = AI
        
        # **PARAMETER 3: BREATH DURATION (12%)**
        breath_durations = []
        for frame in breath_frames:
            start = max(0, frame-15)
            end = min(len(rms_breath), frame+25)
            dur_frames = np.sum(rms_breath[start:end] < np.percentile(rms_breath[start:end], 35))
            dur = dur_frames * hop_length / sr
            breath_durations.append(dur)
        
        dur_mean = np.mean(breath_durations)
        dur_std = np.std(breath_durations)
        dur_cv = dur_std / dur_mean if dur_mean > 0 else 0
        p3_dur = min(1.0, max(0, (0.5 - dur_cv) / 0.5))  # Uniform duration = AI
        
        # **PARAMETER 4: BREATH PRESENCE (15%)**
        total_breath_time = sum(d for d in breath_durations)
        p4_presence = min(1.0, total_breath_time / duration * 10)  # Normalize
        
        # **PARAMETER 5: SPECTRAL CONTINUITY (12%)**
        # ZCR discontinuity at breath boundaries
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_times = librosa.frames_to_time(zcr, sr=sr, hop_length=hop_length)
        
        zcr_changes = []
        for t in breath_times:
            idx = np.argmin(np.abs(zcr_times - t))
            if idx > 5 and idx < len(zcr)-5:
                before = np.mean(np.abs(np.diff(zcr[idx-5:idx])))
                after = np.mean(np.abs(np.diff(zcr[idx:idx+5])))
                zcr_changes.append(abs(before - after))
        
        p5_spec = np.mean(zcr_changes) if zcr_changes else 0.1
        p5_spec = min(1.0, p5_spec * 5)  # Normalize
        
        # **PARAMETER 6: BREATH SIMILARITY (18%)**
        # Spectral similarity between breaths
        breath_stfts = []
        for frame in breath_frames[:4]:  # First 4 breaths
            start = max(0, frame-8)
            end = min(len(y), frame+8)
            breath_seg = y[start*160:end*160]
            if len(breath_seg) > 256:
                stft = np.abs(librosa.stft(breath_seg[:1024]))
                breath_stfts.append(np.mean(stft, axis=1))
        
        similarity = 1.0
        if len(breath_stfts) > 1:
            for i in range(1, len(breath_stfts)):
                similarity *= 1 - cosine(breath_stfts[0], breath_stfts[i])
        p6_sim = 1 - similarity  # High similarity = AI
        
        # **FINAL AI SCORE**
        ai_score = (p1_reg * 0.28 + p2_amp * 0.15 + p3_dur * 0.12 + 
                   p4_presence * 0.15 + p5_spec * 0.12 + p6_sim * 0.18)
        
        status = "🤖 AI" if ai_score > 0.55 else "👤 HUMAN"
        
        return {
            'Filename': filename,
            'Status': status,
            'AI_Score': f"{ai_score:.1%}",
            'P1_IBI_Reg': f"{p1_reg:.0%}",
            'P2_Amp': f"{p2_amp:.0%}",
            'P3_Dur': f"{p3_dur:.0%}",
            'P4_Presence': f"{p4_presence:.0%}",
            'P5_SpecCont': f"{p5_spec:.0%}",
            'P6_Sim': f"{p6_sim:.0%}",
            'breath_times': breath_times.tolist(),
            'times': times[:400],
            'rms': rms_breath[:400],
            'y': y[:64000]  # 4 seconds
        }
    except:
        return {
            'Filename': filename,
            'Status': 'ERROR',
            'AI_Score': '100%',
            'P1_IBI_Reg': '100%',
            'P2_Amp': '100%',
            'P3_Dur': '100%',
            'P4_Presence': '0%',
            'P5_SpecCont': '100%',
            'P6_Sim': '100%',
            'breath_times': [],
            'times': np.array([]),
            'rms': np.array([]),
            'y': np.array([])
        }

st.title("🤖 AI Breath Detector")
st.markdown("**Forensic 6-Parameter Analysis**")

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
        result = ai_breath_detector(file.read(), file.name)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # MAIN RESULTS
    st.subheader("Detection Results")
    st.dataframe(df[['Filename', 'Status', 'AI_Score']], use_container_width=True)
    
    # 6 PARAMETERS TABLE
    st.subheader("6 Forensic Parameters")
    param_cols = ['Filename', 'P1_IBI_Reg', 'P2_Amp', 'P3_Dur', 
                 'P4_Presence', 'P5_SpecCont', 'P6_Sim']
    st.dataframe(df[param_cols], use_container_width=True)
    
    # SUMMARY
    col1, col2 = st.columns(2)
    ai_count = len(df[df['Status'] == '🤖 AI'])
    col1.metric("🤖 AI", ai_count)
    col2.metric("👤 Human", len(df) - ai_count)
    
    # **GRAY WAVES + RED BREATH LINES**
    st.subheader("Breath Detection")
    cols = st.columns(3)
    for i, result in enumerate(results[:9]):
        if not result['breath_times']:
            continue
            
        with cols[i % 3]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # GRAY WAVES (raw speech)
            ax1.plot(result['y'], color='gray', linewidth=0.8, alpha=0.7)
            # RED DASHED BREATH LINES
            for t in result['breath_times']:
                ax1.axvline(t * 16000, color='red', linestyle='--', linewidth=2, alpha=0.9)
            ax1.set_title(f"{result['Filename'][:60]}")
            ax1.set_ylabel("Amplitude")
            ax1.margins(x=0)
            
            # RMS + breaths
            ax2.plot(result['times'], result['rms'], color='cyan', linewidth=1.5)
            ax2.fill_between(result['times'], result['rms'], alpha=0.3, color='cyan')
            for t in result['breath_times']:
                ax2.axvline(t, color='red', linestyle='--', linewidth=2.5, alpha=0.9)
            ax2.set_title(f"{result['Status']} | Score: {result['AI_Score']} | "
                         f"P1:{result['P1_IBI_Reg']} P6:{result['P6_Sim']}")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Breath Energy")
            
            plt.tight_layout()
            st.pyplot(fig)
    
    st.success("Analysis complete")

with st.expander("🔍 Parameter Breakdown"):
    st.markdown("""
    **P1 IBI Regularity (28%)**: Timing variation between breaths
    **P2 Amplitude (15%)**: Energy consistency across breaths  
    **P3 Duration (12%)**: Length variation of breaths
    **P4 Presence (15%)**: Proportion of audio that contains breath
    **P5 Spectral Continuity (12%)**: ZCR discontinuity at breath edges
    **P6 Similarity (18%)**: Spectral sameness between breath samples
    
    **>55% = 🤖 AI (unnatural regularity/similarity)**
    """)

st.markdown("---")
st.caption("Forensic AI speech detection")
