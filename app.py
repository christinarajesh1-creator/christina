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

st.set_page_config(layout="wide")

def forensic_ai_detector(file_bytes, filename):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        duration = len(y) / sr
        
        b, a = butter(4, [100, 1800], btype='band', fs=sr)
        y_filt = filtfilt(b, a, y)
        
        hop_length = 256
        rms_filt = librosa.feature.rms(y=y_filt, frame_length=512, hop_length=hop_length)[0]
        times = librosa.frames_to_time(rms_filt, sr=sr, hop_length=hop_length)
        
        smooth_rms = np.convolve(rms_filt, np.ones(31)/31, mode='same')
        energy_ratio = rms_filt / (smooth_rms + 1e-10)
        breath_mask = (energy_ratio < 0.68) & (rms_filt < np.percentile(rms_filt, 25))
        
        breath_peaks, _ = find_peaks(breath_mask.astype(float), height=0.4, distance=30, prominence=0.22)
        breath_centers = times[breath_peaks]
        breath_centers = breath_centers[(breath_centers > 0.4) & (breath_centers < duration - 1.0)]
        breath_frames = breath_peaks[(breath_peaks * hop_length/sr > 0.4) & (breath_peaks * hop_length/sr < duration - 1.0)]
        
        n_breaths = len(breath_centers)
        
        if n_breaths < 2:
            return {
                'Filename': filename,
                'Status': '🤖 AI',
                'AI_Score': '92%',
                'P1_IBI_Reg': 'N/A',
                'P2_Amp': 'N/A',
                'P3_Dur': 'N/A',
                'P4_Presence': '0%',
                'P5_SpecCont': 'N/A',
                'P6_Sim': 'N/A',
                'breath_times': [],
                'times': times[:300],
                'rms': rms_filt[:300],
                'speech': y[:48000]
            }
        
        # P1: IBI REGULARITY
        ibis = np.diff(breath_centers)
        ibi_cv = np.std(ibis) / np.mean(ibis)
        p1_regularity = 1 - min(1.0, ibi_cv / 0.45)
        
        # P2: AMPLITUDE VARIATION
        breath_amps = rms_filt[breath_frames]
        amp_cv = np.std(breath_amps) / np.mean(breath_amps)
        p2_amp_var = 1 - min(1.0, amp_cv / 0.35)
        
        # P3: DURATION VARIATION
        breath_durs = []
        for frame in breath_frames:
            start = max(0, frame-20)
            end = min(len(rms_filt), frame+30)
            seg = rms_filt[start:end]
            low_energy_frames = np.sum(seg < np.percentile(seg, 32))
            dur = low_energy_frames * hop_length / sr
            breath_durs.append(dur)
        dur_cv = np.std(breath_durs) / np.mean(breath_durs)
        p3_dur_var = 1 - min(1.0, dur_cv / 0.4)
        
        # P4: PRESENCE
        total_breath_dur = sum(breath_durs)
        p4_presence = min(0.99, total_breath_dur / duration * 8)
        
        # P5: SPECTRAL CONTINUITY (ZCR jumps)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_times = librosa.frames_to_time(zcr, sr=sr, hop_length=hop_length)
        zcr_jumps = []
        for t in breath_centers:
            idx = np.argmin(np.abs(zcr_times - t))
            if 10 < idx < len(zcr)-10:
                pre = np.mean(np.abs(np.diff(zcr[idx-8:idx])))
                post = np.mean(np.abs(np.diff(zcr[idx:idx+8])))
                zcr_jumps.append(abs(pre - post))
        p5_spec = np.mean(zcr_jumps) * 20 if zcr_jumps else 0
        p5_spec_cont = min(1.0, p5_spec)
        
        # P6: BREATH SIMILARITY (MFCC cosine)
        breath_spectra = []
        for frame in breath_frames[:6]:
            start_sample = max(0, int((times[frame]-0.12)*sr))
            end_sample = min(len(y), int((times[frame]+0.12)*sr))
            chunk = y[start_sample:end_sample]
            if len(chunk) > 512:
                mfcc = librosa.feature.mfcc(y=chunk[:2048], sr=sr, n_mfcc=13).mean(axis=1)
                breath_spectra.append(mfcc)
        
        p6_similarity = 0
        if len(breath_spectra) > 1:
            ref = breath_spectra[0]
            for spec in breath_spectra[1:]:
                sim = 1 - cosine(ref, spec)
                p6_similarity += sim
            p6_similarity /= len(breath_spectra) - 1
        p6_sim = p6_similarity
        
        # WEIGHTED SCORE
        ai_score = (p1_regularity * 0.28 + p2_amp_var * 0.15 + p3_dur_var * 0.12 + 
                   p4_presence * 0.15 + p5_spec_cont * 0.12 + p6_sim * 0.18)
        
        status = "🤖 AI" if ai_score > 0.58 else "👤 HUMAN"
        
        return {
            'Filename': filename,
            'Status': status,
            'AI_Score': f"{ai_score:.1%}",
            'P1_IBI_Reg': f"{p1_regularity:.0%}",
            'P2_Amp': f"{p2_amp_var:.0%}",
            'P3_Dur': f"{p3_dur_var:.0%}",
            'P4_Presence': f"{p4_presence:.0%}",
            'P5_SpecCont': f"{p5_spec_cont:.0%}",
            'P6_Sim': f"{p6_sim:.0%}",
            'breath_times': breath_centers.tolist(),
            'times': times[:400],
            'rms': rms_filt[:400],
            'speech': y[:64000]
        }
    except:
        return {
            'Filename': filename,
            'Status': 'ERROR',
            'AI_Score': '100%',
            'P1_IBI_Reg': 'N/A',
            'P2_Amp': 'N/A',
            'P3_Dur': 'N/A',
            'P4_Presence': '0%',
            'P5_SpecCont': 'N/A',
            'P6_Sim': 'N/A',
            'breath_times': [],
            'times': np.array([]),
            'rms': np.array([]),
            'speech': np.array([])
        }

st.title("🤖 AI Breath Detector")
st.markdown("Forensic 6-parameter analysis")

uploaded_files = st.file_uploader(
    "Upload speech", type=['wav','mp3','m4a','flac'], 
    accept_multiple_files=True
)

if uploaded_files:
    progress = st.progress(0)
    results = []
    
    for i, file in enumerate(uploaded_files):
        progress.progress((i+1)/len(uploaded_files))
        file.seek(0)
        result = forensic_ai_detector(file.read(), file.name)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    st.subheader("Results")
    st.dataframe(df[['Filename', 'Status', 'AI_Score']], use_container_width=True)
    
    st.subheader("Parameters")
    st.dataframe(df[['Filename', 'P1_IBI_Reg', 'P2_Amp', 'P3_Dur', 
                    'P4_Presence', 'P5_SpecCont', 'P6_Sim']], use_container_width=True)
    
    col1, col2 = st.columns(2)
    ai_count = len(df[df['Status'] == '🤖 AI'])
    col1.metric("🤖 AI", ai_count)
    col2.metric("👤 Human", len(df) - ai_count)
    
    st.subheader("Detection Visualization")
    cols = st.columns(3)
    for i, result in enumerate(results[:12]):
        if not result['breath_times']:
            continue
            
        with cols[i % 3]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # GRAY WAVES
            ax1.plot(result['speech'], color='gray', linewidth=0.6, alpha=0.8)
            # RED DASHED BREATHS
            for t in result['breath_times']:
                ax1.axvline(t * 16000, color='red', linestyle='--', linewidth=2.5, alpha=0.95)
            ax1.set_title(result['Filename'][:70])
            ax1.set_ylabel("Raw Speech")
            
            # BREATH ENERGY
            ax2.plot(result['times'], result['rms'], color='cyan', linewidth=2)
            ax2.fill_between(result['times'], result['rms'], alpha=0.4, color='cyan')
            for t in result['breath_times']:
                ax2.axvline(t, color='red', linestyle='--', linewidth=3, alpha=0.95)
            ax2.set_title(f"{result['Status']} | {result['AI_Score']}")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Breath Energy")
            
            plt.tight_layout()
            st.pyplot(fig)

with st.expander("Parameters"):
    st.write("P1 IBI Reg (28%): Timing regularity")
    st.write("P2 Amp (15%): Amplitude uniformity")
    st.write("P3 Dur (12%): Duration consistency")
    st.write("P4 Presence (15%): Breath proportion")
    st.write("P5 SpecCont (12%): Spectral jumps")
    st.write("P6 Sim (18%): Breath sample reuse")

st.markdown("---")
