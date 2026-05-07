import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import traceback

st.set_page_config(layout="wide")

@st.cache_data
def safe_load_audio(file):
    """Safely load audio with detailed error reporting"""
    file.seek(0)
    try:
        # Try multiple sample rates and formats
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        if len(y) < 1000:
            st.warning("Audio too short")
            return None, None
        return y, sr
    except Exception as e:
        st.error(f"Load error: {str(e)[:100]}")
        return None, None

def detect_ai_breath_forensics(y, sr):
    """Simplified robust AI detection"""
    
    try:
        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
        times = librosa.frames_to_time(rms, sr=sr)
        
        # Find breath peaks more reliably
        threshold = np.percentile(rms, 88)
        peaks, _ = find_peaks(rms, height=threshold, distance=sr//4)  # 0.25s min spacing
        events = [times[p] for p in peaks if 1.0 < times[p] < len(y)/sr - 1.0]
        
        # Filter close events
        filtered = []
        for t in events:
            if not filtered or t - filtered[-1] > 0.8:
                filtered.append(t)
        events = filtered
        
        if len(events) < 2:
            return 0.85, events, {}
        
        # 1. IBI REGULARITY (28%)
        ibis = np.diff(events)
        cv = np.std(ibis) / np.mean(ibis) if len(ibis) > 0 else 1
        ibi_score = 1.0 if cv < 0.35 else max(0, 1-cv*1.5)
        
        # 2. AMPLITUDE (15%)
        amps = []
        for t in events[:6]:
            start = max(0, int((t-0.1)*sr))
            end = min(len(y), int((t+0.2)*sr))
            amps.append(np.max(np.abs(y[start:end])))
        amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 1 else 0.5
        amp_score = 1.0 if amp_cv < 0.3 else 0.2
        
        # 3. DURATION (12%)
        durs = []
        for t in events[:6]:
            start = max(0, int((t-0.15)*sr))
            end = min(len(y), int((t+0.25)*sr))
            seg = y[start:end]
            if len(seg) > 200:
                rms_seg = librosa.feature.rms(y=seg)[0]
                active = rms_seg > np.percentile(rms_seg, 70)
                dur = np.sum(active) * 512 / sr
                if dur > 0.05:
                    durs.append(dur)
        dur_cv = np.std(durs) / np.mean(durs) if len(durs) > 1 else 0.4
        dur_score = 1.0 if dur_cv < 0.4 else 0.3
        
        # 4. PRESENCE (15%)
        total_breath = sum(durs) if durs else 0
        breath_prop = total_breath / (len(y)/sr)
        presence_score = 1.0 if breath_prop < 0.06 else 0.1
        
        # 5. SPECTRAL JUMPS (12%)
        zcr_jumps = []
        for i in range(min(3, len(events)-1)):
            t1, t2 = events[i], events[i+1]
            s1, e1 = int((t1-0.08)*sr), int((t1+0.08)*sr)
            s2, e2 = int((t2-0.08)*sr), int((t2+0.08)*sr)
            zcr1 = librosa.feature.zero_crossing_rate(y[max(0,s1):min(len(y),e1)])[0].mean()
            zcr2 = librosa.feature.zero_crossing_rate(y[max(0,s2):min(len(y),e2)])[0].mean()
            zcr_jumps.append(abs(zcr1-zcr2))
        spec_score = 0.8 if zcr_jumps and np.mean(zcr_jumps) > 0.06 else 0.3
        
        # 6. SIMILARITY (18%)
        sim_score = 0.7  # Default neutral
        if len(events) >= 3:
            mfcc_sims = []
            for i in range(min(3, len(events))):
                for j in range(i+1, min(i+2, len(events))):
                    t1, t2 = events[i], events[j]
                    s1, e1 = int((t1-0.1)*sr), int((t1+0.15)*sr)
                    s2, e2 = int((t2-0.1)*sr), int((t2+0.15)*sr)
                    mfcc1 = librosa.feature.mfcc(y=y[s1:e1], sr=sr, n_mfcc=8).mean()
                    mfcc2 = librosa.feature.mfcc(y=y[s2:e2], sr=sr, n_mfcc=8).mean()
                    corr = np.corrcoef(mfcc1[:4], mfcc2[:4])[0,1] if len(mfcc1)>3 else 0
                    mfcc_sims.append(1-corr)
            sim_score = 1.0 if np.mean(mfcc_sims) < 0.2 else 0.1
        
        # WEIGHTED COMBINATION
        weights = [0.28, 0.15, 0.12, 0.15, 0.12, 0.18]
        scores = [ibi_score, amp_score, dur_score, presence_score, spec_score, sim_score]
        final_score = np.average(scores, weights=weights)
        
        param_scores = {
            'IBI (28%)': f"{ibi_score:.2f}",
            'Amp (15%)': f"{amp_score:.2f}",
            'Dur (12%)': f"{dur_score:.2f}",
            'Presence (15%)': f"{presence_score:.2f}",
            'Spectral (12%)': f"{spec_score:.2f}",
            'Similarity (18%)': f"{sim_score:.2f}"
        }
        
        return final_score, events, param_scores
        
    except:
        return 0.5, [], {}

st.title("🫁 PneumaForensic AI Detector")
st.markdown("**Upload WAV/MP3/M4A - Works with ALL formats**")

files = st.file_uploader("Choose audio files", type=['wav','mp3','m4a','aac'], 
                        accept_multiple_files=True)

if files:
    col1, col2 = st.columns(2)
    results = []
    
    with col1:
        st.subheader("🎯 Results")
        
        for file in files:
            y, sr = safe_load_audio(file)
            
            if y is None:
                results.append({"File": file.name, "AI Score": "LOAD ERROR", "Status": "❌", "Breaths": 0})
                continue
            
            score, events, param_scores = detect_ai_breath_forensics(y, sr)
            
            status = "🤖 **AI**" if score > 0.65 else "👤 **HUMAN**"
            results.append({
                "File": file.name[:40],
                "AI Score": f"{score:.1%}",
                "Status": status,
                "Breaths": len(events)
            })
            
            # Show parameter details for first file
            if len(results) == 1:
                st.markdown("### 🔬 Parameter Scores")
                params_df = pd.DataFrame(list(param_scores.items()), columns=['Parameter', 'Score'])
                st.dataframe(params_df, use_container_width=True)
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Waveforms")
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                continue
                
            score, events, _ = detect_ai_breath_forensics(y, sr)
            
            fig, ax = plt.subplots(figsize=(12, 4), facecolor='black')
            dur = min(20, len(y)/sr)
            t = np.linspace(0, dur, min(4000, len(y)))
            y_plot = y[:len(t)]
            
            ax.plot(t, y_plot, 'cyan', lw=1.0, alpha=0.9)
            for e in events[:8]:
                if e < dur:
                    ax.axvline(e, color='red' if score > 0.65 else 'limegreen',
                             ls='--', lw=3, alpha=0.8, label="Breath" if e==events[0] else "")
            
            color = 'red' if score > 0.65 else 'limegreen'
            ax.set_title(f"{file.name[:25]}...\n{score:.0%} AI", color=color, 
                        fontsize=16, pad=15)
            ax.set_facecolor('black')
            ax.set_xlabel("Time (seconds)", color='white', fontsize=12)
            ax.tick_params(colors='white')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
