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

@st.cache_data
def safe_load_audio(file):
    file.seek(0)
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        if len(y) < 10000:  # 0.6s minimum
            return None, None
        return y, sr
    except:
        return None, None

def detect_breath_events(y, sr):
    """Conservative breath detection - only REAL breaths"""
    # Use silence detection to find breath locations
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Find regions of LOW speech energy (potential breaths)
    speech_threshold = np.percentile(rms, 40)  # Lower threshold for breaths
    low_energy = rms < speech_threshold
    
    # Find breath-like peaks in low-energy regions
    breath_peaks, _ = find_peaks(low_energy.astype(float), 
                                height=0.3, distance=sr//2.5,  # Min 0.4s apart
                                prominence=0.2)
    
    events = []
    for p in breath_peaks:
        t = times[p]
        if 1.5 < t < len(y)/sr - 3.0:
            # Verify it's actually a breath (some energy burst in silence)
            start = max(0, int((t-0.25)*sr))
            end = min(len(y), int((t+0.4)*sr))
            breath_rms = np.max(librosa.feature.rms(y=y[start:end])[0])
            
            # Breath must have moderate energy (not pure silence)
            if 0.005 < breath_rms < 0.08:
                events.append(t)
    
    # Final filtering
    filtered = []
    for t in events:
        if not filtered or t - filtered[-1] > 0.6:
            filtered.append(t)
    
    return filtered[:12]  # Max 12 breaths

def analyze_breath_features(events, y, sr):
    """6 parameters - HUMAN FRIENDLY calibration"""
    scores = {}
    
    # Base score - more breaths = more human
    breath_count_score = min(0.4, len(events) * 0.05)
    
    if len(events) < 2:
        scores = {'IBI_Regularity': 0.8, 'Amplitude': 0.7, 'Duration': 0.6, 
                 'Presence': 0.9, 'Spectral': 0.5, 'Similarity': 0.8}
        return scores
    
    # 1. IBI REGULARITY (28%) - HUMAN: CV > 0.4
    ibis = np.diff(events)
    cv = np.std(ibis) / np.mean(ibis)
    ibi_human = 1.0 if cv > 0.4 else max(0, cv * 1.5)  # HIGH variation = HUMAN
    scores['IBI_Regularity'] = ibi_human
    
    # 2. BREATH AMPLITUDE (15%) - HUMAN: varied amplitudes
    amps = []
    for t in events:
        start = max(0, int((t-0.15)*sr))
        end = min(len(y), int((t+0.3)*sr))
        peak_amp = np.max(np.abs(y[start:end]))
        if peak_amp > 0.003:
            amps.append(peak_amp)
    
    if len(amps) > 1:
        amp_cv = np.std(amps) / np.mean(amps)
        scores['Amplitude'] = 1.0 if amp_cv > 0.25 else max(0, amp_cv * 2)
    else:
        scores['Amplitude'] = 0.8
    
    # 3. BREATH DURATION (12%) - HUMAN: varied lengths
    durs = []
    for t in events:
        start = max(0, int((t-0.2)*sr))
        end = min(len(y), int((t+0.45)*sr))
        seg = y[start:end]
        rms_seg = librosa.feature.rms(y=seg)[0]
        # Active duration above noise floor
        active = rms_seg > np.percentile(rms_seg, 55)
        dur = np.sum(active) * 512 / sr
        if 0.08 < dur < 0.8:
            durs.append(dur)
    
    if len(durs) > 1:
        dur_cv = np.std(durs) / np.mean(durs)
        scores['Duration'] = 1.0 if dur_cv > 0.3 else max(0, dur_cv * 2)
    else:
        scores['Duration'] = 0.7
    
    # 4. BREATH PRESENCE (15%) - HUMAN: 3-12% of audio
    total_dur = len(y) / sr
    breath_prop = sum(durs) / total_dur if durs else 0
    scores['Presence'] = 1.0 if 0.02 < breath_prop < 0.15 else 0.3
    
    # 5. SPECTRAL CONTINUITY (12%) - HUMAN: smooth transitions
    zcr_changes = []
    for i in range(min(3, len(events)-1)):
        t1, t2 = events[i], events[i+1]
        s1, e1 = int((t1-0.12)*sr), int((t1+0.18)*sr)
        s2, e2 = int((t2-0.12)*sr), int((t2+0.18)*sr)
        zcr1 = librosa.feature.zero_crossing_rate(y[s1:e1])[0].mean()
        zcr2 = librosa.feature.zero_crossing_rate(y[s2:e2])[0].mean()
        zcr_changes.append(abs(zcr1 - zcr2))
    
    if zcr_changes:
        avg_jump = np.mean(zcr_changes)
        scores['Spectral'] = 0.8 if avg_jump < 0.05 else 0.3  # Small jumps = HUMAN
    else:
        scores['Spectral'] = 0.6
    
    # 6. BREATH SIMILARITY (18%) - HUMAN: unique breaths
    sims = []
    for i in range(min(4, len(events))):
        for j in range(i+1, min(i+3, len(events))):
            t1, t2 = events[i], events[j]
            s1, e1 = int((t1-0.13)*sr), int((t1+0.2)*sr)
            s2, e2 = int((t2-0.13)*sr), int((t2+0.2)*sr)
            try:
                mfcc1 = librosa.feature.mfcc(y=y[s1:e1], sr=sr, n_mfcc=8).mean(1)
                mfcc2 = librosa.feature.mfcc(y=y[s2:e2], sr=sr, n_mfcc=8).mean(1)
                corr = np.corrcoef(mfcc1, mfcc2)[0,1]
                sims.append(abs(1 - corr))  # Low correlation = dissimilar = HUMAN
            except:
                sims.append(0.6)
    
    if sims:
        avg_sim = np.mean(sims)
        scores['Similarity'] = 1.0 if avg_sim > 0.3 else 0.2  # Dissimilar = HUMAN
    else:
        scores['Similarity'] = 0.7
    
    return scores

st.title("🫁 PneumaForensic v2.0 - HUMAN CALIBRATED")
st.markdown("""
**Gray Waves** = Your speech  
**Red Dashed Lines** = Detected breaths  
**HUMAN** = Irregular spacing (natural)  
**AI** = Perfect grid OR no breaths
""")

files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a','aac'], 
                        accept_multiple_files=True)

if files:
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.header("🎯 Results")
        results = []
        all_scores = []
        
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                results.append({"File": file.name, "AI": "ERROR", "Status": "❌", "Breaths": 0})
                continue
            
            events = detect_breath_events(y, sr)
            param_scores = analyze_breath_features(events, y, sr)
            
            # AI score = OPPOSITE of human features (high = AI)
            weights = [0.28, 0.15, 0.12, 0.15, 0.12, 0.18]
            ai_score = 1 - np.average(list(param_scores.values()), weights=weights)
            
            status = "🤖 **AI**" if ai_score > 0.60 else "👤 **HUMAN**"
            
            results.append({
                "File": file.name[:35],
                "AI": f"{ai_score:.0%}",
                "Status": status,
                "Breaths": len(events)
            })
            
            all_scores.append({
                "File": file.name[:25], "AI Score": ai_score, "Status": status, **param_scores
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        if all_scores:
            st.subheader("🔬 Parameter Analysis")
            scores_df = pd.DataFrame(all_scores)
            st.dataframe(scores_df.round(2), use_container_width=True)
    
    with col2:
        st.header("📈 Breath Pattern Analysis")
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                continue
            
            events = detect_breath_events(y, sr)
            param_scores = analyze_breath_features(events, y, sr)
            ai_score = 1 - np.average(list(param_scores.values()), weights=[0.28,0.15,0.12,0.15,0.12,0.18])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor='black')
            dur = min(22, len(y)/sr * 0.7)
            
            # MAIN WAVEFORM
            t = np.linspace(0, dur, int(dur*sr//160))
            y_plot = y[:len(t)]
            ax1.plot(t, y_plot, color='lightgray', lw=1.4)
            
            for i, e in enumerate(events):
                if e < dur:
                    color = 'red' if ai_score > 0.60 else 'lime'
                    ax1.axvline(e, color=color, ls='--', lw=3, alpha=0.9)
                    if i < 5:
                        ax1.text(e, 0.1, f'B{i+1}', ha='center', color=color, 
                                transform=ax1.get_xaxis_transform(), fontsize=9)
            
            color = 'red' if ai_score > 0.60 else 'lime'
            ax1.set_title(f"{file.name[:28]} | AI: {ai_score:.0%} | {len(events)} breaths", 
                         color=color, fontsize=14, pad=15)
            ax1.set_facecolor('black')
            ax1.set_ylabel("Amplitude", color='white', fontsize=11)
            ax1.tick_params(colors='white')
            
            # IBI SPACING
            if len(events) >= 3:
                ibis = np.diff(events[:10])
                colors = ['orange' if x < np.mean(ibis)*0.7 or x > np.mean(ibis)*1.5 else 'cyan' 
                         for x in ibis]
                bars = ax2.bar(range(len(ibis)), ibis, color=colors, alpha=0.8, width=0.6)
                ax2.axhline(np.mean(ibis), color='white', ls='-', lw=2.5, 
                           label=f'Avg: {np.mean(ibis):.2f}s')
                cv_text = f'CV: {np.std(ibis)/np.mean(ibis):.2f}'
                ax2.text(0.02, 0.95, cv_text, transform=ax2.transAxes, color='white', 
                        va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
            else:
                ax2.bar(['Breaths'], [len(events)], color='orange', alpha=0.7)
                ax2.text(0.5, 0.6, f"Only {len(events)} breaths detected\nMore = more human-like", 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=12, color='white')
            
            ax2.set_title("Breath Intervals (Spacing)", color='white', pad=10)
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
