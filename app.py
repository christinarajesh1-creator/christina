import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import traceback

st.set_page_config(layout="wide")

@st.cache_data
def safe_load_audio(file):
    file.seek(0)
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        if len(y) < 5000:  # Minimum 0.3s
            return None, None
        return y, sr
    except:
        return None, None

def detect_breath_events(y, sr):
    """Robust breath detection"""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Find high-energy breath-like events
    peaks, props = find_peaks(rms, height=np.percentile(rms, 85), 
                             distance=sr//3, prominence=np.percentile(rms, 75))
    
    events = []
    for p in peaks:
        t = times[p]
        if 1.0 < t < len(y)/sr - 2.0:
            # Confirm it's a breath (short high-energy burst)
            start = max(0, int((t-0.2)*sr))
            end = min(len(y), int((t+0.3)*sr))
            breath_rms = np.mean(librosa.feature.rms(y=y[start:end])[0])
            if breath_rms > np.percentile(rms, 80):
                events.append(t)
    
    # Filter too-close events
    filtered = [events[0]] if events else []
    for t in events[1:]:
        if t - filtered[-1] > 0.7:
            filtered.append(t)
    
    return filtered

def analyze_breath_features(events, y, sr):
    """Calculate 6 forensic parameters"""
    if len(events) < 2:
        return {
            'IBI_Regularity': 0.9,      # Few breaths = AI
            'Amplitude': 0.8,
            'Duration': 0.7,
            'Presence': 0.9,
            'Spectral': 0.6,
            'Similarity': 0.8
        }
    
    scores = {}
    
    # 1. IBI REGULARITY (28%) - HUMAN has HIGH variation
    ibis = np.diff(events)
    cv = np.std(ibis) / np.mean(ibis)
    scores['IBI_Regularity'] = max(0, 1 - cv * 2) if cv < 1.5 else 0.1  # Low CV = AI
    
    # 2. BREATH AMPLITUDE (15%) - HUMAN varies volume
    amps = []
    for t in events[:8]:
        start = max(0, int((t-0.12)*sr))
        end = min(len(y), int((t+0.22)*sr))
        amps.append(np.max(np.abs(y[start:end])))
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 1 else 0
    scores['Amplitude'] = max(0, 1 - amp_cv * 3) if amp_cv < 0.8 else 0.2  # Low CV = AI
    
    # 3. BREATH DURATION (12%) - HUMAN varies length
    durs = []
    for t in events[:8]:
        start = max(0, int((t-0.18)*sr))
        end = min(len(y), int((t+0.35)*sr))
        seg = y[start:end]
        rms_seg = librosa.feature.rms(y=seg, frame_length=512)[0]
        active_frames = np.sum(rms_seg > np.percentile(rms_seg, 60))
        dur = active_frames * 512 / sr
        if 0.05 < dur < 1.0:
            durs.append(dur)
    dur_cv = np.std(durs) / np.mean(durs) if len(durs) > 1 else 0
    scores['Duration'] = max(0, 1 - dur_cv * 2.5) if dur_cv < 0.7 else 0.3
    
    # 4. BREATH PRESENCE (15%) - HUMAN has reasonable amount
    total_dur = len(y) / sr
    breath_prop = sum(durs) / total_dur if durs else 0
    scores['Presence'] = 0.9 if breath_prop < 0.04 else 0.2  # Too little = AI
    
    # 5. SPECTRAL CONTINUITY (12%) - AI has jumps
    zcr_changes = []
    for i in range(min(4, len(events)-1)):
        t1, t2 = events[i], events[i+1]
        s1, e1 = int((t1-0.1)*sr), int((t1+0.15)*sr)
        s2, e2 = int((t2-0.1)*sr), int((t2+0.15)*sr)
        zcr1 = librosa.feature.zero_crossing_rate(y[s1:e1])[0].mean()
        zcr2 = librosa.feature.zero_crossing_rate(y[s2:e2])[0].mean()
        zcr_changes.append(abs(zcr1 - zcr2))
    scores['Spectral'] = 0.8 if zcr_changes and np.mean(zcr_changes) > 0.07 else 0.3
    
    # 6. BREATH SIMILARITY (18%) - AI reuses samples
    sims = []
    for i in range(min(4, len(events))):
        for j in range(i+1, min(i+3, len(events))):
            t1, t2 = events[i], events[j]
            s1, e1 = int((t1-0.12)*sr), int((t1+0.18)*sr)
            s2, e2 = int((t2-0.12)*sr), int((t2+0.18)*sr)
            try:
                mfcc1 = librosa.feature.mfcc(y=y[s1:e1], sr=sr, n_mfcc=10).mean(1)
                mfcc2 = librosa.feature.mfcc(y=y[s2:e2], sr=sr, n_mfcc=10).mean(1)
                corr = np.corrcoef(mfcc1[:5], mfcc2[:5])[0,1]
                sims.append(1 - abs(corr))
            except:
                sims.append(0.5)
    scores['Similarity'] = 0.9 if sims and np.mean(sims) < 0.25 else 0.1
    
    return scores

st.title("🫁 PneumaForensic - Breath Pattern AI Detector")
st.markdown("""
**The Gray Waves**: Your actual speech  
**Red Dashed Lines**: Detected breaths  
**HUMAN**: Uneven red line spacing (natural breathing)  
**AI**: Perfect grid spacing OR missing breaths entirely
""")

files = st.file_uploader("Upload audio", type=['wav','mp3','m4a','aac'], 
                        accept_multiple_files=True)

if files:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.header("📊 Forensic Results")
        results = []
        all_scores = []
        
        for file in files:
            y, sr = safe_load_audio(file)
            
            if y is None:
                results.append({"File": file.name, "AI": "LOAD ERROR", "Status": "❌", "Breaths": 0})
                continue
            
            events = detect_breath_events(y, sr)
            param_scores = analyze_breath_features(events, y, sr)
            
            # Weighted AI score (high = AI)
            weights = [0.28, 0.15, 0.12, 0.15, 0.12, 0.18]
            score = np.average(list(param_scores.values()), weights=weights)
            
            status = "🤖 **AI**" if score > 0.55 else "👤 **HUMAN**"
            
            results.append({
                "File": file.name[:35],
                "AI Score": f"{score:.0%}",
                "Status": status,
                "Breaths": len(events)
            })
            
            all_scores.append({**{"File": file.name, "Score": score, "Status": "AI" if score > 0.55 else "HUMAN"}, **param_scores})
        
        # Results table
        df = pd.DataFrame(results)
        st.dataframe(df.style.highlight_max(subset=['AI Score']), use_container_width=True)
        
        # Parameter breakdown table
        if all_scores:
            st.subheader("🔬 6 Forensic Parameters")
            scores_df = pd.DataFrame(all_scores)
            numeric_cols = scores_df.select_dtypes(include=[np.number]).columns
            styled_df = scores_df.style.format({col: '{:.1%}' for col in numeric_cols if col != 'Score'})
            styled_df = styled_df.background_gradient(subset=numeric_cols, cmap='Reds')
            st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        st.header("📈 Gray Waves + Red Breath Lines")
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                continue
            
            events = detect_breath_events(y, sr)
            param_scores = analyze_breath_features(events, y, sr)
            score = np.average(list(param_scores.values()), weights=[0.28,0.15,0.12,0.15,0.12,0.18])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor='black')
            dur = min(25, len(y)/sr * 0.8)
            
            # Plot 1: Full waveform with breaths
            t = np.linspace(0, dur, int(dur*sr//100 + 1))
            y_plot = y[:len(t)]
            ax1.plot(t, y_plot, 'lightgray', lw=1.2, alpha=0.9)
            
            for e in events:
                if e < dur:
                    ax1.axvline(e, color='red', ls='--', lw=3, alpha=0.9)
            
            color = 'red' if score > 0.55 else 'lime'
            ax1.set_title(f"{file.name[:25]}...\n{score:.0%} AI | {len(events)} breaths", 
                         color=color, fontsize=14, pad=10)
            ax1.set_facecolor('black')
            ax1.set_ylabel("Amplitude", color='white')
            ax1.tick_params(colors='white')
            
            # Plot 2: IBI regularity visualization
            if len(events) > 2:
                ibis = np.diff(events[:8])
                ax2.bar(range(len(ibis)), ibis, color='orange', alpha=0.7)
                ax2.axhline(np.mean(ibis), color='white', ls='-', lw=2, label=f'Mean: {np.mean(ibis):.1f}s')
                cv = np.std(ibis)/np.mean(ibis)
                ax2.set_title(f"Breath Intervals (CV={cv:.2f})", color='white')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, f"Only {len(events)} breaths\nFew breaths = AI", 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=14, color='orange')
            
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
