import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks

st.set_page_config(layout="wide")

def detect_ai_breath_forensics(y, sr):
    """AI Detection based on 6 weighted breath parameters"""
    
    # Find breath-like energy peaks
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Detect breath peaks (high energy, local maxima)
    peaks, _ = find_peaks(rms, height=np.percentile(rms, 90), distance=800)
    events = [times[p] for p in peaks if times[p] > 1.0]
    
    # Filter minimum spacing (1 second apart)
    filtered_events = []
    last_t = 0
    for t in events:
        if t - last_t > 1.0:
            filtered_events.append(t)
            last_t = t
    
    if len(filtered_events) < 2:
        return 0.90, filtered_events  # Few breaths = AI
    
    events = filtered_events
    
    scores = {}
    
    # 1. IBI REGULARITY (28%) - AI has low CV (too regular)
    ibis = np.diff(events)
    cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 10
    scores['ibi_regularity'] = 1.0 if cv < 0.30 else 0.1  # AI: very regular
    # Weight: 28%
    
    # 2. BREATH AMPLITUDE (15%) - AI has uniform volume
    amps = []
    for t in events:
        start = max(0, int((t-0.12)*sr))
        end = min(len(y), int((t+0.18)*sr))
        breath = y[start:end]
        if len(breath) > 100:
            amps.append(np.max(np.abs(breath)))
    
    if amps:
        amp_cv = np.std(amps) / np.mean(amps)
        scores['amplitude'] = 1.0 if amp_cv < 0.25 else 0.2  # AI: consistent
    else:
        scores['amplitude'] = 0.8
    # Weight: 15%
    
    # 3. BREATH DURATION (12%) - AI has copy-pasted lengths
    durations = []
    for t in events:
        start = max(0, int((t-0.15)*sr))
        end = min(len(y), int((t+0.25)*sr))
        breath = y[start:end]
        # Find actual breath duration via energy threshold
        breath_rms = librosa.feature.rms(y=breath, frame_length=256, hop_length=64)[0]
        active = breath_rms > np.mean(breath_rms) * 1.5
        if np.any(active):
            dur_frames = np.sum(active) * 64 / sr
            durations.append(dur_frames)
    
    if durations:
        dur_cv = np.std(durations) / np.mean(durations)
        scores['duration'] = 1.0 if dur_cv < 0.35 else 0.3  # AI: uniform lengths
    else:
        scores['duration'] = 0.7
    # Weight: 12%
    
    # 4. BREATH PRESENCE (15%) - AI has unnaturally low breath proportion
    total_dur = len(y) / sr
    breath_total_dur = sum(durations) if durations else 0
    breath_prop = breath_total_dur / total_dur
    scores['presence'] = 1.0 if breath_prop < 0.08 else 0.1  # AI: too little breathing
    # Weight: 15%
    
    # 5. SPECTRAL CONTINUITY (12%) - AI has abrupt spectral changes
    zcr_changes = []
    for i in range(min(4, len(events)-1)):
        t1, t2 = events[i], events[i+1]
        start1 = max(0, int((t1-0.1)*sr))
        end1 = min(len(y), int((t1+0.1)*sr))
        start2 = max(0, int((t2-0.1)*sr))
        end2 = min(len(y), int((t2+0.1)*sr))
        
        zcr1 = librosa.feature.zero_crossing_rate(y[start1:end1])[0].mean()
        zcr2 = librosa.feature.zero_crossing_rate(y[start2:end2])[0].mean()
        zcr_changes.append(abs(zcr1 - zcr2))
    
    if zcr_changes:
        avg_zcr_jump = np.mean(zcr_changes)
        scores['spectral'] = 1.0 if avg_zcr_jump > 0.08 else 0.2  # AI: abrupt changes
    else:
        scores['spectral'] = 0.6
    # Weight: 12%
    
    # 6. BREATH SIMILARITY (18%) - AI reuses same breath samples
    mfcc_similarities = []
    for i in range(min(5, len(events))):
        for j in range(i+1, min(i+3, len(events))):
            t1, t2 = events[i], events[j]
            start1 = max(0, int((t1-0.12)*sr))
            end1 = min(len(y), int((t1+0.18)*sr))
            start2 = max(0, int((t2-0.12)*sr))
            end2 = min(len(y), int((t2+0.18)*sr))
            
            mfcc1 = librosa.feature.mfcc(y=y[start1:end1], sr=sr, n_mfcc=13).mean(1)
            mfcc2 = librosa.feature.mfcc(y=y[start2:end2], sr=sr, n_mfcc=13).mean(1)
            sim = 1 - np.corrcoef(mfcc1, mfcc2)[0,1]
            mfcc_similarities.append(sim)
    
    if mfcc_similarities:
        avg_similarity = np.mean(mfcc_similarities)
        scores['similarity'] = 1.0 if avg_similarity < 0.15 else 0.1  # AI: breaths too similar
    else:
        scores['similarity'] = 0.9
    # Weight: 18%
    
    # FINAL WEIGHTED SCORE
    weights = {
        'ibi_regularity': 0.28,
        'amplitude': 0.15,
        'duration': 0.12,
        'presence': 0.15,
        'spectral': 0.12,
        'similarity': 0.18
    }
    
    final_score = sum(scores[k] * weights[k] for k in weights)
    
    return final_score, events, scores

st.title("🫁 PneumaForensic AI Detector")
st.markdown("**Detects AI speech by analyzing 6 weighted breath parameters**")

files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a'], accept_multiple_files=True)

if files:
    col1, col2 = st.columns(2)
    results = []
    
    with col1:
        st.subheader("📊 Detection Results")
        detailed_scores = []
        
        for file in files:
            file.seek(0)
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
                score, events, param_scores = detect_ai_breath_forensics(y, sr)
                
                status = "🤖 **AI**" if score > 0.65 else "👤 **HUMAN**"
                confidence = f"{score:.1%}"
                
                results.append({
                    "File": file.name,
                    "AI Score": confidence,
                    "Status": status,
                    "Breaths": len(events)
                })
                
                # Store detailed scores for display
                detailed_scores.append({
                    "File": file.name,
                    "Score": score,
                    "Status": "AI" if score > 0.65 else "HUMAN",
                    **param_scores
                })
                
            except Exception as e:
                results.append({"File": file.name, "AI Score": "ERROR", "Status": "Failed", "Breaths": 0})
        
        df = pd.DataFrame(results)
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
        
        if detailed_scores:
            st.subheader("🔬 Parameter Breakdown")
            detail_df = pd.DataFrame(detailed_scores)
            detail_df = detail_df.round(3)
            st.dataframe(detail_df, use_container_width=True)
    
    with col2:
        st.subheader("🎧 Audio Waveforms")
        for file in files:
            file.seek(0)
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
                score, events, _ = detect_ai_breath_forensics(y, sr)
                
                fig, ax = plt.subplots(figsize=(10, 4), facecolor='black')
                dur = min(25, len(y)/sr)
                t = np.linspace(0, dur, int(dur*sr//100))
                y_plot = y[:len(t)]
                
                ax.plot(t, y_plot, 'cyan', lw=0.8, alpha=0.9)
                
                # Mark breaths
                for e in events:
                    if e < dur:
                        ax.axvline(e, color='red' if score > 0.65 else 'lime', 
                                 ls='--', lw=2.5, alpha=0.8)
                
                color = 'red' if score > 0.65 else 'lime'
                ax.set_title(f"{file.name[:30]}...\n{score:.1%} AI", 
                           color=color, fontsize=18, pad=20)
                ax.set_facecolor('black')
                ax.set_xlabel("Time (s)", color='white')
                ax.tick_params(colors='white')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except:
                st.error(f"Could not process {file.name}")
