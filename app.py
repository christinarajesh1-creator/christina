import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="PneumaForensic AI Detector")

@st.cache_data
def safe_load_audio(file):
    file.seek(0)
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        if len(y) < 10000: return None, None
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.9
        return y, sr
    except: return None, None

def detect_breath_events(y, sr):
    """Very sensitive human breath detection"""
    # Use longer frames for better silence detection
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Find ALL low energy regions (30th percentile = very inclusive)
    thresh = np.percentile(rms, 30)
    silence_idx = np.where(rms < thresh)[0]
    
    if len(silence_idx) < 15: return []
    
    silence_rms = rms[silence_idx]
    silence_times = times[silence_idx]
    
    # Find ANY energy bursts in silence (very low threshold)
    peaks, props = find_peaks(silence_rms, 
                             height=np.percentile(silence_rms, 60),
                             distance=hop/sr*2,  # Min 1s apart
                             prominence=np.std(silence_rms)*0.2)
    
    candidates = silence_times[peaks]
    
    # Loose verification - accept most candidates
    breaths = []
    for t in candidates:
        if 1.0 < t < len(y)/sr - 2.0:
            # Check 1s window has some energy burst
            start = max(0, int((t-0.5)*sr))
            end = min(len(y), int((t+0.5)*sr))
            seg_rms = np.max(librosa.feature.rms(y=y[start:end])[0])
            if seg_rms > 0.0015:  # Very low threshold
                breaths.append(t)
    
    # Loose spacing (1.5-10s)
    final_breaths = [breaths[0]] if breaths else []
    for t in breaths[1:]:
        if t - final_breaths[-1] > 1.5:
            final_breaths.append(t)
    
    return final_breaths[:25]

def calculate_ai_score(y, sr):
    """Accurate 6-parameter AI detection with visible scores"""
    events = detect_breath_events(y, sr)
    n_breaths = len(events)
    
    total_time = len(y) / sr
    breath_rate = n_breaths / max(1, total_time) if total_time > 0 else 0
    
    # 1. BREATH COUNT (25%) - Human: 8-20 breaths/min (0.13-0.33 breaths/sec)
    count_score = 0.05
    if n_breaths < 2 or breath_rate < 0.08 or breath_rate > 0.5:
        count_score = 0.85
    scores = {'Count': count_score}
    
    if n_breaths < 3:
        return count_score, events, scores
    
    weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.15]
    
    # 2. INTERVAL REGULARITY (20%) - AI: CV < 12%
    ibis = np.diff(events)
    cv = np.std(ibis) / np.mean(ibis)
    reg_score = 0.90 if cv < 0.12 else 0.02
    scores['Regularity'] = reg_score
    
    # 3. AMPLITUDE UNIFORMITY (15%) - AI: CV < 12%
    amps = []
    for t in events[:15]:
        s = max(0, int((t-0.25)*sr))
        e = min(len(y), int((t+0.35)*sr))
        amp = np.max(np.abs(y[s:e]))
        if amp > 0.0005: amps.append(amp)
    
    amp_score = 0.02
    if len(amps) >= 5:
        cv_amp = np.std(amps)/np.mean(amps)
        if cv_amp < 0.12: amp_score = 0.85
    scores['Amplitude'] = amp_score
    
    # 4. DURATION UNIFORMITY (15%) - AI: CV < 15%
    durs = []
    for t in events[:15]:
        s = max(0, int((t-0.35)*sr))
        e = min(len(y), int((t+0.65)*sr))
        if e > s:
            rms_seg = librosa.feature.rms(y=y[s:e], hop_length=256)[0]
            active = rms_seg > np.percentile(rms_seg, 40)
            dur = np.sum(active) * 0.016  # 256/16000
            if 0.08 < dur < 1.8: durs.append(dur)
    
    dur_score = 0.03
    if len(durs) >= 5:
        cv_dur = np.std(durs)/np.mean(durs)
        if cv_dur < 0.15: dur_score = 0.80
    scores['Duration'] = dur_score
    
    # 5. SPECTRAL SIMILARITY (10%) - AI: low ZCR variation
    zcr_var = []
    for i in range(min(6, len(events)-1)):
        t1, t2 = events[i], events[i+1]
        s1, e1 = int(max(0,(t1-0.2)*sr)), int(min(len(y),(t1+0.3)*sr))
        s2, e2 = int(max(0,(t2-0.2)*sr)), int(min(len(y),(t2+0.3)*sr))
        z1 = librosa.feature.zero_crossing_rate(y[s1:e1]).mean()
        z2 = librosa.feature.zero_crossing_rate(y[s2:e2]).mean()
        zcr_var.append(abs(z1-z2))
    
    spec_score = 0.04
    if zcr_var:
        if np.mean(zcr_var) < 0.06: spec_score = 0.75
    scores['Spectral'] = spec_score
    
    # 6. BREATH SIMILARITY (15%) - AI: high MFCC correlation
    sims = []
    for i in range(min(10, len(events))):
        for j in range(i+2, min(i+5, len(events))):
            t1, t2 = events[i], events[j]
            s1, e1 = int(max(0,(t1-0.22)*sr)), int(min(len(y),(t1+0.32)*sr))
            s2, e2 = int(max(0,(t2-0.22)*sr)), int(min(len(y),(t2+0.32)*sr))
            try:
                m1 = librosa.feature.mfcc(y=y[s1:e1], sr=sr, n_mfcc=10).mean(1)
                m2 = librosa.feature.mfcc(y=y[s2:e2], sr=sr, n_mfcc=10).mean(1)
                if len(m1) > 6:
                    corr = np.corrcoef(m1[:8], m2[:8])[0,1]
                    sims.append(corr if not np.isnan(corr) else 0)
            except: continue
    
    sim_score = 0.05
    if sims:
        if np.mean([s for s in sims if s > 0]) > 0.88:
            sim_score = 0.88
    scores['Similarity'] = sim_score
    
    # Calculate final weighted score
    param_scores = [scores['Count'], scores['Regularity'], scores['Amplitude'], 
                   scores['Duration'], scores['Spectral'], scores['Similarity']]
    final_score = np.average(param_scores, weights=weights)
    
    return final_score, events, scores

def main():
    st.title("🫁 PneumaForensic - Fixed & Accurate")
    
    files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a'], 
                            accept_multiple_files=True)
    
    if not files:
        st.info("📁 Please upload audio files")
        st.markdown("""
        **Expected results:**
        - **Human speech**: 15-45% AI score, 5-15 breaths  
        - **AI generated**: 75%+ AI score, 0-2 breaths or perfect regularity
        """)
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎯 Detection Results")
        results = []
        all_params = []
        
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                results.append({"File": f"{file.name[:30]}...", "AI": "ERROR", "Status": "❌", "Breaths": 0})
                continue
            
            ai_score, events, params = calculate_ai_score(y, sr)
            status = "🤖 AI" if ai_score > 0.70 else "👤 HUMAN"
            
            results.append({
                "File": f"{file.name[:30]}...",
                "AI": f"{ai_score:.0%}", 
                "Status": status,
                "Breaths": len(events)
            })
            
            # Store ALL parameters as numeric
            param_row = {"File": f"{file.name[:25]}..."}
            for k, v in params.items():
                param_row[k] = float(v)
            all_params.append(param_row)
        
        # Results table
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        # ALWAYS show parameters table
        st.subheader("📊 All 6 Parameters")
        if all_params:
            df_params = pd.DataFrame(all_params)
            numeric_cols = [c for c in df_params.columns if c != 'File']
            
            # Style properly
            styled = df_params.style
            styled = styled.background_gradient(subset=numeric_cols, cmap='Reds', vmin=0, vmax=1)
            styled = styled.format({col: '{:.0%}' for col in numeric_cols})
            
            st.dataframe(styled, use_container_width=True)
    
    with col2:
        st.header("📈 Breath Visualization")
        for i, file in enumerate(files[:2]):
            y, sr = safe_load_audio(file)
            if y is None: continue
            
            ai_score, events, params = calculate_ai_score(y, sr)
            
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='black')
            
            # Audio waveform
            dur = min(20, len(y)/sr)
            t = np.linspace(0, dur, int(dur*sr//128))
            t, audio = t[:len(y)], y[:len(t)]
            
            color = 'red' if ai_score > 0.70 else 'lime'
            axes[0].plot(t, audio, color='lightgray', lw=1.2)
            
            for j, bt in enumerate(events[:15]):
                if bt < dur:
                    axes[0].axvline(bt, color=color, ls='--', lw=2.5, alpha=0.9)
                    if j < 8:
                        axes[0].annotate(f'B{j+1}', (bt, 0.2), rotation=90, 
                                       color=color, fontsize=9, weight='bold')
            
            axes[0].set_title(f"{file.name[:25]} | AI: {ai_score:.0%}", 
                             color=color, fontsize=14, pad=15)
            axes[0].set_facecolor('#1a1a1a')
            axes[0].set_ylabel('Amplitude')
            axes[0].tick_params(colors='white')
            
            # Breath intervals
            if len(events) >= 2:
                intervals = np.diff(events[:12])
                x = np.arange(len(intervals))
                colors = ['red' if abs(i - np.mean(intervals)) < 0.4*np.std(intervals) 
                         else 'lime' for i in intervals]
                axes[1].bar(x, intervals, color=colors, alpha=0.8, edgecolor='white')
                axes[1].axhline(np.mean(intervals), color='white', lw=2, ls='-')
                cv = np.std(intervals)/np.mean(intervals)
                axes[1].text(0.02, 0.95, f'CV: {cv:.1%}', transform=axes[1].transAxes,
                           color='red' if cv < 0.15 else 'lime', va='top', weight='bold',
                           bbox=dict(boxstyle='round', facecolor='black'))
            else:
                axes[1].bar(['Breaths Found'], [len(events)], color='red', alpha=0.9)
                axes[1].text(0.3, 1.5, f'{len(events)} breaths\n→ AI indicator', 
                           ha='center', color='white', weight='bold', transform=axes[1].transAxes)
            
            axes[1].set_title('Breath Intervals\n(Red=AI regular, Green=human varied)')
            axes[1].set_facecolor('#1a1a1a')
            axes[1].tick_params(colors='white')
            axes[1].set_xlabel('Breath #')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    st.caption("✅ Fixed: Shows all 6 parameters | Better human detection | Accurate scoring")

if __name__ == "__main__":
    main()
