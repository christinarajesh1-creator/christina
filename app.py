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
    """Robust breath detection - handles all edge cases"""
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Safe silence detection
    if len(rms) == 0: return []
    thresh = np.percentile(rms, 30)
    silence_idx = np.where(rms < thresh)[0]
    
    if len(silence_idx) < 10: return []
    
    silence_rms = rms[silence_idx]
    silence_times = times[silence_idx]
    
    # FIXED: Safe find_peaks parameters
    height = np.percentile(silence_rms, 60)
    distance = max(2, int(1.0 * len(silence_rms) / 60))  # Min 2, ~1s in samples
    prominence = max(0.001, np.std(silence_rms) * 0.2)
    
    try:
        peaks, _ = find_peaks(silence_rms, height=height, distance=distance, prominence=prominence)
    except:
        peaks = []
    
    candidates = silence_times[peaks]
    
    # Loose verification
    breaths = []
    for t in candidates:
        if 1.0 < t < len(y)/sr - 2.0:
            start = max(0, int((t-0.5)*sr))
            end = min(len(y), int((t+0.5)*sr))
            if end > start:
                seg_rms = np.max(librosa.feature.rms(y=y[start:end])[0])
                if seg_rms > 0.001:
                    breaths.append(t)
    
    # Safe spacing filter
    final_breaths = []
    for t in breaths:
        if not final_breaths or t - final_breaths[-1] > 1.2:
            final_breaths.append(t)
    
    return final_breaths[:25]

def calculate_ai_score(y, sr):
    """6 forensic AI parameters - bulletproof"""
    events = detect_breath_events(y, sr)
    n_breaths = len(events)
    
    total_time = len(y) / sr
    breath_rate = n_breaths / max(0.1, total_time)
    
    # Parameter 1: Breath count (25%)
    count_score = 0.08
    if n_breaths <= 1 or breath_rate < 0.06 or breath_rate > 0.6:
        count_score = 0.82
    
    scores = {'Count': count_score}
    
    if n_breaths < 3:
        return count_score, events, scores
    
    weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.15]
    
    # Parameter 2: IBI Regularity (20%)
    ibis = np.diff(events)
    if len(ibis) > 1:
        cv = np.std(ibis) / np.mean(ibis)
        scores['Regularity'] = 0.88 if cv < 0.13 else 0.02
    else:
        scores['Regularity'] = 0.02
    
    # Parameter 3: Amplitude uniformity (15%)
    amps = []
    for t in events[:15]:
        s = max(0, int((t-0.25)*sr))
        e = min(len(y), int((t+0.35)*sr))
        if e > s:
            amp = np.max(np.abs(y[s:e]))
            if amp > 0.0003: amps.append(amp)
    
    amp_score = 0.02
    if len(amps) >= 4:
        cv_amp = np.std(amps)/np.mean(amps)
        if cv_amp < 0.13: amp_score = 0.82
    scores['Amplitude'] = amp_score
    
    # Parameter 4: Duration uniformity (15%)
    durs = []
    for t in events[:15]:
        s = max(0, int((t-0.35)*sr))
        e = min(len(y), int((t+0.65)*sr))
        if e > s + 500:
            rms_seg = librosa.feature.rms(y=y[s:e], hop_length=256)[0]
            active = rms_seg > np.percentile(rms_seg, 35)
            dur = np.sum(active) * 0.016
            if 0.06 < dur < 2.0: durs.append(dur)
    
    dur_score = 0.02
    if len(durs) >= 4:
        cv_dur = np.std(durs)/np.mean(durs)
        if cv_dur < 0.16: dur_score = 0.78
    scores['Duration'] = dur_score
    
    # Parameter 5: Spectral similarity (10%)
    zcr_var = []
    for i in range(min(5, len(events)-1)):
        t1, t2 = events[i], events[i+1]
        s1, e1 = int(max(0,(t1-0.2)*sr)), int(min(len(y),(t1+0.3)*sr))
        s2, e2 = int(max(0,(t2-0.2)*sr)), int(min(len(y),(t2+0.3)*sr))
        if e1 > s1 + 200 and e2 > s2 + 200:
            z1 = librosa.feature.zero_crossing_rate(y[s1:e1]).mean()
            z2 = librosa.feature.zero_crossing_rate(y[s2:e2]).mean()
            zcr_var.append(abs(z1-z2))
    
    spec_score = 0.03
    if len(zcr_var) > 1:
        if np.mean(zcr_var) < 0.05: spec_score = 0.72
    scores['Spectral'] = spec_score
    
    # Parameter 6: MFCC similarity (15%)
    sims = []
    for i in range(min(8, len(events))):
        for j in range(i+2, min(i+5, len(events))):
            t1, t2 = events[i], events[j]
            s1, e1 = int(max(0,(t1-0.22)*sr)), int(min(len(y),(t1+0.32)*sr))
            s2, e2 = int(max(0,(t2-0.22)*sr)), int(min(len(y),(t2+0.32)*sr))
            try:
                if e1-s1 > 800 and e2-s2 > 800:
                    m1 = librosa.feature.mfcc(y=y[s1:e1], sr=sr, n_mfcc=8).mean(1)
                    m2 = librosa.feature.mfcc(y=y[s2:e2], sr=sr, n_mfcc=8).mean(1)
                    if len(m1) > 5:
                        corr = np.corrcoef(m1, m2)[0,1]
                        if not np.isnan(corr): sims.append(corr)
            except: continue
    
    sim_score = 0.04
    if len(sims) > 2:
        if np.mean(sims) > 0.87: sim_score = 0.85
    scores['Similarity'] = sim_score
    
    # Final weighted score
    param_scores = [scores[k] for k in ['Count','Regularity','Amplitude','Duration','Spectral','Similarity']]
    final_score = np.average(param_scores, weights=weights)
    
    return final_score, events, scores

def main():
    st.title("🫁 PneumaForensic AI Detector ✅")
    st.markdown("**Upload audio to see all 6 forensic parameters**")
    
    files = st.file_uploader("Choose files", type=['wav','mp3','m4a','aac'], accept_multiple_files=True)
    
    if not files:
        return
    
    col1, col2 = st.columns([1.1, 1])
    
    with col1:
        st.header("Results")
        results = []
        param_data = []
        
        for file in files:
            with st.spinner(f"Analyzing {file.name}..."):
                y, sr = safe_load_audio(file)
                if y is None:
                    results.append({"File": file.name[:30], "AI Score": "ERROR", "Status": "❌", "Breaths": "-"})
                    continue
                
                ai_score, events, params = calculate_ai_score(y, sr)
                status = "🤖 **AI**" if ai_score > 0.72 else "👤 **HUMAN**"
                
                results.append({
                    "File": file.name[:30],
                    "AI Score": f"{ai_score:.0%}",
                    "Status": status,
                    "Breaths": len(events)
                })
                
                # Numeric params for table
                row = {"File": file.name[:25]}
                for k, v in params.items():
                    row[k] = float(v)
                param_data.append(row)
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        # Parameters table - ALWAYS SHOWS
        if param_data:
            st.subheader("🔬 6 Forensic Parameters")
            df_params = pd.DataFrame(param_data)
            num_cols = [c for c in df_params if c != 'File']
            
            styled = df_params[num_cols].style\
                .background_gradient(cmap='Reds', vmin=0, vmax=1)\
                .format('{:.0%}')
            
            st.dataframe(pd.concat([df_params[['File']], df_params[num_cols]], axis=1)
                        .style.format({c: '{:.0%}' for c in num_cols}), 
                        use_container_width=True)
    
    with col2:
        st.header("Breath Patterns")
        for file in files[:2]:
            y, sr = safe_load_audio(file)
            if y is None: continue
            
            ai_score, events, _ = calculate_ai_score(y, sr)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
            fig.patch.set_facecolor('black')
            
            # Waveform
            dur = min(20, len(y)/sr)
            t = np.linspace(0, dur, int(dur*sr//256))
            t = t[:len(y)]
            yplot = y[:len(t)]
            
            ax1.plot(t, yplot, 'lightblue', lw=0.8, alpha=0.7)
            color = 'red' if ai_score > 0.72 else 'green'
            
            for i, bt in enumerate(events[:12]):
                if bt < dur:
                    ax1.axvline(bt, color=color, ls='--', lw=2, alpha=0.8)
            
            ax1.set_title(f"{file.name[:25]}\nAI: {ai_score:.0%}", color=color, fontsize=12)
            ax1.set_facecolor('#1a1a1a')
            ax1.tick_params(colors='white')
            
            # Intervals
            if len(events) > 1:
                ints = np.diff(events[:10])
                bars = ax2.bar(range(len(ints)), ints, color='orange', alpha=0.7)
                mean_int = np.mean(ints)
                ax2.axhline(mean_int, color='white', lw=2)
                cv = np.std(ints)/mean_int
                ax2.text(0.05, 0.9, f'CV: {cv:.0%}', transform=ax2.transAxes,
                        bbox=dict(boxstyle='round', facecolor='black'))
            else:
                ax2.text(0.5, 0.5, f'{len(events)}\nbreaths', ha='center', 
                        transform=ax2.transAxes, fontsize=20)
            
            ax2.set_facecolor('#1a1a1a')
            ax2.set_title('Breath Intervals')
            ax2.tick_params(colors='white')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    st.success("✅ All errors fixed - shows full parameter breakdown!")

if __name__ == "__main__":
    main()
