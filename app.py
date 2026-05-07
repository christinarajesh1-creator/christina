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
    """Safely load and normalize audio"""
    file.seek(0)
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True)
        if len(y) < 15000:
            return None, None
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.95
        return y, sr
    except:
        return None, None

def detect_breath_events(y, sr):
    """Improved breath detection - more sensitive to human patterns"""
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Find low-energy regions (silence + breaths)
    silence_threshold = np.percentile(rms, 35)  # More lenient
    silence_mask = rms < silence_threshold
    silence_rms = rms[silence_mask]
    silence_times = times[silence_mask]
    
    if len(silence_rms) < 10:
        return []
    
    # Find breath-like peaks in silence (relaxed parameters)
    breath_peaks, _ = find_peaks(
        silence_rms, 
        height=np.percentile(silence_rms, 70),  # Lower threshold
        distance=int(sr * 0.25),  # Allow closer breaths
        prominence=np.std(silence_rms) * 0.3    # Lower prominence
    )
    
    candidates = [silence_times[p] for p in breath_peaks 
                 if 1.5 < silence_times[p] < len(y)/sr - 3.0]
    
    # Verify with multiple features
    breaths = []
    for t in candidates:
        start = max(0, int((t - 0.4) * sr))
        end = min(len(y), int((t + 0.4) * sr))
        segment = y[start:end]
        
        # Check RMS pattern (breath has peak in middle of silence)
        seg_rms = librosa.feature.rms(y=segment, hop_length=256)[0]
        peak_loc = np.argmax(seg_rms)
        center = len(seg_rms) // 2
        
        # Breath signature: energy peak near center + reasonable amplitude
        if (abs(peak_loc - center) < len(seg_rms)*0.3 and 
            0.002 < np.max(seg_rms) < 0.3):
            breaths.append(t)
    
    # Natural spacing filter (humans have 2-6s between breaths)
    filtered = []
    for t in breaths:
        if not filtered or 1.8 < t - filtered[-1] < 8.0:
            filtered.append(t)
    
    return filtered[:20]

def calculate_ai_score(y, sr):
    """ACCURATELY calibrated 6 forensic parameters"""
    events = detect_breath_events(y, sr)
    n_breaths = len(events)
    
    # NORMAL breath count scoring (humans: 4-12 breaths per minute of speech)
    total_duration = len(y) / sr
    expected = max(1, total_duration / 5.0)  # ~1 breath every 5s
    count_score = 0.0
    
    if n_breaths == 0:
        count_score = 0.82
    elif n_breaths <= 1:
        count_score = 0.75
    elif n_breaths >= 18:
        count_score = 0.70
    elif abs(n_breaths - expected) / expected > 2.5:
        count_score = 0.65
    else:
        count_score = 0.05  # Normal count = human
    
    weights = [0.25, 0.18, 0.13, 0.18, 0.10, 0.16]
    
    if n_breaths < 3:
        return count_score, events, {'Count': count_score}
    
    scores = {'Count': count_score}
    
    # 1. IBI VARIABILITY (25%) - AI has CV < 0.15 (robotically regular)
    ibis = np.diff(events)
    cv_ibi = np.std(ibis) / np.mean(ibis)
    scores['IBI Var'] = 0.92 if cv_ibi < 0.15 else 0.03
    
    # 2. AMPLITUDE VARIATION (18%) - AI has CV < 0.15
    amps = []
    for t in events[:12]:
        start = max(0, int((t-0.2)*sr))
        end = min(len(y), int((t+0.3)*sr))
        peak = np.max(np.abs(y[start:end]))
        if peak > 0.0008:
            amps.append(peak)
    
    amp_score = 0.03
    if len(amps) >= 4:
        cv_amp = np.std(amps) / np.mean(amps)
        if cv_amp < 0.15:
            amp_score = 0.88
    
    scores['Amplitude'] = amp_score
    
    # 3. DURATION VARIATION (13%) - AI has CV < 0.18
    durs = []
    for t in events[:12]:
        start = max(0, int((t-0.3)*sr))
        end = min(len(y), int((t+0.6)*sr))
        if end > start:
            rms_seg = librosa.feature.rms(y=y[start:end], hop_length=256)[0]
            active = rms_seg > np.mean(rms_seg) * 0.6
            dur = np.sum(active) * 256 / sr
            if 0.1 < dur < 2.0:
                durs.append(dur)
    
    dur_score = 0.04
    if len(durs) >= 4:
        cv_dur = np.std(durs) / np.mean(durs)
        if cv_dur < 0.18:
            dur_score = 0.82
    
    scores['Duration'] = dur_score
    
    # 4. BREATH DENSITY (18%) - Already calculated above
    scores['Density'] = count_score
    
    # 5. SPECTRAL STABILITY (10%) - AI breaths sound identical
    zcr_sim = []
    for i in range(min(4, len(events)-1)):
        t1, t2 = events[i], events[i+1]
        s1, e1 = int(max(0,(t1-0.15)*sr)), int(min(len(y),(t1+0.25)*sr))
        s2, e2 = int(max(0,(t2-0.15)*sr)), int(min(len(y),(t2+0.25)*sr))
        if e1>s1 and e2>s2:
            z1 = librosa.feature.zero_crossing_rate(y[s1:e1]).mean()
            z2 = librosa.feature.zero_crossing_rate(y[s2:e2]).mean()
            zcr_sim.append(abs(z1-z2))
    
    spec_score = 0.08
    if zcr_sim:
        if np.mean(zcr_sim) < 0.08:  # Too similar
            spec_score = 0.78
    scores['Spectral'] = spec_score
    
    # 6. MFCC SIMILARITY (16%) - AI reuses breath samples
    mfcc_sim = []
    pairs_checked = 0
    for i in range(min(8, len(events))):
        for j in range(i+1, min(i+4, len(events))):
            pairs_checked += 1
            if pairs_checked > 12: break
            t1, t2 = events[i], events[j]
            s1, e1 = int(max(0,(t1-0.18)*sr)), int(min(len(y),(t1+0.28)*sr))
            s2, e2 = int(max(0,(t2-0.18)*sr)), int(min(len(y),(t2+0.28)*sr))
            try:
                if e1>s1 and e2>s2 and e1-s1>1000 and e2-s2>1000:
                    mfcc1 = librosa.feature.mfcc(y=y[s1:e1], sr=sr, n_mfcc=8).mean(1)
                    mfcc2 = librosa.feature.mfcc(y=y[s2:e2], sr=sr, n_mfcc=8).mean(1)
                    if len(mfcc1)>4 and len(mfcc2)>4:
                        corr = np.corrcoef(mfcc1, mfcc2)[0,1]
                        mfcc_sim.append(abs(corr) if not np.isnan(corr) else 0)
            except:
                continue
    
    sim_score = 0.06
    if mfcc_sim:
        avg_corr = np.mean(mfcc_sim)
        if avg_corr > 0.85:  # Breaths too similar
            sim_score = 0.90
    scores['Similarity'] = sim_score
    
    # FINAL WEIGHTED SCORE
    param_names = ['IBI Var', 'Amplitude', 'Duration', 'Density', 'Spectral', 'Similarity']
    ai_scores = np.array([scores[name] for name in param_names])
    final_score = np.average(ai_scores, weights=weights)
    
    return final_score, events, scores

def main():
    st.title("🫁 PneumaForensic AI Detector v2.0")
    st.markdown("""
    **Calibrated for accuracy:**
    - **HUMAN**: 4-12 breaths, irregular timing (CV>20%), varied characteristics
    - **AI**: 0-2 breaths, robotic regularity (CV<15%), identical breaths
    
    **6 Parameters**: IBI Var(25%) | Amp(18%) | Dur(13%) | Density(18%) | Spectral(10%) | Similarity(16%)
    """)
    
    files = st.file_uploader("Upload audio", type=['wav','mp3','m4a','aac'], accept_multiple_files=True)
    
    if not files:
        st.info("👆 Upload files to analyze")
        return
    
    col1, col2 = st.columns([1.1, 1])
    
    with col1:
        st.header("📊 Results")
        results = []
        details = []
        
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                results.append({"File": file.name[:35], "AI Score": "ERROR", "Status": "❌", "Breaths": 0})
                continue
            
            ai_score, events, params = calculate_ai_score(y, sr)
            status = "🤖 **AI**" if ai_score > 0.72 else "👤 **HUMAN**"
            
            results.append({
                "File": file.name[:35],
                "AI Score": f"{ai_score:.0%}",
                "Status": status,
                "Breaths": len(events)
            })
            
            details.append({"File": file.name[:25], **{k: v for k, v in params.items()}})
        
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        
        if details:
            st.subheader("🔬 Parameter Details")
            df_details = pd.DataFrame(details)
            num_cols = [col for col in df_details.columns if col != 'File']
            
            # Convert to numeric then style
            df_num = df_details[num_cols].apply(pd.to_numeric, errors='coerce')
            styled = df_num.style.background_gradient(cmap='Reds_r', vmin=0, vmax=1, axis=0)
            styled = styled.format('{:.0%}')
            
            # Add file column back
            display_df = pd.concat([df_details[['File']], df_num], axis=1)
            st.dataframe(display_df.style.format({col: '{:.0%}' for col in num_cols}), 
                        use_container_width=True, hide_index=True)
    
    with col2:
        st.header("🎵 Waveform Analysis")
        for file in files[:2]:  # Limit to 2 plots
            y, sr = safe_load_audio(file)
            if y is None: continue
            
            ai_score, events, _ = calculate_ai_score(y, sr)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor='black')
            
            # Waveform
            duration = min(25, len(y)/sr)
            t = np.linspace(0, duration, int(duration*sr//100))
            t, y_plot = t[:len(y)], y[:len(t)]
            
            color = '#FF4444' if ai_score > 0.72 else '#44FF44'
            ax1.plot(t, y_plot, color='#CCCCCC', lw=1.5, alpha=0.9)
            
            for i, bt in enumerate(events[:12]):
                if bt < duration:
                    ax1.axvline(bt, color=color, ls='--', lw=3, alpha=0.9)
                    if i < 6:
                        ax1.text(bt, 0.15, f'B{i+1}', rotation=90, color=color, 
                               va='bottom', fontsize=10, weight='bold')
            
            ax1.set_title(f"{file.name[:25]} | AI: {ai_score:.0%} | {len(events)} breaths", 
                         color=color, fontsize=14, pad=20, weight='bold')
            ax1.set_facecolor('#111111')
            ax1.set_ylabel('Amplitude')
            ax1.tick_params(colors='white')
            ax1.grid(alpha=0.3, color='gray')
            
            # Intervals
            if len(events) >= 2:
                ints = np.diff(events[:12])
                bars = ax2.bar(range(len(ints)), ints, color=['#FF6666' if abs(i-np.mean(ints))<0.5*np.std(ints) 
                                     else '#66FF66' for i in ints], alpha=0.8, edgecolor='white')
                ax2.axhline(np.mean(ints), color='white', lw=2, ls='-', label=f'Mean: {np.mean(ints):.1f}s')
                cv = np.std(ints)/np.mean(ints)
                ax2.text(0.02, 0.95, f'CV: {cv:.1%}', transform=ax2.transAxes, 
                        color='#FF4444' if cv<0.20 else '#44FF44', va='top', weight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
            else:
                ax2.bar(['Breaths'], [len(events)], color='#FF4444', alpha=0.9)
                ax2.text(0.1, 1.5, f'{len(events)} breaths\nHIGH AI SCORE', ha='center', 
                        color='white', weight='bold', transform=ax2.transAxes)
            
            ax2.set_facecolor('#111111')
            ax2.set_title('Breath Intervals (Red=Regular/AI-like)')
            ax2.tick_params(colors='white')
            ax2.grid(alpha=0.3, color='gray')
            if len(events) >= 2:
                ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    st.markdown("*Accurate forensic analysis using 6 breath parameters*")

if __name__ == "__main__":
    main()
