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
        if len(y) < 15000:  # Minimum 1 second
            return None, None
        # Normalize to prevent amplitude bias
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.95
        return y, sr
    except Exception as e:
        st.error(f"Load failed: {str(e)[:80]}")
        return None, None

def detect_breath_events(y, sr):
    """Detect real human-like breaths in silence regions"""
    hop_length = 256
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Step 1: Find silence regions (potential breath locations)
    silence_threshold = np.percentile(rms, 28)
    silence_mask = rms < silence_threshold
    silence_rms = rms[silence_mask]
    silence_times = times[silence_mask]
    
    if len(silence_rms) < 20:
        return []
    
    # Step 2: Find energy bursts WITHIN silence (actual breaths)
    breath_peaks, properties = find_peaks(
        silence_rms, 
        height=np.percentile(silence_rms, 78),
        distance=int(sr * 0.35),  # Min 0.35s between breaths
        prominence=np.std(silence_rms) * 0.6
    )
    
    candidate_events = [silence_times[p] for p in breath_peaks 
                       if 2.0 < silence_times[p] < len(y)/sr - 4.0]
    
    # Step 3: Verify each candidate is a real breath
    verified_breaths = []
    for t in candidate_events:
        # Extract 0.6s window around candidate
        start_frame = max(0, int((t - 0.3) * sr))
        end_frame = min(len(y), int((t + 0.3) * sr))
        breath_segment = y[start_frame:end_frame]
        
        # Breath characteristics check
        breath_rms_peak = np.max(librosa.feature.rms(y=breath_segment)[0])
        breath_duration = 0.6
        
        # REAL BREATH: moderate energy burst in silence (not speech, not pure silence)
        if 0.006 < breath_rms_peak < 0.15:
            verified_breaths.append(t)
    
    # Final temporal filtering
    filtered_breaths = [verified_breaths[0]] if verified_breaths else []
    for t in verified_breaths[1:]:
        if t - filtered_breaths[-1] > 0.5:
            filtered_breaths.append(t)
    
    return filtered_breaths[:16]  # Cap at 16 breaths

def calculate_ai_score(events, y, sr):
    """6 forensic parameters with exact weights - AI traits only"""
    weights = np.array([0.28, 0.15, 0.12, 0.15, 0.12, 0.18])
    
    # Base score for breath count
    n_breaths = len(events)
    if n_breaths == 0:
        return 0.92, events, {'Breath Count': 0.92}
    if n_breaths <= 1:
        return 0.78, events, {'Breath Count': 0.78}
    
    scores = {}
    
    # 1. IBI REGULARITY (28%) - AI: CV < 0.28 (too regular)
    ibis = np.diff(events)
    cv_ibi = np.std(ibis) / np.mean(ibis)
    scores['IBI Regularity'] = 1.0 if cv_ibi < 0.28 else 0.05
    
    # 2. BREATH AMPLITUDE (15%) - AI: uniform volume (CV < 0.22)
    amps = []
    for t in events[:12]:
        start = max(0, int((t-0.14)*sr))
        end = min(len(y), int((t+0.24)*sr))
        peak_amp = np.max(np.abs(y[start:end]))
        if peak_amp > 0.001:
            amps.append(peak_amp)
    
    amp_ai = 0.9 if len(amps) >= 3 and np.std(amps)/np.mean(amps) < 0.22 else 0.1
    scores['Amplitude'] = amp_ai
    
    # 3. BREATH DURATION (12%) - AI: copy-pasted lengths (CV < 0.27)
    durs = []
    for t in events[:12]:
        start = max(0, int((t-0.22)*sr))
        end = min(len(y), int((t+0.45)*sr))
        seg_rms = librosa.feature.rms(y=y[start:end], frame_length=512, hop_length=128)[0]
        active_mask = seg_rms > np.percentile(seg_rms, 62)
        duration = np.sum(active_mask) * 128 / sr
        if 0.07 < duration < 1.1:
            durs.append(duration)
    
    dur_ai = 0.85 if len(durs) >= 3 and np.std(durs)/np.mean(durs) < 0.27 else 0.12
    scores['Duration'] = dur_ai
    
    # 4. BREATH PRESENCE (15%) - AI: unnaturally low (or high)
    total_duration = len(y) / sr
    expected_breaths = total_duration / 4.5  # ~1 breath every 4.5s
    presence_ratio = n_breaths / expected_breaths
    scores['Presence'] = 0.88 if presence_ratio < 0.25 or presence_ratio > 3.0 else 0.08
    
    # 5. SPECTRAL CONTINUITY (12%) - AI: abrupt ZCR changes (>0.12)
    zcr_changes = []
    for i in range(min(6, len(events)-1)):
        t1, t2 = events[i], events[i+1]
        s1, e1 = int(max(0, (t1-0.11)*sr)), int(min(len(y), (t1+0.16)*sr))
        s2, e2 = int(max(0, (t2-0.11)*sr)), int(min(len(y), (t2+0.16)*sr))
        zcr1 = librosa.feature.zero_crossing_rate(y[s1:e1])[0].mean()
        zcr2 = librosa.feature.zero_crossing_rate(y[s2:e2])[0].mean()
        zcr_changes.append(abs(zcr1 - zcr2))
    
    spec_ai = 0.75 if zcr_changes and np.mean(zcr_changes) > 0.12 else 0.15
    scores['Spectral'] = spec_ai
    
    # 6. BREATH SIMILARITY (18%) - AI: reuses samples (high MFCC correlation)
    similarities = []
    for i in range(min(7, len(events))):
        for j in range(i+1, min(i+4, len(events))):
            t1, t2 = events[i], events[j]
            s1, e1 = int(max(0, (t1-0.13)*sr)), int(min(len(y), (t1+0.19)*sr))
            s2, e2 = int(max(0, (t2-0.13)*sr)), int(min(len(y), (t2+0.19)*sr))
            try:
                mfcc1 = librosa.feature.mfcc(y=y[s1:e1], sr=sr, n_mfcc=13).mean(axis=1)
                mfcc2 = librosa.feature.mfcc(y=y[s2:e2], sr=sr, n_mfcc=13).mean(axis=1)
                if len(mfcc1) > 3 and len(mfcc2) > 3:
                    corr = abs(np.corrcoef(mfcc1[:8], mfcc2[:8])[0,1])
                    similarities.append(1 - corr)  # High similarity = AI
            except:
                similarities.append(0.4)
    
    sim_ai = 0.92 if similarities and np.mean(similarities) < 0.18 else 0.09
    scores['Similarity'] = sim_ai
    
    # FINAL WEIGHTED AI SCORE
    ai_scores = np.array([scores['IBI Regularity'], scores['Amplitude'], scores['Duration'],
                         scores['Presence'], scores['Spectral'], scores['Similarity']])
    final_score = np.average(ai_scores, weights=weights)
    
    return final_score, events, scores

def main():
    st.title("🫁 PneumaForensic AI Detector")
    st.markdown("""
    **How it works:**
    - **Gray waves** = Your speech
    - **Red dashed lines** = Detected breaths  
    - **HUMAN** = Irregular spacing, 6-14 breaths, varied heights
    - **AI** = Perfect grid spacing, 0-3 breaths, uniform everything
    
    **6 Forensic Parameters (weighted):**
    28% IBI Regularity | 15% Amplitude | 12% Duration | 15% Presence | 12% Spectral | 18% Similarity
    """)
    
    files = st.file_uploader("Upload WAV/MP3/M4A", 
                            type=['wav','mp3','m4a','aac'], 
                            accept_multiple_files=True)
    
    if not files:
        st.info("👆 Upload audio files to analyze")
        return
    
    col1, col2 = st.columns([1.1, 1])
    
    with col1:
        st.header("📊 Detection Results")
        results = []
        detailed_scores = []
        
        for file in files:
            y, sr = safe_load_audio(file)
            
            if y is None:
                results.append({
                    "File": file.name[:35], 
                    "AI Score": "LOAD ERROR", 
                    "Status": "❌", 
                    "Breaths": 0
                })
                continue
            
            ai_score, events, params = calculate_ai_score(events=None, y=y, sr=sr)
            
            status = "🤖 **AI GENERATED**" if ai_score > 0.65 else "👤 **HUMAN**"
            
            results.append({
                "File": file.name[:35],
                "AI Score": f"{ai_score:.0%}",
                "Status": status,
                "Breaths": len(events)
            })
            
            detailed_scores.append({
                "File": file.name[:25],
                "AI": f"{ai_score:.0%}",
                **{k: f"{v:.0%}" for k, v in params.items()}
            })
        
        # Main results table
        df_results = pd.DataFrame(results)
        st.dataframe(df_results.style.highlight_max(subset=['AI Score'], color='salmon'), 
                    use_container_width=True)
        
        # Detailed parameter table
        if detailed_scores:
            st.subheader("🔬 Parameter Breakdown")
            df_params = pd.DataFrame(detailed_scores)
            numeric_cols = [col for col in df_params.columns if col != 'File']
            styled_df = df_params.style.format({col: '{:.0%}' for col in numeric_cols})
            styled_df = styled_df.background_gradient(subset=numeric_cols, cmap='Reds_r')
            st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        st.header("🎵 Audio Forensics")
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                st.error(f"Could not process {file.name}")
                continue
            
            ai_score, events, _ = calculate_ai_score(events=None, y=y, sr=sr)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), facecolor='black')
            
            # Plot 1: Speech waveform with breath markers
            duration = min(30, len(y)/sr)
            t = np.linspace(0, duration, int(duration * sr // 120))
            if len(t) > len(y):
                t = t[:len(y)]
            y_plot = y[:len(t)]
            
            ax1.plot(t, y_plot, color='#D3D3D3', linewidth=1.6, alpha=0.95)
            
            breath_color = '#FF4444' if ai_score > 0.65 else '#44FF44'
            line_alpha = 0.92 if ai_score > 0.65 else 0.88
            
            for i, breath_time in enumerate(events[:14]):
                if breath_time < duration:
                    ax1.axvline(breath_time, color=breath_color, linestyle='--', 
                               linewidth=4, alpha=line_alpha)
                    # Label first 8 breaths
                    if i < 8:
                        ax1.annotate(f'B{i+1}', (breath_time, 0.12), 
                                   xytext=(8, 12), textcoords='offset points',
                                   fontsize=11, color=breath_color, 
                                   rotation=90, weight='bold')
            
            title_color = '#FF4444' if ai_score > 0.65 else '#44FF44'
            ax1.set_title(f"{file.name[:28]}...\nAI Probability: {ai_score:.0%} | {len(events)} breaths", 
                         color=title_color, fontsize=16, pad=25, weight='bold')
            ax1.set_facecolor('#0A0A0A')
            ax1.set_ylabel("Amplitude", color='white', fontsize=13)
            ax1.tick_params(colors='white', labelsize=10)
            ax1.grid(True, alpha=0.2, color='gray')
            
            # Plot 2: Breath interval analysis
            if len(events) >= 2:
                intervals = np.diff(events[:14])
                bar_positions = np.arange(len(intervals))
                regularity_colors = ['#FF6666' if abs(interval - np.mean(intervals)) < np.std(intervals)*0.35 
                                   else '#66FF66' for interval in intervals]
                
                bars = ax2.bar(bar_positions, intervals, color=regularity_colors, 
                              alpha=0.85, width=0.75, edgecolor='white', linewidth=0.5)
                
                mean_interval = np.mean(intervals)
                ax2.axhline(mean_interval, color='white', linestyle='-', linewidth=3, 
                           label=f'Mean: {mean_interval:.1f}s')
                
                cv = np.std(intervals) / mean_interval
                regularity_ai = 1.0 if cv < 0.28 else 0.0
                ax2.text(0.03, 0.97, f'Regularity Score: {regularity_ai:.0%} AI', 
                        transform=ax2.transAxes, color='#FF4444' if regularity_ai > 0.5 else '#44FF44',
                        va='top', fontsize=12, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='black', alpha=0.9))
            else:
                ax2.bar(['Breaths\nDetected'], [len(events)], color='#FF4444', alpha=0.9, width=0.6)
                ax2.set_ylim(0, 3)
                breath_ai = 0.9 if len(events) < 2 else 0.1
                ax2.text(0.05, 1.8, f'{len(events)} breaths\nAI Score: {breath_ai:.0%}', 
                        transform=ax2.transAxes, color='#FF4444', fontsize=14, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.9))
            
            ax2.set_title("Breath Intervals\n(Red=Regular/AI-like, Green=Varied/Human-like)", 
                         color='white', fontsize=14, pad=15, weight='bold')
            ax2.set_facecolor('#0A0A0A')
            ax2.set_xlabel("Breath Number", color='white', fontsize=12)
            ax2.set_ylabel("Interval (seconds)", color='white', fontsize=12)
            ax2.tick_params(colors='white', labelsize=10)
            ax2.grid(True, alpha=0.2, color='gray')
            if len(events) >= 2:
                ax2.legend(loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    st.markdown("*Powered by 6 weighted forensic breath parameters*")

if __name__ == "__main__":
    main()
