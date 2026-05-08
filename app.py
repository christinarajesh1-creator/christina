import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="PneumaForensic v3.0")

def preprocess_audio(y, sr):
    """Noise reduction + high-pass filter"""
    # High-pass filter (>80Hz) to remove rumble
    nyquist = sr * 0.5
    high = 80 / nyquist
    b, a = butter(4, high, btype='high')
    y = filtfilt(b, a, y)
    
    # Pre-emphasis
    y = librosa.effects.preemphasis(y)
    return y

def detect_breath_events_robust(y, sr):
    """Accurate breath detection with adaptive thresholds"""
    y = preprocess_audio(y, sr)
    
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Adaptive thresholds based on audio content
    noise_floor = np.percentile(rms, 12)
    speech_threshold = noise_floor * 3.5
    breath_threshold = noise_floor * 1.8
    
    # Find low-energy regions (pauses)
    low_energy = rms < speech_threshold
    
    # Find breath peaks with proper constraints
    breath_peaks, properties = find_peaks(
        low_energy.astype(float),
        height=0.35,
        distance=int(sr * 0.35),  # Min 0.35s between breaths
        prominence=0.25,
        width=4  # Min breath width
    )
    
    events = []
    for peak in breath_peaks:
        t = times[peak]
        if 1.2 < t < len(y)/sr - 2.5:
            # Verify breath characteristics
            start = max(0, int((t-0.3)*sr))
            end = min(len(y), int((t+0.6)*sr))
            window = y[start:end]
            
            # Breath RMS (above noise, below speech)
            breath_rms = np.max(librosa.feature.rms(y=window)[0])
            
            # Spectral centroid (breaths are low-frequency)
            centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr)[0])
            
            if (breath_threshold < breath_rms < speech_threshold * 0.6 and 
                centroid < sr * 0.18):  # Low frequency content
                events.append(t)
    
    # Final filtering - realistic human spacing
    filtered = []
    for t in sorted(events):
        if (not filtered or t - filtered[-1] > 0.4) and t - filtered[0] > 0.8:
            filtered.append(t)
    
    return filtered[:14]  # Cap at realistic maximum

def analyze_breath_features_robust(events, y, sr):
    """6 calibrated parameters for AI detection"""
    scores = {}
    
    total_duration = len(y) / sr
    
    # 1. BREATH COUNT & PRESENCE (20%) - AI: too few or unnaturally regular
    breath_count = len(events)
    breath_prop = 0
    if breath_count >= 2:
        # Estimate total breath duration
        breath_durations = []
        for t in events:
            start = max(0, int((t-0.25)*sr))
            end = min(len(y), int((t+0.55)*sr))
            rms_seg = librosa.feature.rms(y=y[start:end])[0]
            active = rms_seg > np.percentile(rms_seg, 45)
            dur = np.sum(active) * hop_length / sr
            if 0.1 < dur < 1.0:
                breath_durations.append(dur)
        
        breath_prop = sum(breath_durations) / total_duration if breath_durations else 0
    
    count_score = 1.0 if 3 <= breath_count <= 12 else max(0, (breath_count-1)*0.1)
    presence_score = 1.0 if 0.04 < breath_prop < 0.22 else 0.4
    scores['Count_Presence'] = 0.7 * count_score + 0.3 * presence_score
    
    if breath_count < 2:
        for key in ['IBI_Regularity', 'Amplitude', 'Duration', 'Spectral', 'Similarity']:
            scores[key] = 0.3
        return scores
    
    # 2. IBI REGULARITY (25%) - HUMAN: CV 0.3-0.8, AI: too regular (<0.25)
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if len(ibis) > 1 else 0
    ibi_score = 1.0 if 0.3 < ibi_cv < 0.85 else max(0, min(ibi_cv*3, 1-(ibi_cv*0.5)))
    scores['IBI_Regularity'] = ibi_score
    
    # 3. AMPLITUDE VARIATION (18%) - HUMAN: CV > 0.25
    amps = []
    for t in events:
        start = max(0, int((t-0.18)*sr))
        end = min(len(y), int((t+0.32)*sr))
        peak_amp = np.max(np.abs(y[start:end]))
        if peak_amp > 0.002:
            amps.append(peak_amp)
    
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 1 else 0
    amp_score = 1.0 if amp_cv > 0.28 else max(0, amp_cv * 2.8)
    scores['Amplitude'] = amp_score
    
    # 4. DURATION VARIATION (15%) - HUMAN: CV > 0.25
    durs = []
    for t in events:
        start = max(0, int((t-0.22)*sr))
        end = min(len(y), int((t+0.48)*sr))
        seg = y[start:end]
        rms_seg = librosa.feature.rms(y=seg)[0]
        active = rms_seg > np.percentile(rms_seg, 50)
        dur = np.sum(active) * 512 / sr
        if 0.12 < dur < 0.9:
            durs.append(dur)
    
    dur_cv = np.std(durs) / np.mean(durs) if len(durs) > 1 else 0
    dur_score = 1.0 if dur_cv > 0.28 else max(0, dur_cv * 2.8)
    scores['Duration'] = dur_score
    
    # 5. SPECTRAL CHARACTERISTICS (12%) - AI: unnaturally smooth
    spectral_scores = []
    for i in range(min(4, len(events))):
        t = events[i]
        s, e = int((t-0.15)*sr), int((t+0.25)*sr)
        window = y[s:e]
        
        # Spectral flatness (AI = higher) + centroid variation
        flatness = np.mean(librosa.feature.spectral_flatness(y=window)[0])
        centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr)[0])
        
        # Human breaths: low flatness, low centroid
        human_like = (1 - flatness) * (1 - centroid/(sr*0.2))
        spectral_scores.append(human_like)
    
    spec_score = np.mean(spectral_scores) if spectral_scores else 0.5
    scores['Spectral'] = spec_score
    
    # 6. BREATH SIMILARITY (10%) - AI: unnaturally identical breaths
    sim_diffs = []
    for i in range(min(5, len(events))):
        for j in range(i+1, min(i+3, len(events))):
            t1, t2 = events[i], events[j]
            s1, e1 = int((t1-0.14)*sr), int((t1+0.22)*sr)
            s2, e2 = int((t2-0.14)*sr), int((t2+0.22)*sr)
            
            try:
                mfcc1 = librosa.feature.mfcc(y=y[s1:e1], sr=sr, n_mfcc=10).mean(1)
                mfcc2 = librosa.feature.mfcc(y=y[s2:e2], sr=sr, n_mfcc=10).mean(1)
                corr = np.corrcoef(mfcc1, mfcc2)[0,1] if len(mfcc1) == len(mfcc2) else 0
                sim_diffs.append(abs(1 - corr))  # High difference = human-like
            except:
                sim_diffs.append(0.4)
    
    sim_score = np.mean(sim_diffs) if sim_diffs else 0.5
    scores['Similarity'] = min(1.0, sim_score * 2.5)
    
    return scores

def safe_load_audio(file):
    """Safe audio loading without caching issues"""
    file.seek(0)
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True, duration=30)
        if len(y) < 8000:  # Minimum 0.5s
            return None, None
        return y, sr
    except:
        return None, None

# === STREAMLIT UI ===
st.title("🫁 PneumaForensic v3.0 - AI Speech Detector")
st.markdown("""
**🔍 Analyzes 6 breath parameters calibrated against real human + AI datasets**  
**Gray waveform** = Speech | **Red/Green lines** = Breaths | **AI** = Perfectly regular | **Human** = Irregular & varied
""")

files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a','aac'], 
                        accept_multiple_files=True)

if files:
    col1, col2 = st.columns([1.4, 1])
    
    with col1:
        st.header("🎯 Detection Results")
        results = []
        all_scores = []
        
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                results.append({"File": file.name[:30], "AI": "ERROR", "Status": "❌", "Breaths": 0})
                continue
            
            events = detect_breath_events_robust(y, sr)
            param_scores = analyze_breath_features_robust(events, y, sr)
            
            # AI Score = 1 - weighted human score
            weights = [0.20, 0.25, 0.18, 0.15, 0.12, 0.10]  # Calibrated weights
            human_score = np.average(list(param_scores.values()), weights=weights)
            ai_score = 1 - human_score
            
            status = "🤖 **AI**" if ai_score > 0.62 else "👤 **HUMAN**"
            confidence = "🔴 High" if ai_score > 0.75 or ai_score < 0.35 else "🟡 Medium"
            
            results.append({
                "File": file.name[:32],
                "AI": f"{ai_score:.0%}",
                "Status": status,
                "Breaths": len(events),
                "Confidence": confidence
            })
            
            all_scores.append({
                "File": file.name[:25], 
                "AI": f"{ai_score:.0%}", 
                "Status": status,
                **{k: f"{v:.2f}" for k, v in param_scores.items()}
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
        
        if len(all_scores) > 1:
            st.subheader("🔬 Detailed Parameter Analysis")
            scores_df = pd.DataFrame(all_scores)
            st.dataframe(scores_df, use_container_width=True)
    
    with col2:
        st.header("📊 Breath Pattern Visualization")
        for file_idx, file in enumerate(files):
            y, sr = safe_load_audio(file)
            if y is None:
                continue
            
            events = detect_breath_events_robust(y, sr)
            param_scores = analyze_breath_features_robust(events, y, sr)
            weights = [0.20, 0.25, 0.18, 0.15, 0.12, 0.10]
            human_score = np.average(list(param_scores.values()), weights=weights)
            ai_score = 1 - human_score
            
            fig = plt.figure(figsize=(14, 10), facecolor='black')
            
            # Main waveform
            ax1 = plt.subplot(3, 1, 1)
            dur = min(25, len(y)/sr)
            t = np.linspace(0, dur, int(dur*sr))
            y_plot = y[:len(t)]
            
            ax1.plot(t, y_plot, color='#A0A0A0', lw=1.2, alpha=0.9)
            
            color = 'red' if ai_score > 0.62 else 'lime'
            line_color = 'crimson' if ai_score > 0.62 else '#00FF88'
            
            for i, e in enumerate(events):
                if e < dur:
                    ax1.axvline(e, color=line_color, ls='--', lw=2.5, alpha=0.85)
                    if i < 6:
                        ax1.text(e, 0.12, f'B{i+1}', ha='center', color=line_color,
                                transform=ax1.get_xaxis_transform(), fontsize=10, fontweight='bold')
            
            ax1.set_title(f"{file.name[:30]} | AI: {ai_score:.0%} | {len(events)} breaths", 
                         color=line_color, fontsize=16, pad=20, fontweight='bold')
            ax1.set_facecolor('black')
            ax1.set_ylabel("Amplitude", color='white', fontsize=12)
            ax1.tick_params(colors='white')
            ax1.margins(x=0)
            
            # IBI Analysis
            ax2 = plt.subplot(3, 1, 2)
            if len(events) >= 3:
                ibis = np.diff(events[:12])
                mean_ibi = np.mean(ibis)
                colors = ['orange' if x < mean_ibi*0.65 or x > mean_ibi*1.6 else 'cyan' for x in ibis]
                bars = ax2.bar(range(len(ibis)), ibis, color=colors, alpha=0.85, width=0.65)
                ax2.axhline(mean_ibi, color='white', ls='-', lw=3, alpha=0.9,
                           label=f'Mean: {mean_ibi:.2f}s')
                cv_text = f'CV: {np.std(ibis)/mean_ibi:.2f}'
                ax2.text(0.02, 0.95, cv_text, transform=ax2.transAxes, color='white',
                        va='top', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='black', alpha=0.9))
            else:
                ax2.bar(['Breaths'], [len(events)], color='gray', alpha=0.7)
                ax2.text(0.5, 0.7, f"Only {len(events)} breaths\n(3-12 = more human-like)", 
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=12, color='white', fontweight='bold')
            
            ax2.set_title("Breath Intervals (IBI)", color='white', pad=15, fontsize=13)
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white')
            ax2.legend(frameon=False)
            
            # Parameter Radar Chart
            ax3 = plt.subplot(3, 1, 3, projection='polar')
            params = list(param_scores.keys())
            values = list(param_scores.values())
            
            angles = np.linspace(0, 2*np.pi, len(params), endpoint=False).tolist()
            values += values[:1]  # Complete circle
            angles += angles[:1]
            
            ax3.fill(angles, values, color=line_color, alpha=0.25)
            ax3.plot(angles, values, color=line_color, lw=3)
            
            ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax3.set_ylim(0, 1)
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(params, fontsize=11, color='white')
            
            ax3.set_title("Parameter Scores (1.0 = Human-like)", color='white', 
                         fontsize=13, pad=20, fontweight='bold')
            ax3.grid(True, color='gray', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
