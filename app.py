import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="PneumaForensic v3.1 - AI Optimized")

def preprocess_audio(y, sr):
    """Enhanced preprocessing for AI detection"""
    nyquist = sr * 0.5
    high = 90 / nyquist
    b, a = butter(4, high, btype='high')
    y = filtfilt(b, a, y)
    y = y / (np.max(np.abs(y)) + 1e-8)
    return librosa.effects.preemphasis(y)

def detect_breath_events_ai_optimized(y, sr):
    """AI-optimized breath detection - conservative"""
    y = preprocess_audio(y, sr)
    
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    noise_floor = np.percentile(rms, 10)
    speech_threshold = noise_floor * 4.2
    breath_threshold = noise_floor * 2.1
    
    low_energy = rms < speech_threshold
    
    breath_peaks, _ = find_peaks(
        low_energy.astype(float),
        height=0.45,
        distance=int(sr * 0.42),
        prominence=0.35,
        width=5
    )
    
    events = []
    for peak in breath_peaks:
        t = times[peak]
        if 1.5 < t < len(y)/sr - 3.0:
            start = max(0, int((t-0.28)*sr))
            end = min(len(y), int((t+0.55)*sr))
            window = y[start:end]
            
            breath_rms = np.max(librosa.feature.rms(y=window)[0])
            centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr)[0])
            flatness = np.mean(librosa.feature.spectral_flatness(y=window)[0])
            
            if (breath_threshold < breath_rms < speech_threshold * 0.45 and 
                centroid < sr * 0.16 and flatness < 0.65):
                events.append(t)
    
    filtered = []
    for t in sorted(events):
        if not filtered or 0.38 < t - filtered[-1] < 2.5:
            filtered.append(t)
    
    return filtered[:10]

def analyze_breath_features_ai_focused(events, y, sr):
    """AI-focused 6 parameters - HIGH SCORE = AI"""
    scores = {}
    
    breath_count = len(events)
    # 1. COUNT - AI sweet spot 4-8
    count_score = 0.9 if 4 <= breath_count <= 8 else 0.3
    count_score = min(count_score, 0.2 if breath_count > 12 else count_score)
    scores['Count'] = count_score
    
    if breath_count < 3:
        for key in ['IBI', 'Amp', 'Dur', 'Spec', 'Sim']:
            scores[key] = 0.85
        return scores
    
    # 2. IBI - AI = LOW CV
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis)
    ibi_ai = 1.0 if ibi_cv < 0.28 else max(0, 1 - (ibi_cv * 1.8))
    scores['IBI'] = ibi_ai
    
    # 3. AMPLITUDE - AI = LOW variation
    amps = []
    for t in events:
        start = max(0, int((t-0.16)*sr))
        end = min(len(y), int((t+0.28)*sr))
        peak_amp = np.max(np.abs(y[start:end]))
        amps.append(peak_amp)
    
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 1 else 0
    amp_ai = 1.0 if amp_cv < 0.22 else max(0, 1 - (amp_cv * 2.2))
    scores['Amp'] = amp_ai
    
    # 4. DURATION - AI = uniform
    durs = []
    for t in events:
        start = max(0, int((t-0.2)*sr))
        end = min(len(y), int((t+0.45)*sr))
        rms_seg = librosa.feature.rms(y=y[start:end])[0]
        active = rms_seg > np.percentile(rms_seg, 48)
        dur = np.sum(active) * 512 / sr
        if 0.15 < dur < 0.85:
            durs.append(dur)
    
    dur_cv = np.std(durs) / np.mean(durs) if len(durs) > 1 else 0
    dur_ai = 1.0 if dur_cv < 0.25 else max(0, 1 - (dur_cv * 2.0))
    scores['Dur'] = dur_ai
    
    # 5. SPECTRAL - AI = smooth
    spec_ai_scores = []
    for t in events[:6]:
        s, e = int((t-0.16)*sr), int((t+0.26)*sr)
        mfcc = librosa.feature.mfcc(y=y[s:e], sr=sr, n_mfcc=12)
        mfcc_var = np.var(np.diff(mfcc.mean(1)))
        flatness = librosa.feature.spectral_flatness(y=y[s:e])[0].mean()
        ai_like = (1 - mfcc_var * 50) * (0.7 + flatness * 0.3)
        spec_ai_scores.append(max(0, min(1, ai_like)))
    
    scores['Spec'] = np.mean(spec_ai_scores) if spec_ai_scores else 0.8
    
    # 6. SIMILARITY - AI = identical breaths
    sim_similarities = []
    for i in range(min(4, len(events))):
        for j in range(i+1, min(i+2, len(events))):
            t1, t2 = events[i], events[j]
            s1, e1 = int((t1-0.15)*sr), int((t1+0.23)*sr)
            s2, e2 = int((t2-0.15)*sr), int((t2+0.23)*sr)
            try:
                mfcc1 = librosa.feature.mfcc(y=y[s1:e1], sr=sr, n_mfcc=8).mean(1)
                mfcc2 = librosa.feature.mfcc(y=y[s2:e2], sr=sr, n_mfcc=8).mean(1)
                n = min(len(mfcc1), len(mfcc2))
                corr = np.corrcoef(mfcc1[:n], mfcc2[:n])[0,1]
                sim_similarities.append(corr)
            except:
                sim_similarities.append(0.7)
    
    scores['Sim'] = np.mean(sim_similarities) if sim_similarities else 0.6
    
    return scores

def safe_load_audio(file):
    file.seek(0)
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True, duration=35)
        if len(y) < 10000:
            return None, None
        return y, sr
    except:
        return None, None

# === UI ===
st.title("🫁 PneumaForensic v3.1 - AI Detector")
st.markdown("**Detects ElevenLabs, Azure TTS, etc. | Red=AI (regular) | Green=Human (varied)**")

files = st.file_uploader("Upload audio", type=['wav','mp3','m4a','aac'], accept_multiple_files=True)

if files:
    col1, col2 = st.columns([1.4, 1])
    
    with col1:
        st.header("🎯 Results")
        results = []
        all_scores = []
        
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                results.append({"File": file.name[:30], "AI": "ERROR", "Status": "❌", "Breaths": 0})
                continue
            
            events = detect_breath_events_ai_optimized(y, sr)
            param_scores = analyze_breath_features_ai_focused(events, y, sr)
            
            weights = [0.22, 0.28, 0.17, 0.15, 0.10, 0.08]
            ai_score = np.average(list(param_scores.values()), weights=weights)
            
            status = "🤖 **AI**" if ai_score > 0.68 else "👤 **HUMAN**"
            
            results.append({
                "File": file.name[:32],
                "AI": f"{ai_score:.0%}",
                "Status": status,
                "Breaths": len(events)
            })
            
            all_scores.append({
                "File": file.name[:25],
                "AI": f"{ai_score:.0%}",
                "Status": status,
                **{k: f"{v:.3f}" for k, v in param_scores.items()}
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        if all_scores:
            st.subheader("📊 Parameters (High = AI-like)")
            scores_df = pd.DataFrame(all_scores)
            st.dataframe(scores_df, use_container_width=True)
    
    with col2:
        st.header("🔍 Analysis")
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                continue
            
            events = detect_breath_events_ai_optimized(y, sr)
            param_scores = analyze_breath_features_ai_focused(events, y, sr)
            weights = [0.22, 0.28, 0.17, 0.15, 0.10, 0.08]
            ai_score = np.average(list(param_scores.values()), weights=weights)
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), facecolor='black')
            
            # Waveform
            dur = min(28, len(y)/sr)
            t = np.linspace(0, dur, int(dur*sr))
            y_plot = y[:len(t)]
            
            color = 'crimson' if ai_score > 0.68 else '#00FF88'
            ax1.plot(t, y_plot, color='#888888', lw=1.3)
            
            for i, e in enumerate(events):
                if e < dur:
                    ax1.axvline(e, color=color, ls='--', lw=2.8, alpha=0.9)
                    if i < 5:
                        ax1.text(e, 0.15, f'B{i+1}', ha='center', color=color,
                                transform=ax1.get_xaxis_transform(), fontweight='bold')
            
            ax1.set_title(f"{file.name[:28]} | AI: {ai_score:.0%}", 
                         color=color, fontsize=16, pad=20, fontweight='bold')
            ax1.set_facecolor('black')
            ax1.set_ylabel("Amplitude", color='white')
            ax1.tick_params(colors='white')
            
            # IBI
            if len(events) >= 3:
                ibis = np.diff(events[:10])
                mean_ibi = np.mean(ibis)
                colors = ['red' if 0.75*mean_ibi <= x <= 1.3*mean_ibi else 'lime' for x in ibis]
                ax2.bar(range(len(ibis)), ibis, color=colors, alpha=0.8, width=0.7)
                ax2.axhline(mean_ibi, color='white', lw=2.5, label=f'Mean: {mean_ibi:.2f}s')
            else:
                ax2.text(0.5, 0.5, f'{len(events)} breaths\n(Few=AI-like)', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=14, color='orange', fontweight='bold')
            
            ax2.set_title("Spacing (Regular=AI)", color='white', pad=15)
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white')
            ax2.legend()
            
            # Parameters
            params = list(param_scores.keys())
            values = list(param_scores.values())
            colors = ['red' if v > 0.68 else 'lime' for v in values]
            
            bars = ax3.bar(params, values, color=colors, alpha=0.8)
            ax3.axhline(0.68, color='white', ls='--', lw=2, label='AI Threshold')
            ax3.set_ylim(0, 1.05)
            
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax3.set_title("AI Scores", color='white', pad=15, fontweight='bold')
            ax3.tick_params(colors='white')
            ax3.legend()
            ax3.set_facecolor('black')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

st.markdown("---")
st.markdown("*v3.1 AI-Optimized | 68% threshold | Catches modern TTS*")
