import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="PneumaForensic v4.0")

def preprocess_audio(y, sr):
    nyquist = sr * 0.5
    high = 85 / nyquist
    b, a = butter(4, high, btype='high')
    y = filtfilt(b, a, y)
    return y / (np.max(np.abs(y)) + 1e-8)

def detect_breath_events(y, sr):
    y = preprocess_audio(y, sr)
    
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    noise_floor = np.percentile(rms, 12)
    breath_threshold = noise_floor * 1.8
    
    low_energy = rms < breath_threshold * 1.2
    breath_peaks, _ = find_peaks(
        low_energy.astype(float),
        height=0.4,
        distance=int(sr * 0.25),
        prominence=0.25,
        width=4
    )
    
    events = []
    for peak in breath_peaks:
        t = times[peak]
        if 1.0 < t < len(y)/sr - 2.0:
            start = max(0, int((t-0.25)*sr))
            end = min(len(y), int((t+0.55)*sr))
            window = y[start:end]
            
            breath_rms = np.max(librosa.feature.rms(y=window)[0])
            centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr)[0])
            
            if breath_threshold < breath_rms < noise_floor * 4.5 and centroid < sr * 0.22:
                events.append(t)
    
    filtered = []
    for t in sorted(events):
        if not filtered or 0.25 < t - filtered[-1] < 2.8:
            filtered.append(t)
    
    return filtered[:15]

def analyze_breath_features(events, y, sr):
    scores = {}
    breath_count = len(events)
    
    # Always initialize all 6 parameters
    scores = {
        'Count': 0.15 if 3 <= breath_count <= 12 else 0.75,
        'IBI': 0.0,
        'Amp': 0.0,
        'Dur': 0.0,
        'Spec': 0.45,
        'ZCR': 0.4
    }
    
    if breath_count < 2:
        for key in ['IBI', 'Amp', 'Dur', 'Spec', 'ZCR']:
            scores[key] = 0.7
        return scores
    
    # 2. IBI REGULARITY (25%)
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis)
    scores['IBI'] = 0.85 if ibi_cv < 0.22 or ibi_cv > 1.1 else 0.25
    
    # 3. AMPLITUDE VARIATION (20%)
    amps = [np.max(np.abs(y[int((t-0.12)*sr):int((t+0.28)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 1 else 1.0
    scores['Amp'] = 0.8 if amp_cv < 0.20 else 0.22
    
    # 4. DURATION VARIATION (15%)
    durs = []
    for t in events:
        start = max(0, int((t-0.22)*sr))
        end = min(len(y), int((t+0.45)*sr))
        rms_seg = librosa.feature.rms(y=y[start:end])[0]
        active = rms_seg > np.percentile(rms_seg, 40)
        dur = np.sum(active) * 512 / sr
        if 0.12 < dur < 0.9:
            durs.append(dur)
    
    dur_cv = np.std(durs) / np.mean(durs) if len(durs) > 1 else 1.0
    scores['Dur'] = 0.75 if dur_cv < 0.25 else 0.20
    
    # 5. SPECTRAL CHARACTERISTICS (15%)
    if len(events) >= 3:
        centroids = []
        for t in events[:6]:
            window = y[int((t-0.18)*sr):int((t+0.28)*sr)]
            centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr)[0])
            centroids.append(centroid)
        spec_cv = np.std(centroids) / np.mean(centroids)
        scores['Spec'] = 0.8 if spec_cv < 0.18 else 0.28
    
    # 6. ZCR VARIATION (10%)
    if len(events) >= 3:
        zcrs = []
        for t in events[:5]:
            window = y[int((t-0.15)*sr):int((t+0.25)*sr)]
            zcr = np.mean(librosa.feature.zero_crossing_rate(window))
            zcrs.append(zcr)
        zcr_cv = np.std(zcrs) / np.mean(zcrs)
        scores['ZCR'] = 0.75 if zcr_cv < 0.16 else 0.25
    
    return scores

def safe_load_audio(file):
    file.seek(0)
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True, duration=40)
        if len(y) < 12000:
            return None, None
        return y, sr
    except:
        return None, None

st.title("PneumaForensic v4.0")
st.markdown("Upload audio files for AI/Human detection")

files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a','aac'], accept_multiple_files=True)

if files:
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.header("Results")
        results = []
        all_scores = []
        
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                results.append({"File": file.name[:30], "AI": "ERROR", "Status": "❌", "Breaths": 0})
                continue
            
            events = detect_breath_events(y, sr)
            param_scores = analyze_breath_features(events, y, sr)
            
            # FIXED: Ensure exactly 6 scores and matching weights
            score_values = list(param_scores.values())
            weights = [0.15, 0.25, 0.20, 0.15, 0.15, 0.10]
            ai_score = np.average(score_values, weights=weights)
            
            status = "🤖 AI" if ai_score > 0.68 else "👤 HUMAN"
            
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
                **param_scores
            })
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        if all_scores:
            st.subheader("Detailed Scores")
            st.dataframe(pd.DataFrame(all_scores).round(2), use_container_width=True)
    
    with col2:
        st.header("Analysis")
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                continue
            
            events = detect_breath_events(y, sr)
            param_scores = analyze_breath_features(events, y, sr)
            
            score_values = list(param_scores.values())
            weights = [0.15, 0.25, 0.20, 0.15, 0.15, 0.10]
            ai_score = np.average(score_values, weights=weights)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor='black')
            
            dur = min(28, len(y)/sr)
            t = np.linspace(0, dur, int(dur*sr))
            ax1.plot(t[:len(y[:int(dur*sr)])], y[:int(dur*sr)], color='#888', lw=1)
            
            color = 'red' if ai_score > 0.68 else 'lime'
            for i, e in enumerate(events):
                if e < dur:
                    ax1.axvline(e, color=color, ls='--', lw=2, alpha=0.9)
                    if i < 8:
                        ax1.text(e, 0.1, f'B{i+1}', ha='center', color=color,
                                transform=ax1.get_xaxis_transform(), fontweight='bold', fontsize=9)
            
            ax1.set_title(f"{file.name[:25]} | AI: {ai_score:.0%}", color=color, fontsize=14, pad=15)
            ax1.set_facecolor('black')
            ax1.set_ylabel("Amplitude", color='white')
            ax1.tick_params(colors='white')
            
            if len(events) >= 2:
                ibis = np.diff(events[:12])
                ax2.bar(range(len(ibis)), ibis, color='orange', alpha=0.7, width=0.5)
                ax2.axhline(np.mean(ibis), color='white', lw=2)
                ax2.text(0.02, 0.92, f'IBI CV: {np.std(ibis)/np.mean(ibis):.2f}', 
                        transform=ax2.transAxes, color='white', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, f'{len(events)} breaths', ha='center', 
                        transform=ax2.transAxes, fontsize=18, color='white')
            
            ax2.set_title("Breath Intervals", color='white', pad=10)
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white')
            
            param_text = "\n".join([f"{k}: {v:.1f}" for k,v in param_scores.items()])
            ax2.text(0.02, 0.02, param_text, transform=ax2.transAxes, 
                    fontsize=9, color='cyan', verticalalignment='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.9))
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
