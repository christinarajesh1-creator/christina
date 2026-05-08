import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="PneumaForensic v4.1")

def preprocess_audio(y, sr):
    nyquist = sr * 0.5
    high = 90 / nyquist
    b, a = butter(3, high, btype='high')
    y = filtfilt(b, a, y)
    return y / (np.max(np.abs(y)) + 1e-8)

def detect_breath_events(y, sr):
    y = preprocess_audio(y, sr)
    
    frame_length = 1024
    hop_length = 256
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    noise_floor = np.percentile(rms, 10)
    breath_threshold = noise_floor * 2.2
    
    # More sensitive breath detection
    low_energy = rms < noise_floor * 3.0
    breath_peaks, properties = find_peaks(
        low_energy.astype(float),
        height=0.35,
        distance=int(sr * 0.2),
        prominence=0.2,
        width=2
    )
    
    events = []
    for peak in breath_peaks:
        t = times[peak]
        if 0.8 < t < len(y)/sr - 1.5:
            start = max(0, int((t-0.2)*sr))
            end = min(len(y), int((t+0.4)*sr))
            window = y[start:end]
            
            if len(window) > 100:
                breath_rms = np.max(librosa.feature.rms(y=window)[0])
                zcr = librosa.feature.zero_crossing_rate(window).mean()
                
                if noise_floor * 1.5 < breath_rms < noise_floor * 5.0 and zcr < 0.08:
                    events.append(t)
    
    # Filter close events
    filtered = [events[0]] if events else []
    for t in events[1:]:
        if t - filtered[-1] > 0.2:
            filtered.append(t)
    
    return filtered[:20]

def analyze_breath_features(events, y, sr):
    scores = {
        'Count': 0.0,
        'IBI': 0.0,
        'Amp': 0.0,
        'Dur': 0.0,
        'Spec': 0.0,
        'ZCR': 0.0
    }
    
    breath_count = len(events)
    
    # 1. COUNT (15%) - Prefer 4-10 breaths
    if 4 <= breath_count <= 10:
        scores['Count'] = 0.1
    elif breath_count >= 2:
        scores['Count'] = 0.3
    else:
        scores['Count'] = 0.8
    
    if breath_count < 2:
        return scores
    
    # 2. IBI REGULARITY (25%) - Human CV: 0.25-0.9
    ibis = np.diff(events)
    ibi_mean = np.mean(ibis)
    ibi_cv = np.std(ibis) / ibi_mean if ibi_mean > 0 else 1.0
    
    if 0.25 <= ibi_cv <= 0.9:
        scores['IBI'] = 0.15
    else:
        scores['IBI'] = 0.75
    
    # 3. AMPLITUDE VARIATION (20%) - Human CV: 0.25-1.0
    amps = []
    for t in events:
        start = max(0, int((t-0.15)*sr))
        end = min(len(y), int((t+0.25)*sr))
        amp = np.max(np.abs(y[start:end]))
        if amp > 0.01:
            amps.append(amp)
    
    if len(amps) > 1:
        amp_cv = np.std(amps) / np.mean(amps)
        if 0.25 <= amp_cv <= 1.0:
            scores['Amp'] = 0.18
        else:
            scores['Amp'] = 0.7
    else:
        scores['Amp'] = 0.3
    
    # 4. DURATION (15%) - Human: 0.15-0.6s with variation
    durs = []
    for t in events:
        start = max(0, int((t-0.2)*sr))
        end = min(len(y), int((t+0.4)*sr))
        rms_seg = librosa.feature.rms(y=y[start:end], hop_length=256)[0]
        noise = np.percentile(rms_seg, 30)
        active = rms_seg > noise
        dur = np.sum(active) * 0.016  # 256/16000
        if 0.1 < dur < 0.8:
            durs.append(dur)
    
    if len(durs) > 1:
        dur_cv = np.std(durs) / np.mean(durs)
        if 0.2 <= dur_cv <= 0.8:
            scores['Dur'] = 0.12
        else:
            scores['Dur'] = 0.65
    else:
        scores['Dur'] = 0.25
    
    # 5. SPECTRAL (15%) - Human has more variation
    specs = []
    for t in events[:8]:
        start = max(0, int((t-0.15)*sr))
        end = min(len(y), int((t+0.25)*sr))
        window = y[start:end]
        if len(window) > 50:
            spec = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr)[0])
            specs.append(spec)
    
    if len(specs) > 2:
        spec_cv = np.std(specs) / np.mean(specs)
        if spec_cv > 0.12:
            scores['Spec'] = 0.15
        else:
            scores['Spec'] = 0.7
    else:
        scores['Spec'] = 0.3
    
    # 6. ZCR (10%) - Human more noisy
    zcrs = []
    for t in events[:8]:
        start = max(0, int((t-0.15)*sr))
        end = min(len(y), int((t+0.25)*sr))
        window = y[start:end]
        if len(window) > 50:
            zcr = np.mean(librosa.feature.zero_crossing_rate(window))
            zcrs.append(zcr)
    
    if len(zcrs) > 2:
        zcr_cv = np.std(zcrs) / np.mean(zcrs)
        if zcr_cv > 0.18:
            scores['ZCR'] = 0.08
        else:
            scores['ZCR'] = 0.6
    else:
        scores['ZCR'] = 0.2
    
    return scores

def safe_load_audio(file):
    file.seek(0)
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, mono=True, duration=45)
        if len(y) < 8000:
            return None, None
        return y, sr
    except:
        return None, None

st.title("PneumaForensic v4.1 - Accurate Detection")
files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a','aac'], accept_multiple_files=True)

if files:
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.header("Detection Results")
        results = []
        all_scores = []
        
        for file in files:
            y, sr = safe_load_audio(file)
            if y is None:
                results.append({"File": file.name[:30], "AI": "ERROR", "Status": "❌", "Breaths": 0})
                continue
            
            events = detect_breath_events(y, sr)
            param_scores = analyze_breath_features(events, y, sr)
            
            weights = [0.15, 0.25, 0.20, 0.15, 0.15, 0.10]
            ai_score = np.average(list(param_scores.values()), weights=weights)
            
            status = "🤖 AI" if ai_score > 0.55 else "👤 HUMAN"
            
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
        st.subheader("Parameter Breakdown")
        st.dataframe(pd.DataFrame(all_scores).round(2), use_container_width=True)
    
    with col2:
        st.header("Breath Visualization")
        for file in files[:2]:  # Limit to 2 plots
            y, sr = safe_load_audio(file)
            if y is None:
                continue
            
            events = detect_breath_events(y, sr)
            param_scores = analyze_breath_features(events, y, sr)
            weights = [0.15, 0.25, 0.20, 0.15, 0.15, 0.10]
            ai_score = np.average(list(param_scores.values()), weights=weights)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor='black')
            
            dur = min(30, len(y)/sr)
            t = np.linspace(0, dur, int(dur*sr))
            ax1.plot(t, y[:len(t)], color='#666', lw=0.8, alpha=0.9)
            
            color = 'red' if ai_score > 0.55 else 'lime'
            for i, e in enumerate(events):
                if e < dur:
                    ax1.axvline(e, color=color, ls='--', lw=2.5, alpha=0.9)
                    ax1.text(e, 0.15, f'B{i+1}', ha='center', color=color,
                            transform=ax1.get_xaxis_transform(), fontweight='bold', fontsize=10)
            
            ax1.set_title(f"{file.name[:25]} | AI: {ai_score:.0%}", color=color, fontsize=14, pad=15)
            ax1.set_facecolor('black')
            ax1.set_ylabel("Amplitude", color='white')
            ax1.tick_params(colors='white')
            ax1.set_xlim(0, dur)
            
            # IBI plot
            if len(events) >= 2:
                ibis = np.diff(events[:10])
                bars = ax2.bar(range(len(ibis)), ibis, color='orange', alpha=0.8, width=0.6)
                ax2.axhline(np.mean(ibis), color='white', lw=2, ls='-', alpha=0.9)
                cv_text = f'CV: {np.std(ibis)/np.mean(ibis):.2f}'
                ax2.text(0.02, 0.95, cv_text, transform=ax2.transAxes, 
                        color='white', fontweight='bold', fontsize=11)
            else:
                ax2.text(0.5, 0.5, f'Found {len(events)} breaths', ha='center', 
                        transform=ax2.transAxes, fontsize=20, color='white')
            
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white')
            ax2.set_title("Inter-Breath Intervals", color='white', pad=10)
            
            # Parameter table
            params = '\n'.join([f"{k}: {v:.2f}" for k,v in param_scores.items()])
            ax2.text(0.68, 0.02, params, transform=ax2.transAxes, fontsize=9, 
                    color='cyan', verticalalignment='bottom',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='black', alpha=0.9))
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
