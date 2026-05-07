import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="PneumaForensic")

@st.cache_data
def get_forensic_score(y, sr, events, duration):
    # 1. TIMING REGULARITY (30%) - ROGER KILLER
    if len(events) > 1:
        ibis = np.diff(events)
        ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 999
    else:
        ibi_cv = 999
    p1 = np.clip(2.0 * ibi_cv, 0, 1)

    # 2. SPECTRAL PURITY (20%)
    flatness = []
    for t in events:
        start = int(max(0, t*sr - 0.1*sr))
        end = int(min(len(y), (t+0.3)*sr))
        if end > start:
            spec_flat = np.mean(librosa.feature.spectral_flatness(y=y[start:end]))
            flatness.append(spec_flat)
    avg_flat = np.mean(flatness) if flatness else 0.8
    p2 = np.clip((avg_flat - 0.3) * 2.5, 0, 1)

    # 3. BREATH DENSITY (15%)
    bpm = (len(events) / max(duration, 1)) * 60
    p3 = 1.0 if (bpm < 10 or bpm > 40) else 0.2

    # 4. NOISE FLOOR (15%)
    silence_regions = []
    for i in range(len(events)-1):
        mid_silence = (events[i] + events[i+1]) / 2
        start = int(max(0, (mid_silence-0.4)*sr))
        end = int(min(len(y), (mid_silence+0.4)*sr))
        if end > start:
            silence_std = np.std(y[start:end])
            silence_regions.append(silence_std)
    
    noise_floor = np.mean(silence_regions) if silence_regions else 0
    p4 = np.clip(1.0 - (noise_floor * 8000), 0, 1)

    # 5. BREATH AMPLITUDE (10%)
    amps = []
    for t in events:
        start = int(max(0, (t-0.1)*sr))
        end = int(min(len(y), (t+0.4)*sr))
        if end > start:
            peak_amp = np.max(np.abs(y[start:end]))
            amps.append(peak_amp)
    
    amp_cv = np.std(amps) / np.mean(amps) if amps and np.mean(amps) > 0 else 0
    p5 = np.clip(amp_cv * 3.0, 0, 1)

    # 6. SIMILARITY (10%)
    mfccs = []
    for t in events:
        start = int(max(0, t*sr))
        end = int(min(len(y), (t+0.4)*sr))
        if end > start + sr//10:
            mfcc = np.mean(librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=13), axis=1)
            mfccs.append(mfcc)
    
    p6 = 0.0
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) 
                for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        avg_dist = np.mean(dists)
        p6 = np.clip(1.0 - (avg_dist / 80), 0, 1)

    score = (p1*0.30) + (p2*0.20) + (p3*0.15) + (p4*0.15) + (p5*0.10) + (p6*0.10)
    
    status = "🤖 AI" if score > 0.6 else "👤 HUMAN"
    metrics = {
        "Status": status,
        "AI Score": f"{score:.0%}",
        "Timing": f"{p1:.0%}",
        "Purity": f"{p2:.0%}",
        "Density": f"{p3:.0%}",
        "Noise": f"{p4:.0%}",
        "Amp": f"{p5:.0%}",
        "Sim": f"{p6:.0%}",
        "BPM": f"{bpm:.1f}"
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

def detect_breaths(y, sr):
    """Aggressive breath detection for Roger AI"""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).flatten()
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    
    # Multiple detection strategies
    events = []
    
    # Strategy 1: High energy peaks
    speech_level = np.percentile(rms, 75)
    breath_threshold = speech_level * 1.8
    peaks = (rms > breath_threshold) & (rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:])
    for i in np.where(peaks)[0] + 1:
        if times[i] > 2.0:
            events.append(times[i])
    
    # Strategy 2: Amplitude bursts
    envelope = np.abs(y)
    burst_idx = np.where((envelope[1:] > 3*np.mean(envelope)) & 
                        (envelope[1:] > envelope[:-1]) & 
                        (envelope[1:] > envelope[2:]))[0] + 1
    burst_times = (burst_idx / sr).tolist()
    events.extend([t for t in burst_times if t > 2.0])
    
    # Remove duplicates and enforce minimum spacing
    events = sorted(set(events))
    filtered_events = []
    last_t = -10
    for t in events:
        if t - last_t > 1.2:
            filtered_events.append(t)
            last_t = t
    
    return filtered_events[:20]  # Max 20 breaths

st.title("🫁 PneumaForensic")

files = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'm4a'], accept_multiple_files=True)

if files:
    all_data = []
    plot_data = []
    
    for f in files:
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
            if np.max(np.abs(y)) < 0.01:
                continue
                
            events = detect_breaths(y, sr)
            duration = len(y) / sr
            
            score, metrics = get_forensic_score(y, sr, events, duration)
            
            row = {"File": f.name, **metrics}
            all_data.append(row)
            plot_data.append({"y": y, "events": events, "dur": duration, "name": f.name, "score": score})
            
        except:
            continue

    if all_data:
        df = pd.DataFrame(all_data)
        st.dataframe(df, use_container_width=True)

        for p in plot_data:
            fig, ax = plt.subplots(figsize=(15, 2))
            ax.plot(np.linspace(0, p['dur'], len(p['y'])), p['y'], 'gray', alpha=0.4, lw=0.5)
            for e in p['events']:
                ax.axvline(e, color='red', lw=2, alpha=0.8)
            color = "red" if p['score'] > 0.6 else "green"
            ax.set_title(f"{p['name']} | {p['score']:.0%}", color=color, fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
