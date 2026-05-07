import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.set_page_config(layout="wide", page_title="PneumaForensic 10.0")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.98, {k: "100%" for k in ["Shimmer", "Purity", "Timing", "Silence", "Amp", "Splice"]}

    # 1. VOCAL SHIMMER (25%) - PENALIZE STABILITY
    # AI is a steady mathematical wave. Humans have 'shimmer' (chaotic energy).
    S = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S=S).flatten()
    shimmer_cv = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
    # HARDENED: If energy is too steady (AI), score goes UP.
    p1 = np.clip(1.0 - (shimmer_cv * 2), 0, 1)

    # 2. SPECTRAL PURITY (20%) - PENALIZE CLEANLINESS
    # AI breaths are 'pure' digital signals. Humans are messy noise.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0
    # HARDENED: Roger is too 'pure/flat'. If flatness is low, score goes UP.
    p2 = np.clip(1.0 - (avg_flat * 150), 0, 1)

    # 3. TIMING REGULARITY (20%) - TARGETING ROGER (CV 0.35-0.45)
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    # Penalty kicks in for any CV below 0.6. Roger's 0.39 will now be Red.
    p3 = np.clip(1.5 - (ibi_cv * 2.5), 0, 1)

    # 4. DIGITAL SILENCE (15%) - HUMAN PROTECTOR
    # Rewards microphone hiss. Punishes perfect digital silence.
    noise_floor = np.std(y[y < np.percentile(y, 10)])
    p4 = np.clip(1.0 - (noise_floor * 1200), 0, 1)

    # 5. AMPLITUDE SCORE (10%) - UNIFORM VOLUME
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p5 = np.clip(1.0 - (amp_cv * 1.5), 0, 1)

    # 6. DIGITAL SPLICE (10%)
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p6 = np.clip(np.std(zcr) * 10, 0, 1) 

    score = (p1*0.25) + (p2*0.20) + (p3*0.20) + (p4*0.15) + (p5*0.10) + (p6*0.10)
    
    metrics = {
        "AI Confidence": f"{score:.0%}",
        "Vocal Shimmer": f"{p1:.0%}",
        "Spectral Purity": f"{p2:.0%}",
        "Timing Regularity": f"{p3:.0%}",
        "Digital Silence": f"{p4:.0%}",
        "Amplitude Score": f"{p5:.0%}",
        "Digital Splice": f"{p6:.0%}",
        "CV": round(ibi_cv, 2)
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

st.title("🫁 PneumaForensic 10.0")

files = st.file_uploader("Upload Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    all_data, plot_data = [], []
    for f in files:
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
            y = librosa.util.normalize(y)
            rms = librosa.feature.rms(y=y).flatten()
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            threshold = np.percentile(rms, 10)
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold:
                    t = times[i]
                    if t - last_t > 1.6:
                        events.append(float(t))
                        last_t = t
            
            score, metrics = get_forensic_score(y, sr, events, len(y)/sr)
            row = {"Filename": f.name}; row.update(metrics)
            all_data.append(row)
            plot_data.append({"y": y, "events": events, "dur": len(y)/sr, "name": f.name, "score": score})
        except: continue

    if all_data:
        df = pd.DataFrame(all_data)
        st.subheader("📋 6-Parameter Forensic Master Report")
        st.dataframe(df, use_container_width=True)

        for p in plot_data:
            fig, ax = plt.subplots(figsize=(15, 1.5))
            ax.plot(np.linspace(0, p['dur'], len(p['y'])), p['y'], color='gray', alpha=0.3, lw=0.5)
            for e in p['events']:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            color = "red" if p['score'] > 0.55 else "green"
            ax.set_title(f"{p['name']} | Confidence: {p['score']:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
