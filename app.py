import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.98, {"Timing": 1.0, "Purity": 1.0, "Density": 1.0, "Amp": 1.0, "Splice": 1.0, "Sim": 1.0, "CV": 0.0}

    # 1. Timing Regularity (28%) - ROGER KILLER
    # Penalizes 0.40 CV. Humans must be > 0.60 to be green.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1 = np.clip(1.3 - (ibi_cv * 2), 0, 1)

    # 2. Spectral Purity (18%)
    # AI breaths are 'too clean' (Low flatness flux). Human breaths are messy.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    p2 = np.clip(1.0 - (np.mean(flatness) * 45), 0, 1) if flatness else 0.5

    # 3. Breath Density (15%)
    bpm = (len(events) / duration) * 60
    p3 = 1.0 if bpm > 22 or bpm < 6 else 0.1

    # 4. Digital Silence Floor (15%)
    # AI Roger has zero noise floor. Human mics have hiss.
    noise_floor = np.std(y[y < np.percentile(y, 10)])
    p4 = np.clip(1.0 - (noise_floor * 500), 0, 1)

    # 5. Amplitude Uniformity (12%)
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p5 = np.clip(1.0 - (amp_cv / 0.5), 0, 1)

    # 6. Similarity Score (12%)
    mfccs = [np.mean(librosa.feature.mfcc(y=y[int(t*sr):int((t+0.4)*sr)], sr=sr, n_mfcc=13), axis=1) for t in events]
    p6 = 0.0
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        p6 = np.clip(1.0 - (np.mean(dists) / 200), 0, 1)

    score = (p1*0.28) + (p2*0.18) + (p3*0.15) + (p4*0.15) + (p5*0.12) + (p6*0.12)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Timing (28%)": f"{p1:.0%}",
        "Purity (18%)": f"{p2:.0%}",
        "Density (15%)": f"{p3:.0%}",
        "Silence (15%)": f"{p4:.0%}",
        "Amp (12%)": f"{p5:.0%}",
        "Sim (12%)": f"{p6:.0%}",
        "CV": round(ibi_cv, 2)
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

st.title("🫁 PneumaForensic")

files = st.file_uploader("Upload Batch", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    all_data = []
    plot_data = []
    
    for f in files:
        try:
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000)
            y = librosa.util.normalize(y)
            rms = librosa.feature.rms(y=y).flatten()
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            threshold = np.percentile(rms, 12)
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold:
                    t = times[i]
                    if t - last_t > 1.6:
                        events.append(float(t))
                        last_t = t
            
            score, metrics = get_forensic_score(y, sr, events, len(y)/sr)
            
            # Store for Table
            row = {"Filename": f.name}
            row.update(metrics)
            all_data.append(row)
            
            # Store for Visualization
            plot_data.append({"y": y, "events": events, "dur": len(y)/sr, "name": f.name, "score": score})
        except: continue

    # 1. Master Forensic Table
    st.subheader("📋 Batch Analysis Report")
    summary_df = pd.DataFrame(all_data)
    st.dataframe(summary_df, use_container_width=True)

    # 2. Individual Waveform Analysis
    st.subheader("📊 Breath Pattern Analysis")
    for p in plot_data:
        fig, ax = plt.subplots(figsize=(15, 1.8))
        ax.plot(np.linspace(0, p['dur'], len(p['y'])), p['y'], color='gray', alpha=0.3, lw=0.5)
        for e in p['events']:
            ax.axvline(e, color='red', linestyle='--', lw=1.5)
        
        status_color = "red" if p['score'] > 0.55 else "green"
        ax.set_title(f"{p['name']} | AI Confidence: {p['score']:.0%}", color=status_color, loc='left', fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)
        st.divider()

    st.download_button("Export Results", summary_df.to_csv(index=False).encode('utf-8'), "pneuma_results.csv")
