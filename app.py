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
        return 0.98, {"Status": "Abnormal (Insufficient Breaths)"}

    # 1. Vocal Fatigue (20%)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_drift = np.std(pitch_values) / np.mean(pitch_values) if len(pitch_values) > 0 else 0
    p1 = np.clip(1.0 - (pitch_drift * 15), 0, 1)

    # 2. Spectral Purity (20%)
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0.5
    p2 = np.clip(1.0 - (avg_flat * 45), 0, 1)

    # 3. Timing Regularity (20%)
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p3 = np.clip(1.3 - (ibi_cv * 2), 0, 1)

    # 4. Digital Silence (15%)
    noise_floor = np.std(y[y < np.percentile(y, 10)])
    p4 = np.clip(1.0 - (noise_floor * 550), 0, 1)

    # 5. Digital Splice (15%)
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p5 = np.clip(np.std(zcr) * 30, 0, 1) 

    # 6. Breath Density (10%)
    bpm = (len(events) / duration) * 60
    p6 = 1.0 if bpm > 22 or bpm < 6 else 0.1

    score = (p1*0.20) + (p2*0.20) + (p3*0.20) + (p4*0.15) + (p5*0.15) + (p6*0.10)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Vocal Fatigue": f"{p1:.0%}",
        "Spectral Purity": f"{p2:.0%}",
        "Timing Regularity": f"{p3:.0%}",
        "Digital Silence": f"{p4:.0%}",
        "Digital Splice": f"{p5:.0%}",
        "Breath Density": f"{p6:.0%}",
        "IBI CV": round(ibi_cv, 2)
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

st.title("🫁 PneumaForensic")

files = st.file_uploader("Upload", type=['wav', 'mp3'], accept_multiple_files=True)

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
            row = {"Filename": f.name}
            row.update(metrics)
            all_data.append(row)
            plot_data.append({"y": y, "events": events, "dur": len(y)/sr, "name": f.name, "score": score})
        except: continue

    if all_data:
        df = pd.DataFrame(all_data)
        st.dataframe(df, use_container_width=True)

        for p in plot_data:
            fig, ax = plt.subplots(figsize=(15, 1.8))
            ax.plot(np.linspace(0, p['dur'], len(p['y'])), p['y'], color='gray', alpha=0.3, lw=0.5)
            for e in p['events']:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
            color = "red" if p['score'] > 0.55 else "green"
            ax.set_title(f"{p['name']} | AI Confidence: {p['score']:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            st.divider()

    st.download_button("Export CSV", df.to_csv(index=False).encode('utf-8'), "pneuma_results.csv")
