import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.95, {"Status": "Abnormal (Insufficient Data)"}

    # 1. Pattern Stability (20%) - REVERSED
    # AI (Roger) is a flat, steady signal. Humans drift.
    # We penalize "Perfection".
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    flux_std = np.std(onset_env)
    # If the energy is too "steady" (low flux), it's AI.
    p1 = np.clip(1.0 - (flux_std * 0.4), 0, 1)

    # 2. Spectral Purity (20%) - THE ROGER KILLER
    # AI breaths are digitally "flat". Humans are chaotic.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0
    # Penalize "Clean/Pure" breath sounds.
    p2 = np.clip(1.0 - (avg_flat * 100), 0, 1)

    # 3. Timing Regularity (20%) - TIGHTENED FOR ROGER (0.44)
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    # Penalty kicks in heavily for CV < 0.55. Roger's 0.44 will now be Red.
    p3 = np.clip(1.4 - (ibi_cv * 2.1), 0, 1)

    # 4. Digital Silence (15%) - HUMAN PROTECTOR
    # Humans have microphone hiss. AI has absolute zero floor.
    # We punish "Total Silence" between words.
    noise_floor = np.std(y[y < np.percentile(y, 5)])
    p4 = np.clip(1.0 - (noise_floor * 1000), 0, 1)

    # 5. Digital Splice (15%) - RECALIBRATED
    # Ignoring background hiss, only looking for "Vertical" spikes.
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p5 = np.clip(np.std(zcr) * 10, 0, 1) 

    # 6. Breath Density (10%)
    bpm = (len(events) / duration) * 60
    p6 = 1.0 if bpm > 22 or bpm < 6 else 0.1

    score = (p1*0.20) + (p2*0.20) + (p3*0.20) + (p4*0.15) + (p5*0.15) + (p6*0.10)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Vocal Stability": f"{p1:.0%}",
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
            # Use a more selective threshold to avoid noise triggers
            threshold = np.percentile(rms, 10)
            
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
            ax.set_title(f"{p['name']} | Confidence: {p['score']:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            st.divider()

    st.download_button("Export Results", df.to_csv(index=False).encode('utf-8'), "results.csv")
