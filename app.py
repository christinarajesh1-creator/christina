import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.98, {"Status": "Abnormal (Zero Breaths)"}

    # 1. Rhythmic Entropy (35%) - THE ROGER KILLER
    # Penalizes 0.3-0.5 CV range. Humans must be > 0.65 for Green.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1 = np.clip(1.4 - (ibi_cv * 2.2), 0, 1)

    # 2. Spectral Flatness flux (25%) - HUMAN PROTECTOR
    # AI breaths are 'flat/smooth'. Human breaths are 'chaotic noise'.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    # AI has lower entropy. If it's too 'clean' (low flatness flux), it's AI.
    p2 = np.clip(1.0 - (np.mean(flatness) * 60), 0, 1) if flatness else 0.5

    # 3. Digital Silence floor (20%)
    # Humans have mic hiss. AI has zero noise floor.
    noise_floor = np.std(y[y < np.percentile(y, 10)])
    p3 = np.clip(1.0 - (noise_floor * 500), 0, 1)

    # 4. Digital Continuity (20%) - DE-WEIGHTED
    # Lowered to 20% so noisy human mics stop triggering 100% AI flags.
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p4 = np.clip(np.std(zcr) * 8, 0, 1) 

    score = (p1 * 0.35) + (p2 * 0.25) + (p3 * 0.20) + (p4 * 0.20)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Timing Regularity": f"{p1:.0%}",
        "Spectral Purity": f"{p2:.0%}",
        "Digital Silence": f"{p3:.0%}",
        "Digital Splice": f"{p4:.0%}",
        "IBI CV": round(ibi_cv, 2)
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
            threshold = np.percentile(rms, 10)
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold:
                    t = times[i]
                    if t - last_t > 1.7:
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
        st.subheader("📋 Forensic Report")
        st.dataframe(df, use_container_width=True)

        for p in plot_data:
            fig, ax = plt.subplots(figsize=(15, 1.5))
            ax.plot(np.linspace(0, p['dur'], len(p['y'])), p['y'], color='gray', alpha=0.3, lw=0.5)
            for e in p['events']:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
            color = "red" if p['score'] > 0.55 else "green"
            ax.set_title(f"{p['name']} | AI Confidence: {p['score']:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            st.divider()

    st.download_button("Export Results", df.to_csv(index=False).encode('utf-8'), "results.csv")
