import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.99, {"Status": "Abnormal (Silent/AI)"}

    # 1. HARDENED REGULARITY (35%)
    # Roger is 0.38. We now penalize ANYTHING under 0.55 heavily.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1 = np.clip(1.5 - (ibi_cv * 2.5), 0, 1)

    # 2. SPECTRAL PURITY (25%) - ROGER KILLER
    # AI breaths are 'flat/clean'. Humans are chaotic noise.
    # Higher Flatness = Human. Lower Flatness = AI.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0
    # Penalty for being 'too clean/pure' (AI)
    p2 = np.clip(1.0 - (avg_flat * 150), 0, 1)

    # 3. VOCAL STABILITY (20%)
    # AI is steady. Humans drift in energy.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    flux_std = np.std(onset_env)
    # If the energy is too 'steady' (Low Flux), it's AI.
    p3 = np.clip(1.0 - (flux_std * 0.6), 0, 1)

    # 4. DIGITAL SILENCE (20%) - HUMAN PROTECTOR
    # Rewards microphone hiss. Punishes perfect digital silence.
    noise_floor = np.std(y[y < np.percentile(y, 8)])
    p4 = np.clip(1.0 - (noise_floor * 1200), 0, 1)

    # FINAL WEIGHTED SCORE
    score = (p1*0.35) + (p2*0.25) + (p3*0.20) + (p4*0.20)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Regularity (35%)": f"{p1:.0%}",
        "Purity (25%)": f"{p2:.0%}",
        "Stability (20%)": f"{p3:.0%}",
        "Silence (20%)": f"{p4:.0%}",
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
            # Threshold to find clear breaths in your files
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
        # Master Forensic Table
        df = pd.DataFrame(all_data)
        st.subheader("📋 Forensic Results")
        st.dataframe(df, use_container_width=True)

        # Charts
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

    st.download_button("Export CSV", df.to_csv(index=False).encode('utf-8'), "results.csv")
