import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 2:
        return 0.99, {"Status": "Digital Silence/AI"}

    # 1. BIOLOGICAL ENTROPY (35%) - THE HUMAN PROTECTOR
    # Humans have messy breaths. AI (Roger) has 'clean' digital noise.
    # Higher Flatness = Biological Noise = Human. Lower = AI.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0.5
    p1 = np.clip(1.0 - (avg_flat * 120), 0, 1)

    # 2. DIGITAL SILENCE (25%) - THE MIC HISS TEST
    # Humans have mic hiss. AI has absolute zero floor. 
    # This rewards the 'messy' noise floor of your human WAV files.
    noise_floor = np.std(y[y < np.percentile(y, 10)])
    p2 = np.clip(1.0 - (noise_floor * 1800), 0, 1)

    # 3. PATTERN STABILITY (20%) - THE STEADINESS TEST
    # Humans drift in energy (Fatigue). AI is a steady mathematical stream.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    flux_std = np.std(onset_env)
    p3 = np.clip(1.0 - (flux_std * 0.45), 0, 1)

    # 4. TIMING REGULARITY (20%) - CATCHING THE 0.4 CV
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    # Penalty kicks in for any CV below 0.55 (Roger's 0.40 is now RED).
    p4 = np.clip(1.4 - (ibi_cv * 2.2), 0, 1)

    # TOTAL CALCULATION
    score = (p1 * 0.35) + (p2 * 0.25) + (p3 * 0.20) + (p4 * 0.20)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Spectral Purity (35%)": f"{p1:.0%}",
        "Digital Silence (25%)": f"{p2:.0%}",
        "Vocal Stability (20%)": f"{p3:.0%}",
        "Timing Regularity (20%)": f"{p4:.0%}",
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
            # Threshold calibrated for human microphone noise
            threshold = np.percentile(rms, 15) 
            
            events, last_t = [], -5.0
            for i, val in enumerate(rms):
                if val < threshold * 0.7:
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
        # Master Report Table
        df = pd.DataFrame(all_data)
        st.subheader("📋 Master Forensic Report")
        st.dataframe(df, use_container_width=True)

        # Visualizations (Waves + Breaths)
        for p in plot_data:
            fig, ax = plt.subplots(figsize=(15, 1.5))
            ax.plot(np.linspace(0, p['dur'], len(p['y'])), p['y'], color='gray', alpha=0.3, lw=0.5)
            for e in p['events']:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
            status_color = "red" if p['score'] > 0.55 else "green"
            ax.set_title(f"{p['name']} | Confidence: {p['score']:.0%}", color=status_color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            st.divider()

    st.download_button("Export Results", df.to_csv(index=False).encode('utf-8'), "forensic_results.csv")
