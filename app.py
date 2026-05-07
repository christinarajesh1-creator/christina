import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.98, {"Status": "Abnormal (Silent/AI)"}

    # 1. Rhythmic Entropy (35%) - TARGETING THE ROGER CV (0.44)
    # Humans are chaotic (CV > 0.65). AI mimics variety (CV 0.3-0.5).
    # We punish the 'Pseudo-Random' zone where Roger hides.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1 = np.clip(1.5 - (ibi_cv * 2.3), 0, 1)

    # 2. Spectral Noise Flux (25%) - HUMAN PROTECTOR
    # Humans have 'messy' chaotic breath noise. AI has 'flat' smooth noise.
    # Higher Flatness Flux = Human. Lower Flatness Flux = AI.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0
    # Penalty for 'too clean' (AI)
    p2 = np.clip(1.0 - (avg_flat * 100), 0, 1)

    # 3. Digital Silence Floor (20%) - REVERSED LOGIC
    # Humans have mic hiss. AI has zero noise floor.
    # We punish 'Perfect Silence' and reward 'Microphone Hiss'.
    noise_floor = np.std(y[y < np.percentile(y, 10)])
    p3 = np.clip(1.0 - (noise_floor * 800), 0, 1)

    # 4. Digital Continuity (20%)
    # Lowered weight so noisy human mics don't trigger AI flags.
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p4 = np.clip(np.std(zcr) * 10, 0, 1) 

    score = (p1 * 0.35) + (p2 * 0.25) + (p3 * 0.20) + (p4 * 0.20)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Regularity (35%)": f"{p1:.0%}",
        "Spectral Purity (25%)": f"{p2:.0%}",
        "Digital Silence (20%)": f"{p3:.0%}",
        "Digital Splice (20%)": f"{p4:.0%}",
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
