import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 3:
        return 0.98, {"Status": "Suspiciously Low Breaths"}

    # 1. Rhythmic Chaos (30%) - THE ROGER KILLER
    # Roger mimics variety (CV 0.44). Humans are chaotic (CV > 0.65).
    # We penalize any pattern that is "pseudo-random" (0.3 - 0.5 range).
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1_timing = np.clip(1.5 - (ibi_cv * 2.2), 0, 1)

    # 2. Spectral Texture (25%) - HUMAN PROTECTION
    # AI breaths are "flat/clean". Human breaths are "chaotic noise".
    # High flatness flux = Human. Low flatness flux = AI.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    p2_texture = np.clip(1.0 - (np.mean(flatness) * 40), 0, 1) if flatness else 0.5

    # 3. Noise Floor Entropy (20%)
    # Humans have microphone hiss (entropy). AI has zero floor (pure digital silence).
    noise_floor = np.std(y[y < np.percentile(y, 8)])
    p3_silence = np.clip(1.0 - (noise_floor * 800), 0, 1)

    # 4. Digital Continuity (25%) - HARDENED
    # Lowered sensitivity so background noise doesn't trigger 100% AI flags.
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p4_splice = np.clip(np.std(zcr) * 12, 0, 1) 

    # FINAL WEIGHTED SCORE
    score = (p1_timing * 0.30) + (p2_texture * 0.25) + (p3_silence * 0.20) + (p4_splice * 0.25)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Timing Regularity": f"{p1_timing:.0%}",
        "Spectral Purity": f"{p2_texture:.0%}",
        "Digital Silence": f"{p3_silence:.0%}",
        "Digital Splice": f"{p4_splice:.0%}",
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
            # Use a more aggressive gate to find clear inhales
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
        # Master Report Table
        df = pd.DataFrame(all_data)
        st.subheader("📋 Forensic Report")
        st.dataframe(df, use_container_width=True)

        # Waveform Visuals
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
