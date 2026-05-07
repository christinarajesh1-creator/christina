import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def get_forensic_score(y, sr, events, duration):
    if len(events) < 2:
        return 0.99, {"Status": "Digital Silence/No Breaths"}

    # 1. BIOLOGICAL CHAOS (35%) - THE HUMAN PROTECTOR
    # Humans have messy breaths. AI (Roger) has 'clean' breaths.
    # We measure Spectral Flatness. High flatness = Chaotic/Human.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0.5
    # If the sound is 'too clean' (Low Flatness), the AI score goes UP.
    p1_chaos = np.clip(1.0 - (avg_flat * 100), 0, 1)

    # 2. DIGITAL SILENCE (25%) - THE MIC HISS TEST
    # Humans have mic hiss (noise floor). AI has absolute zero.
    # We penalize "Perfect Silence" between words.
    noise_floor = np.std(y[y < np.percentile(y, 10)])
    # If it's too silent (AI), score goes UP. If it's noisy (Human), score goes DOWN.
    p2_silence = np.clip(1.0 - (noise_floor * 1500), 0, 1)

    # 3. PATTERN STABILITY (20%) - THE STEADINESS TEST
    # Humans drift in energy. AI is a steady mathematical stream.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    flux_std = np.std(onset_env)
    # If the voice is "too steady" (Low Flux), it's AI.
    p3_stability = np.clip(1.0 - (flux_std * 0.4), 0, 1)

    # 4. TIMING REGULARITY (20%) - TARGETING ROGER (0.4 CV)
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    # Penalty kicks in for any CV below 0.55 (Roger's zone).
    p4_timing = np.clip(1.3 - (ibi_cv * 2.2), 0, 1)

    # FINAL TOTAL
    score = (p1_chaos * 0.35) + (p2_silence * 0.25) + (p3_stability * 0.20) + (p4_timing * 0.20)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Spectral Purity": f"{p1_chaos:.0%}",
        "Digital Silence": f"{p2_silence:.0%}",
        "Vocal Stability": f"{p3_stability:.0%}",
        "Timing Regularity": f"{p4_timing:.0%}",
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
            # Threshold to ignore the background noise in your WAV files
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
        # 1. Master Table (Compare everything at once)
        df = pd.DataFrame(all_data)
        st.subheader("📋 Forensic Results Table")
        st.dataframe(df, use_container_width=True)

        # 2. Waveforms
        st.subheader("📊 Visual Waveform Analysis")
        for p in plot_data:
            fig, ax = plt.subplots(figsize=(15, 1.6))
            ax.plot(np.linspace(0, p['dur'], len(p['y'])), p['y'], color='gray', alpha=0.3, lw=0.5)
            for e in p['events']:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
            color = "red" if p['score'] > 0.55 else "green"
            ax.set_title(f"{p['name']} | AI Confidence: {p['score']:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            st.divider()

    st.download_button("Export Results", df.to_csv(index=False).encode('utf-8'), "pneuma_results.csv")
