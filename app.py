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
        return 0.98, {k: "100%" for k in ["Timing", "Shimmer", "Purity", "Presence", "Amplitude", "Splice", "Similarity", "CV"]}

    # 1. BIOLOGICAL SHIMMER (25%) - THE ROGER KILLER
    # Humans have chaotic micro-variations in energy. AI is a steady stream.
    # Higher energy flux = Human. Lower flux = AI.
    S = np.abs(librosa.stft(y))
    shimmer = np.std(librosa.feature.rms(S=S)) / np.mean(librosa.feature.rms(S=S))
    p1_shimmer = np.clip(1.2 - (shimmer * 10), 0, 1)

    # 2. SPECTRAL ENTROPY (20%) - HUMAN PROTECTOR
    # AI breaths are 'flat/pure'. Human breaths are chaotic white noise.
    # We reward the 'noise' in your human WAV files.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0
    # Penalty for being 'too pure' (AI). Roger is clean, humans are messy.
    p2_purity = np.clip(1.0 - (avg_flat * 140), 0, 1)

    # 3. TIMING IRREGULARITY (20%) - TARGETING CV 0.35-0.45
    # Recalibrated: Roger's 0.44 is now flagged heavily. Humans at 0.6+ are Green.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p3_timing = np.clip(1.5 - (ibi_cv * 2.5), 0, 1)

    # 4. DIGITAL SILENCE FLOOR (15%)
    # Humans have mic hiss. AI has absolute zero. Rewards mic floor noise.
    noise_floor = np.std(y[y < np.percentile(y, 8)])
    p4_silence = np.clip(1.0 - (noise_floor * 1200), 0, 1)

    # 5. AMPLITUDE UNIFORMITY (10%)
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p5_amp = np.clip(1.0 - (amp_cv / 0.5), 0, 1)

    # 6. DIGITAL SPLICE (10%) - HARDENED
    # Reduced weight so noisy human mics don't trigger AI flags.
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p6_splice = np.clip(np.std(zcr) * 8, 0, 1) 

    score = (p1_shimmer*0.25) + (p2_purity*0.20) + (p3_timing*0.20) + (p4_silence*0.15) + (p5_amp*0.10) + (p6_splice*0.10)
    
    return round(np.clip(score, 0, 1), 2), {
        "AI Confidence": f"{score:.0%}",
        "Vocal Shimmer (25%)": f"{p1_shimmer:.0%}",
        "Spectral Purity (20%)": f"{p2_purity:.0%}",
        "Timing Regularity (20%)": f"{p3_timing:.0%}",
        "Digital Silence (15%)": f"{p4_silence:.0%}",
        "Amplitude Score (10%)": f"{p5_amp:.0%}",
        "Digital Splice (10%)": f"{p6_splice:.0%}",
        "IBI CV": round(ibi_cv, 2)
    }

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
        st.subheader("📋 6-Parameter Master Report (Hardened)")
        df = pd.DataFrame(all_data)
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
            st.divider()

    st.download_button("Export Report", df.to_csv(index=False).encode('utf-8'), "forensic_report.csv")
