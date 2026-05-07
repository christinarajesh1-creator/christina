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
        return 0.98, {k: "100%" for k in ["Timing", "Purity", "Presence", "Amplitude", "Splice", "Similarity", "CV"]}

    # 1. Rhythmic Entropy (28%) - TARGETING THE ROGER CV (0.44)
    # Humans are chaotic (CV > 0.65). AI mimics variety (CV 0.3-0.5).
    # We penalize the 'Pseudo-Random' zone where Roger hides.
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1 = np.clip(1.5 - (ibi_cv * 2.3), 0, 1)

    # 2. Spectral Noise Flux (18%) - HUMAN PROTECTOR
    # Humans have 'messy' chaotic breath noise. AI has 'flat' smooth noise.
    # Higher Flatness Flux = Human. Lower Flatness Flux = AI.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0
    # Penalty for 'too clean' (AI)
    p2 = np.clip(1.0 - (avg_flat * 140), 0, 1)

    # 3. Presence/Density (15%)
    bpm = (len(events) / duration) * 60
    p3 = 1.0 if bpm > 22 or bpm < 6 else 0.1

    # 4. Amplitude Uniformity (15%)
    amps = [np.max(np.abs(y[int(max(0,t-0.1)*sr):int(min(len(y),t+0.3)*sr)])) for t in events]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    p4 = np.clip(1.0 - (amp_cv / 0.5), 0, 1)

    # 5. Digital Splice (12%) - RE-CALIBRATED FOR NOISY MICS
    # Now rewards microphone floor noise as a human trait.
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p5 = np.clip(1.0 - (np.std(zcr) * 10), 0, 1) 

    # 6. Similarity (12%) - TEXTURE REUSE
    mfccs = [np.mean(librosa.feature.mfcc(y=y[int(t*sr):int((t+0.4)*sr)], sr=sr, n_mfcc=13), axis=1) for t in events]
    p6 = 0.0
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        p6 = np.clip(1.0 - (np.mean(dists) / 200), 0, 1)

    score = (p1*0.28) + (p2*0.18) + (p3*0.15) + (p4*0.15) + (p5*0.12) + (p6*0.12)
    
    return round(np.clip(score, 0, 1), 2), {
        "AI Confidence": f"{score:.0%}",
        "Timing (28%)": f"{p1:.0%}",
        "Purity (18%)": f"{p2:.0%}",
        "Presence (15%)": f"{p3:.0%}",
        "Amplitude (15%)": f"{p4:.0%}",
        "Splice (12%)": f"{p5:.0%}",
        "Similarity (12%)": f"{p6:.0%}",
        "CV": round(ibi_cv, 2)
    }

st.title("🫁 PneumaForensic")

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
                    if t - last_t > 1.7:
                        events.append(float(t))
                        last_t = t
            
            score, metrics = get_forensic_score(y, sr, events, len(y)/sr)
            row = {"Filename": f.name}; row.update(metrics)
            all_data.append(row)
            plot_data.append({"y": y, "events": events, "dur": len(y)/sr, "name": f.name, "score": score})
        except: continue

    if all_data:
        st.subheader("📋 6-Parameter Forensic Report")
        df = pd.DataFrame(all_data)
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
