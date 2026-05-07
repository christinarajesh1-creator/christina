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

    # 1. VOCAL FATIGUE (20%) - NEW
    # Humans drift in pitch/intensity as they breathe. AI is perfectly stable.
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    # High variance in pitch = Human. Low variance = AI.
    pitch_drift = np.std(pitch_values) / np.mean(pitch_values) if len(pitch_values) > 0 else 0
    p1_fatigue = np.clip(1.0 - (pitch_drift * 15), 0, 1)

    # 2. SPECTRAL PURITY (20%) - ROGER KILLER
    # AI breaths are 'too clean'. Human breaths are chaotic noise.
    flatness = [np.mean(librosa.feature.spectral_flatness(y=y[int(t*sr):int((t+0.3)*sr)])) for t in events]
    avg_flat = np.mean(flatness) if flatness else 0.5
    # Penalize 'Digital Cleanliness'
    p2_purity = np.clip(1.0 - (avg_flat * 45), 0, 1)

    # 3. TIMING REGULARITY (20%) - CATCHING THE 0.4 CV
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    # Recalibrated: Roger's 0.40 CV is now flagged as High AI Risk.
    p3_reg = np.clip(1.3 - (ibi_cv * 2), 0, 1)

    # 4. DIGITAL SILENCE (15%) - HUMAN PROTECTOR
    # Humans have mic hiss. AI has zero noise floor. 
    # Rewards 'messy' human recordings.
    noise_floor = np.std(y[y < np.percentile(y, 10)])
    p4_silence = np.clip(1.0 - (noise_floor * 550), 0, 1)

    # 5. DIGITAL SPLICE (15%)
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    p5_splice = np.clip(np.std(zcr) * 30, 0, 1) 

    # 6. BREATH DENSITY (10%)
    bpm = (len(events) / duration) * 60
    p6_density = 1.0 if bpm > 22 or bpm < 6 else 0.1

    # FINAL HARDENED SCORE
    score = (p1_fatigue*0.20) + (p2_purity*0.20) + (p3_reg*0.20) + (p4_silence*0.15) + (p5_splice*0.15) + (p6_density*0.10)
    
    metrics = {
        "AI Score": f"{score:.0%}",
        "Vocal Fatigue (20%)": f"{p1_fatigue:.0%}",
        "Spectral Purity (20%)": f"{p2_purity:.0%}",
        "Regularity (20%)": f"{p3_reg:.0%}",
        "Digital Silence (15%)": f"{p4_silence:.0%}",
        "Digital Splice (15%)": f"{p5_splice:.0%}",
        "Density (10%)": f"{p6_density:.0%}",
        "IBI CV": round(ibi_cv, 2)
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

st.title("🫁 PneumaForensic: Advanced Deep-Scan")
st.markdown("Forensic analysis optimized to distinguish biological prosody from ElevenLabs Roger AI.")

files = st.file_uploader("Batch Upload (WAV/MP3)", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    all_data = []
    plot_data = []
    
    for f in files:
        try:
            # Low sample rate for faster processing of batch files
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
        except Exception as e:
            st.error(f"Could not process {f.name}: {e}")

    if all_data:
        # 1. Master Results Table
        st.subheader("📋 Batch Analysis Report")
        df = pd.DataFrame(all_data)
        st.dataframe(df, use_container_width=True)

        # 2. Waveform Visuals
        st.subheader("📊 Breath Pattern Analysis")
        for p in plot_data:
            fig, ax = plt.subplots(figsize=(15, 1.8))
            # The Gray Waves
            ax.plot(np.linspace(0, p['dur'], len(p['y'])), p['y'], color='gray', alpha=0.3, lw=0.5)
            # The Red Dashed Lines
            for e in p['events']:
                ax.axvline(e, color='red', linestyle='--', lw=1.5)
            
            color = "red" if p['score'] > 0.55 else "green"
            ax.set_title(f"{p['name']} | AI Confidence: {p['score']:.0%}", color=color, loc='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            st.divider()

    st.download_button("Export Forensic Report", df.to_csv(index=False).encode('utf-8'), "pneuma_forensics.csv")
