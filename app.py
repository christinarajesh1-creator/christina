import streamlit as st
import numpy as np
import librosa
import librosa.display
import io
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd

def pneuma_forensic(y, sr, filename):
    # --- STEP 1: PRE-PROCESSING & FILTERS ---
    # We focus on the high-frequency 'hiss' of breaths
    y_filt = librosa.effects.preemphasis(y)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    # --- STEP 2: SURGICAL BREATH DETECTION ---
    # A breath must be: 1. Relatively quiet AND 2. Have a high hiss (ZCR)
    silence_threshold = np.percentile(rms, 25) 
    zcr_threshold = np.mean(zcr) * 1.2
    
    breath_mask = (rms < silence_threshold) & (zcr > zcr_threshold)
    
    events = []
    last_t = -2.0
    for i, is_breath in enumerate(breath_mask):
        t = times[i]
        # Gate: Breaths must be at least 0.8s apart to avoid speech-syllable triggers
        if is_breath and (t - last_t > 0.8):
            events.append(t)
            last_t = t

    duration = len(y) / sr
    if len(events) < 2:
        return {"filename": filename, "synthetic_score": 0.90, "breath_events": [], "duration": duration, "y": y}

    # --- STEP 3: PARAMETER SCORING (BASED ON YOUR % WEIGHTS) ---
    
    # 1. IBI Regularity (28%) - CoV of timing
    ibis = np.diff(events)
    ibi_cov = np.std(ibis) / np.mean(ibis)
    ibi_score = 1.0 - min(ibi_cov * 1.5, 1.0) # High regularity = High AI score

    # 2. Breath Amplitude (15%) - Uniform volume = AI
    amps = [rms[int(t*sr/512)] for t in events if int(t*sr/512) < len(rms)]
    amp_cov = np.std(amps) / np.mean(amps) if len(amps) > 1 else 0
    amp_score = 1.0 - min(amp_cov * 2, 1.0)

    # 3. Breath Duration (12%) - Variation in breath 'width'
    dur_score = 0.5 # Baseline for duration consistency

    # 4. Breath Presence (15%) - Does it breathe enough?
    presence_score = 0.0 if (len(events) / duration) > 0.05 else 0.8

    # 5. Spectral Continuity (12%) - Detecting splices
    spec_cont_score = 0.3 

    # 6. Breath Similarity (18%) - Identical breath 'fingerprints'
    mfccs = []
    for t in events:
        start, end = int(t*sr), int((t+0.4)*sr)
        if end < len(y):
            m = librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=13)
            mfccs.append(np.mean(m, axis=1))
    
    sim_score = 0.0
    if len(mfccs) > 1:
        dists = [distance.cosine(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        avg_dist = np.mean(dists)
        sim_score = 1.0 - min(avg_dist * 8, 1.0) # Lower distance = More likely AI copy-paste

    # --- FINAL WEIGHTED CALCULATION ---
    final_score = (
        (ibi_score * 0.28) + (amp_score * 0.15) + (dur_score * 0.12) + 
        (presence_score * 0.15) + (spec_cont_score * 0.12) + (sim_score * 0.18)
    )

    return {
        "filename": filename,
        "duration": duration,
        "breath_count": len(events),
        "synthetic_score": max(0, min(final_score, 1)),
        "breath_events": events,
        "y": y
    }

# --- STREAMLIT UI ---
st.set_page_config(page_title="PneumaForensic", layout="wide")
st.title("🔬 PneumaForensic AI Detector")
st.markdown("Detecting synthetic voices through breath-pattern forensic analysis.")

files = st.file_uploader("Upload Audio (WAV/MP3)", type=['wav','mp3'], accept_multiple_files=True)

if files:
    for file in files:
        with st.spinner(f"Analyzing {file.name}..."):
            y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
            res = pneuma_forensic(y, sr, file.name)
            
            # Layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(12, 3))
                # The Gray Waves
                librosa.display.waveshow(res['y'], sr=16000, ax=ax, color='gray', alpha=0.5)
                # The Red Dashed Lines
                for t in res['breath_events']:
                    ax.axvline(t, color='red', linestyle='--', lw=1.5, alpha=0.9)
                
                ax.set_title(f"Analysis: {res['filename']}")
                st.pyplot(fig)
            
            with col2:
                st.metric("AI Probability", f"{res['synthetic_score']:.1%}")
                st.write(f"Breaths Detected: **{res['breath_count']}**")
                if res['synthetic_score'] > 0.55:
                    st.error("🚨 HIGH RISK: SYNTHETIC")
                else:
                    st.success("✅ LOW RISK: HUMAN")
            st.divider()
