import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.set_page_config(layout="wide", page_title="PneumaForensic")

def analyze_audio(y, sr, filename):
    try:
        y = librosa.util.normalize(y)
        duration = len(y) / sr
        
        # Feature Extraction
        rms = librosa.feature.rms(y=y)[0] 
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        
        # Detection Logic
        threshold = np.percentile(rms, 15)
        breath_candidates = times[rms < threshold * 0.7]
        
        events = []
        last_t = -5.0
        for t in breath_candidates:
            if t - last_t > 1.8: 
                events.append(float(t))
                last_t = t
        
        # Default AI score if no biological breathing is found
        if len(events) < 2:
            return {
                "filename": filename, "score": 0.85, "events": [], 
                "y": y, "sr": sr, "duration": duration, "details": "No Breaths Detected"
            }

        # --- 6 PARAMETER CALCULATIONS ---
        
        # 1. IBI Regularity (28%)
        ibis = np.diff(events)
        p1 = 1.0 - (np.std(ibis) / np.mean(ibis)) if np.mean(ibis) > 0 else 1.0
        
        # 2. Breath Amplitude (15%)
        amps = []
        for t in events:
            start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
            amps.append(np.max(np.abs(y[start:end])) if start < end else 0.01)
        p2 = 1.0 - (np.std(amps) / np.mean(amps)) if np.mean(amps) > 0 else 1.0
        
        # 3. Breath Duration (12%)
        # Calculates if the silences are identical in length (Copy-Paste)
        p3 = 1.0 if len(set([round(x, 1) for x in ibis])) < (len(ibis) / 2) else 0.2
        
        # 4. Breath Presence (15%)
        # Scoring high if breaths are suspiciously absent
        p4 = np.clip(1.0 - (len(events) / (duration / 8)), 0, 1)
        
        # 5. Spectral Continuity (12%)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        p5 = np.clip(np.std(zcr) * 20, 0, 1)
        
        # 6. Breath Similarity (18%)
        mfccs = []
        for t in events:
            start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
            if (end - start) > 512:
                m = librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=13)
                mfccs.append(np.mean(m, axis=1))
        
        p6 = 0.5
        if len(mfccs) > 1:
            dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
            p6 = np.clip(1 - (np.mean(dists) / 400), 0, 1)

        final_score = (p1*0.28) + (p2*0.15) + (p3*0.12) + (p4*0.15) + (p5*0.12) + (p6*0.18)
        
        return {
            "filename": filename,
            "score": round(np.clip(final_score, 0, 1), 2),
            "events": events,
            "y": y,
            "sr": sr,
            "duration": duration
        }
    except Exception as e:
        return {"filename": filename, "score": 0.0, "error": str(e)}

# --- STREAMLIT UI ---
st.title("🫁 PneumaForensic")

uploaded_files = st.file_uploader("Upload Files for Batch Analysis", type=['wav', 'mp3', 'm4a'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    for uploaded_file in uploaded_files:
        try:
            # Read file once
            file_bytes = uploaded_file.read()
            data, sample_rate = librosa.load(io.BytesIO(file_bytes), sr=16000)
            analysis = analyze_audio(data, sample_rate, uploaded_file.name)
            if analysis:
                results_list.append(analysis)
        except Exception as load_error:
            st.error(f"Load Error: {uploaded_file.name} - {load_error}")

    if results_list:
        # Batch Summary Table
        summary_df = pd.DataFrame([{"File": r['filename'], "AI Score": f"{r.get('score', 0):.0%}"} for r in results_list])
        st.table(summary_df)

        # Charts
        for r in results_list:
            if "y" in r:
                fig, ax = plt.subplots(figsize=(15, 2))
                
                # The Gray Waves (Audio signal)
                t_axis = np.linspace(0, r['duration'], len(r['y']))
                ax.plot(t_axis, r['y'], color='gray', alpha=0.3, lw=0.5)
                
                # The Red Dashed Lines (Detected breaths)
                for breath_time in r.get('events', []):
                    ax.axvline(breath_time, color='red', linestyle='--', lw=1.5)
                
                color = "red" if r['score'] > 0.6 else "green"
                ax.set_title(f"{r['filename']} | AI Confidence: {r['score']:.0%}", color=color, loc='left', fontweight='bold')
                ax.set_xlim(0, r['duration'])
                ax.set_ylim(-1, 1)
                ax.axis('off')
                st.pyplot(fig)
                st.divider()

        st.download_button("Download Forensic Report", summary_df.to_csv(index=False).encode('utf-8'), "pneuma_results.csv")
