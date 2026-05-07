import streamlit as st
import numpy as np
import librosa
import io
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd

def pneuma_forensic(y, sr, filename):
    """6-parameter forensic AI detection with strict weighting"""
    try:
        y = librosa.util.normalize(y)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        duration = len(y) / sr
        
        # Breath detection
        silence_threshold = np.percentile(rms, 15)
        breath_candidates = times[rms < silence_threshold * 0.7]
        
        events = []
        last_t = -5.0
        for t in breath_candidates:
            if t - last_t > 1.8: # Prevent double-counting one breath
                events.append(float(t))
                last_t = t
        
        # 1. IBI Regularity (28%) - AI is too rhythmic
        if len(events) > 1:
            ibis = np.diff(events)
            # Low CV (Standard Deviation/Mean) = High Regularity = AI
            ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
            p1 = np.clip(1.0 - (ibi_cv * 2), 0, 1) 
        else:
            p1 = 1.0 # High suspicion if only one breath

        # 2. Breath Amplitude (15%) - AI is uniform volume
        amps = []
        for t in events:
            start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
            amps.append(np.max(np.abs(y[start:end])) if start < end else 0.01)
        amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 1 else 0
        p2 = np.clip(1.0 - (amp_cv * 2), 0, 1)

        # 3. Breath Duration (12%) - AI has fixed length
        # Using ZCR to approximate breath "texture" duration
        p3 = 0.8 if len(set([round(x, 1) for x in (np.diff(events) if len(events)>1 else [0])])) < len(events)/2 else 0.2

        # 4. Breath Presence (15%) - Lack of breathing
        expected = duration / 8
        p4 = np.clip(1.0 - (len(events) / expected), 0, 1)

        # 5. Spectral Continuity (12%) - Splice artifacts (ZCR spikes)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        p5 = np.clip(np.std(zcr) * 15, 0, 1)

        # 6. Breath Similarity (18%) - Copy-pasted breath samples
        mfccs = []
        for t in events:
            start, end = int(max(0, t-0.1)*sr), int(min(len(y), t+0.3)*sr)
            if (end - start) > 512:
                m = librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=13)
                mfccs.append(np.mean(m, axis=1))
        
        p6 = 0.0
        if len(mfccs) > 1:
            dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
            # Low distance = High similarity = AI
            p6 = np.clip(1 - (np.mean(dists) / 300), 0, 1)
        else:
            p6 = 0.8 # Suspiciously unique or absent

        # Weighting calculation
        synthetic_score = (p1 * 0.28) + (p2 * 0.15) + (p3 * 0.12) + (p4 * 0.15) + (p5 * 0.12) + (p6 * 0.18)

        return {
            "filename": filename[:30], 
            "duration": round(duration,1), 
            "breath_count": len(events),
            "synthetic_score": round(np.clip(synthetic_score, 0, 1), 2),
            "breath_events": events,
            "y": y
        }
    except:
        return {"filename": filename[:30], "synthetic_score": 0.9, "breath_count": 0}

st.set_page_config(layout="wide", page_title="PneumaForensic")
st.title("🫁 PneumaForensic")

uploaded_files = st.file_uploader("Upload Batch", type=['wav','mp3','m4a'], accept_multiple_files=True)

if uploaded_files:
    results = []
    for file in uploaded_files:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
        results.append(pneuma_forensic(y, sr, file.name))
    
    df = pd.DataFrame(results)
    st.dataframe(df[['filename', 'breath_count', 'synthetic_score']], use_container_width=True)

    for res in results:
        fig, ax = plt.subplots(figsize=(15, 2))
        t_axis = np.linspace(0, res['duration'], len(res['y']))
        ax.plot(t_axis, res['y'], color='gray', alpha=0.4, lw=0.5) # Gray Waves
        
        for e in res['breath_events']:
            ax.axvline(e, color='red', ls='--', lw=1.5) # Red Dashed Lines
        
        color = 'red' if res['synthetic_score'] > 0.55 else 'green'
        ax.set_title(f"{res['filename']} (AI Score: {res['synthetic_score']:.0%})", color=color, loc='left', fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)
        st.divider()

    st.download_button("Download Report", df.to_csv().encode(), "forensic_report.csv")
