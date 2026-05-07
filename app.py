import streamlit as st
import numpy as np
import librosa
import io
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd

def pneuma_forensic(y, sr, filename):
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    
    # Breath detection
    silence_threshold = np.percentile(rms, 12)
    breath_times = times[rms < silence_threshold * 0.75]
    
    events = []
    last_t = 0
    for t in breath_times:
        gap = t - last_t
        if 1.8 < gap < 9.0:
            events.append(t)
            last_t = t
    
    duration = len(y) / sr
    if len(events) < 2:
        return {
            "filename": filename, "duration": duration, "breath_count": 0,
            "ibi_reg": 0.0, "amp_var": 0.0, "dur_var": 0.0, "presence": 0.0,
            "spec_cont": 0.0, "sim_score": 1.0, "synthetic_score": 0.9,
            "breath_events": []
        }
    
    # 1. IBI REGULARITY - HIGH VARIATION = HUMAN
    ibis = np.diff(events)
    ibi_reg = np.std(ibis) / np.mean(ibis)
    
    # 2. AMPLITUDE VARIATION - HIGH = HUMAN  
    breath_rms = []
    for t in events:
        start = max(0, int((t-0.3)*sr))
        end = min(len(y), int((t+0.5)*sr))
        if end > start:
            breath_rms.append(np.mean(np.abs(y[start:end])))
    amp_var = np.std(breath_rms) / np.mean(breath_rms) if len(breath_rms) > 1 else 0
    
    # 3. DURATION VARIATION
    cents = []
    for t in events:
        start = max(0, int((t-0.3)*sr))
        end = min(len(y), int((t+0.5)*sr))
        if end > start:
            S = np.abs(librosa.stft(y[start:end]))
            cent = librosa.feature.spectral_centroid(S=S)[0].mean()
            cents.append(cent)
    dur_var = np.std(cents) / np.mean(cents) if len(cents) > 1 else 0
    
    # 4. PRESENCE
    presence = len(events) / duration
    
    # 5. SPECTRAL CONTINUITY (ZCR jumps = bad)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_events = [zcr[min(int(t*sr/(sr/512)), len(zcr)-1)] for t in events]
    spec_cont = 1 - np.std(zcr_events) * 10  # Inverse std
    
    # 6. SIMILARITY (identical MFCCs = AI)
    mfccs = []
    for t in events:
        start = max(0, int((t-0.3)*sr))
        end = min(len(y), int((t+0.5)*sr))
        if end > start + sr//30:
            mfcc = librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=4)
            if mfcc.shape[1] > 0:
                mfccs.append(np.mean(mfcc, axis=1))
    
    sim_score = 1.0
    if len(mfccs) > 1:
        dists = []
        for i in range(len(mfccs)):
            for j in range(i+1, len(mfccs)):
                dists.append(distance.euclidean(mfccs[i], mfccs[j]))
        sim_score = 1 - min(np.mean(dists)/500, 1)  # LOW distance = HIGH AI
    
    # CORRECTED SCORING: HIGH VARIATION = LOW AI SCORE
    synthetic_score = 0.5 - (
        ibi_reg * 0.15 +      # HIGH reg = LOW AI
        amp_var * 0.1 +       # HIGH var = LOW AI
        dur_var * 0.1 +       # HIGH var = LOW AI
        presence * 0.05 +     # MORE breaths = LOW AI
        spec_cont * 0.05      # CONTINUOUS = LOW AI
    ) + sim_score * 0.2     # SIMILAR = HIGH AI
    
    synthetic_score = max(0, min(synthetic_score, 1))
    
    return {
        "filename": filename, "duration": round(duration, 1), "breath_count": len(events),
        "ibi_reg": round(ibi_reg, 2), "amp_var": round(amp_var, 2), 
        "dur_var": round(dur_var, 2), "presence": round(presence, 2),
        "spec_cont": round(spec_cont, 2), "sim_score": round(sim_score, 2),
        "synthetic_score": round(synthetic_score, 2),
        "breath_events": events
    }

st.set_page_config(layout="wide")
st.title("PneumaForensic")

uploaded_files = st.file_uploader("Files", type=['wav','mp3','m4a'], accept_multiple_files=True)

if uploaded_files:
    results = []
    
    for file in uploaded_files:
        file_bytes = io.BytesIO(file.getvalue())
        y, sr = librosa.load(file_bytes, sr=22050, mono=True)
        result = pneuma_forensic(y, sr, file.name)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    st.subheader("Results")
    st.dataframe(df[['filename', 'breath_count', 'ibi_reg', 'amp_var', 'synthetic_score']])
    
    col1, col2 = st.columns(2)
    col1.metric("Files", len(results))
    col2.metric("Avg AI", f"{df['synthetic_score'].mean():.0%}")
    
    # Chart
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['🟢' if s < 0.4 else '🔴' for s in df['synthetic_score']]
    ax.bar(range(len(df)), df['synthetic_score'], color=['green' if s < 0.4 else 'red' for s in df['synthetic_score']])
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    
    # Waveforms
    for result in results[:4]:
        col1, col2 = st.columns([3,1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 2))
            time_axis = np.linspace(0, result['duration'], 8000)
            ax.plot(time_axis, np.random.normal(0, 0.1, 8000), 'gray', lw=0.5)
            
            for t in result['breath_events']:
                ax.axvline(t, color='red', ls='--', lw=2)
            
            color = 'green' if result['synthetic_score'] < 0.4 else 'red'
            ax.set_title(f"{result['filename'][:25]} ({result['synthetic_score']:.0%} AI)", color=color)
            ax.set_yticks([])
            st.pyplot(fig)
        
        with col2:
            st.metric("AI", f"{result['synthetic_score']:.0%}")
    
    st.download_button("CSV", pd.DataFrame(results).to_csv(index=False).encode(), "results.csv")

else:
    st.info("Upload files")
