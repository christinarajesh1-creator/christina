import streamlit as st
import numpy as np
import librosa
import io
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd

def analyze_breaths(y, sr, filename):
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    
    # More accurate breath detection
    silence_threshold = np.percentile(rms, 5)  # Stricter
    breath_times = times[rms < silence_threshold * 0.6]
    
    # Human rhythm only (3-10s gaps)
    events = []
    last_t = 0
    for t in breath_times:
        gap = t - last_t
        if 3.0 < gap < 10.0:
            events.append(t)
            last_t = t
    
    duration = len(y) / sr
    
    if len(events) < 3:
        return {
            "filename": filename,
            "duration": duration,
            "breath_count": len(events),
            "avg_ibi": 0,
            "ibi_cv": 1.0,
            "human_score": 0.1,
            "breath_events": events
        }
    
    # Accurate IBI analysis
    ibis = np.diff(events)
    avg_ibi = np.mean(ibis)
    ibi_cv = np.std(ibis) / avg_ibi  # Coefficient of variation
    
    # Human-like breathing score (HIGH = HUMAN)
    breath_density = len(events) / duration
    human_score = min(ibi_cv * 0.7 + breath_density * 3 + 0.2, 1.0)
    
    return {
        "filename": filename,
        "duration": duration,
        "breath_count": len(events),
        "avg_ibi": avg_ibi,
        "ibi_cv": ibi_cv,
        "human_score": human_score,
        "breath_events": events
    }

st.set_page_config(page_title="PneumaForensic Batch", layout="wide")

st.title("🫁 PneumaForensic Batch")

uploaded_files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a'], 
                                 accept_multiple_files=True)

if uploaded_files:
    results = []
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        filename = file.name
        file_bytes = io.BytesIO(file.read())
        
        y, sr = librosa.load(file_bytes, sr=22050)
        result = analyze_breaths(y, sr, filename)
        results.append(result)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    df = pd.DataFrame(results)
    
    # Results table
    st.subheader("Results")
    st.dataframe(df[['filename', 'breath_count', 'ibi_cv', 'human_score']].round(3), 
                use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Files", len(results))
    col2.metric("Avg Human Score", f"{df['human_score'].mean():.1%}")
    col3.metric("Breaths Total", df['breath_count'].sum())
    
    # Bar chart
    st.subheader("Human Scores")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if s > 0.6 else 'orange' if s > 0.3 else 'red' for s in df['human_score']]
    ax.bar(range(len(df)), df['human_score'], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Human Score')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f[:12] for f in df['filename']], rotation=45)
    st.pyplot(fig)
    
    # Waveforms (top 6)
    st.subheader("Breath Detection")
    for i, result in enumerate(results[:6]):
        col1, col2 = st.columns([3, 1])
        with col1:
            file_bytes = io.BytesIO(uploaded_files[i].read())
            y_plot, sr_plot = librosa.load(file_bytes, sr=22050)
            
            fig, ax = plt.subplots(figsize=(12, 3))
            time_axis = np.linspace(0, len(y_plot)/sr_plot, len(y_plot))
            ax.plot(time_axis, y_plot, 'gray', alpha=0.7, linewidth=0.5)
            
            breath_times = result['breath_events']
            if len(breath_times) > 0:
                for t in breath_times:
                    ax.axvline(x=t, color='red', linestyle='--', linewidth=2)
                ax.scatter(breath_times, np.zeros(len(breath_times)), 
                          color='red', s=60, marker='v')
            
            ax.set_title(f"{result['filename']} ({result['breath_count']} breaths)")
            ax.set_xlabel('Time (s)')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.metric("Human", f"{result['human_score']:.0%}")
            st.metric("Breaths", result['breath_count'])
    
    # Download
    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "results.csv")

else:
    st.info("Upload multiple audio files")
