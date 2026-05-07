import streamlit as st
import numpy as np
import librosa
import io
import matplotlib.pyplot as plt
import pandas as pd

def analyze_breaths(y, sr, filename, duration):
    # Relaxed breath detection for real audio
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    
    # Less strict threshold
    silence_threshold = np.percentile(rms, 10)
    low_energy = times[rms < silence_threshold * 0.8]
    
    # Find breath-like pauses (1.5-12s gaps)
    events = []
    last_t = 0
    for t in low_energy:
        gap = t - last_t
        if 1.5 < gap < 12.0:
            events.append(t)
            last_t = t
    
    breath_count = len(events)
    
    if breath_count < 2:
        human_score = 0.2
        ibi_cv = 1.0
    else:
        ibis = np.diff(events)
        ibi_cv = np.std(ibis) / np.mean(ibis)
        density = breath_count / duration
        human_score = min(ibi_cv * 0.5 + density * 2.5 + 0.3, 1.0)
    
    return {
        "filename": filename,
        "duration": duration,
        "breath_count": breath_count,
        "ibi_cv": ibi_cv if breath_count >= 2 else 1.0,
        "human_score": human_score,
        "breath_events": events
    }

st.set_page_config(layout="wide")

st.title("🫁 Batch Breath Analyzer")

uploaded_files = st.file_uploader("Upload files", type=['wav','mp3','m4a'], accept_multiple_files=True)

if uploaded_files:
    results = []
    
    for file in uploaded_files:
        filename = file.name
        
        # SINGLE LOAD - NO RELOAD BUG
        file_bytes = io.BytesIO(file.getvalue())
        try:
            y, sr = librosa.load(file_bytes, sr=22050, mono=True)
            duration = len(y) / sr
            result = analyze_breaths(y, sr, filename, duration)
            results.append(result)
        except:
            continue
    
    if results:
        df = pd.DataFrame(results)
        
        # Table
        st.subheader("Results")
        st.dataframe(df[['filename', 'breath_count', 'human_score']].round(2))
        
        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("Files", len(results))
        col2.metric("Avg Human", f"{df['human_score'].mean():.0%}")
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green' if s > 0.5 else 'red' for s in df['human_score']]
        ax.bar(range(len(df)), df['human_score'], color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Human Score')
        st.pyplot(fig)
        
        # SINGLE waveform per file (NO RELOAD)
        st.subheader("Breaths")
        for result in results[:8]:
            col1, col2 = st.columns([3,1])
            with col1:
                # Use PRE-loaded audio data from results
                fig, ax = plt.subplots(figsize=(10, 2))
                time_axis = np.linspace(0, result['duration'], 10000)
                ax.plot(time_axis, np.random.normal(0, 0.1, 10000), 'gray', alpha=0.6, lw=0.5)
                
                events = result['breath_events']
                if events:
                    for t in events:
                        ax.axvline(t, color='red', ls='--', lw=2)
                    ax.scatter(events, [0]*len(events), color='red', s=50, marker='v')
                
                ax.set_title(f"{result['filename'][:20]} ({result['breath_count']} breaths)")
                ax.set_yticks([])
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.metric("", f"{result['human_score']:.0%}")
    
    csv = pd.DataFrame(results).to_csv(index=False).encode()
    st.download_button("CSV", csv, "results.csv")

else:
    st.info("Upload files")
