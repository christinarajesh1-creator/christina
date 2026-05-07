import streamlit as st
import numpy as np
import librosa
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

class PneumaForensic:
    @staticmethod
    def advanced_breath_analysis(y, sr, filename=""):
        """Batch-ready breath analysis"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            
            silence_threshold = np.percentile(rms, 8)
            breath_times = times[rms < silence_threshold * 0.7]
            
            events = []
            last_t = 0
            for t in breath_times:
                gap = t - last_t
                if 2.0 < gap < 8.0:
                    events.append(t)
                    last_t = t
            
            # Quick synthetic score calculation
            if len(events) < 2:
                return {
                    "filename": filename,
                    "duration": len(y)/sr,
                    "breath_count": 0,
                    "ibi_reg": 1.0, "synthetic_score": 0.95
                }
            
            ibis = np.diff(events)
            ibi_reg = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 1.0
            ibi_reg = min(ibi_reg, 2.0)
            
            # Simplified weighted score for batch
            synthetic_score = max(0, min(1 - ibi_reg * 0.3 + (1 - len(events)/10), 1.0))
            
            return {
                "filename": filename,
                "duration": len(y)/sr,
                "breath_count": len(events),
                "ibi_reg": ibi_reg,
                "synthetic_score": synthetic_score,
                "breath_events": events  # For plotting
            }
        except:
            return {
                "filename": filename, "duration": 0, 
                "breath_count": 0, "ibi_reg": 1.0, "synthetic_score": 0.95,
                "breath_events": []
            }

# === BATCH STREAMLIT APP ===
st.set_page_config(page_title="🫁 PneumaForensic Batch", layout="wide")

st.title("🫁 **PneumaForensic BATCH ANALYZER**")
st.markdown("**Upload MULTIPLE audio files for side-by-side AI detection**")

# File uploader (MULTIPLE)
uploaded_files = st.file_uploader("📁 Upload audio files (hold Ctrl/Cmd for multiple)", 
                                 type=['wav','mp3','m4a','flac'], accept_multiple_files=True)

if uploaded_files:
    # === BATCH PROCESSING ===
    st.markdown("---")
    progress_bar = st.progress(0)
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        st.info(f"🔍 Analyzing {filename}...")
        
        y, sr = librosa.load(uploaded_file, sr=22050)
        result = PneumaForensic.advanced_breath_analysis(y, sr, filename)
        results.append(result)
        
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
    
    # === RESULTS TABLE ===
    df = pd.DataFrame(results)
    st.subheader("📊 **Batch Results**")
    st.dataframe(df, use_container_width=True)
    
    # === SUMMARY ===
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files", len(results))
    with col2:
        avg_ai = df['synthetic_score'].mean()
        st.metric("Avg AI Score", f"{avg_ai:.1%}")
    with col3:
        human_count = len(df[df['synthetic_score'] < 0.4])
        st.metric("Human Files", human_count)
    
    # === BAR CHART ===
    st.subheader("🎯 **AI Probability Distribution**")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if score < 0.4 else 'orange' if score < 0.7 else 'red' 
              for score in df['synthetic_score']]
    bars = ax.bar(range(len(df)), df['synthetic_score'], color=colors, alpha=0.7)
    ax.set_xlabel('Files')
    ax.set_ylabel('AI Probability')
    ax.set_title('Batch Analysis - AI Detection Scores')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{f[:15]}..." for f in df['filename']], rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # === INDIVIDUAL WAVEFORMS ===
    st.markdown("---")
    st.subheader("🎵 **Individual Breath Analysis**")
    
    # Show first 6 files max (for performance)
    for i, result in enumerate(results[:6]):
        with st.expander(f"📈 {result['filename']} (AI: {result['synthetic_score']:.0%})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Waveform with breaths
                fig, ax = plt.subplots(figsize=(12, 4))
                y, sr = librosa.load(uploaded_files[i], sr=22050)
                time_axis = np.linspace(0, len(y)/sr, len(y))
                
                # Gray speech waves
                ax.plot(time_axis, y, 'gray', alpha=0.7, linewidth=0.8, label='Speech')
                
                # Red dashed breath lines
                breath_times = result['breath_events']
                if len(breath_times) > 0:
                    for t in breath_times:
                        ax.axvline(x=t, color='red', linestyle='--', linewidth=2, alpha=0.8)
                    ax.scatter(breath_times, np.zeros(len(breath_times)), 
                              color='red', s=80, marker='v', zorder=5, label=f'{len(breath_times)} Breaths')
                else:
                    ax.text(0.5, 0.5, '❌ NO BREATHS\n(STRONG AI)', 
                           transform=ax.transAxes, ha='center', fontsize=14,
                           bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
                
                ax.set_title(f"Breath Detection: {result['filename']}")
                ax.set_xlabel('Time (s)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.metric("Breaths", result['breath_count'])
                st.metric("Duration", f"{result['duration']:.1f}s")
                st.metric("AI Score", f"{result['synthetic_score']:.0%}")

    # === DOWNLOAD RESULTS ===
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "💾 Download Results CSV",
        csv_buffer.getvalue(),
        "pneumaforensic_results.csv",
        "text/csv"
    )

else:
    st.info("""
    👆 **Upload multiple audio files** (hold Ctrl/Cmd to select many)
    
    **Perfect for:**
    - Dataset testing
    - ElevenLabs vs Real comparison  
    - RVC model validation
    - Batch AI detection
    """)
    
    st.markdown("""
    ### 🎯 **What to look for:**
    **🟢 HUMAN**: Uneven red breath lines, 4-12 breaths  
    **🔴 AI**: Perfect grid OR no breaths at all
    """)
