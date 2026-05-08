import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

HOP_LENGTH = 512
SR = 22050

def detect_breaths_batch(files):
    """Batch breath detection - ultra simple"""
    results = []
    
    for filename, file_bytes in files:
        try:
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=SR, mono=True)
            duration = len(y) / sr
            
            # Raw RMS
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP_LENGTH)[0]
            times = librosa.frames_to_time(rms, sr=SR, hop_length=HOP_LENGTH)
            
            # Simple smooth
            smooth_rms = np.convolve(rms, np.ones(15)/15, mode='same')
            
            # Breath mask: RMS < 65% smooth
            breath_mask = rms < smooth_rms * 0.65
            
            # Find breath regions
            breath_regions = []
            i = 0
            while i < len(breath_mask):
                if breath_mask[i]:
                    start = i
                    while i < len(breath_mask) and breath_mask[i]:
                        i += 1
                    end = i
                    # Min 10 frames = ~0.23s
                    if end - start >= 10:
                        breath_regions.append((times[start], times[end-1]))
                else:
                    i += 1
            
            breath_count = len(breath_regions)
            
            # SIMPLE SCORING
            if breath_count >= 5:
                ai_score = 0.15
                status = "👤 HUMAN"
            elif breath_count >= 3:
                ai_score = 0.45
                status = "👤 HUMAN"
            elif breath_count >= 1:
                ai_score = 0.70
                status = "🤖 AI"
            else:
                ai_score = 0.92
                status = "🤖 AI"
            
            results.append({
                'filename': filename,
                'duration_s': round(duration, 2),
                'breaths': breath_count,
                'ai_score': round(ai_score, 3),
                'status': status,
                'waveform': y[:60000],
                'rms': rms[:300],
                'smooth_rms': smooth_rms[:300],
                'times': times[:300],
                'breath_times': [t[0] for t in breath_regions],
                'breath_regions': breath_regions
            })
            
        except:
            results.append({
                'filename': filename,
                'duration_s': 0,
                'breaths': 0,
                'ai_score': 1.0,
                'status': 'ERROR',
                'waveform': np.array([]),
                'rms': np.array([]),
                'times': np.array([]),
                'breath_times': [],
                'breath_regions': []
            })
    
    return results

st.title("🔬 Batch Breath Detector")
st.markdown("**Upload multiple files - detects ALL breaths**")

uploaded_files = st.file_uploader(
    "Choose audio files", 
    type=['wav','mp3','m4a','flac','ogg'], 
    accept_multiple_files=True
)

if uploaded_files:
    # Prepare file list
    file_list = [(f.name, f.read()) for f in uploaded_files]
    
    progress = st.progress(0)
    results = detect_breaths_batch(file_list)
    progress.progress(1.0)
    
    df = pd.DataFrame(results)
    
    # RESULTS TABLE
    st.subheader("📊 Batch Results")
    st.dataframe(df[['filename', 'duration_s', 'breaths', 'status', 'ai_score']], 
                use_container_width=True, hide_index=False)
    
    # SUMMARY
    human_count = len(df[df['status'] == '👤 HUMAN'])
    ai_count = len(df[df['status'] == '🤖 AI'])
    col1, col2 = st.columns(2)
    col1.metric("👤 Human", human_count)
    col2.metric("🤖 AI", ai_count)
    
    # VISUALIZATIONS - FIRST 6 FILES
    st.subheader("📈 Breath Visualization")
    cols = st.columns(3)
    
    for i, result in enumerate(results[:6]):
        if len(result['breath_regions']) == 0:
            continue
            
        with cols[i%3]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Waveform with breaths
            ax1.plot(result['waveform'], 'gray', alpha=0.6, linewidth=0.8)
            color = 'green' if result['status'] == '👤 HUMAN' else 'red'
            for start_t, end_t in result['breath_regions'][:8]:
                if start_t * SR < len(result['waveform']):
                    ax1.axvspan(start_t*SR, min(end_t*SR, len(result['waveform'])), 
                               color=color, alpha=0.4)
            ax1.set_title(f"{result['filename'][:40]}")
            ax1.set_ylabel("Amplitude")
            
            # RMS with breaths
            ax2.plot(result['times'], result['rms'], 'cyan', linewidth=1.5, label='RMS')
            ax2.plot(result['times'], result['smooth_rms'], 'orange', linewidth=1, label='Smooth')
            for start_t, end_t in result['breath_regions'][:8]:
                if start_t < result['times'][-1]:
                    ax2.axvspan(start_t, min(end_t, result['times'][-1]), 
                               color=color, alpha=0.5)
            ax2.set_title(f"{result['status']} | {result['breaths']} breaths | Score: {result['ai_score']}")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("RMS Energy")
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
    
    st.success(f"✅ Analyzed {len(results)} files")

# SIMPLE HOWTO
with st.expander("ℹ️ How it works"):
    st.markdown("""
    **Ultra-simple algorithm:**
    
    1. Compute RMS energy envelope
    2. Smooth with 15-frame average  
    3. Find regions where RMS < 65% smooth
    4. Regions >0.23s = breaths
    5. 3+ breaths = human voice
    
    **No complex filtering - pure energy drops**
    """)
