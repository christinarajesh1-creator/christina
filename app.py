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

@st.cache_data
def detect_breaths_simple(file_bytes, filename):
    """SIMPLEST breath detection that actually works"""
    try:
        # Load raw
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True)
        duration = len(y) / sr
        
        st.write(f"DEBUG: {filename} - duration: {duration:.2f}s, samples: {len(y)}")
        
        # Basic RMS - NO fancy filtering
        hop_length = 512
        frame_length = 1024
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(rms, sr=sr, hop_length=hop_length)
        
        # SIMPLE moving average smooth
        window = 15
        smooth_rms = np.convolve(rms, np.ones(window)/window, mode='same')
        
        # **SUPER SIMPLE BREATH DETECTION**
        # Just find where RMS drops below 60% of local smooth
        breath_threshold = 0.6
        breath_mask = rms < (smooth_rms * breath_threshold)
        
        # Count consecutive low energy frames as single breath
        breath_regions = []
        in_breath = False
        breath_start = 0
        
        for i in range(len(breath_mask)):
            if breath_mask[i] and not in_breath:
                in_breath = True
                breath_start = i
            elif not breath_mask[i] and in_breath:
                in_breath = False
                # Minimum 8 frames (~0.16s) to count as breath
                if i - breath_start > 8:
                    breath_regions.append((breath_start, i))
        
        # Convert to times
        breath_times = [(times[start], times[end]) for start, end in breath_regions]
        breath_count = len(breath_times)
        
        # VERY CONSERVATIVE AI SCORING
        if breath_count >= 4:
            ai_score = 0.2  # Many breaths = human
            status = "👤 HUMAN"
        elif breath_count >= 2:
            ai_score = 0.5
            status = "❓ UNCLEAR"
        else:
            ai_score = 0.85
            status = "🤖 AI"
        
        return {
            'filename': filename,
            'duration': f"{duration:.2f}s",
            'samples': len(y),
            'rms_frames': len(rms),
            'breaths': breath_count,
            'ai_score': ai_score,
            'status': status,
            'threshold': breath_threshold,
            'times': times,
            'rms': rms,
            'smooth_rms': smooth_rms,
            'breath_mask': breath_mask,
            'breath_regions': breath_regions,
            'breath_times': breath_times,
            'waveform': y[:60000]  # First 2.7s
        }
    except Exception as e:
        st.error(f"Error {filename}: {e}")
        return None

st.title("🔬 Breath Detector - DEBUG VERSION")
st.markdown("**Finding why breaths aren't detected**")

tab1, tab2, tab3 = st.tabs(["Upload", "Results", "Debug Plots"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload ONE audio file for debugging",
        type=['wav', 'mp3', 'm4a', 'flac'],
        accept_multiple_files=False
    )
    
    if uploaded_files:
        result = detect_breaths_simple(uploaded_files.read(), uploaded_files.name)
        if result:
            st.session_state.result = result
            st.success("Analysis complete! Check Results tab")

with tab2:
    if 'result' in st.session_state:
        result = st.session_state.result
        df = pd.DataFrame([result])
        
        st.subheader("Detection Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", result['status'])
        col2.metric("Breaths Found", result['breaths'])
        col3.metric("AI Score", f"{result['ai_score']:.2f}")
        col4.metric("Duration", result['duration'])
        
        st.dataframe(df[['filename', 'breaths', 'ai_score', 'status', 'threshold']], 
                    use_container_width=True)

with tab3:
    if 'result' in st.session_state:
        result = st.session_state.result
        
        st.subheader("DEBUG: Raw Signal")
        fig_debug, axes = plt.subplots(4, 1, figsize=(14, 16))
        
        # 1. Raw waveform
        axes[0].plot(result['waveform'])
        axes[0].set_title("Raw Waveform (first 2.7s)")
        axes[0].set_ylabel("Amplitude")
        
        # 2. RMS envelope
        axes[1].plot(result['times'][:300], result['rms'][:300], 'blue', label='Raw RMS')
        axes[1].plot(result['times'][:300], result['smooth_rms'][:300], 'red', label='Smooth RMS')
        axes[1].set_title("RMS Energy Envelope")
        axes[1].set_ylabel("RMS")
        axes[1].legend()
        
        # 3. Breath mask
        mask_times = result['times'][:300]
        mask_data = result['breath_mask'][:300]
        axes[2].plot(mask_times, mask_data, 'green', linewidth=2)
        axes[2].axhline(y=0.6, color='red', linestyle='--', label='Threshold')
        axes[2].set_title("Breath Probability (RMS < 60% smooth)")
        axes[2].set_ylabel("Probability")
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].legend()
        
        # 4. Detected breaths
        axes[3].plot(result['times'][:300], result['rms'][:300], 'blue', alpha=0.7)
        axes[3].plot(result['times'][:300], result['smooth_rms'][:300], 'red', alpha=0.7)
        
        # Mark breath regions
        for start_frame, end_frame in result['breath_regions'][:300]:
            if start_frame < 300 and end_frame < 300:
                axes[3].axvspan(start_frame * hop_length/sr, end_frame * hop_length/sr, 
                               alpha=0.4, color='orange', label='Breath' if start_frame == result['breath_regions'][0][0] else "")
        
        axes[3].set_title("DETECTED BREATHS (orange regions)")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("RMS")
        axes[3].legend()
        
        plt.tight_layout()
        st.pyplot(fig_debug)
        
        st.subheader("DEBUG INFO")
        st.json({
            'total_frames': len(result['rms']),
            'breath_threshold': result['threshold'],
            'min_breath_frames': 8,
            'breath_regions_found': len(result['breath_regions']),
            'sample_rate': 22050,
            'hop_length': 512
        })

