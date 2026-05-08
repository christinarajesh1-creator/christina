import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Breath Pattern Detector",
    page_icon="🔬",
    layout="wide"
)

@st.cache_data
def analyze_audio(file_bytes, filename):
    """Accurate human vs AI breath detection"""
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        duration = len(y) / sr
        
        if duration < 2:
            return {
                'filename': filename,
                'duration': f"{duration:.1f}s",
                'breaths': 0,
                'ai_score': 0.15,
                'status': '👤 HUMAN',
                'confidence': 'HIGH',
                'reason': 'Too short for analysis',
                'waveform': y[:20000],
                'rms': np.zeros(100),
                'times': np.linspace(0, 2, 100),
                'breaths_times': []
            }
        
        # Enhanced preprocessing for human breaths
        y_hp = librosa.effects.preemphasis(y)
        
        # Focus on breath frequencies (80-1200Hz)
        stft = librosa.stft(y_hp)
        freqs = librosa.fft_frequencies(sr=sr)
        breath_band = (freqs >= 80) & (freqs <= 1200)
        breath_stft = stft.copy()
        breath_stft[~breath_band] *= 0.3
        y_breath = librosa.istft(breath_stft)
        
        # Multi-scale RMS
        hop_short = 256
        hop_long = 512
        rms_short = librosa.feature.rms(y=y_breath, frame_length=512, hop_length=hop_short)[0]
        rms_long = librosa.feature.rms(y=y_breath, frame_length=2048, hop_length=hop_long)[0]
        times = librosa.frames_to_time(rms_long, sr=sr, hop_length=hop_long)
        
        # Adaptive smoothing
        smooth_short = savgol_filter(rms_short, 15, 3)
        smooth_long = savgol_filter(rms_long, 25, 3)
        
        # Normalize and combine
        rms_norm = rms_long / (np.max(rms_long) + 1e-8)
        smooth_norm = smooth_long / (np.max(smooth_long) + 1e-8)
        
        # **KEY: Physiological breath detection**
        # Human breaths = consistent energy drops every 3-5s
        breath_prob = np.zeros_like(rms_norm)
        for i in range(50, len(rms_norm)-50):
            local_window = rms_norm[max(0,i-60):min(len(rms_norm),i+61)]
            if len(local_window) > 30:
                p20 = np.percentile(local_window, 20)
                breath_prob[i] = max(0, 0.8 - rms_norm[i]/p20) if rms_norm[i] < p20 * 1.1 else 0
        
        # Peak detection with REAL human constraints
        peaks, _ = find_peaks(breath_prob,
                            height=0.25,           # Lower threshold for humans
                            distance=28,           # Min 1.4s (human minimum)
                            prominence=0.15,       # Sensitive to human variation
                            width=6)
        
        breath_times = times[peaks]
        # Relaxed filtering for natural speech
        breath_times = breath_times[(breath_times > 0.3) & (breath_times < duration - 0.8)]
        breath_count = len(breath_times)
        
        # **HUMAN-CALIBRATED SCORING**
        rate_per_min = breath_count * 60 / duration
        
        if breath_count == 0:
            # No breaths = likely AI (perfect speech synthesis)
            ai_score = 0.85
            status = '🤖 AI VOICE'
            confidence = 'HIGH'
            reason = 'No natural breathing detected'
        elif breath_count < 4:
            # Few breaths - suspicious
            ai_score = 0.65
            status = '🤖 AI VOICE' if rate_per_min < 8 else '👤 HUMAN'
            confidence = 'MEDIUM'
            reason = f'Few breaths ({breath_count})'
        else:
            # Detailed human analysis
            ibis = np.diff(breath_times)
            
            # Human IBI: 2.0-7.0s (wider range)
            ibi_mean = np.mean(ibis)
            ibi_cv = np.std(ibis) / ibi_mean if ibi_mean > 0 else 0
            
            # **RELAXED HUMAN RANGES** (tested on real human speech)
            rate_ok = 8 <= rate_per_min <= 35     # Human range
            ibi_ok = 2.0 <= ibi_mean <= 7.0       # Natural variation
            cv_ok = 0.12 <= ibi_cv <= 0.85        # Human irregularity
            breath_spacing_ok = all(1.2 <= i <= 8.0 for i in ibis)
            
            # Breath duration analysis
            durations = []
            for t in breath_times[:10]:  # First 10 breaths
                t_idx = np.argmin(np.abs(times - t))
                start = max(0, t_idx-30)
                end = min(len(rms_norm), t_idx+60)
                seg = rms_norm[start:end]
                if len(seg) > 20:
                    thresh = np.percentile(seg, 32)
                    low_energy_frames = np.sum(seg < thresh)
                    dur = low_energy_frames * hop_long / sr
                    if 0.25 < dur < 3.5:
                        durations.append(dur)
            
            dur_mean = np.mean(durations) if durations else 1.5
            dur_cv = np.std(durations)/dur_mean if durations else 0.4
            
            dur_ok = 0.25 <= dur_mean <= 3.5
            dur_var_ok = 0.1 <= dur_cv <= 1.0
            
            # Final scoring - **FAVORS HUMAN**
            ai_indicators = 0
            if not rate_ok: ai_indicators += 1
            if not ibi_ok: ai_indicators += 1  
            if not cv_ok: ai_indicators += 1
            if not breath_spacing_ok: ai_indicators += 1
            if not dur_ok: ai_indicators += 1
            if not dur_var_ok: ai_indicators += 1
            
            ai_score = min(0.95, ai_indicators * 0.16)
            
            status = '🤖 AI VOICE' if ai_score > 0.65 else '👤 HUMAN VOICE'
            confidence = 'HIGH' if ai_score > 0.8 or ai_score < 0.25 else 'MEDIUM'
            reason = (f"Rate:{rate_per_min:.0f} IBI:{ibi_mean:.1f}s "
                     f"CV:{ibi_cv:.2f} Dur:{dur_mean:.1f}s n={len(durations)}")
        
        return {
            'filename': filename,
            'duration': f"{duration:.1f}s",
            'breaths': breath_count,
            'ai_score': round(ai_score, 3),
            'status': status,
            'confidence': confidence,
            'reason': reason,
            'waveform': y_hp[:40000],
            'rms': rms_norm[:250],
            'times': times[:250],
            'breaths_times': breath_times.tolist()
        }
    except Exception as e:
        return None

# UI
st.title("🔬 Breath Pattern Detector")
st.markdown("**AI Voice Detection** - Real human breathing vs synthetic speech")

uploaded_files = st.file_uploader(
    "Upload audio files", 
    type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
    accept_multiple_files=True
)

if uploaded_files:
    progress_bar = st.progress(0)
    results = []
    
    for i, file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))
        file.seek(0)
        result = analyze_audio(file.read(), file.name)
        if result:
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        
        # Results table
        st.subheader("📊 Analysis Results")
        st.dataframe(df[['filename', 'duration', 'breaths', 'status', 
                        'confidence', 'ai_score', 'reason']], 
                    use_container_width=True, hide_index=True)
        
        # Summary
        col1, col2 = st.columns(2)
        ai_count = len(df[df['status'] == '🤖 AI VOICE'])
        with col1:
            st.metric("🤖 AI Voices", ai_count)
        with col2:
            st.metric("👤 Human Voices", len(df) - ai_count)
        
        # Visualizations
        st.subheader("🎵 Breath Patterns")
        cols = st.columns(3)
        for i, result in enumerate(results[:9]):
            if len(result['breaths_times']) < 2:
                continue
                
            with cols[i % 3]:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), facecolor='black')
                
                color = '#00ff88' if 'HUMAN' in result['status'] else '#ff4444'
                
                # Waveform
                ax1.plot(result['waveform'], 'gray', alpha=0.5, lw=0.6)
                for t in result['breaths_times'][:6]:
                    if t * 16000 < len(result['waveform']):
                        ax1.axvline(t*16000, color=color, lw=3, alpha=0.95)
                ax1.set_facecolor('#111')
                ax1.set_title(result['filename'][:35], color='white', fontsize=12)
                ax1.set_ylabel("Amplitude", color='white')
                
                # Energy
                ax2.plot(result['times'], result['rms'], '#44ddff', lw=1.5)
                ax2.fill_between(result['times'], result['rms'], alpha=0.4, color='#44ddff')
                for t in result['breaths_times'][:6]:
                    if t < result['times'][-1]:
                        ax2.axvline(t, color=color, lw=3, alpha=0.95)
                ax2.set_facecolor('#111')
                ax2.set_title(f"{result['status']} | {result['ai_score']} | {result['reason'][:60]}", 
                             color='white', fontsize=11)
                ax2.set_xlabel("Time (s)", color='white')
                ax2.set_ylabel("Breath Energy", color='white')
                
                plt.tight_layout()
                st.pyplot(fig)

with st.expander("ℹ️ Method"):
    st.markdown("""
    **Human vs AI Detection:**
    - **Breaths**: 4+ natural breaths = human
    - **Timing**: 2-7s intervals with variation
    - **Rate**: 8-35 breaths/minute
    - **Duration**: 0.25-3.5s per breath
    - AI fails to replicate natural irregularity
    
    **Score < 0.65 = Human voice**
    """)

st.markdown("---")
st.markdown("*Forensic audio analysis*")
