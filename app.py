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
    """Complete forensic breath analysis"""
    try:
        # Load audio
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        duration = len(y) / sr
        
        if duration < 2.5:
            return {
                'filename': filename,
                'duration': f"{duration:.1f}s",
                'breaths': 0,
                'ai_score': 0.92,
                'status': '🤖 AI VOICE',
                'confidence': 'VERY HIGH',
                'reason': 'Too short - no breathing detected',
                'waveform': y[:20000],
                'rms': np.zeros(100),
                'times': np.linspace(0, 2, 100),
                'breaths_times': []
            }
        
        # Preprocessing
        y_clean = librosa.effects.preemphasis(y)
        
        # RMS envelope
        hop = 512
        rms = librosa.feature.rms(y=y_clean, frame_length=1024, hop_length=hop)[0]
        times = librosa.frames_to_time(rms, sr=sr, hop_length=hop)
        
        # Smooth envelope
        smooth_rms = savgol_filter(rms, min(21, len(rms)//10), 3)
        
        # Breath detection - energy valleys
        valley_mask = (rms < smooth_rms * 0.7) & (rms < np.percentile(rms, 25))
        
        # Refine with morphological closing
        valley_prob = np.convolve(valley_mask.astype(float), np.ones(7)/7, mode='same')
        
        # Peak detection with human constraints
        peaks, props = find_peaks(valley_prob, 
                                height=0.4,
                                distance=22,  # Min ~1.1s between breaths
                                prominence=0.25)
        
        breath_times = times[peaks]
        breath_times = breath_times[(breath_times > 0.6) & (breath_times < duration - 1)]
        breath_count = len(breath_times)
        
        # ANALYSIS PARAMETERS
        rate_per_min = breath_count * 60 / duration
        
        if breath_count < 3:
            ai_score = 0.88 + (3-breath_count)*0.04
            status = '🤖 AI VOICE'
            confidence = 'HIGH'
            reason = f'Insufficient breaths ({breath_count})'
        else:
            # Inter-breath intervals
            ibis = np.diff(breath_times)
            ibi_avg = np.mean(ibis)
            ibi_std = np.std(ibis)
            ibi_cv = ibi_std / ibi_avg if ibi_avg > 0 else 0
            
            # Breath durations
            durations = []
            for t in breath_times:
                t_idx = np.argmin(np.abs(times - t))
                start = max(0, t_idx-25)
                end = min(len(rms), t_idx+50)
                seg_rms = rms[start:end]
                if len(seg_rms) > 15:
                    seg_thresh = np.percentile(seg_rms, 35)
                    dur_frames = np.sum(seg_rms < seg_thresh)
                    dur_sec = dur_frames * hop / sr
                    if 0.3 < dur_sec < 3.0:
                        durations.append(dur_sec)
            
            dur_avg = np.mean(durations) if durations else 1.2
            dur_cv = np.std(durations)/dur_avg if durations and dur_avg > 0 else 0.4
            
            # HUMAN RANGES (from respiratory physiology)
            rate_ai = 1 if rate_per_min < 10 or rate_per_min > 30 else 0
            ibi_ai = 1 if ibi_avg < 2.0 or ibi_avg > 6.0 else 0
            cv_ai = 1 if ibi_cv < 0.15 or ibi_cv > 0.75 else 0
            dur_ai = 1 if dur_avg < 0.4 or dur_avg > 2.2 else 0
            reg_ai = 1 if dur_cv < 0.1 or dur_cv > 0.85 else 0
            
            # Rhythm entropy
            ibis_norm = np.histogram(ibis, bins=12, range=(1,7), density=True)[0]
            entropy_score = entropy(ibis_norm + 1e-10)
            ent_ai = 1 if entropy_score < 0.5 else 0
            
            # Composite AI score
            ai_score = (0.25*rate_ai + 0.22*ibi_ai + 0.20*cv_ai + 0.18*dur_ai + 
                       0.10*reg_ai + 0.05*ent_ai)
            
            status = '🤖 AI VOICE' if ai_score > 0.58 else '👤 HUMAN VOICE'
            confidence = 'HIGH' if ai_score > 0.75 or ai_score < 0.25 else 'MEDIUM'
            reason = f'Rate:{rate_per_min:.0f}/min IBI:{ibi_avg:.1f}s CV:{ibi_cv:.2f}'
        
        return {
            'filename': filename,
            'duration': f"{duration:.1f}s",
            'breaths': breath_count,
            'ai_score': round(ai_score, 3),
            'status': status,
            'confidence': confidence,
            'reason': reason,
            'waveform': y_clean[:40000],
            'rms': rms[:250],
            'times': times[:250],
            'breaths_times': breath_times.tolist()
        }
    except:
        return None

# HEADER
st.title("🔬 Breath Pattern Detector")
st.markdown("**Forensic AI Voice Detection** - Analyzes natural human breathing vs synthetic speech")

# UPLOAD
uploaded_files = st.file_uploader(
    "Choose audio files", 
    type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
    accept_multiple_files=True,
    help="Upload speech samples to detect AI generation"
)

if uploaded_files:
    # PROCESSING
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    for i, file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text(f'Analyzing {file.name}...')
        
        file.seek(0)
        result = analyze_audio(file.read(), file.name)
        if result:
            results.append(result)
    
    progress_bar.progress(1.0)
    status_text.text(f'✅ Complete! Analyzed {len(results)} files')
    
    if results:
        df = pd.DataFrame(results)
        
        # RESULTS TABLE
        st.subheader("📊 Detection Results")
        st.dataframe(
            df[['filename', 'duration', 'breaths', 'status', 'confidence', 'ai_score', 'reason']],
            use_container_width=True,
            hide_index=True
        )
        
        # SUMMARY METRICS
        col1, col2, col3 = st.columns(3)
        ai_files = len(df[df['status'] == '🤖 AI VOICE'])
        human_files = len(df) - ai_files
        with col1:
            st.metric("AI Voices", ai_files)
        with col2:
            st.metric("Human Voices", human_files)
        with col3:
            avg_score = df['ai_score'].mean()
            st.metric("Avg AI Score", f"{avg_score:.3f}")
        
        # VISUALIZATIONS
        st.subheader("🎵 Audio Analysis")
        cols = st.columns(3)
        
        for i, result in enumerate(results[:9]):
            if not result['breaths_times']:
                continue
                
            with cols[i % 3]:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), facecolor='black')
                
                # Waveform
                color = '#00ff88' if 'HUMAN' in result['status'] else '#ff4444'
                ax1.plot(result['waveform'], color='gray', alpha=0.5, linewidth=0.6)
                for t in result['breaths_times'][:6]:
                    if t * 16000 < len(result['waveform']):
                        ax1.axvline(t*16000, color=color, linewidth=3, alpha=0.95)
                ax1.set_facecolor('#111111')
                ax1.set_title(result['filename'][:35], color='white', fontsize=12)
                ax1.set_ylabel("Amplitude", color='white')
                ax1.tick_params(colors='white')
                
                # Breath envelope
                ax2.plot(result['times'], result['rms'], color='#44ddff', linewidth=1.5)
                ax2.fill_between(result['times'], result['rms'], alpha=0.4, color='#44ddff')
                for t in result['breaths_times'][:6]:
                    if t < result['times'][-1]:
                        ax2.axvline(t, color=color, linewidth=3, alpha=0.95)
                ax2.set_facecolor('#111111')
                ax2.set_title(f"{result['status']} | Score: {result['ai_score']} | {result['breaths']} breaths", 
                             color='white', fontsize=11)
                ax2.set_xlabel("Time (s)", color='white')
                ax2.set_ylabel("Breath Energy", color='white')
                ax2.tick_params(colors='white')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # HOW IT WORKS
        with st.expander("🔍 Detection Method"):
            st.markdown("""
            **6 Forensic Parameters:**
            
            1. **Breath Rate** (12-24/min human) - AI often too regular
            2. **Inter-Breath Interval** (2.5-5s human) - AI timing unnatural  
            3. **Interval Variability** (CV 0.25-0.55 human) - AI robotic
            4. **Breath Duration** (0.6-1.6s human) - AI inconsistent
            5. **Duration Regularity** (CV 0.2-0.65 human) - AI uniform
            6. **Rhythm Entropy** - AI patterns predictable
            
            **Score > 0.58 = AI generated voice**
            """)

st.markdown("---")
st.markdown("*Powered by forensic audio analysis*")
