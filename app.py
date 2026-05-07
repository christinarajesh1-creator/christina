import streamlit as st
import numpy as np
import librosa
from scipy.spatial import distance
import matplotlib.pyplot as plt

class PneumaForensic:
    @staticmethod
    def advanced_breath_analysis(y, sr):
        """6-parameter forensic analysis"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            
            # STRICT breath detection
            silence_threshold = np.percentile(rms, 8)
            breath_times = times[rms < silence_threshold * 0.7]
            
            # Human breathing rhythm filter (2-8s gaps)
            events = []
            last_t = 0
            for t in breath_times:
                gap = t - last_t
                if 2.0 < gap < 8.0:
                    events.append(t)
                    last_t = t
            
            if len(events) < 2:
                return {
                    "ibi_reg": 1.0, "amp_var": 0.0, "dur_var": 0.0,
                    "presence": 0.01, "spec_cont": 1.0, "sim_score": 1.0,
                    "synthetic_score": 0.95, "breath_events": events
                }
            
            # Calculate all 6 parameters (same as before)
            ibis = np.diff(events)
            ibi_reg = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 1.0
            ibi_reg = min(ibi_reg, 2.0)
            
            breath_rms = []
            for t in events:
                start = max(0, int((t-0.4)*sr))
                end = min(len(y), int((t+0.6)*sr))
                if end > start:
                    breath_rms.append(np.mean(np.abs(y[start:end])))
            amp_var = np.std(breath_rms) / np.mean(breath_rms) if breath_rms and np.mean(breath_rms) > 0 else 0.0
            amp_var = min(amp_var, 1.5)
            
            cents = []
            for t in events:
                start = max(0, int((t-0.4)*sr))
                end = min(len(y), int((t+0.6)*sr))
                if end > start:
                    cent = librosa.feature.spectral_centroid(y=y[start:end], sr=sr)[0]
                    cents.append(np.mean(cent) if len(cent) > 0 else 0)
            dur_var = np.std(cents) / np.mean(cents) if cents and np.mean(cents) > 0 else 0.0
            dur_var = min(dur_var, 2.0)
            
            total_breath_duration = sum(np.diff(events + [events[-1]])) if len(events) > 1 else 0
            times_max = max(times[-1], 1.0)
            presence = min(total_breath_duration / times_max, 0.4)
            
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_events = []
            for t in events:
                idx = min(int(t*sr / (sr/512)), len(zcr)-1)
                if 0 <= idx < len(zcr):
                    zcr_events.append(zcr[idx])
            spec_cont = np.std(zcr_events) if len(zcr_events) > 1 else 1.0
            spec_cont = min(spec_cont * 2, 1.5)
            
            mfccs = []
            for t in events:
                start = max(0, int((t-0.4)*sr))
                end = min(len(y), int((t+0.6)*sr))
                if end > start + sr//20:
                    mfcc = librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=5, n_fft=2048)
                    if mfcc.shape[1] > 0:
                        mfcc_mean = np.mean(mfcc, axis=1)
                        mfccs.append(mfcc_mean)
            
            sim_score = 1.0
            if len(mfccs) > 1:
                dists = []
                for i in range(len(mfccs)):
                    for j in range(i+1, len(mfccs)):
                        dists.append(distance.euclidean(mfccs[i], mfccs[j]))
                sim_score = np.mean(dists) if dists else 1.0
                sim_score = min(sim_score / 1000, 1.0)
            
            # Weighted synthetic score
            weights = {'ibi_reg': 0.28, 'amp_var': 0.15, 'dur_var': 0.12, 
                      'presence': 0.15, 'spec_cont': 0.12, 'sim_score': 0.18}
            synthetic_score = (
                (1 - ibi_reg * 0.5) * weights['ibi_reg'] +
                (1 - amp_var * 0.4) * weights['amp_var'] +
                (1 - dur_var * 0.3) * weights['dur_var'] +
                (1 - presence * 2) * weights['presence'] +
                spec_cont * weights['spec_cont'] +
                sim_score * weights['sim_score']
            )
            synthetic_score = max(0, min(synthetic_score, 1.0))
            
            return {
                "ibi_reg": float(ibi_reg), "amp_var": float(amp_var), 
                "dur_var": float(dur_var), "presence": float(presence),
                "spec_cont": float(spec_cont), "sim_score": float(sim_score),
                "synthetic_score": float(synthetic_score),
                "breath_events": events  # Return actual breath times for plotting
            }
        except:
            return None

# === STREAMLIT APP ===
st.set_page_config(page_title="🫁 PneumaForensic", page_icon="🫁", layout="wide")

st.title("🫁 PneumaForensic")
st.markdown("**AI Voice Detection - Forensic Breath Analysis**")

# File uploader
uploaded_file = st.file_uploader("Upload audio (WAV, MP3, M4A)", type=['wav','mp3','m4a','flac'])

if uploaded_file is not None:
    # Audio player
    st.audio(uploaded_file)
    
    # Analyze
    with st.spinner("🔍 Detecting breaths..."):
        y, sr = librosa.load(uploaded_file, sr=22050)
        results = PneumaForensic.advanced_breath_analysis(y, sr)
    
    if results:
        duration = len(y) / sr
        
        # === MAIN VERDICT ===
        col1, col2 = st.columns([2,1])
        with col1:
            st.metric("🎯 AI Probability", f"{results['synthetic_score']:.0%}")
        with col2:
            if results['synthetic_score'] > 0.7:
                st.error("🔴 **HIGHLY SYNTHETIC**")
            elif results['synthetic_score'] > 0.4:
                st.warning("🟡 **SUSPICIOUS**")
            else:
                st.success("🟢 **HUMAN**")
        
        # === BREATH WAVEFORM (YOUR REQUEST) ===
        st.markdown("---")
        st.subheader("🎵 **Breath Detection Waveform**")
        st.markdown("""
        **🟤 Gray waves**: The actual sound of the person speaking  
        **🔴 Red dashed lines**: Every detected breath  
        
        **Natural/Human**: Red lines at **uneven intervals** (natural pauses)  
        **Synthetic/AI**: Red lines in **perfect grid** or **missing entirely**
        """)
        
        # Create PERFECT waveform plot
        fig, ax = plt.subplots(figsize=(14, 6))
        time_axis = np.linspace(0, duration, len(y))
        
        # Gray waveform
        ax.plot(time_axis, y, 'gray', alpha=0.7, linewidth=0.8, label='Speech')
        
        # Red dashed breath lines (EXACTLY as requested)
        breath_times = results['breath_events']
        if len(breath_times) > 0:
            for t in breath_times:
                ax.axvline(x=t, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.scatter(breath_times, np.zeros(len(breath_times)), 
                      color='red', s=100, marker='v', zorder=5, label=f'{len(breath_times)} Breaths')
        else:
            ax.text(0.5, 0.5, '❌ NO BREATHS DETECTED\n(STRONG AI INDICATOR)', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(f'Breath Analysis ({len(breath_times)} breaths detected)', fontsize=14, pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # === 6 PARAMETERS ===
        st.markdown("---")
        st.subheader("🔬 6 Forensic Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("IBI Regularity (28%)", f"{results['ibi_reg']:.3f}")
            st.metric("Amplitude Var (15%)", f"{results['amp_var']:.3f}")
            st.metric("Duration Var (12%)", f"{results['dur_var']:.3f}")
        with col2:
            st.metric("Breath Presence (15%)", f"{results['presence']:.3f}")
            st.metric("Spectral Cont (12%)", f"{results['spec_cont']:.3f}")
            st.metric("Similarity (18%)", f"{results['sim_score']:.3f}")

else:
    st.info("👆 Upload audio to analyze breaths")
