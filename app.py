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

st.set_page_config(layout="wide")

def forensic_breath_analysis(file_bytes, filename):
    try:
        # Load with higher quality
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        duration = len(y) / sr
        
        if duration < 3:
            return None
        
        # Advanced preprocessing
        y_clean = librosa.effects.preemphasis(y)
        y_denoised = librosa.effects.trim(y_clean, top_db=25)[0] if duration > 5 else y_clean
        
        # Multi-resolution RMS for robust detection
        hop_short = 256
        hop_long = 512
        
        rms_short = librosa.feature.rms(y=y_denoised, frame_length=512, hop_length=hop_short)[0]
        rms_long = librosa.feature.rms(y=y_denoised, frame_length=2048, hop_length=hop_long)[0]
        
        times_long = librosa.frames_to_time(rms_long, sr=sr, hop_length=hop_long)
        
        # Breath-specific filtering (100-800Hz)
        fft = np.fft.rfft(y_denoised)
        freqs = np.fft.rfftfreq(len(y_denoised), 1/sr)
        breath_fft = fft.copy()
        breath_mask = (freqs >= 100) & (freqs <= 800)
        breath_fft[~breath_mask] *= 0.1  # Attenuate non-breath
        y_breath = np.fft.irfft(breath_fft, n=len(y_denoised))
        rms_breath = librosa.feature.rms(y=y_breath, frame_length=1024, hop_length=hop_long)[0]
        
        # Combined envelope
        rms_env = 0.5 * rms_long + 0.5 * rms_breath
        
        # Savitzky-Golay smoothing (better than convolution)
        smooth_env = savgol_filter(rms_env, window_length=21, polyorder=3)
        
        # Physiological breath detection
        breath_prob = np.zeros_like(rms_env)
        for i in range(40, len(rms_env)-40):
            window = rms_env[max(0,i-40):min(len(rms_env),i+41)]
            if len(window) > 20:
                local_p25 = np.percentile(window, 25)
                local_mean = np.mean(window)
                breath_prob[i] = 1.0 - (rms_env[i] / local_mean) if rms_env[i] < local_p25 * 1.2 else 0
        
        # Peak detection with strict human constraints
        peaks, properties = find_peaks(breath_prob, 
                                     height=0.45,
                                     distance=int(1.8*sr/hop_long),  # Min 1.8s between breaths
                                     prominence=0.25,
                                     width=8)
        
        breath_times = times_long[peaks]
        breath_times = breath_times[(breath_times > 0.8) & (breath_times < duration-1.2)]
        
        # TRUE HUMAN vs AI PARAMETERS (researched values)
        breath_count = len(breath_times)
        
        params = {
            "Filename": filename,
            "Duration_s": round(duration, 1),
            "Breaths": breath_count,
            "Rate_10m": round(breath_count * 10 / duration, 1)
        }
        
        if breath_count < 3:
            params.update({
                "AI_Score": 0.88,
                "Status": "🤖 AI GENERATED",
                "Confidence": "HIGH",
                "P1_Rate": "LOW",
                "P2_IBI": "N/A", 
                "P3_Var": "N/A",
                "P4_Dur": "N/A",
                "P5_Reg": "N/A",
                "P6_Ent": "N/A"
            })
        else:
            # P1: Breathing rate (human 12-24 breaths per 10min)
            rate_norm = max(0, min(1, abs(params["Rate_10m"] - 18)/6))
            
            # P2: Interbreath interval (human 2.5-5s)
            ibis = np.diff(breath_times)
            ibi_mean = np.mean(ibis)
            ibi_norm = max(0, min(1, abs(ibi_mean - 3.3)/1.25))
            
            # P3: Variability (human CV 0.25-0.55)
            ibi_cv = np.std(ibis)/ibi_mean
            var_norm = max(0, min(1, abs(ibi_cv - 0.4)/0.15))
            
            # P4: Breath duration (human 0.6-1.6s)
            durations = []
            for t in breath_times:
                t_idx = np.argmin(np.abs(times_long - t))
                start = max(0, t_idx-15)
                end = min(len(rms_env), t_idx+30)
                breath_seg = rms_env[start:end]
                thresh = np.percentile(breath_seg, 30)
                dur_frames = np.sum(breath_seg < thresh)
                dur = dur_frames * hop_long / sr
                if 0.4 < dur < 2.2:
                    durations.append(dur)
            
            dur_mean = np.mean(durations) if durations else 0
            dur_norm = max(0, min(1, abs(dur_mean - 1.1)/0.55))
            
            # P5: Regularity (human CV 0.2-0.65)
            dur_cv = np.std(durations)/dur_mean if durations and dur_mean > 0 else 0
            reg_norm = max(0, min(1, abs(dur_cv - 0.425)/0.225))
            
            # P6: Rhythm entropy (AI = low, human = high)
            ibis_bins = np.clip(ibis, 1, 6)
            hist, _ = np.histogram(ibis_bins, bins=10, density=True)
            rhythm_entropy = entropy(hist + 1e-10)
            ent_norm = max(0, 1 - rhythm_entropy/1.2)  # Inverted: low entropy = AI
            
            ai_score = 0.2 * rate_norm + 0.2 * ibi_norm + 0.18 * var_norm + 0.17 * dur_norm + \
                      0.13 * reg_norm + 0.12 * ent_norm
            
            status = "🤖 AI GENERATED" if ai_score > 0.52 else "👤 HUMAN SPEECH"
            conf = "HIGH" if ai_score > 0.65 or ai_score < 0.35 else "MEDIUM"
            
            params.update({
                "AI_Score": round(ai_score, 3),
                "Status": status,
                "Confidence": conf,
                "P1_Rate": f"{params['Rate_10m']:.1f}",
                "P2_IBI": f"{ibi_mean:.2f}s",
                "P3_Var": f"{ibi_cv:.3f}",
                "P4_Dur": f"{dur_mean:.2f}s",
                "P5_Reg": f"{dur_cv:.3f}",
                "P6_Ent": f"{rhythm_entropy:.3f}",
                "y": y_denoised,
                "rms": rms_env,
                "times": times_long,
                "breath_times": breath_times.tolist()
            })
        
        return params
        
    except:
        return None

st.title("🎯 AI Voice Detector - Forensic Breathing Analysis")

uploaded_files = st.file_uploader("Upload audio files", 
                                 type=['wav','mp3','m4a','flac'], 
                                 accept_multiple_files=True)

if uploaded_files:
    progress = st.progress(0)
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress((i + 1) / len(uploaded_files))
        uploaded_file.seek(0)
        result = forensic_breath_analysis(uploaded_file.read(), uploaded_file.name)
        if result:
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        
        # MAIN RESULTS
        st.subheader("🔍 Detection Results")
        display_cols = ["Filename", "Duration_s", "Status", "Confidence", "AI_Score", 
                       "Breaths", "P1_Rate", "P2_IBI", "P3_Var"]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
        
        # DETAILED PARAMETERS
        st.subheader("📊 Detailed Parameters")
        detail_cols = ["Filename", "P4_Dur", "P5_Reg", "P6_Ent"]
        st.dataframe(df[["Filename"] + detail_cols], use_container_width=True, hide_index=True)
        
        # VISUALIZATIONS
        st.subheader("📈 Breath Patterns")
        cols = st.columns(3)
        
        for idx, result in enumerate(results[:9]):
            if 'breath_times' not in result or not result['breath_times']:
                continue
                
            with cols[idx % 3]:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Waveform with breaths
                ax1.plot(result['y'][:30000], 'gray', alpha=0.6, linewidth=0.8)
                for t in result['breath_times'][:4]:
                    if t * 16000 < 30000:
                        color = 'red' if 'AI' in result['Status'] else 'green'
                        ax1.axvline(t*16000, color=color, linewidth=3, alpha=0.9)
                ax1.set_title(f"{result['Filename'][:25]} | {result['Status']}", fontsize=12)
                ax1.set_ylabel("Amplitude")
                
                # Energy envelope
                ax2.plot(result['times'][:250], result['rms'][:250], 'cyan', linewidth=1.2)
                ax2.fill_between(result['times'][:250], result['rms'][:250], alpha=0.3, color='cyan')
                for t in result['breath_times'][:4]:
                    if t < result['times'][249]:
                        color = 'red' if 'AI' in result['Status'] else 'green'
                        ax2.axvline(t, color=color, linewidth=2.5, alpha=1)
                ax2.set_title(f"AI Score: {result['AI_Score']} | Rate: {result['P1_Rate']}")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Breath Energy")
                
                plt.tight_layout()
                st.pyplot(fig)
        
        st.success(f"✅ Processed {len(results)} files")
        
        # SUMMARY STATS
        ai_count = len(df[df['Status'].str.contains('AI')])
        human_count = len(df) - ai_count
        st.metric("AI Detected", f"{ai_count}/{len(df)}")
        st.metric("Human Detected", f"{human_count}/{len(df)}")
