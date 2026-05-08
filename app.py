import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import euclidean

st.set_page_config(layout="wide")

def forensic_analysis(y, sr):
    duration = len(y) / sr
    
    # Pre-process: normalize and denoise
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    
    # 1. BREATH DETECTION - Multi-band RMS + envelope
    hop_length = 512
    frame_length = 2048
    
    # Low freq (50-300Hz) for breath fundamentals
    low_band = librosa.stft(y, n_fft=2048, hop_length=hop_length)
    low_band = np.abs(low_band) * (np.arange(low_band.shape[0])[:, None] < 300//(sr/frame_length)*2)
    low_rms = np.sqrt(np.mean(np.abs(low_band)**2, axis=0))
    
    # High freq (2-8kHz) for fricatives/breath noise
    high_band = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    high_band = high_band * (np.arange(high_band.shape[0])[:, None] > 2000//(sr/frame_length)*2)
    high_rms = np.sqrt(np.mean(np.abs(high_band)**2, axis=0))
    
    times = librosa.frames_to_time(low_rms, sr=sr, hop_length=hop_length)
    breath_score = 0.7 * low_rms + 0.3 * high_rms
    peaks, _ = signal.find_peaks(breath_score, height=np.percentile(breath_score, 82), 
                                distance=sr//4, prominence=np.std(breath_score)*0.8)
    
    events = times[peaks]
    events = events[(events > 1.0) & (events < duration - 1.0)]
    
    # Filter clustered events
    filtered_events = []
    for t in sorted(events):
        if not filtered_events or t - filtered_events[-1] > 0.8:
            filtered_events.append(t)
    
    events = filtered_events[:12]
    
    if len(events) < 2:
        return {
            "File": "N/A", "AI Score": "98%", "Status": "AI", 
            "Breaths": 0, "IBI_CV": "0.00", "Breath_Purity": "100%",
            "Noise_Floor": "100%", "Amp_CV": "0%", "Formant_Stable": "100%",
            "Spectral_Roll": "100%"
        }, []
    
    # 1. IBI REGULARITY (Human: high CV 0.15-0.35, AI: low <0.12)
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis)
    timing_human = max(0, min(1.0, (ibi_cv - 0.12) / 0.23))
    
    # 2. BREATH SPECTRAL PURITY (AI: unnaturally clean >0.6 flatness)
    purity_scores = []
    formant_stability = []
    for t in events:
        start = max(0, int((t-0.2)*sr))
        end = min(len(y), int((t+0.3)*sr))
        breath = y[start:end]
        
        if len(breath) > sr//5:
            # Spectral flatness (AI = high purity = low flatness)
            flatness = np.mean(librosa.feature.spectral_flatness(y=breath))
            purity_scores.append(flatness)
            
            # Formant stability (AI = unnaturally stable)
            f0, voiced_flag, voiced_probs = signal.hpsv(breath, f0_prior=150)
            formant_stability.append(np.std(voiced_probs[voiced_flag]) if np.sum(voiced_flag)>10 else 0)
    
    avg_purity = np.mean(purity_scores)
    purity_ai = max(0, min(1.0, (avg_purity - 0.45) / 0.25))  # High flatness = AI
    formant_ai = max(0, min(1.0, 1.0 - np.mean(formant_stability) * 10))
    
    # 3. NOISE FLOOR VARIABILITY (Human: natural room noise)
    non_breath_mask = np.ones_like(y, dtype=bool)
    for t in events:
        start = max(0, int((t-0.3)*sr))
        end = min(len(y), int((t+0.3)*sr))
        non_breath_mask[start:end] = False
    
    noise_std = np.std(y[non_breath_mask]) if np.sum(non_breath_mask) > sr else 0
    noise_floor_ai = max(0, min(1.0, 1.0 - noise_std * 12000))
    
    # 4. AMPLITUDE VARIABILITY (Human: natural variation)
    amps = []
    for t in events:
        start = max(0, int((t-0.15)*sr))
        end = min(len(y), int((t+0.25)*sr))
        amps.append(np.max(np.abs(y[start:end])))
    
    amp_cv = np.std(amps) / np.mean(amps)
    amp_ai = max(0, min(1.0, (0.25 - amp_cv) / 0.25))
    
    # 5. SPECTRAL ROLLOFF STABILITY (AI: unnaturally consistent)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_cv_breath = []
    for t in events:
        frame_start = max(0, int((t-0.2)*sr/hop_length))
        frame_end = min(len(rolloff[0]), int((t+0.2)*sr/hop_length))
        if frame_end > frame_start:
            rolloff_cv_breath.append(np.std(rolloff[0, frame_start:frame_end]))
    
    spectral_ai = max(0, min(1.0, 1.0 - np.mean(rolloff_cv_breath) * 50))
    
    # COMPOSITE AI SCORE (higher = more AI-like)
    ai_score = (0.25 * (1-timing_human) + 0.20 * purity_ai + 0.20 * noise_floor_ai + 
                0.15 * amp_ai + 0.10 * formant_ai + 0.10 * spectral_ai)
    
    status = "AI" if ai_score > 0.55 else "HUMAN"
    
    return {
        "File": "Audio", "AI Score": f"{ai_score:.0%}", "Status": status,
        "Breaths": len(events), "IBI_CV": f"{ibi_cv:.2f}", 
        "Breath_Purity": f"{purity_ai:.0%}", "Noise_Floor": f"{noise_floor_ai:.0%}",
        "Amp_CV": f"{amp_ai:.0%}", "Formant_Stable": f"{formant_ai:.0%}",
        "Spectral_Roll": f"{spectral_ai:.0%}"
    }, events

st.title("🔬 PneumaForensic v2.0 - AI Voice Detector")

files = st.file_uploader("Upload audio files", type=['wav','mp3','m4a','flac'], 
                        accept_multiple_files=True)

if files:
    tab1, tab2 = st.tabs(["📊 Analysis Results", "📈 Waveforms + Detection"])
    
    all_results = []
    
    with tab1:
        st.subheader("Breathing Pattern Analysis")
        for file in files:
            file.seek(0)
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, duration=30)
                metrics, events = forensic_analysis(y, sr)
                metrics["File"] = file.name
                all_results.append(metrics)
            except Exception as e:
                all_results.append({"File": file.name, "AI Score": "ERROR", "Status": "Failed"})
        
        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if len(all_results) > 1:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall AI Likelihood", 
                         f"{np.mean([float(r['AI Score'][:-1])/100 for r in all_results if r['AI Score'] != 'ERROR']):.0%}")
            with col2:
                st.metric("Files Analyzed", len([r for r in all_results if r['AI Score'] != 'ERROR']))
    
    with tab2:
        for file in files:
            file.seek(0)
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000, duration=30)
                metrics, events = forensic_analysis(y, sr)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                             facecolor='#0a0a0a', height_ratios=[3,1])
                
                # Waveform
                dur = min(25, len(y)/sr)
                t = np.linspace(0, dur, min(5000, len(y)))
                y_short = y[:len(t)]
                
                ax1.plot(t, y_short, color='#00d4ff', lw=0.7, alpha=0.85)
                
                # Breath markers
                for i, e in enumerate(events):
                    if e < dur:
                        ax1.axvline(e, color='red' if float(metrics['AI Score'][:-1])/100 > 0.55 else 'lime',
                                   ls='--', lw=2.5, alpha=0.9)
                        ax1.text(e, np.max(y_short)*0.7, f"B{i+1}", 
                               ha='center', va='center', fontweight='bold', 
                               color='white', fontsize=10)
                
                color = '#ff4444' if float(metrics['AI Score'][:-1])/100 > 0.55 else '#44ff44'
                ax1.set_title(f"{file.name} | AI: {metrics['AI Score']} | {metrics['Status']}", 
                            color=color, fontsize=16, pad=20)
                ax1.set_facecolor('#111111')
                ax1.set_xlim(0, dur)
                ax1.margins(x=0)
                
                # IBI plot
                if len(events) > 1:
                    ibis = np.diff(events[:8])
                    ax2.bar(range(len(ibis)), ibis, color='orange', alpha=0.7)
                    ax2.axhline(np.mean(ibis), color='white', ls='-', lw=2, alpha=0.8)
                    ax2.set_title(f"Inter-Breath Intervals (CV: {metrics['IBI_CV']})")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
            except:
                st.error(f"Could not process: {file.name}")

else:
    st.info("👆 Upload .wav, .mp3, .m4a, or .flac files to detect AI vs human breathing patterns")
    st.markdown("**Key AI Indicators Detected:**\n- Regular breath timing\n- Unnaturally clean breath spectra\n- Silent noise floors\n- Consistent amplitudes")
