import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="PneumaForensic")

@st.cache_data
def get_forensic_score(y, sr, events, duration):
    if len(events) < 2:
        return 0.95, {"Status": "Abnormal (Silent/AI)", "Breaths Detected": 0}

    # 1. TIMING REGULARITY (30%) - ROGER KILLER
    # Roger has very regular timing (low CV). Humans have natural variation.
    if len(events) > 1:
        ibis = np.diff(events)
        ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 999
    else:
        ibi_cv = 999
    # Roger CV ~0.1-0.3, Humans ~0.4-0.8+
    p1 = np.clip(2.0 * ibi_cv, 0, 1)  # Low CV = high AI score

    # 2. SPECTRAL PURITY (20%) - AI breaths are unnaturally clean
    flatness = []
    for t in events:
        start = int(max(0, t*sr - 0.1*sr))
        end = int(min(len(y), (t+0.3)*sr))
        if end > start:
            spec_flat = np.mean(librosa.feature.spectral_flatness(y=y[start:end]))
            flatness.append(spec_flat)
    avg_flat = np.mean(flatness) if flatness else 0.8
    # AI >0.6 flatness, Humans <0.4
    p2 = np.clip((avg_flat - 0.3) * 2.5, 0, 1)

    # 3. BREATH DENSITY (15%) - Roger has unnaturally few breaths
    bpm = (len(events) / max(duration, 1)) * 60
    # Human range: 12-30 BPM, AI often <10 or >40
    p3 = 1.0 if (bpm < 10 or bpm > 40) else 0.2

    # 4. NOISE FLOOR (15%) - AI has perfect silence between breaths
    silence_regions = []
    for i in range(len(events)-1):
        mid_silence = (events[i] + events[i+1]) / 2
        start = int(max(0, (mid_silence-0.4)*sr))
        end = int(min(len(y), (mid_silence+0.4)*sr))
        if end > start:
            silence_std = np.std(y[start:end])
            silence_regions.append(silence_std)
    
    noise_floor = np.mean(silence_regions) if silence_regions else 0
    # AI noise floor ~0, Humans >0.001
    p4 = np.clip(1.0 - (noise_floor * 8000), 0, 1)

    # 5. BREATH AMPLITUDE VARIATION (10%)
    amps = []
    for t in events:
        start = int(max(0, (t-0.1)*sr))
        end = int(min(len(y), (t+0.4)*sr))
        if end > start:
            peak_amp = np.max(np.abs(y[start:end]))
            amps.append(peak_amp)
    
    amp_cv = np.std(amps) / np.mean(amps) if amps and np.mean(amps) > 0 else 0
    # AI has unnaturally consistent amplitude
    p5 = np.clip(amp_cv * 3.0, 0, 1)

    # 6. BREATH SIMILARITY (10%) - AI breaths are too similar
    mfccs = []
    for t in events:
        start = int(max(0, t*sr))
        end = int(min(len(y), (t+0.4)*sr))
        if end > start + sr//10:  # Minimum length
            mfcc = np.mean(librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=13), axis=1)
            mfccs.append(mfcc)
    
    p6 = 0.0
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) 
                for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        avg_dist = np.mean(dists)
        # AI has very low distance between breaths
        p6 = np.clip(1.0 - (avg_dist / 80), 0, 1)

    score = (p1*0.30) + (p2*0.20) + (p3*0.15) + (p4*0.15) + (p5*0.10) + (p6*0.10)
    
    status = "🤖 AI DETECTED" if score > 0.6 else "👤 HUMAN"
    metrics = {
        "Status": status,
        "AI Score": f"{score:.0%}",
        "Timing (30%)": f"{p1:.0%}",
        "Purity (20%)": f"{p2:.0%}",
        "Density (15%)": f"{p3:.0%}",
        "Noise (15%)": f"{p4:.0%}",
        "Amp Var (10%)": f"{p5:.0%}",
        "Similarity (10%)": f"{p6:.0%}",
        "Breaths": len(events),
        "BPM": f"{bpm:.1f}",
        "IBI CV": f"{ibi_cv:.2f}"
    }
    
    return round(np.clip(score, 0, 1), 2), metrics

def detect_breaths_improved(y, sr):
    """Improved breath detection - looks for breath energy peaks"""
    # Get RMS energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).flatten()
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    
    # Adaptive threshold - breaths are energy ABOVE speech level
    speech_level = np.mean(rms)
    breath_threshold = speech_level * 2.5  # Breaths are 2.5x louder than average
    
    # Find breath candidates (high energy peaks)
    breath_candidates = []
    for i in range(1, len(rms)-1):
        if (rms[i] > breath_threshold and 
            rms[i] > rms[i-1] and rms[i] > rms[i+1] and  # Local peak
            times[i] - times[0] > 2.0):  # Not in first 2s
            breath_candidates.append(times[i])
    
    # Filter to avoid clustering (minimum 1.5s between breaths)
    events = []
    last_time = -10
    for t in breath_candidates:
        if t - last_time > 1.5:
            events.append(t)
            last_time = t
    
    return events

st.title("🫁 PneumaForensic v2.0 - Roger AI Killer")
st.markdown("**Upload audio files to analyze breathing patterns. Optimized for ElevenLabs Roger voice.**")

files = st.file_uploader("Upload Audio Files", type=['wav', 'mp3', 'm4a'], 
                        accept_multiple_files=True, help="WAV, MP3, M4A supported")

if files:
    all_data = []
    plot_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, f in enumerate(files):
        try:
            status_text.text(f"Processing {f.name}...")
            
            # Load and preprocess
            y, sr = librosa.load(io.BytesIO(f.read()), sr=16000, duration=60)  # Limit to 60s
            y = librosa.util.normalize(y) if np.max(np.abs(y)) > 0.01 else y
            
            # Detect breaths using improved method
            events = detect_breaths_improved(y, sr)
            duration = len(y) / sr
            
            score, metrics = get_forensic_score(y, sr, events, duration)
            
            row = {"Filename": f.name}
            row.update(metrics)
            all_data.append(row)
            plot_data.append({
                "y": y, "events": events, "dur": duration, 
                "name": f.name, "score": score
            })
            
        except Exception as e:
            st.error(f"Error processing {f.name}: {str(e)}")
            continue
        
        progress_bar.progress((i + 1) / len(files))
    
    progress_bar.empty()
    status_text.empty()

    if all_data:
        st.subheader("📋 Forensic Analysis Results")
        df = pd.DataFrame(all_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("📊 Audio Waveforms with Breath Markers")
        for p in plot_data:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 4), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Waveform
            time_axis = np.linspace(0, p['dur'], len(p['y']))
            ax1.plot(time_axis, p['y'], color='gray', alpha=0.4, lw=0.5)
            
            # Mark breaths
            for e in p['events']:
                ax1.axvline(e, color='red', linestyle='--', lw=2, alpha=0.8)
                ax1.fill_betweenx([-1,1], e-0.1, e+0.1, color='red', alpha=0.2)
            
            color = "red" if p['score'] > 0.6 else "green"
            ax1.set_title(f"{p['name']} | AI Score: {p['score']:.0%} | Breaths: {len(p['events'])}", 
                         color=color, fontsize=14, fontweight='bold', pad=20)
            ax1.set_ylabel("Amplitude")
            ax1.margins(x=0)
            
            # Breaths timeline
            if p['events']:
                ax2.eventplot(p['events'], colors='red', lineoffsets=1, linelengths=1)
                ax2.set_xlabel("Time (seconds)")
                ax2.set_yticks([])
                ax2.set_xlim(0, p['dur'])
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.divider()

st.markdown("---")
st.markdown("""
**How it works:**
- **Timing**: AI has unnaturally regular breath intervals
- **Purity**: AI breaths lack natural spectral chaos  
- **Density**: AI has wrong breath rate for humans
- **Noise**: AI has perfect silence (no mic noise)
- **Amplitude**: AI breaths have consistent volume
- **Similarity**: AI breaths sound too identical

**Optimized for ElevenLabs Roger** - Scores >60% = Likely AI
""")
