import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PneumaForensic")

def detect_ai_breathing(y, sr):
    """AI Detection - Low score = Human, High score = AI"""
    
    # Get breath locations - high energy bursts
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Find breath candidates (high RMS peaks)
    breath_threshold = np.percentile(rms, 88)
    breath_idx = np.where(rms > breath_threshold)[0]
    
    events = []
    last_time = 0
    for i in breath_idx:
        t = times[i]
        if t > 2.0 and t - last_time > 1.2:  # Min 1.2s between breaths
            events.append(t)
            last_time = t
    
    # Minimum 3 breaths for analysis
    if len(events) < 3:
        return 0.9, events, "FEW BREATHS - AI"
    
    # AI DETECTION METRICS
    ibis = np.diff(events)
    
    # 1. TIMING: AI has LOW variation (regular)
    timing_cv = np.std(ibis) / np.mean(ibis)
    timing_ai = max(0, 1.0 - timing_cv * 3)  # Regular = AI
    
    # 2. BREATH PURITY: AI breaths are CLEAN
    purity_ai = []
    for t in events[:6]:
        start = max(0, int((t-0.15)*sr))
        end = min(len(y), int((t+0.25)*sr))
        breath = y[start:end]
        if len(breath) > sr//4:
            purity = np.mean(librosa.feature.spectral_flatness(breath))
            purity_ai.append(purity)
    
    avg_purity = np.mean(purity_ai) if purity_ai else 0.9
    purity_score = max(0, (avg_purity - 0.4) * 2.5)
    
    # 3. NOISE FLOOR: AI = silent
    quiet_parts = y[np.abs(y) < np.std(y)*0.3]
    noise_floor = np.std(quiet_parts) if len(quiet_parts) > 10000 else 0
    noise_ai = max(0, 1.0 - noise_floor * 15000)
    
    # 4. BREATH RATE: AI often wrong
    duration = events[-1] - events[0]
    bpm = len(events) / duration * 60 if duration > 0 else 0
    rate_ai = 1.0 if bpm < 10 or bpm > 35 else 0.1
    
    # TOTAL AI SCORE (0-1, >0.6 = AI)
    ai_score = (timing_ai * 0.4 + purity_score * 0.25 + noise_ai * 0.2 + rate_ai * 0.15)
    
    status = "🤖 AI" if ai_score > 0.6 else "👤 HUMAN"
    
    return ai_score, events, {
        "Score": f"{ai_score:.0%}",
        "Status": status,
        "Timing": f"{timing_ai:.0%}",
        "Purity": f"{purity_score:.0%}",
        "Noise": f"{noise_ai:.0%}",
        "BPM": f"{bpm:.1f}",
        "Breaths": len(events)
    }

st.title("🫁 PneumaForensic v3")

st.markdown("**🟢 LOW % = HUMAN** | **🔴 HIGH % = AI**")

files = st.file_uploader("Upload audio files", type=['wav', 'mp3', 'm4a'], accept_multiple_files=True)

if files:
    results = []
    
    # Results table
    st.subheader("📊 Analysis Results")
    result_df = []
    
    # Graphs
    st.subheader("📈 Breath Detection")
    cols = st.columns(3)
    
    for idx, file in enumerate(files):
        try:
            # Reload file bytes
            file.seek(0)
            y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
            
            ai_score, events, metrics = detect_ai_breathing(y, sr)
            metrics["File"] = file.name[:30]
            results.append(metrics)
            result_df.append(metrics)
            
            # Graph in columns
            with cols[idx % 3]:
                fig, ax = plt.subplots(figsize=(5, 3))
                duration = min(20, len(y)/sr)
                t_axis = np.linspace(0, duration, min(2000, len(y)))
                y_plot = y[:len(t_axis)]
                
                ax.plot(t_axis, y_plot, color='#4a90e2', linewidth=0.8, alpha=0.7)
                
                # Red breath markers
                for e in events:
                    if e < duration:
                        ax.axvline(e, color='red', linestyle='--', linewidth=3, alpha=0.9)
                
                color = 'red' if ai_score > 0.6 else 'green'
                ax.set_title(f"{ai_score:.0%}\n{len(events)} breaths", 
                           color=color, fontsize=12, pad=5)
                ax.set_facecolor('#0f0f0f')
                ax.set_xlim(0, duration)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
                
        except Exception as e:
            st.error(f"Error {file.name}: {e}")
    
    # Show table
    if result_df:
        df = pd.DataFrame(result_df)
        st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("**Roger AI fails on Timing + Purity**")
