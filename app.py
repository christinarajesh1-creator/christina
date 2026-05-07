import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def analyze_audio_real(y, sr):
    """Real AI detection - humans score LOW"""
    
    # 1. Breath timing - AI = too regular (low CV)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(rms, sr=sr)
    
    # Find energy peaks as breaths
    peaks = []
    for i in range(1000, len(rms)-1000):  # Skip start/end
        if rms[i] > np.percentile(rms, 85) and rms[i] > rms[i-1] and rms[i] > rms[i+1]:
            peaks.append(times[i])
    
    # Filter close peaks
    events = []
    last_peak = 0
    for p in peaks:
        if p - last_peak > 1.5:
            events.append(p)
            last_peak = p
    
    if len(events) < 2:
        return 0.8, events, "No breaths - AI"
    
    # 2. Timing regularity (AI = low variation)
    ibis = np.diff(events)
    timing_cv = np.std(ibis) / np.mean(ibis)
    p1 = 1.0 - min(timing_cv * 2.5, 1.0)  # Low CV = high AI score
    
    # 3. Spectral purity (AI breaths = clean sound)
    purity_scores = []
    for t in events[:5]:
        start = max(0, int((t-0.2)*sr))
        end = min(len(y), int((t+0.2)*sr))
        breath_audio = y[start:end]
        if len(breath_audio) > 100:
            purity = np.mean(librosa.feature.spectral_flatness(y=breath_audio))
            purity_scores.append(purity)
    
    avg_purity = np.mean(purity_scores) if purity_scores else 0.8
    p2 = max((avg_purity - 0.3) * 3, 0)
    
    # 4. Silence quality (AI = perfect silence)
    silence = y[np.abs(y) < 0.005]
    noise_level = np.std(silence) if len(silence) > 1000 else 0
    p4 = max(1.0 - noise_level * 20000, 0)
    
    # 5. Breath rate
    bpm = len(events) / (events[-1] - events[0]) * 60 if len(events) > 1 else 0
    p3 = 1.0 if bpm < 8 or bpm > 35 else 0.1
    
    # Final score - HUMAN = LOW SCORE
    score = p1*0.35 + p2*0.25 + p3*0.15 + p4*0.15 + 0.1
    
    status = "🤖 AI" if score > 0.5 else "👤 HUMAN"
    
    return score, events, {
        "AI Score": f"{score:.0%}",
        "Status": status,
        "Timing": f"{p1:.0%}",
        "Purity": f"{p2:.0%}",
        "BPM": f"{bpm:.1f}",
        "Noise": f"{p4:.0%}",
        "Breaths": len(events)
    }

st.title("🫁 PneumaForensic - Fixed for Humans")

st.info("👤 **HUMANS = GREEN/LOW SCORE** | 🤖 **AI = RED/HIGH SCORE**")

files = st.file_uploader("Upload", type=['wav','mp3','m4a'], accept_multiple_files=True)

if files:
    col1, col2 = st.columns(2)
    
    results = []
    
    with col1:
        st.subheader("📊 Results")
        
        for file in files:
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
                score, events, metrics = analyze_audio_real(y, sr)
                metrics["File"] = file.name
                results.append(metrics)
                
            except:
                results.append({"File": file.name, "AI Score": "ERROR", "Status": "Bad file"})
    
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Breath Patterns")
        
        for file in files:
            try:
                y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
                score, events, metrics = analyze_audio_real(y, sr)
                
                fig, ax = plt.subplots(figsize=(8, 3))
                duration = min(len(y)/sr, 20)
                t = np.linspace(0, duration, len(y[:int(duration*sr)]))
                demo_y = y[:len(t)]
                
                ax.plot(t, demo_y, 'lightgray', linewidth=1, alpha=0.8)
                
                for e in events:
                    if e < duration:
                        ax.axvline(e, color='red', linestyle='--', linewidth=4, alpha=0.9)
                
                color = 'red' if score > 0.5 else 'green'
                ax.set_title(f"{file.name}\n{score:.0%} {metrics['Status']}", 
                           color=color, fontsize=14, pad=10)
                ax.set_facecolor('#1a1a1a')
                ax.set_xlim(0, duration)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
                
            except:
                st.error(f"Graph failed: {file.name}")

with st.expander("🔍 Detection Logic"):
    st.markdown("""
    **Why humans score LOW:**
    - **Irregular timing** between breaths (high CV)
    - **Noisy/chaotic** breath sounds (low purity)  
    - **Mic noise** in silences
    - **Normal BPM** (12-25)
    
    **Why AI scores HIGH:**
    - **Perfect regular** breath timing
    - **Clean synthetic** breath sounds
    - **Zero noise floor**
    - **Wrong BPM rate**
    """)
