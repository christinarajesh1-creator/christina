import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import gc

st.set_page_config(page_title="Deepfake Voice Detection", layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        # High-res load to detect micro-textures in neural voices
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True, duration=30)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # 1. High-Precision Breath Tracking (1D Flattened for stability)
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Identify breath candidates
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.1, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {
            "File": name, "Status": "AI (No Breaths)", "AI Prob": "95%", 
            "IBI CV": 0, "Amp CV": 0, "Presence": "0%", "B-Sim": 0
        }, []

    # --- PARAMETER EXTRACTION ---
    # 1. IBI Regularity (Timing spacing)
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    
    # 2. Amplitude Variance (Breath loudness)
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    # 3. Presence (Ratio of audio that is breath)
    presence_ratio = (len(breaths) * 0.48) / duration # Tuned for neural breath lengths

    # 4. Neural Texture Similarity (Copy-paste check)
    textures = []
    for b in breaths[:5]:
        start = int(b * sr)
        seg = y_norm[start:start+int(0.2*sr)]
        if len(seg) >= int(0.1*sr):
            m = np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1)
            textures.append(m)
    
    # B-Sim measures if breaths sound like "clones" (AI trait)
    sim_val = np.mean(np.std(textures, axis=0)) if len(textures) > 1 else 10.0

    # --- AGGRESSIVE SCORING ENGINE (v11.0) ---
    ai_points = 20  # Base level
    
    # Flag 1: The AI Timing "Sweet Spot" (Catches samples from your screenshot)
    if 0.16 < ibi_cv < 0.31: 
        ai_points += 45 
    elif ibi_cv < 0.15: 
        ai_points += 55 # High robotic penalty
        
    # Flag 2: Uniform Amplitude (AI lacks organic volume changes)
    if amp_cv < 0.19: 
        ai_points += 30
    
    # Flag 3: Clone Check (Detects reused breath samples)
    if sim_val < 1.35: 
        ai_points += 45
    
    # Flag 4: Over-Breathing (AI often over-injects breaths for 'realism')
    if presence_ratio > 0.26: 
        ai_points += 35

    # Result Calculation
    final_prob = min(99, ai_points)
    # Threshold at 50% for AI classification
    status = "AI" if final_prob >= 50 else "HUMAN"
    
    return {
        "File": name, "Status": status, "AI Prob": f"{final_prob}%",
        "IBI CV": round(ibi_cv, 3), "Amp CV": round(amp_cv, 3), 
        "Presence": f"{presence_ratio:.1%}", "B-Sim": round(sim_val, 3)
    }, breaths

# --- STREAMLIT UI ---
st.title("🔬 PneumaForensic v11.0")
st.markdown("### Forensic Parameter Analysis for Synthetic Voice Detection")

files = st.file_uploader("Upload Batch (Handles 150+ files)", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    results_list = []
    
    for f in files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            try:
                metrics, b_times = forensic_analysis(y, sr, f.name)
                results_list.append(metrics)
                
                # Render Individual Graph (Gray Waves/Red Dashed)
                with st.expander(f"Visual Scan: {f.name}"):
                    fig, ax = plt.subplots(figsize=(12, 1.2))
                    t = np.linspace(0, len(y)/sr, len(y))
                    ax.plot(t, y, color='gray', alpha=0.4, linewidth=0.6) 
                    for bt in b_times:
                        ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2)
                    ax.set_title(f"{metrics['Status']} ({metrics['AI Prob']})", color='white', loc='right', fontsize=9)
                    ax.set_facecolor('#0e1117')
                    fig.patch.set_facecolor('#0e1117')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")
            
            # Explicit Memory Management
            del y
            gc.collect()

    st.divider()
    if results_list:
        df = pd.DataFrame(results_list)
        
        def color_status(val):
            color = "#ff4b4b" if "AI" in val else "#00f900"
            return f'color: {color}; font-weight: bold'
        
        st.subheader("📊 Analytical Results")
        try:
            st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
            st.download_button("Download Report (CSV)", df.to_csv(index=False), "forensic_report.csv")
        except:
            st.dataframe(df, use_container_width=True)
else:
    st.info("Awaiting audio files for biometric parameter check.")
