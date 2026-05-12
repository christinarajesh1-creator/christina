import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import gc

st.set_page_config(page_title="PneumaForensic v9.2", layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        # Load 20s to balance accuracy and memory for 150+ files
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True, duration=20)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # Breath Tracking (RMS Envelope)
    hop = 512
    rms = librosa.feature.rms(y=y_norm, hop_length=hop)[0]
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Peak Finding
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.1, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    # 1. IBI Regularity (28%)
    ibi = np.diff(breaths) if len(breaths) > 1 else np.array([0])
    ibi_cv = np.std(ibi) / np.mean(ibi) if np.mean(ibi) > 0 else 0
    ibi_score = 1.0 if (0.17 < ibi_cv < 0.26 or ibi_cv < 0.12) else 0.0

    # 2. Breath Amplitude (15%)
    amps = [rms_smooth[p] for p in peaks] if len(peaks) > 0 else [0]
    amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0
    amp_score = 1.0 if (amp_cv < 0.18) else 0.0

    # 3. Breath Duration (12%)
    widths = signal.peak_widths(rms_smooth, peaks, rel_height=0.5)[0] / (sr/hop) if len(peaks) > 0 else [0]
    dur_cv = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0
    dur_score = 1.0 if (dur_cv < 0.15) else 0.0

    # 4. Breath Presence (15%)
    presence_ratio = (len(breaths) * 0.45) / duration 
    presence_score = 1.0 if (presence_ratio > 0.28 or presence_ratio < 0.03) else 0.0

    # 5. Spectral Continuity (12%)
    zcr = librosa.feature.zero_crossing_rate(y_norm)[0]
    zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-10)
    cont_score = 1.0 if (zcr_cv > 0.42) else 0.0

    # 6. Breath Similarity (18%)
    mfccs = []
    for b in breaths[:4]:
        start = int(b * sr)
        seg = y_norm[start:start+int(0.2*sr)]
        if len(seg) >= int(0.1*sr):
            mfccs.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    sim_val = np.mean(np.std(mfccs, axis=0)) if len(mfccs) > 1 else 10.0
    sim_score = 1.0 if (sim_val < 1.4) else 0.0

    # Final Weights
    total = (ibi_score * 0.28) + (amp_score * 0.15) + (dur_score * 0.12) + \
            (presence_score * 0.15) + (cont_score * 0.12) + (sim_score * 0.18)
    
    status = "AI" if total >= 0.40 else "HUMAN"
    
    return {
        "File": name, "Status": status, "AI Prob": f"{min(99, total*140):.0%}",
        "IBI Reg (28%)": round(ibi_cv, 3), "Amp Var (15%)": round(amp_cv, 3), 
        "Dur Var (12%)": round(dur_cv, 3), "Presence (15%)": f"{presence_ratio:.1%}", 
        "S-Cont (12%)": round(zcr_cv, 3), "B-Sim (18%)": round(sim_val, 3)
    }, breaths

# --- UI ---
st.title("🔬 PneumaForensic v9.2")

files = st.file_uploader("Upload Batch (Up to 150 files)", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    results = []
    
    # Process files
    for f in files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            m, b_times = forensic_analysis(y, sr, f.name)
            results.append(m)
            
            # Individual Visualizer
            with st.container():
                fig, ax = plt.subplots(figsize=(12, 1.5))
                t = np.linspace(0, len(y)/sr, len(y))
                ax.plot(t, y, color='gray', alpha=0.5, linewidth=0.7) # Gray Waves
                for bt in b_times:
                    ax.axvline(x=bt, color='red', linestyle='--', linewidth=1.2) # Red Dashed
                ax.set_title(f"{f.name} | {m['Status']} ({m['AI Prob']})", color='white', loc='left', fontsize=10)
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            
            del y
            gc.collect()

    st.divider()
    st.subheader("📊 Forensic Parameter Table")
    df = pd.DataFrame(results)
    
    def color_status(s):
        return "color: #ff4b4b; font-weight: bold" if s == "AI" else "color: #00f900; font-weight: bold"

    if not df.empty:
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
        st.download_button("Export Results", df.to_csv(index=False), "forensic_report.csv")
