import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import gc

st.set_page_config(page_title="PneumaForensic v9.0", layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        # Load mono at 16kHz for consistent forensic analysis
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True, duration=30)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr

    # 1. Advanced Breath Tracking (RMS Envelope)
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')

    # Identify breath candidates with a slightly higher sensitivity
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.15, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]

    # Lack of Bio-markers is an immediate AI flag
    if len(breaths) < 2:
        return {
            "File": name, "Status": "AI", "AI Prob": "98%", 
            "IBI Reg": 0.0, "Amp Var": 0.0, "Presence": "0%", "ZCR Var": 0.0, "Sim Val": 0.0
        }, []

    # --- THE 6 RE-CALIBRATED PARAMETERS ---
    
    # 1. IBI Regularity (28%): Flags the "Uncanny Valley"
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    ibi_flag = 1.0 if (0.17 < ibi_cv < 0.26 or ibi_cv < 0.12) else 0.0

    # 2. Breath Amplitude (15%): Flags uniform volume
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    amp_flag = 1.0 if (amp_cv < 0.18) else 0.0

    # 3. Breath Duration (12%): Flags identical length breaths
    # Calculate peak widths as a proxy for breath duration
    widths = signal.peak_widths(rms_smooth, peaks, rel_height=0.5)[0] / (sr/hop)
    dur_cv = np.std(widths) / np.mean(widths) if len(widths) > 0 else 0
    dur_flag = 1.0 if (dur_cv < 0.15) else 0.0

    # 4. Breath Presence (15%): Flags "Over-Breathing"
    presence_ratio = (len(breaths) * 0.45) / duration 
    presence_flag = 1.0 if (presence_ratio > 0.28 or presence_ratio < 0.03) else 0.0

    # 5. Spectral Continuity (12%): Flags digital texture artifacts
    zcr = librosa.feature.zero_crossing_rate(y_norm).flatten()
    zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-10)
    cont_flag = 1.0 if (zcr_cv > 0.42) else 0.0

    # 6. Breath Similarity (18%): Flags copy-pasted audio segments
    mfccs = []
    for b in breaths[:4]:
        start = int(b * sr)
        seg = y_norm[start:start+int(0.25*sr)]
        if len(seg) >= int(0.2*sr):
            mfccs.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))

    sim_val = np.mean(np.std(mfccs, axis=0)) if len(mfccs) > 1 else 10.0
    sim_flag = 1.0 if (sim_val < 1.4) else 0.0

    # --- WEIGHTED AI SCORING ---
    final_score = (ibi_flag * 0.28) + (amp_flag * 0.15) + (dur_flag * 0.12) + \
                  (presence_flag * 0.15) + (cont_flag * 0.12) + (sim_flag * 0.18)

    status = "AI" if final_score >= 0.40 else "HUMAN"

    return {
        "File": name, "Status": status, "AI Prob": f"{min(99, final_score*140):.0%}",
        "IBI Reg": round(ibi_cv, 3), "Amp Var": round(amp_cv, 3), 
        "Presence": f"{presence_ratio:.1%}", "ZCR Var": round(zcr_cv, 3), 
        "Sim Val": round(sim_val, 3)
    }, breaths

# --- UI SECTION ---
st.title("🔬 PneumaForensic v9.0")
st.caption("Forensic Analysis: Bio-metric Breath Integrity Scan")

uploaded_files = st.file_uploader("Upload Audio Batch", type=['wav', 'mp3', 'm4a'], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    for f in uploaded_files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            metrics, peaks = forensic_analysis(y, sr, f.name)
            results_list.append(metrics)
            
            # Rendering Plot: Gray Waves & Red Dashed Lines
            with st.expander(f"Visual Scan: {f.name}"):
                fig, ax = plt.subplots(figsize=(12, 1.8))
                t = np.linspace(0, len(y)/sr, len(y))
                ax.plot(t, y, color='gray', alpha=0.5)
                for p in peaks:
                    ax.axvline(x=p, color='red', linestyle='--', linewidth=1.2)
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax.set_title(f"{metrics['Status']} ({metrics['AI Prob']})", color='white', loc='right')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            
            del y
            gc.collect()

    st.divider()
    st.subheader("📊 Analytical Metrics")
    if results_list:
        df = pd.DataFrame(results_list)

        def style_status(val):
            color = '#ff4b4b' if val == "AI" else '#00f900'
            return f'color: {color}; font-weight: bold'

        try:
            st.dataframe(df.style.map(style_status, subset=['Status']), use_container_width=True)
        except:
            st.dataframe(df, use_container_width=True)
