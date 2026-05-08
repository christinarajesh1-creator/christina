import streamlit as st
import numpy as np
import librosa
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(layout="wide")

@st.cache_data
def load_audio(file_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        return y, sr
    except:
        return None, None

def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = librosa.get_duration(y=y_norm, sr=sr)
    
    # 1. Advanced Breath Detection (RMS + Peak Analysis)
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop)[0]
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Dynamic peak finding tailored for breath envelopes
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.1, distance=sr//hop)
    
    # Extract timing and segments
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    # Default result for no breaths
    if len(breaths) < 2:
        return {
            "File": name, "Status": "AI", "AI Prob": "98%", 
            "IBI": "0.0", "Amp": "0.0", "Dur": "0.0", "Presence": "0%", "Continuity": "0.0", "Similarity": "0.0"
        }, []

    # --- PARAMETER 1: IBI Regularity (28%) ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi)
    # AI often has CV < 0.1 (perfect) or CV > 0.5 (random stitching)
    ibi_score = 1.0 if (ibi_cv < 0.15 or ibi_cv > 0.45) else 0.0

    # --- PARAMETER 2: Breath Amplitude (15%) ---
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps)
    amp_score = 1.0 if (amp_cv < 0.12) else 0.0

    # --- PARAMETER 3: Breath Duration (12%) ---
    # Measure width of RMS peaks at half-height
    widths = signal.peak_widths(rms_smooth, peaks, rel_height=0.5)[0] / (sr/hop)
    dur_cv = np.std(widths) / np.mean(widths)
    dur_score = 1.0 if (dur_cv < 0.15) else 0.0

    # --- PARAMETER 4: Breath Presence (15%) ---
    presence_ratio = (len(breaths) * 0.5) / duration # Assumes avg breath is 0.5s
    # AI either lacks breaths or over-injects them (Ratio > 0.25 is sus)
    presence_score = 1.0 if (presence_ratio < 0.02 or presence_ratio > 0.25) else 0.0

    # --- PARAMETER 5: Spectral Continuity (12%) ---
    zcr = librosa.feature.zero_crossing_rate(y_norm)[0]
    zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-10)
    # High ZCR CV indicates digital "noise" or splice artifacts
    cont_score = 1.0 if zcr_cv > 0.45 else 0.0

    # --- PARAMETER 6: Breath Similarity (18%) ---
    # Compare MFCCs of the first few breaths
    mfccs = []
    for b in breaths[:5]:
        start = int(b * sr)
        seg = y_norm[start:start+int(0.3*sr)]
        if len(seg) > 0: mfccs.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    
    sim_val = np.mean(np.std(mfccs, axis=0)) if len(mfccs) > 1 else 0
    # Low sim_val means breaths are "copy-pasted" clones
    sim_score = 1.0 if (sim_val < 0.8) else 0.0

    # --- WEIGHTED DETECTION CALCULATION ---
    final_score = (ibi_score * 0.28) + (amp_score * 0.15) + (dur_score * 0.12) + \
                  (presence_score * 0.15) + (cont_score * 0.12) + (sim_score * 0.18)
    
    status = "AI" if final_score > 0.45 else "HUMAN"
    
    return {
        "File": name, "Status": status, "AI Prob": f"{final_score:.0%}",
        "IBI Reg": f"{ibi_cv:.3f}", "Amp Var": f"{amp_cv:.3f}", 
        "Dur Var": f"{dur_cv:.3f}", "Presence": f"{presence_ratio:.1%}",
        "ZCR Var": f"{zcr_cv:.3f}", "Sim Val": f"{sim_val:.3f}"
    }, breaths

# --- UI LOGIC ---
st.title("🔬 PneumaForensic v8.0")

files = st.file_uploader("", type=['wav', 'mp3'], accept_multiple_files=True)

if files:
    all_results = []
    for f in files:
        f.seek(0)
        y, sr = load_audio(f.read())
        if y is not None:
            res, peaks = forensic_analysis(y, sr, f.name)
            all_results.append(res)
            
            # --- GRAPHING: Gray Waves & Red Dashed Lines ---
            fig, ax = plt.subplots(figsize=(12, 2))
            t = np.linspace(0, len(y)/sr, len(y))
            ax.plot(t, y, color='gray', alpha=0.6, label="Sound Waves")
            
            for p in peaks:
                ax.axvline(x=p, color='red', linestyle='--', linewidth=1.5, label="Breath Detected" if p == peaks[0] else "")
            
            ax.set_title(f"{f.name} | Status: {res['Status']} ({res['AI Prob']})", color='white')
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

    st.subheader("Forensic Parameter Breakdown")
    df = pd.DataFrame(all_results)
    
    def highlight(s):
        return "color: #ff4b4b" if s == "AI" else "color: #00f900"
    
    try:
        st.dataframe(df.style.map(highlight, subset=['Status']), use_container_width=True)
    except:
        st.dataframe(df, use_container_width=True)
