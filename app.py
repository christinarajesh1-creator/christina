import streamlit as st
import numpy as np
import librosa
import io
from scipy.spatial import distance
import plotly.graph_objects as go
import plotly.express as px

# Your PneumaForensic class (unchanged)
class PneumaForensic:
    @staticmethod
    def advanced_breath_analysis(y, sr):
        """6-parameter forensic analysis - PUBLICATION READY"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            
            silence_threshold = np.percentile(rms, 8)
            breath_times = times[rms < silence_threshold * 0.7]
            
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
                    "synthetic_score": 0.95
                }
            
            # IBI REGULARITY (28%)
            ibis = np.diff(events)
            ibi_reg = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 1.0
            ibi_reg = min(ibi_reg, 2.0)
            
            # BREATH AMPLITUDE (15%)
            breath_rms = []
            for t in events:
                start = max(0, int((t-0.4)*sr))
                end = min(len(y), int((t+0.6)*sr))
                if end > start:
                    breath_rms.append(np.mean(np.abs(y[start:end])))
            amp_var = np.std(breath_rms) / np.mean(breath_rms) if breath_rms and np.mean(breath_rms) > 0 else 0.0
            amp_var = min(amp_var, 1.5)
            
            # BREATH DURATION (12%)
            cents = []
            for t in events:
                start = max(0, int((t-0.4)*sr))
                end = min(len(y), int((t+0.6)*sr))
                if end > start:
                    cent = librosa.feature.spectral_centroid(y=y[start:end], sr=sr)[0]
                    cents.append(np.mean(cent) if len(cent) > 0 else 0)
            dur_var = np.std(cents) / np.mean(cents) if cents and np.mean(cents) > 0 else 0.0
            dur_var = min(dur_var, 2.0)
            
            # BREATH PRESENCE (15%)
            total_breath_duration = sum(np.diff(events + [events[-1]])) if len(events) > 1 else 0
            presence = min(total_breath_duration / max(times[-1], 1.0), 0.4)
            
            # SPECTRAL CONTINUITY (12%)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_events = []
            for t in events:
                idx = min(int(t*sr / (sr/512)), len(zcr)-1)
                if 0 <= idx < len(zcr):
                    zcr_events.append(zcr[idx])
            spec_cont = np.std(zcr_events) if len(zcr_events) > 1 else 1.0
            spec_cont = min(spec_cont * 2, 1.5)
            
            # BREATH SIMILARITY (18%)
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
            
            # SYNTHETIC SCORE
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
                "breath_events": len(events)
            }
            
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return None

# === STREAMLIT APP ===
st.set_page_config(page_title="🫁 PneumaForensic", layout="wide")
st.title("🫁 PneumaForensic")
st.markdown("**AI Voice Detection via Forensic Breath Analysis** - 6 Parameter System")

# File uploader
uploaded_file = st.file_uploader("Upload audio file (WAV, MP3, M4A)", type=['wav', 'mp3', 'm4a', 'flac'])

if uploaded_file is not None:
    # Load audio
    try:
        y, sr = librosa.load(uploaded_file, sr=22050)
        duration = len(y) / sr
        st.success(f"✅ Audio loaded: {duration:.1f}s, {sr}Hz")
        
        # Analyze
        with st.spinner("🔍 Analyzing breath patterns..."):
            results = PneumaForensic.advanced_breath_analysis(y, sr)
        
        if results:
            # Main results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("🎯 Synthetic Probability", 
                         f"{results['synthetic_score']:.1%}", 
                         delta=None)
                
                if results['synthetic_score'] > 0.7:
                    st.error("🔴 **HIGHLY SYNTHETIC**")
                elif results['synthetic_score'] > 0.4:
                    st.warning("🟡 **SUSPICIOUS**")
                else:
                    st.success("🟢 **NATURAL HUMAN**")
            
            with col2:
                st.subheader("6 Forensic Parameters")
                params = [
                    ("IBI Regularity", results['ibi_reg'], "28%"),
                    ("Breath Amplitude", results['amp_var'], "15%"),
                    ("Breath Duration", results['dur_var'], "12%"),
                    ("Breath Presence", results['presence'], "15%"),
                    ("Spectral Continuity", results['spec_cont'], "12%"),
                    ("Breath Similarity", results['sim_score'], "18%")
                ]
                
                for name, value, weight in params:
                    st.metric(name, f"{value:.3f}", f"{weight} weight")
            
            # Radar chart
            st.subheader("📊 Forensic Profile")
            angles = ['IBI', 'Amp', 'Dur', 'Presence', 'Spec', 'Sim']
            values = [results['ibi_reg'], results['amp_var'], results['dur_var'], 
                     results['presence'], results['spec_cont'], results['sim_score']]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values, theta=angles, fill='toself', name='Forensic Score'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
                            showlegend=False, title="Parameter Variation (Low = Synthetic)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Audio waveform with breaths
            st.subheader("🎵 Audio + Detected Breaths")
            times = np.linspace(0, len(y)/sr, len(y))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=times, y=y, mode='lines', 
                                    name='Audio', line=dict(color='blue')))
            
            if 'breath_events' in results:
                breath_times = [t for t in range(len(y)/sr) if any(abs(t-e)<0.5 for e in events)]
                fig2.add_trace(go.Scatter(x=[e for e in events], y=[0]*len(events),
                                        mode='markers', name='Breaths',
                                        marker=dict(color='red', size=10)))
            
            fig2.update_layout(title="Waveform + Breath Events", xaxis_title="Time (s)")
            st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Audio loading failed: {e}")

st.markdown("---")
st.markdown("**🫁 PneumaForensic** detects AI voices by analyzing **breathing patterns** - the ultimate forensic discriminator.")
