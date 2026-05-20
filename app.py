import streamlit as st
import numpy as np
import librosa
import librosa.feature
import io
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# Secure the Excel exporter dependency safely
try:
    import xlsxwriter
except ImportError:
    st.error("Missing dependency: Please run 'pip install xlsxwriter' in your terminal.")

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Forensic Voice Authenticator", layout="wide")

# ==========================================
# 2. BOUNDARY CRITERIA (Appendix A)
# ==========================================
# These constants match the paper's exact specifications
BOUNDARIES = {
    "amplitude_drop_min_db": 30.0,      # Minimum 30 dB drop
    "amplitude_drop_max_db": 60.0,     # Maximum 60 dB drop
    "duration_min_ms": 100.0,           # 100 ms minimum
    "duration_max_ms": 2500.0,         # 2500 ms maximum
    "decay_r_squared": 0.85,           # R² > 0.85 required
    "ratio_min": 0.05,                 # Balance index min
    "ratio_max": 0.35,                 # Balance index max
    "spectral_min_hz": 300.0,          # 300 Hz minimum
    "spectral_max_hz": 2000.0,         # 2000 Hz maximum
    "recovery_min_ms": 300.0,          # 300 ms minimum
    "centroid_max_hz": 2150.0,         # For < 2 breaths fallback
    "rolloff_max_hz": 4100.0            # For < 2 breaths fallback
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def exponential_decay(x, a, b):
    """Natural exponential decay model for breath envelope fitting."""
    return a * np.exp(-b * x)

def compute_r_squared(y_true, y_pred):
    """Compute R² for decay curve validation."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-9))

def compute_spectral_band_energy(y, sr, hop_length, band_min, band_max):
    """
    Compute spectral energy within specified frequency band (300-2000 Hz).
    Returns the proportion of energy in the human fricative band.
    """
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Find indices for the frequency band
    band_min_idx = np.searchsorted(freqs, band_min)
    band_max_idx = np.searchsorted(freqs, band_max)
    
    # Compute band energy
    band_energy = np.sum(S[band_min_idx:band_max_idx, :] ** 2)
    total_energy = np.sum(S ** 2) + 1e-9
    
    return band_energy / total_energy

# ==========================================
# 4. HYBRID BIOMETRIC & SPECTRAL ENGINE
# ==========================================
def extract_hybrid_forensic_features(y, sr):
    """
    Extracts the 6-parameter biological breath matrix as specified in the paper.
    
    Parameters (from Section 4.2):
    1. Amplitude: Inhalation drop (dB) relative to peak speech
    2. Duration: Breath event duration (ms)
    3. Decay Rate: Exponential fit R²
    4. Speech-to-Breath Ratio: Balance index (0.05-0.35)
    5. Spectral Flow: Fricative band containment (300-2000 Hz)
    6. Recovery Intervals: Time between breath events (ms)
    """
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr  # Total duration in seconds
    
    hop_length = 256
    frame_time = hop_length / sr
    
    # --- Compute global speech metrics for amplitude reference ---
    rms = librosa.feature.rms(y=y_norm, hop_length=hop_length).flatten()
    rms_smooth = np.convolve(rms, np.ones(3)/3, mode='same')
    peak_speech_rms = np.percentile(rms_smooth, 95)  # Peak speech amplitude
    peak_speech_db = 20 * np.log10(peak_speech_rms + 1e-6)
    
    # --- Extract acoustic features ---
    flatness = librosa.feature.spectral_flatness(y=y_norm, hop_length=hop_length).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=y_norm, hop_length=hop_length).flatten()
    
    rms_min, rms_max = np.min(rms_smooth), np.max(rms_smooth)
    rms_rel = (rms_smooth - rms_min) / (rms_max - rms_min + 1e-6)
    
    # --- Identify breath zones ---
    # Breath = quiet (0.02 < rms_rel < 0.30) + high flatness + moderate zcr
    breath_indices = []
    for i in range(len(rms_rel)):
        if (0.02 < rms_rel[i] < 0.30) and (flatness[i] > 0.08) and (zcr[i] > 0.10):
            breath_indices.append(i)
    
    # --- Cluster into discrete breath events ---
    breath_events = []
    if breath_indices:
        current_event = [breath_indices[0]]
        for idx in breath_indices[1:]:
            if idx == current_event[-1] + 1:
                current_event.append(idx)
            else:
                dur_ms = len(current_event) * frame_time * 1000.0
                # Apply paper's exact duration bounds: 100-2500 ms
                if BOUNDARIES["duration_min_ms"] <= dur_ms <= BOUNDARIES["duration_max_ms"]:
                    breath_events.append(current_event)
                current_event = [idx]
        
        # Handle final event
        dur_ms = len(current_event) * frame_time * 1000.0
        if BOUNDARIES["duration_min_ms"] <= dur_ms <= BOUNDARIES["duration_max_ms"]:
            breath_events.append(current_event)
    
    num_breaths = len(breath_events)
    
    # --- Initialize 6-parameter matrix ---
    raw_metrics = {
        "amplitude": -60.0,           # Parameter 1: dB drop relative to speech
        "duration": 0.0,             # Parameter 2: Mean duration (ms)
        "decay_rate": 0.0,            # Parameter 3: R² of exponential fit
        "speech_breath_ratio": 0.0,  # Parameter 4: Balance index
        "spectral_flow": 0.0,        # Parameter 5: Band containment ratio
        "recovery_intervals": 0.0,  # Parameter 6: Mean gap (ms)
        "guardrail_rolloff": 0.0,    # Spectral guardrail
        "guardrail_centroid": 0.0   # Spectral guardrail
    }
    
    # --- Compute spectral guardrails ---
    rolloff = librosa.feature.spectral_rolloff(y=y_norm, sr=sr, roll_percent=0.85)
    raw_metrics["guardrail_rolloff"] = float(np.mean(rolloff))
    
    centroid = librosa.feature.spectral_centroid(y=y_norm, sr=sr)
    raw_metrics["guardrail_centroid"] = float(np.mean(centroid))
    
    # Early exit if insufficient breath events (Section 6.3.1)
    if num_breaths < 2:
        return raw_metrics, num_breaths
    
    # --- PARAMETER 1: Amplitude Drop (dB) ---
    # Compute mean amplitude of breath events
    all_breath_frames = [frame for evt in breath_events for frame in evt]
    breath_rms_vals = rms_smooth[all_breath_frames]
    breath_amplitude_db = 20 * np.log10(np.mean(breath_rms_vals) + 1e-6)
    
    # Amplitude drop = peak speech dB - breath dB
    amplitude_drop = peak_speech_db - breath_amplitude_db
    raw_metrics["amplitude"] = float(amplitude_drop)
    
    # --- PARAMETER 2: Duration (ms) ---
    durations = [len(evt) * frame_time * 1000.0 for evt in breath_events]
    raw_metrics["duration"] = float(np.mean(durations))
    
    # --- PARAMETER 3: Decay Rate (R² of exponential fit) ---
    # Fit exponential decay to each breath event's RMS envelope
    r_squared_values = []
    for evt in breath_events:
        if len(evt) >= 5:  # Need minimum frames for fitting
            env = rms_smooth[evt]
            x = np.arange(len(env))
            try:
                # Normalize envelope for fitting
                env_norm = env / (np.max(env) + 1e-9)
                popt, _ = curve_fit(exponential_decay, x, env_norm, p0=[1.0, 0.1], maxfev=1000)
                y_pred = exponential_decay(x, *popt)
                r2 = compute_r_squared(env_norm, y_pred)
                r_squared_values.append(r2)
            except:
                r_squared_values.append(0.0)
    
    raw_metrics["decay_rate"] = float(np.mean(r_squared_values)) if r_squared_values else 0.0
    
    # --- PARAMETER 4: Speech-to-Breath Ratio (Balance Index) ---
    total_breath_time = sum(len(evt) * frame_time for evt in breath_events)
    total_speech_time = duration - total_breath_time
    
    # Paper's balance index: speech_time / (speech_time + breath_time)
    # Must be between 0.05 and 0.35
    raw_metrics["speech_breath_ratio"] = float(
        total_speech_time / (duration + 1e-6)
    )
    
    # --- PARAMETER 5: Spectral Flow (300-2000 Hz band) ---
    # Compute spectral band containment for breath events
    band_energy_ratios = []
    for evt in breath_events:
        breath_y = y_norm[int(evt[0] * hop_length):int(evt[-1] * hop_length) + hop_length]
        if len(breath_y) > hop_length:
            band_ratio = compute_spectral_band_energy(
                breath_y, sr, hop_length,
                BOUNDARIES["spectral_min_hz"],
                BOUNDARIES["spectral_max_hz"]
            )
            band_energy_ratios.append(band_ratio)
    
    raw_metrics["spectral_flow"] = float(np.mean(band_energy_ratios)) if band_energy_ratios else 0.0
    
    # --- PARAMETER 6: Recovery Intervals (ms) ---
    event_times = [evt[0] * frame_time for evt in breath_events]
    if len(event_times) >= 2:
        gaps_ms = np.diff(event_times) * 1000.0  # Convert to ms
        raw_metrics["recovery_intervals"] = float(np.mean(gaps_ms))
    
    return raw_metrics, num_breaths

# ==========================================
# 5. QUANTITATIVE RULE EVALUATION ENGINE
# ==========================================
def evaluate_hybrid_forensic_verdict(features, num_breaths):
    """
    Evaluates biometric vectors against real-world human baselines.
    Uses the boundary criteria from Appendix A.
    """
    # --- Fallback for insufficient breath events (Section 6.3.1) ---
    if num_breaths < 2:
        # Use spectral guardrails as fallback (from Section 5.2.1 discussion)
        if features["guardrail_centroid"] > BOUNDARIES["centroid_max_hz"] or \
           features["guardrail_rolloff"] > BOUNDARIES["rolloff_max_hz"]:
            excess_freq = max(0, features["guardrail_centroid"] - BOUNDARIES["centroid_max_hz"])
            dynamic_penalty = min(0.11, excess_freq / 1500.0)
            return 0.85 + dynamic_penalty, "AI / DEEPFAKE"
        return 0.35, "HUMAN"
    
    # --- Compute binary alignment with exact paper boundaries ---
    # Parameter 1: Amplitude drop (30-60 dB)
    amp_score = 1.0 if (BOUNDARIES["amplitude_drop_min_db"] <= features["amplitude"] <= 
                       BOUNDARIES["amplitude_drop_max_db"]) else 0.0
    
    # Parameter 2: Duration (100-2500 ms)
    dur_score = 1.0 if (BOUNDARIES["duration_min_ms"] <= features["duration"] <= 
                       BOUNDARIES["duration_max_ms"]) else 0.0
    
    # Parameter 3: Decay R² > 0.85
    decay_score = 1.0 if features["decay_rate"] > BOUNDARIES["decay_r_squared"] else 0.0
    
    # Parameter 4: Speech/Breath ratio (0.05-0.35)
    ratio_score = 1.0 if (BOUNDARIES["ratio_min"] <= features["speech_breath_ratio"] <= 
                         BOUNDARIES["ratio_max"]) else 0.0
    
    # Parameter 5: Spectral flow (300-2000 Hz band dominance check)
    # A valid human breath should have >30% energy in this band
    flow_score = 1.0 if (features["spectral_flow"] > 0.30) else 0.0
    
    # Parameter 6: Recovery intervals >= 300 ms
    recovery_score = 1.0 if features["recovery_intervals"] >= BOUNDARIES["recovery_min_ms"] else 0.0
    
    # --- Weighted human alignment score (from Section 5.2) ---
    human_alignment = (
        (recovery_score * 0.28) +
        (decay_score * 0.18) +
        (amp_score * 0.15) +
        (ratio_score * 0.15) +
        (dur_score * 0.12) +
        (flow_score * 0.12)
    )
    
    # Convert to probability of AI/deepfake
    prob = 1.0 - human_alignment
    
    # --- Apply continuous dynamic overrides (Section 5.2) ---
    if features["guardrail_rolloff"] > BOUNDARIES["rolloff_max_hz"] or \
       features["guardrail_centroid"] > BOUNDARIES["centroid_max_hz"]:
        excess_freq = max(0, features["guardrail_centroid"] - BOUNDARIES["centroid_max_hz"])
        dynamic_penalty = min(0.14, excess_freq / 1000.0)
        prob = max(prob, 0.85 + dynamic_penalty)
    
    # Clamp probability to valid range
    prob = max(0.01, min(0.99, prob))
    
    # Determine status
    status = "AI / DEEPFAKE" if prob >= 0.50 else "HUMAN"
    
    return prob, status

# ==========================================
# 6. EXCEL EXPORT BUFFER UTILITY
# ==========================================
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forensic_Report')
    return output.getvalue()

# ==========================================
# 7. STREAMLIT INTERFACE WORKFLOW
# ==========================================
st.title("🔬 Deepfake Voice Detection Engine")
st.caption("Forensic Hybrid Fusion Pipeline: Validated Non-Linear Respiratory Biometrics")

st.markdown("---")
st.markdown("### 🎤 Upload Forensic Audio Batch")

uploaded_files = st.file_uploader(
    "Upload audio files for batch analysis (WAV, MP3, FLAC)",
    type=['wav', 'mp3', 'flac'],
    accept_multiple_files=True
)

if uploaded_files:
    results_list = []
    progress_bar = st.progress(0)
    
    for i, f in enumerate(uploaded_files):
        f.seek(0)
        try:
            # Load audio at 16kHz for consistency (paper uses PCM processing)
            y, sr = librosa.load(
                io.BytesIO(f.read()),
                sr=16000,
                mono=True,
                duration=15  # Cap at 15 seconds per paper methodology
            )
        except Exception as e:
            st.error(f"Skipping unreadable file {f.name}: {str(e)}")
            continue
        
        if y is not None and len(y) > 0:
            # Extract the 6-parameter biological breath matrix
            metrics, num_breaths = extract_hybrid_forensic_features(y, sr)
            
            # Evaluate against human baseline boundaries
            prob, status = evaluate_hybrid_forensic_verdict(metrics, num_breaths)
            
            # Append results
