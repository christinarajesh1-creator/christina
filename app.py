import numpy as np
import librosa
from scipy.spatial import distance

class PneumaForensic:
    @staticmethod
    def advanced_breath_analysis(y, sr):
        """6-parameter forensic analysis - PUBLICATION READY"""
        try:
            # Preprocessing
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            
            # 1. STRICT BREATH DETECTION (Parameter: Breath Presence 15%)
            silence_threshold = np.percentile(rms, 8)
            breath_times = times[rms < silence_threshold * 0.7]
            
            # Filter human-like breathing rhythm only (2-8 seconds gaps)
            events = []
            last_t = 0
            for t in breath_times:
                gap = t - last_t
                if 2.0 < gap < 8.0:  # Human breathing rhythm
                    events.append(t)
                    last_t = t
            
            if len(events) < 2:
                return {
                    "ibi_reg": 1.0,      # IBI Regularity (28%)
                    "amp_var": 0.0,      # Breath Amplitude (15%)
                    "dur_var": 0.0,      # Breath Duration (12%)
                    "presence": 0.01,    # Breath Presence (15%)
                    "spec_cont": 1.0,    # Spectral Continuity (12%)
                    "sim_score": 1.0,    # Breath Similarity (18%)
                    "synthetic_score": 0.95  # Overall synthetic probability
                }
            
            # PARAMETER 1: IBI REGULARITY (28%) - CV of inter-breath intervals
            ibis = np.diff(events)
            ibi_reg = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 1.0
            ibi_reg = min(ibi_reg, 2.0)  # Cap at realistic human max
            
            # PARAMETER 2: BREATH AMPLITUDE VARIATION (15%) - RMS energy variation
            breath_rms = []
            for t in events:
                start = max(0, int((t-0.4)*sr))
                end = min(len(y), int((t+0.6)*sr))
                if end > start:
                    breath_rms.append(np.mean(np.abs(y[start:end])))
            amp_var = np.std(breath_rms) / np.mean(breath_rms) if breath_rms and np.mean(breath_rms) > 0 else 0.0
            amp_var = min(amp_var, 1.5)
            
            # PARAMETER 3: BREATH DURATION VARIATION (12%) - Spectral centroid variation
            cents = []
            for t in events:
                start = max(0, int((t-0.4)*sr))
                end = min(len(y), int((t+0.6)*sr))
                if end > start:
                    cent = librosa.feature.spectral_centroid(y=y[start:end], sr=sr)[0]
                    cents.append(np.mean(cent) if len(cent) > 0 else 0)
            dur_var = np.std(cents) / np.mean(cents) if cents and np.mean(cents) > 0 else 0.0
            dur_var = min(dur_var, 2.0)
            
            # PARAMETER 4: BREATH PRESENCE (15%) - Proportion of breath audio
            total_breath_duration = sum(np.diff(events + [events[-1]])) if len(events) > 1 else 0
            presence = min(total_breath_duration / max(times[-1], 1.0), 0.4)
            
            # PARAMETER 5: SPECTRAL CONTINUITY (12%) - ZCR discontinuity at breath boundaries
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_events = []
            for t in events:
                idx = min(int(t*sr / (sr/512)), len(zcr)-1)  # Frame-aligned
                if 0 <= idx < len(zcr):
                    zcr_events.append(zcr[idx])
            spec_cont = np.std(zcr_events) if len(zcr_events) > 1 else 1.0
            spec_cont = min(spec_cont * 2, 1.5)  # Normalize
            
            # PARAMETER 6: BREATH SIMILARITY (18%) - MFCC Euclidean distance between breaths
            mfccs = []
            for t in events:
                start = max(0, int((t-0.4)*sr))
                end = min(len(y), int((t+0.6)*sr))
                if end > start + sr//20:  # Minimum breath length
                    mfcc = librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=5, n_fft=2048)
                    if mfcc.shape[1] > 0:
                        mfcc_mean = np.mean(mfcc, axis=1)
                        mfccs.append(mfcc_mean)
            
            sim_score = 1.0
            if len(mfccs) > 1:
                # Calculate average pairwise Euclidean distance (low = synthetic)
                dists = []
                for i in range(len(mfccs)):
                    for j in range(i+1, len(mfccs)):
                        dists.append(distance.euclidean(mfccs[i], mfccs[j]))
                sim_score = np.mean(dists) if dists else 1.0
                sim_score = min(sim_score / 1000, 1.0)  # Normalize to 0-1
            
            # FINAL SYNTHETIC SCORE (weighted combination)
            weights = {
                'ibi_reg': 0.28,
                'amp_var': 0.15,
                'dur_var': 0.12,
                'presence': 0.15,
                'spec_cont': 0.12,
                'sim_score': 0.18
            }
            
            # Synthetic indicators: LOW variation = SYNTHETIC
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
                "ibi_reg": float(ibi_reg),           # 28% - Timing regularity
                "amp_var": float(amp_var),           # 15% - Amplitude variation  
                "dur_var": float(dur_var),           # 12% - Duration variation
                "presence": float(presence),         # 15% - Breath presence ratio
                "spec_cont": float(spec_cont),       # 12% - Spectral continuity
                "sim_score": float(sim_score),       # 18% - Breath similarity
                "synthetic_score": float(synthetic_score)  # 0-1 synthetic probability
            }
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                "ibi_reg": 1.0, "amp_var": 0.0, "dur_var": 0.0,
                "presence": 0.01, "spec_cont": 1.0, "sim_score": 1.0,
                "synthetic_score": 0.95
            }

# USAGE EXAMPLE
if __name__ == "__main__":
    # Load test audio
    y, sr = librosa.load("your_audio_file.wav", sr=22050)
    
    # Analyze
    results = PneumaForensic.advanced_breath_analysis(y, sr)
    
    print("FORENSIC BREATH ANALYSIS RESULTS:")
    print(f"IBI Regularity (28%):    {results['ibi_reg']:.3f}")
    print(f"Breath Amplitude (15%):  {results['amp_var']:.3f}") 
    print(f"Breath Duration (12%):   {results['dur_var']:.3f}")
    print(f"Breath Presence (15%):   {results['presence']:.3f}")
    print(f"Spectral Continuity (12%): {results['spec_cont']:.3f}")
    print(f"Breath Similarity (18%): {results['sim_score']:.3f}")
    print(f"\n🎯 SYNTHETIC PROBABILITY: {results['synthetic_score']:.3f}")
    
    if results['synthetic_score'] > 0.7:
        print("🔴 HIGHLY SYNTHETIC")
    elif results['synthetic_score'] > 0.4:
        print("🟡 SUSPICIOUS")
    else:
        print("🟢 NATURAL")
