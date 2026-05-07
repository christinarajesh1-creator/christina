import numpy as np
import librosa

class PneumaForensic:
    @staticmethod
    def advanced_breath_analysis(y, sr):
        """6-parameter forensic analysis - PUBLICATION READY"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            
            # 1. STRICT breath detection
            silence_threshold = np.percentile(rms, 8)
            breath_times = times[rms < silence_threshold * 0.7]
            
            # Filter human-like gaps only
            events = []
            last_t = 0
            for t in breath_times:
                gap = t - last_t
                if 2.0 < gap < 8.0:  # Human breathing rhythm
                    events.append(t)
                    last_t = t
            
            if len(events) < 2:
                return {
                    "ibi_reg": 1.0, "amp_var": 0.0, "dur_var": 0.0,
                    "presence": 0.01, "spec_cont": 1.0, "sim_score": 1.0
                }
            
            # FEATURE 1: IBI REGULARITY (28%) - CV of intervals
            ibis = np.diff(events)
            ibi_reg = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 1.0
            
            # FEATURE 2: BREATH AMPLITUDE (15%) - RMS variation
            breath_rms = []
            for t in events:
                start = max(0, int((t-0.5)*sr))
                end = min(len(y), int((t+0.5)*sr))
                if end > start:
                    breath_rms.append(np.mean(np.abs(y[start:end])))
            amp_var = np.std(breath_rms) / np.mean(breath_rms) if breath_rms and np.mean(breath_rms) > 0 else 0.0
            
            # FEATURE 3: BREATH DURATION (12%) - Spectral centroid variation
            cents = []
            for t in events:
                start = max(0, int((t-0.5)*sr))
                end = min(len(y), int((t+0.5)*sr))
                if end > start:
                    cent = librosa.feature.spectral_centroid(y=y[start:end], sr=sr)[0]
                    cents.append(np.mean(cent) if len(cent) > 0 else 0)
            dur_var = np.std(cents) / np.mean(cents) if cents and np.mean(cents) > 0 else 0.0
            
            # FEATURE 4: BREATH PRESENCE (15%) - Silence ratio
            presence = len(breath_times) / len(times)
            
            # FEATURE 5: SPECTRAL CONTINUITY (12%) - ZCR jumps
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_events = [zcr[int(t*sr)] for t in events if 0 <= int(t*sr) < len(zcr)]
            spec_cont = np.std(zcr_events) if zcr_events else 1.0
            
            # FEATURE 6: BREATH SIMILARITY (18%) - MFCC distance
            mfccs = []
            for t in events:
                start = max(0, int((t-0.5)*sr))
                end = min(len(y), int((t+0.5)*sr))
                if end > start + sr//10:  # Min length
                    mfcc = librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=3)
                    mfccs.append(np.mean(mfcc, axis=1))
            sim_score = 0.0
            if len(mfccs) > 1:
                from scipy.spatial
