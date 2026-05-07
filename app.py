def get_hardened_forensic_score(y, sr, events, duration):
    """Hardened for ElevenLabs v3 / Roger AI"""
    if len(events) < 2: return 0.95 

    # 1. Purity Check (NEW): AI breaths are often "too clean" spectrally
    purity_scores = []
    for t in events:
        start, end = int(max(0, t-0.15)*sr), int(min(len(y), t+0.35)*sr)
        segment = y[start:end]
        if len(segment) > 512:
            # AI has lower 'Spectral Flatness' in breath artifacts than human noise
            flatness = np.mean(librosa.feature.spectral_flatness(y=segment))
            purity_scores.append(flatness)
    
    # AI often scores lower in 'chaos' (flatness) than natural breath noise
    p_spectral = np.clip(1.0 - (np.mean(purity_scores) * 20), 0, 1) if purity_scores else 0.8

    # 2. Tightened Timing (28%): The 'Grid' check
    ibis = np.diff(events)
    ibi_cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
    p1_timing = np.clip(1.0 - (ibi_cv / 0.35), 0, 1)

    # 3. ElevenLabs Similarity (18%): Fingerprinting texture
    mfccs = [np.mean(librosa.feature.mfcc(y=y[int(t*sr):int((t+0.4)*sr)], sr=sr, n_mfcc=20), axis=1) for t in events]
    p6_sim = 0.0
    if len(mfccs) > 1:
        dists = [distance.euclidean(mfccs[i], mfccs[j]) for i in range(len(mfccs)) for j in range(i+1, len(mfccs))]
        p6_sim = np.clip(1.0 - (np.mean(dists) / 200), 0, 1)

    # Updated weights to prioritize spectral artifacts and timing
    final_score = (p1_timing * 0.30) + (p_spectral * 0.25) + (p6_sim * 0.20) + (0.25 * 0.5) # Basic balance
    return round(np.clip(final_score, 0, 1), 2)
