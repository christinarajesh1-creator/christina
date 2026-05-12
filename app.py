def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.1, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        is_ai = duration > 6
        return {"File": name, "Status": "AI" if is_ai else "HUMAN", "AI Prob": "95%" if is_ai else "15%", "IBI Reg": 0, "Amp Var": 0, "Presence": "0%", "Sim Val": 0}, []

    # PARAMETERS
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    presence = (len(breaths) * 0.45) / duration

    # Sim Val (Fingerprinting Inhale Textures)
    textures = []
    for b in breaths[:6]:
        start, end = int(b * sr), int((b + 0.22) * sr)
        seg = y_norm[start:end]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    
    sim_val = np.mean(np.std(textures, axis=0)) * 10 if len(textures) > 1 else 50.0

    # --- THE BALANCED DECISION ENGINE ---
    # BIOLOGICAL SHIELD: If a voice is sufficiently messy, it's human.
    # Humans have high entropy; we lowered the threshold to 32.0 to protect humans.
    is_biological = (sim_val > 32.0) and (ibi_cv > 0.28 or amp_cv > 0.28)

    if is_biological:
        final_prob = 12
        status = "HUMAN"
    else:
        # If it doesn't meet the "Human Messiness" criteria, it's AI.
        # AI usually sits in the 10-25 Sim Val range.
        if sim_val < 19.0: 
            final_prob = 99
        elif sim_val < 28.0: 
            final_prob = 85
        else: 
            final_prob = 65
        status = "AI"

    return {
        "File": name, "Status": status, "AI Prob": f"{final_prob}%",
        "IBI Reg": round(ibi_cv, 3), "Amp Var": round(amp_cv, 3), 
        "Presence": f"{presence:.1%}", "Sim Val": round(sim_val, 3)
    }, breaths
tus']), use_container_width=True)
