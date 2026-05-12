def forensic_analysis(y, sr, name):
    y_norm = librosa.util.normalize(y)
    duration = len(y_norm) / sr
    
    # 1. High-Sensitivity Breath Tracking
    hop = 256
    rms = librosa.feature.rms(y=y_norm, hop_length=hop).flatten()
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Lowered height to catch subtle AI breaths
    peaks, _ = signal.find_peaks(rms_smooth, height=np.median(rms_smooth)*1.08, distance=sr//hop)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    breaths = [t for t in times if 0.4 < t < (duration - 0.4)]
    
    if len(breaths) < 2:
        return {"File": name, "Status": "AI", "AI Prob": "99%", "IBI Reg": 0, "Amp Var": 0, "Presence": "0%", "Sim Val": 0}, []

    # --- PARAMETER EXTRACTION ---
    ibi = np.diff(breaths)
    ibi_cv = np.std(ibi) / np.mean(ibi) if len(ibi) > 0 else 0
    
    amps = [rms_smooth[p] for p in peaks]
    amp_cv = np.std(amps) / np.mean(amps) if len(amps) > 0 else 0
    
    presence = (len(breaths) * 0.45) / duration

    # The "Sim Val" Trap: Measuring Biological Entropy
    textures = []
    for b in breaths[:6]:
        start, end = int(b * sr), int((b + 0.25) * sr)
        seg = y_norm[start:end]
        if len(seg) >= int(0.1*sr):
            textures.append(np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1))
    
    # Scaled Sim Val: AI usually sits between 10-25. Humans usually > 40.
    sim_val = np.mean(np.std(textures, axis=0)) * 10 if len(textures) > 1 else 50.0

    # --- THE "AI UNTIL PROVEN HUMAN" ENGINE ---
    # We start with a high AI probability and look for reasons to call it HUMAN
    ai_prob = 85 

    # HUMAN CRITERIA: A person must be very messy to be called human
    # If the voice has high biological chaos, we drop the AI score significantly
    is_human = (sim_val > 38.0) and (ibi_cv > 0.32 or amp_cv > 0.32)

    if is_human:
        final_prob = 15
        status = "HUMAN"
    else:
        # If it doesn't meet human messiness, it's AI
        # We refine the probability based on how "perfect" it is
        if sim_val < 20.0: ai_prob = 99
        elif sim_val < 30.0: ai_prob = 85
        else: ai_prob = 65
        
        final_prob = ai_prob
        status = "AI"

    return {
        "File": name, 
        "Status": status, 
        "AI Prob": f"{final_prob}%",
        "IBI Reg": round(ibi_cv, 3), 
        "Amp Var": round(amp_cv, 3), 
        "Presence": f"{presence:.1%}", 
        "Sim Val": round(sim_val, 3)
    }, breaths
atus']), use_container_width=True)
