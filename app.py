def forensic_analysis(y, sr, filename="Audio"):
    if y is None or len(y) < sr:
        # Pass filename to ensure the dict has the 'File' key
        return default_human_result(filename) if (y is not None and len(y) < 5*sr) else default_ai_result(filename), []
    
    # ... (rest of your analysis logic) ...

    # Ensure these keys match exactly what your plotting code calls
    return {
        "File": filename, 
        "AI Probability": f"{ai_prob:.0%}", 
        "Status": status,  # <--- CRITICAL KEY
        "Breaths Detected": len(events), 
        "Timing_Var": f"{ibi_cv:.3f}", 
        "Amp_Var": f"{amp_cv:.3f}"
    }, events

def default_human_result(name):
    return {
        "File": name, 
        "AI Probability": "15%", 
        "Status": "HUMAN", 
        "Breaths Detected": 0
    }, []

def default_ai_result(name):
    return {
        "File": name, 
        "AI Probability": "85%", 
        "Status": "AI (No Breaths)", 
        "Breaths Detected": 0
    }, []
