import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd

st.session_state.setdefault('history', [])

class PneumaEngine:
    @staticmethod
    def analyze(audio_bytes, label="Sample"):
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=22050)
        
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        threshold = np.percentile(rms, 5)
        breath_frames = np.where((rms < threshold) & (zcr < np.mean(zcr)*0.5))
        
        events = []
        if len(breath_frames[0]) > 0:
            diffs = np.diff(breath_frames[0])
            splits = np.where(diffs > 2)[0]
            clusters = np.split(breath_frames[0], splits + 1)
            events
