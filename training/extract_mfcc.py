import librosa
import numpy as np

audio_path = "../dataset/RAVDESS/Actor_01/03-02-04-01-01-01-01.wav"

audio, sample_rate = librosa.load(audio_path, sr=None)

# 13 MFCC features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

# 13 Delta features
delta = librosa.feature.delta(mfcc)

# 13 Delta-Delta features
delta2 = librosa.feature.delta(mfcc, order=2)

# Combine all features
features = np.concatenate(
    [mfcc, delta, delta2],
    axis=0
)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("MFCC shape:", mfcc.shape)
print("Delta shape:", delta.shape)
print("Delta-Delta shape:", delta2.shape)
print("Combined feature shape:", features.shape)