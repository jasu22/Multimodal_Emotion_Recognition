import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "../dataset/RAVDESS"
OUTPUT_FILE = "../features/audio_mfcc_features.csv"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

data = []

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for filename in os.listdir(actor_path):

        if not filename.endswith(".wav"):
            continue

        parts = filename.split("-")
        emotion_code = parts[2]

        if emotion_code not in emotion_map:
            continue

        emotion = emotion_map[emotion_code]

        file_path = os.path.join(actor_path, filename)

        try:
            audio, sr = librosa.load(file_path, sr=22050)

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13
            )

            delta = librosa.feature.delta(mfcc)
            delta_delta = librosa.feature.delta(delta)

            combined = np.concatenate(
                [mfcc, delta, delta_delta],
                axis=0
            )

            feature_vector = np.mean(combined, axis=1)

            row = list(feature_vector)
            row.append(emotion)
            row.append(filename)

            data.append(row)

        except Exception as e:
            print("Error:", filename, e)

columns = [f"mfcc_{i}" for i in range(39)]
columns.append("emotion")
columns.append("filename")

df = pd.DataFrame(data, columns=columns)

df.to_csv(OUTPUT_FILE, index=False)

print("MFCC extraction completed!")
print("Total samples:", len(df))
print("\nEmotion distribution:")
print(df["emotion"].value_counts())