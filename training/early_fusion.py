import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


print("Loading feature files...")

audio = pd.read_csv("../features/audio_mfcc_features.csv")
text = pd.read_csv("../features/roberta_features.csv")
face = pd.read_csv("../features/face_features.csv")

print("Audio shape:", audio.shape)
print("Text shape:", text.shape)
print("Face shape:", face.shape)


# Create matching keys
audio_key = (
    audio["filename"]
    .str.replace(".wav", "", regex=False)
    .str.split("-", n=1)
    .str[1]
)

text_key = (
    text["filename"]
    .str.replace(".wav", "", regex=False)
    .str.split("-", n=1)
    .str[1]
)

face_key = (
    face["filename"]
    .str.replace(".mp4", "", regex=False)
    .str.split("-", n=1)
    .str[1]
)


# Feature columns
audio_columns = [f"mfcc_{i}" for i in range(39)]
text_columns = [f"roberta_{i}" for i in range(768)]
face_columns = [f"feature_{i}" for i in range(512)]


# Create smaller DataFrames
audio_data = audio[audio_columns + ["emotion"]].copy()
audio_data["key"] = audio_key

text_data = text[text_columns + ["emotion"]].copy()
text_data["key"] = text_key

face_data = face[face_columns + ["emotion"]].copy()
face_data["key"] = face_key


print("\nMatching audio + text + face...")


# Merge
merged = face_data.merge(
    audio_data,
    on="key",
    suffixes=("_face", "_audio")
)

merged = merged.merge(
    text_data,
    on="key"
)


print("Matched samples:", len(merged))


# Check labels
print("\nChecking emotion labels...")

audio_mismatch = (
    merged["emotion_face"] != merged["emotion_audio"]
).sum()

text_mismatch = (
    merged["emotion_face"] != merged["emotion"]
).sum()

print("Face vs Audio mismatches:", audio_mismatch)
print("Face vs Text mismatches:", text_mismatch)


# Target
y = merged["emotion_face"].values


# Features
audio_features = merged[audio_columns].values
text_features = merged[text_columns].values
face_features = merged[face_columns].values


print("\nAudio features:", audio_features.shape)
print("Text features:", text_features.shape)
print("Face features:", face_features.shape)


# Early Fusion
X = np.concatenate(
    [
        audio_features,
        text_features,
        face_features
    ],
    axis=1
)

print("\nEarly Fusion feature shape:", X.shape)
print("Labels:", y.shape)


# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Standardization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)


print("\nTraining Early Fusion Model...")

model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Results
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("CORRECT EARLY FUSION RESULTS")
print("==============================")

print("Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)