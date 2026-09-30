import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD FEATURE FILES
# ==========================================

audio = pd.read_csv("../features/audio_mfcc_features.csv")
text = pd.read_csv("../features/roberta_features.csv")
face = pd.read_csv("../features/face_features.csv")


# ==========================================
# 2. CREATE COMMON KEY
# ==========================================

audio["key"] = (
    audio["filename"]
    .str.replace(".wav", "", regex=False)
    .str.split("-", n=1)
    .str[1]
)

text["key"] = (
    text["filename"]
    .str.replace(".wav", "", regex=False)
    .str.split("-", n=1)
    .str[1]
)

face["key"] = (
    face["filename"]
    .str.replace(".mp4", "", regex=False)
    .str.split("-", n=1)
    .str[1]
)


# ==========================================
# 3. SELECT FEATURES
# ==========================================

audio_features = [c for c in audio.columns if c.startswith("mfcc_")]
text_features = [c for c in text.columns if c.startswith("roberta_")]
face_features = [c for c in face.columns if c.startswith("feature_")]


audio_data = audio[audio_features + ["emotion", "key"]]
text_data = text[text_features + ["emotion", "key"]]
face_data = face[face_features + ["emotion", "key"]]


# ==========================================
# 4. MATCH ALL THREE MODALITIES
# ==========================================

data = face_data.merge(
    audio_data,
    on="key",
    suffixes=("_face", "_audio")
)

data = data.merge(
    text_data,
    on="key"
)


# ==========================================
# 5. CHECK DATA
# ==========================================

print("Matched samples:", len(data))

print("\nEmotion labels:")
print(data["emotion"].value_counts())


# ==========================================
# 6. GET FEATURES
# ==========================================

X_audio = data[audio_features].values
X_text = data[text_features].values
X_face = data[face_features].values

y = data["emotion"].values


# ==========================================
# 7. ENCODE EMOTIONS
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# ==========================================
# 8. TRAIN-TEST SPLIT
# ==========================================

indices = np.arange(len(data))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ==========================================
# 9. SCALE EACH MODALITY
# ==========================================

audio_scaler = StandardScaler()
text_scaler = StandardScaler()
face_scaler = StandardScaler()


X_audio_train = audio_scaler.fit_transform(X_audio[train_idx])
X_audio_test = audio_scaler.transform(X_audio[test_idx])

X_text_train = text_scaler.fit_transform(X_text[train_idx])
X_text_test = text_scaler.transform(X_text[test_idx])

X_face_train = face_scaler.fit_transform(X_face[train_idx])
X_face_test = face_scaler.transform(X_face[test_idx])


# ==========================================
# 10. CREATE THREE MODELS
# ==========================================

audio_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)

text_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)

face_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)


# ==========================================
# 11. TRAIN THREE MODELS
# ==========================================

print("\nTraining Audio Model...")
audio_model.fit(
    X_audio_train,
    y_encoded[train_idx]
)

print("Training Text Model...")
text_model.fit(
    X_text_train,
    y_encoded[train_idx]
)

print("Training Face Model...")
face_model.fit(
    X_face_train,
    y_encoded[train_idx]
)


# ==========================================
# 12. GET PREDICTION PROBABILITIES
# ==========================================

audio_probs = audio_model.predict_proba(X_audio_test)

text_probs = text_model.predict_proba(X_text_test)

face_probs = face_model.predict_proba(X_face_test)


# ==========================================
# 13. LATE FUSION
# ==========================================

audio_weight = 0.4
text_weight = 0.3
face_weight = 0.3


final_probs = (
    audio_weight * audio_probs
    + text_weight * text_probs
    + face_weight * face_probs
)


# ==========================================
# 14. FINAL PREDICTION
# ==========================================

final_prediction = np.argmax(
    final_probs,
    axis=1
)


# ==========================================
# 15. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_encoded[test_idx],
    final_prediction
)

print("\n================================")
print("LATE FUSION RESULTS")
print("================================")

print("\nAccuracy:", accuracy)


# ==========================================
# 16. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_encoded[test_idx],
        final_prediction,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)