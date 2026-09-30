import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load features
df = pd.read_csv("../features/audio_mfcc_features.csv")

# Separate features and labels
X = df.drop(["emotion", "filename"], axis=1)
y = df["emotion"]

# Convert emotion names into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create MLP model
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)

# Train
print("Training model...")

model.fit(X_train, y_train)

print("Training completed!")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Audio Model Accuracy:", accuracy)

# Detailed results
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

print("\nEmotion classes:")
print(encoder.classes_)

import joblib

joblib.dump(model, "../api/audio_model.pkl")
joblib.dump(scaler, "../api/audio_scaler.pkl")
joblib.dump(encoder, "../api/audio_label_encoder.pkl")

print("Audio model files saved successfully!")