import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


print("Loading face features...")

df = pd.read_csv("../features/face_features.csv")

print("Dataset shape:", df.shape)


X = df.iloc[:, :512].values
y = df["emotion"].values


label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Training Face Model...")


model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print("==============================")
print("FACE MODEL RESULTS")
print("==============================")

print("Accuracy:", accuracy)

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)
import joblib

joblib.dump(model, "../api/face_model.pkl")
joblib.dump(scaler, "../api/face_scaler.pkl")
joblib.dump(label_encoder, "../api/face_label_encoder.pkl")

print("Face model files saved successfully!")