import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "../features/roberta_features.csv"


print("Loading RoBERTa features...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# Separate features and target
feature_columns = [c for c in df.columns if c.startswith("roberta_")]

X = df[feature_columns]
y = df["emotion"]


# Convert emotion names to numbers
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nEmotion classes:")
print(label_encoder.classes_)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create MLP model
print("\nTraining Text MLP model...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)


model.fit(X_train, y_train)


print("Training completed!")


# Predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nText Model Accuracy:", round(accuracy, 3))


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# Save model, scaler, and label encoder

joblib.dump(model, "../api/text_model.pkl")
joblib.dump(scaler, "../api/text_scaler.pkl")
joblib.dump(label_encoder, "../api/text_label_encoder.pkl")

print("\nText model files saved successfully!")