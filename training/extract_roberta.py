import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

INPUT_FILE = "../features/ravdess_transcripts.csv"
OUTPUT_FILE = "../features/roberta_features.csv"

print("Loading transcript dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total samples:", len(df))

print("Loading RoBERTa tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

print("Loading RoBERTa model...")

model = AutoModel.from_pretrained("roberta-base")

model.eval()

print("RoBERTa model loaded!")

features = []

for i, text in enumerate(df["text"]):

    print(f"Processing {i + 1}/{len(df)}")

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]

    features.append(embedding)

features = np.array(features)

feature_columns = [
    f"roberta_{i}"
    for i in range(768)
]

feature_df = pd.DataFrame(
    features,
    columns=feature_columns
)

feature_df["emotion"] = df["emotion"].values

feature_df["text"] = df["text"].values

feature_df["filename"] = df["filename"].values

feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nRoBERTa feature extraction completed!")

print("Feature shape:", features.shape)

print("Saved to:", OUTPUT_FILE)

print("\nFirst 5 rows:")
print(feature_df.head())