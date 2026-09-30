import os
import pandas as pd
import whisper

DATASET_PATH = "../dataset/RAVDESS"
OUTPUT_FILE = "../features/ravdess_transcripts.csv"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

print("Loading Whisper model...")

model = whisper.load_model("tiny")

print("Whisper model loaded!")

data = []

for actor in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for filename in sorted(os.listdir(actor_path)):

        if not filename.endswith(".wav"):
            continue

        parts = filename.split("-")
        emotion_code = parts[2]

        # Skip Calm
        if emotion_code not in emotion_map:
            continue

        emotion = emotion_map[emotion_code]

        file_path = os.path.join(actor_path, filename)

        print("Processing:", filename)

        try:

            result = model.transcribe(
                file_path,
                fp16=False
            )

            transcript = result["text"].strip()

            data.append([
                filename,
                emotion,
                transcript
            ])

        except Exception as e:

            print("Error:", filename, e)


df = pd.DataFrame(
    data,
    columns=["filename", "emotion", "text"]
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTranscript generation completed!")
print("Total samples:", len(df))

print("\nFirst 10 samples:")
print(df.head(10))