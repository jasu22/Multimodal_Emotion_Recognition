import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import mediapipe as mp
import numpy as np
import pandas as pd


VIDEO_FOLDER = "../dataset/RAVDESS_Video/Actor_01"
OUTPUT_FILE = "../features/face_features.csv"


emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


print("Loading ResNet18...")

resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

resnet = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

resnet.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


print("Loading MediaPipe face detector...")

mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


all_features = []

video_files = sorted([
    f for f in os.listdir(VIDEO_FOLDER)
    if f.endswith(".mp4") and f.startswith("01-")
])


print("Number of videos:", len(video_files))
print()


for video_number, filename in enumerate(video_files, start=1):

    print(
        "Processing",
        video_number,
        "/",
        len(video_files),
        ":",
        filename
    )

    parts = filename.replace(".mp4", "").split("-")

    emotion_code = parts[2]

    if emotion_code not in emotion_map:
        print("Skipping Calm:", filename)
        continue

    emotion = emotion_map[emotion_code]

    video_path = os.path.join(
        VIDEO_FOLDER,
        filename
    )

    cap = cv2.VideoCapture(video_path)

    frame_features = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_detector.process(
            rgb_frame
        )

        if not results.detections:
            continue

        detection = results.detections[0]

        bbox = (
            detection
            .location_data
            .relative_bounding_box
        )

        h, w, _ = frame.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)

        width = int(bbox.width * w)
        height = int(bbox.height * h)

        x = max(0, x)
        y = max(0, y)

        x2 = min(w, x + width)
        y2 = min(h, y + height)

        face = frame[y:y2, x:x2]

        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        input_tensor = transform(face_rgb)

        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():

            feature = resnet(
                input_tensor
            )

        feature = feature.squeeze().numpy()

        frame_features.append(feature)


    cap.release()


    if len(frame_features) == 0:

        print("No face detected!")
        continue


    frame_features = np.array(
        frame_features
    )

    aggregated_feature = np.mean(
        frame_features,
        axis=0
    )


    row = list(aggregated_feature)

    row.append(emotion)
    row.append(filename)

    all_features.append(row)

    print(
        "  Frames:",
        len(frame_features),
        "| Feature shape:",
        aggregated_feature.shape,
        "| Emotion:",
        emotion
    )


face_detector.close()


feature_columns = [
    f"feature_{i}"
    for i in range(512)
]

feature_columns += [
    "emotion",
    "filename"
]


df = pd.DataFrame(
    all_features,
    columns=feature_columns
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("================================")
print("FACE FEATURE EXTRACTION COMPLETE")
print("================================")
print()
print("Total videos processed:", len(df))
print("Feature shape:", df.shape)
print()
print("Saved to:")
print(OUTPUT_FILE)