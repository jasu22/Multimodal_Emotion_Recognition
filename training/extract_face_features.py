import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import mediapipe as mp
import numpy as np

VIDEO_PATH = "../dataset/RAVDESS_Video/Actor_01/01-01-03-01-01-01-01.mp4"

print("Loading ResNet18...")

resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Remove final classification layer
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

print("Loading video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

print("Video opened successfully!")

mp_face_detection = mp.solutions.face_detection

face_features = []

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_detection.process(rgb_frame)

        if not results.detections:
            continue

        detection = results.detections[0]

        bbox = detection.location_data.relative_bounding_box

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

            feature = resnet(input_tensor)

        feature = feature.squeeze().numpy()

        face_features.append(feature)

        if len(face_features) <= 5:

            print(
                "Feature extracted from frame:",
                frame_count
            )

            print(
                "Feature shape:",
                feature.shape
            )

cap.release()

if len(face_features) == 0:

    print("No face features extracted!")

else:

    face_features = np.array(face_features)

    print()
    print("Face feature extraction completed!")

    print(
        "Number of face features:",
        len(face_features)
    )

    print(
        "Feature matrix shape:",
        face_features.shape
    )

    aggregated_feature = np.mean(
        face_features,
        axis=0
    )

    print(
        "Aggregated feature shape:",
        aggregated_feature.shape
    )