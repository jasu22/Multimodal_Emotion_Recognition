import cv2
import mediapipe as mp

VIDEO_PATH = "../dataset/RAVDESS_Video/Actor_01/01-01-03-01-01-01-01.mp4"

print("Loading video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

print("Video opened successfully!")

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    frame_count = 0
    face_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)

        if results.detections:
            face_count += 1

            if face_count <= 5:
                print("Face detected in frame:", frame_count)

cap.release()

print()
print("Face detection completed!")
print("Total frames:", frame_count)
print("Frames containing face:", face_count)