from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
import mediapipe as mp

app = FastAPI(title="Multimodal Emotion Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained Text model
text_model = joblib.load("api/text_model.pkl")
text_scaler = joblib.load("api/text_scaler.pkl")
text_label_encoder = joblib.load("api/text_label_encoder.pkl")


# Load trained Audio model
audio_model = joblib.load("api/audio_model.pkl")
audio_scaler = joblib.load("api/audio_scaler.pkl")
audio_label_encoder = joblib.load("api/audio_label_encoder.pkl")

# Load trained Face model
face_model = joblib.load("api/face_model.pkl")
face_scaler = joblib.load("api/face_scaler.pkl")
face_label_encoder = joblib.load("api/face_label_encoder.pkl")

# Load RoBERTa
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = AutoModel.from_pretrained("roberta-base")
roberta_model.eval()


# Input format
class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Multimodal Emotion Recognition API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/text-model")
def text_model_status():
    return {
        "status": "Text model loaded successfully",
        "classes": text_label_encoder.classes_.tolist()
    }


@app.post("/predict-text")
def predict_text(data: TextInput):

    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = roberta_model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].numpy()

    embedding_scaled = text_scaler.transform(embedding)

    prediction = text_model.predict(embedding_scaled)

    emotion = text_label_encoder.inverse_transform(prediction)[0]

    return {
        "text": data.text,
        "emotion": emotion
    }

@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):

    audio_bytes = await file.read()

    temp_file = "temp_audio.wav"

    with open(temp_file, "wb") as f:
        f.write(audio_bytes)

    y, sr = librosa.load(temp_file, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    delta = librosa.feature.delta(mfcc)

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    mfcc_features = np.concatenate(
        [mfcc, delta, delta_delta],
        axis=0
    )

    features = np.mean(
        mfcc_features,
        axis=1
    )

    features = features.reshape(1, -1)

    features_scaled = audio_scaler.transform(features)

    prediction = audio_model.predict(features_scaled)

    emotion = audio_label_encoder.inverse_transform(
        prediction
    )[0]

    return {
        "filename": file.filename,
        "emotion": emotion
    }

    # Load ResNet18 for face feature extraction
resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

resnet = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

resnet.eval()


# Face image preprocessing
face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# MediaPipe face detector
mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return {"error": "Could not read image"}

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_detector.process(
        rgb_frame
    )

    if not results.detections:
        return {"error": "No face detected"}

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
        return {"error": "Invalid face region"}

    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    input_tensor = face_transform(
        face_rgb
    )

    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():

        feature = resnet(
            input_tensor
        )

    feature = feature.squeeze().numpy()

    features = feature.reshape(1, -1)

    features_scaled = face_scaler.transform(
        features
    )

    prediction = face_model.predict(
        features_scaled
    )

    emotion = face_label_encoder.inverse_transform(
        prediction
    )[0]

    return {
        "filename": file.filename,
        "emotion": emotion
    }

@app.post("/predict-multimodal")
async def predict_multimodal(
    text: str,
    audio: UploadFile = File(...),
    image: UploadFile = File(...)
):

    # =========================
    # TEXT PREDICTION
    # =========================

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = roberta_model(**inputs)

    text_embedding = outputs.last_hidden_state[:, 0, :].numpy()

    text_scaled = text_scaler.transform(
        text_embedding
    )

    text_probabilities = text_model.predict_proba(
        text_scaled
    )[0]


    # =========================
    # AUDIO PREDICTION
    # =========================

    audio_bytes = await audio.read()

    temp_file = "temp_audio.wav"

    with open(temp_file, "wb") as f:
        f.write(audio_bytes)

    y, sr = librosa.load(
        temp_file,
        sr=None
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    delta = librosa.feature.delta(mfcc)

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    mfcc_features = np.concatenate(
        [mfcc, delta, delta_delta],
        axis=0
    )

    audio_features = np.mean(
        mfcc_features,
        axis=1
    )

    audio_features = audio_features.reshape(
        1, -1
    )

    audio_scaled = audio_scaler.transform(
        audio_features
    )

    audio_probabilities = audio_model.predict_proba(
        audio_scaled
    )[0]


    # =========================
    # IMAGE PREDICTION
    # =========================

    image_bytes = await image.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return {"error": "Could not read image"}

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_detector.process(
        rgb_frame
    )

    if not results.detections:
        return {"error": "No face detected"}

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
        return {"error": "Invalid face region"}

    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    input_tensor = face_transform(
        face_rgb
    )

    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():

        feature = resnet(
            input_tensor
        )

    face_features = feature.squeeze().numpy()

    face_features = face_features.reshape(
        1, -1
    )

    face_scaled = face_scaler.transform(
        face_features
    )

    face_probabilities = face_model.predict_proba(
        face_scaled
    )[0]


    # =========================
    # LATE FUSION
    # =========================

    audio_weight = 0.4
    text_weight = 0.3
    face_weight = 0.3
     # =========================
    # LATE FUSION
    # =========================

    classes = [
        "angry",
        "disgust",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised"
    ]

    audio_probabilities_aligned = np.zeros(len(classes))
    text_probabilities_aligned = np.zeros(len(classes))
    face_probabilities_aligned = np.zeros(len(classes))

    for i, emotion in enumerate(audio_label_encoder.classes_):
        audio_probabilities_aligned[
            classes.index(emotion)
        ] = audio_probabilities[i]

    for i, emotion in enumerate(text_label_encoder.classes_):
        text_probabilities_aligned[
            classes.index(emotion)
        ] = text_probabilities[i]

    for i, emotion in enumerate(face_label_encoder.classes_):
        face_probabilities_aligned[
            classes.index(emotion)
        ] = face_probabilities[i]

    audio_weight = 0.4
    text_weight = 0.3
    face_weight = 0.3

    fused_probabilities = (
        audio_weight * audio_probabilities_aligned
        + text_weight * text_probabilities_aligned
        + face_weight * face_probabilities_aligned
    )

    final_index = np.argmax(fused_probabilities)

    final_emotion = classes[final_index]

    return {
        "text": text,
        "audio_filename": audio.filename,
        "image_filename": image.filename,

        "audio_probabilities": {
            classes[i]: round(
                float(audio_probabilities_aligned[i]), 4
            )
            for i in range(len(classes))
        },

        "text_probabilities": {
            classes[i]: round(
                float(text_probabilities_aligned[i]), 4
            )
            for i in range(len(classes))
        },

        "face_probabilities": {
            classes[i]: round(
                float(face_probabilities_aligned[i]), 4
            )
            for i in range(len(classes))
        },

        "fused_probabilities": {
            classes[i]: round(
                float(fused_probabilities[i]), 4
            )
            for i in range(len(classes))
        },

        "audio_emotion": audio_label_encoder.inverse_transform(
            [np.argmax(audio_probabilities)]
        )[0],

        "text_emotion": text_label_encoder.inverse_transform(
            [np.argmax(text_probabilities)]
        )[0],

        "face_emotion": face_label_encoder.inverse_transform(
            [np.argmax(face_probabilities)]
        )[0],

        "final_emotion": final_emotion
    }