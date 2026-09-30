import numpy
import pandas
import sklearn
import librosa
import soundfile
import cv2
import PIL
import torch
import torchvision
import transformers
import fastapi

print("NumPy       : OK")
print("Pandas      : OK")
print("Scikit-learn: OK")
print("Librosa     : OK")
print("SoundFile   : OK")
print("OpenCV      : OK")
print("Pillow      : OK")
print("PyTorch     : OK")
print("Torchvision : OK")
print("Transformers: OK")
print("FastAPI     : OK")

print("\nPyTorch version:", torch.__version__)
print("MPS available :", torch.backends.mps.is_available())