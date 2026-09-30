import matplotlib.pyplot as plt

models = [
    "Audio",
    "Text",
    "Face",
    "Early Fusion",
    "Late Fusion"
]

accuracy = [
    51.6,
    18.0,
    100.0,
    81.82,
    72.73
]

print("MODEL ACCURACY COMPARISON")
print("=========================")

for model, acc in zip(models, accuracy):
    print(f"{model}: {acc:.2f}%")

plt.figure(figsize=(10, 6))
plt.bar(models, accuracy)

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Multimodal Emotion Recognition - Model Comparison")
plt.ylim(0, 110)

for i, acc in enumerate(accuracy):
    plt.text(i, acc + 2, f"{acc:.2f}%", ha="center")

plt.tight_layout()
plt.show()