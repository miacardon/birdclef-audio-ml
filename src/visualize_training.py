import json
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display

# =========================
# 1. CNN Training Curves
# =========================
with open("/home/ubuntu/birdclef_finalproject/results/results.json") as f:
    results = json.load(f)

epochs = range(1, len(results["train_loss"]) + 1)

plt.figure(figsize=(10, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, results["train_loss"], label="Train Loss")
plt.plot(epochs, results["val_loss"], label="Val Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, results["val_accuracy"], label="Val Accuracy")
plt.title("CNN Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("cnn_training_curves.png")
plt.close()

# =========================
# 2. Model Comparison
# =========================
cnn = json.load(open("/home/ubuntu/birdclef_finalproject/results/results.json"))
keras = json.load(open("/home/ubuntu/birdclef_finalproject/results/keras_results.json"))
rf = json.load(open("/home/ubuntu/birdclef_finalproject/results/rf_results.json"))

models = ["CNN", "Keras NN", "Random Forest"]
accuracies = [
    cnn["val_accuracy"][-1],
    keras["keras_val_accuracy"][-1],
    rf["rf_accuracy"]
]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.savefig("model_comparison.png")
plt.close()

# =========================
# 3. Random Forest Feature Importance
# =========================
plt.figure(figsize=(8, 4))
plt.bar(
    range(len(rf["top_feature_importances"])),
    rf["top_feature_importances"]
)
plt.xticks(
    range(len(rf["top_feature_indices"])),
    rf["top_feature_indices"],
    rotation=45
)
plt.title("Top Random Forest Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.close()

# =========================
# 4. Audio Spectrogram
# =========================
audio_path = "my_bird.wav"

if os.path.exists(audio_path):
    y, sr = librosa.load(audio_path, sr=32000, duration=5.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram of Audio Imitation")
    plt.tight_layout()
    plt.savefig("audio_spectrogram.png")
    plt.close()
else:
    print("Audio file not found: my_bird.wav")