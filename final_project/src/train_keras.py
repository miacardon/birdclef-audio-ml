import json
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Load metadata
df = pd.read_csv("train.csv")

# Limit samples for safety
df = df.sample(n=200, random_state=42)

X_features = []
y_labels = []

for _, row in df.iterrows():
    path = f"train_audio/{row['filename']}"
    y, sr = librosa.load(path, sr=32000, duration=5.0)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel)

    # Feature aggregation
    features = np.concatenate([
        mel_db.mean(axis=1),
        mel_db.std(axis=1)
    ])

    X_features.append(features)
    y_labels.append(row["primary_label"])

X = np.array(X_features)
y = pd.factorize(pd.Series(y_labels))[0]

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Build Keras model
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(np.unique(y)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16,
    verbose=1
)

# Save results
keras_results = {
    "keras_train_accuracy": history.history["accuracy"],
    "keras_val_accuracy": history.history["val_accuracy"]
}

with open("keras_results.json", "w") as f:
    json.dump(keras_results, f, indent=4)

print("Keras results saved to keras_results.json")
