import numpy as np
import librosa
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier

# Load training metadata
df = pd.read_csv("/home/ubuntu/birdclef_finalproject/data/train.csv")

# Factorize labels (same mapping as training)
y_encoded, label_names = pd.factorize(df["primary_label"])

# Recreate the same feature extraction
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=32000, duration=5.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    features = np.concatenate([
        mel_db.mean(axis=1),
        mel_db.std(axis=1)
    ])
    return features

# Load training features
X_train = []
for fname in df["filename"][:300]:
    X_train.append(extract_features(f"/home/ubuntu/birdclef_finalproject/data/train_audio/{fname}"))

X_train = np.array(X_train)
y_train = y_encoded[:300]

#Train RF (lightweight)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict my audio
audio_features = extract_features("my_bird.wav").reshape(1, -1)
pred_index = rf.predict(audio_features)[0]
pred_species = label_names[pred_index]

# Get predicted species index
pred_index = rf.predict(audio_features)[0]
pred_species = label_names[pred_index]

# Look up a representative URL from train.csv
matches = df[df["primary_label"] == pred_species]

if len(matches) > 0:
    example_url = matches.iloc[0]["url"]
else:
    example_url = "No URL found for this species."

print("Random Forest predicted species:", pred_species)
print("Example recording URL:", example_url)