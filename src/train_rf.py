import json
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load metadata
df = pd.read_csv("train.csv")

# Limit samples for safety
df = df.sample(n=300, random_state=42)

X_features = []
y_labels = []

for _, row in df.iterrows():
    path = f"train_audio/{row['filename']}"
    y, sr = librosa.load(path, sr=32000, duration=5.0)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel)

    # Aggregate features
    features = np.concatenate([
        mel_db.mean(axis=1),
        mel_db.std(axis=1)
    ])

    X_features.append(features)
    y_labels.append(row["primary_label"])

X = np.array(X_features)
y = pd.factorize(pd.Series(y_labels))[0]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

# Feature importance
importances = rf.feature_importances_
top_features = np.argsort(importances)[-10:][::-1]

rf_results = {
    "rf_accuracy": accuracy,
    "top_feature_indices": top_features.tolist(),
    "top_feature_importances": importances[top_features].tolist()
}

# Save results
with open("rf_results.json", "w") as f:
    json.dump(rf_results, f, indent=4)

print("Random Forest accuracy:", accuracy)
print("Top feature indices:", top_features)
print("Random Forest results saved to rf_results.json")