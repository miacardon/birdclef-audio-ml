import os
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

# format/standardize audio clips = 5 seconds
SAMPLE_RATE = 32000
CLIP_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
N_MELS = 128


# convert string labels from train.csv to integers
class BirdClefDataset(Dataset):
    def __init__(self, csv_path, audio_root):
        self.df = pd.read_csv(csv_path)
        self.audio_root = audio_root

        self.labels = sorted(self.df["primary_label"].unique())
        self.label_to_idx = {
            label: idx for idx, label in enumerate(self.labels)
        }

    # length of dataset
    def __len__(self):
        return len(self.df)

    # loading one audio file
    def _load_audio(self, filepath):
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)

        # enforce exaclty 5 seconds
        if len(y) > NUM_SAMPLES:
            y = y[:NUM_SAMPLES]
        else:
            y = np.pad(y, (0, NUM_SAMPLES - len(y)))

        return y

    # convert waveform to mel spectrogram
    def _waveform_to_mel(self, y):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=1024,
            hop_length=512
        )

        # log scaling (stabalize training and match human perception)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    # put it all together (return (X,y), X shape:(1, 128, T), y shape: scalar integer)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        label = self.label_to_idx[row["primary_label"]]
        audio_path = os.path.join(self.audio_root, row["filename"])

        y = self._load_audio(audio_path)
        mel = self._waveform_to_mel(y)

        # returning tensors in CNN-ready shape
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, label_tensor