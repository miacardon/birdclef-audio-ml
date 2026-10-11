import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from src.dataset import BirdClefDataset
from src.model import BirdClefCNN

torch.set_num_threads(1)

# configuration
BATCH_SIZE = 2
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2

# device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load dataset
dataset = BirdClefDataset(
    csv_path="train.csv",
    audio_root="train_audio"
)

# TEMPORARY: limit dataset size due to EC2 memory constraints
MAX_SAMPLES = 100
if len(dataset) > MAX_SAMPLES:
    dataset = torch.utils.data.Subset(dataset, range(MAX_SAMPLES))

num_classes = len(dataset.dataset.labels) if isinstance(dataset, torch.utils.data.Subset) else len(dataset.labels)
print("Number of classes:", num_classes)

# train / validation split
val_size = int(len(dataset) * VALIDATION_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# initialize model, loss, optimizer
model = BirdClefCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training loop
results = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": []
}

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    results["train_loss"].append(avg_train_loss)

    # validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    results["val_loss"].append(avg_val_loss)
    results["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

# save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to results.json")
