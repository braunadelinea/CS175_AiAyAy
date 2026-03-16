# train_raw_nn.py
# This script trains a CNN directly on raw audio waveforms.
# Author: Adeline Braun

import argparse, pathlib, random
import numpy as np
import soundfile as sf
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

DATA_DIR = pathlib.Path(__file__).parent / "Data"
TARGET_SR = 16_000
FIXED_LEN = TARGET_SR * 5


# Define CNN model
class RawAudioCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 80, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


lang_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
label_names = [d.name for d in lang_dirs]

# Load data
X, y = [], []
for i, lang_dir in enumerate(lang_dirs):
    files = list(lang_dir.glob("*.mp3"))
    chosen = random.sample(files, min(args.samples, len(files)))
    print(f"loading {len(chosen)} clips for {lang_dir.name}...", flush=True)
    for f in chosen:
        try:
            data, sr = sf.read(str(f))
            if data.ndim > 1:
                data = data.mean(axis=1)
            w = torch.tensor(data, dtype=torch.float32)
            if sr != TARGET_SR:
                w = T.Resample(sr, TARGET_SR)(w.unsqueeze(0)).squeeze(0)
            # Pad or truncate to fixed length
            if len(w) < FIXED_LEN:
                w = torch.nn.functional.pad(w, (0, FIXED_LEN - len(w)))
            else:
                w = w[:FIXED_LEN]
            # Append 1D waveform and label
            X.append(w)
            y.append(i)
        except sf.LibsndfileError as e:
            print(f"Error loading {f}: {e}")
            continue

X = torch.stack(X).unsqueeze(1)
y = torch.tensor(y)

Xi, Xt, yi, yt = train_test_split(X, y, test_size=0.2, stratify=y)

model = RawAudioCNN(len(label_names))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Train model
print(f"\nrunning for {args.epochs} epochs over {len(Xi)} samples\n")
train_loader = DataLoader(TensorDataset(Xi, yi), batch_size=32, shuffle=True)
for ep in range(1, args.epochs + 1):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        loss = loss_fn(model(batch_X), batch_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        acc = (model(Xt).argmax(1) == yt).float().mean().item()
    print(
        f"ep {ep}/{args.epochs} | loss: {total_loss/len(train_loader):.3f} | acc: {acc:.3f}"
    )

# Evaluate model
model.eval()
with torch.no_grad():
    preds = model(Xt).argmax(1).numpy()
report = classification_report(yt.numpy(), preds, target_names=label_names)
print("\n" + report)

# Print results to file and save model
out = pathlib.Path(__file__).parent / "results_raw_nn.txt"
out.write_text(
    f"1-D CNN on raw waveform | {args.samples} samples/lang | {args.epochs} epochs\n\n"
    + report
)
torch.save(model.state_dict(), pathlib.Path(__file__).parent / "model_raw_nn.pt")
