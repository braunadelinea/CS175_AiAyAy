# train_mel_nn.py
# This script trains a CNN model on mel spectrograms of audio files.
# Author: Adeline Braun

import argparse, pathlib, random
import soundfile as sf
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
args = parser.parse_args()

DATA_DIR = pathlib.Path(__file__).parent / "Data"
TARGET_SR = 16_000
FIXED_LEN = TARGET_SR * 5


def conv_block(ci, co, stride=1):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(),
    )


# Define CNN model
class MelSpectrogramCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=512, hop_length=160, n_mels=80
        )
        self.log = T.AmplitudeToDB()
        self.aug_f, self.aug_t = T.FrequencyMasking(12), T.TimeMasking(40)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            conv_block(32, 64, stride=2),
            nn.Dropout2d(0.1),
            conv_block(64, 128, stride=2),
            nn.Dropout2d(0.15),
            conv_block(128, 256, stride=2),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.log(self.mel_spec(x))
        if self.training:
            for _ in range(2):
                x = self.aug_t(self.aug_f(x))
        return self.net(x)


lang_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
label_names = [d.name for d in lang_dirs]

# Load data
X, y = [], []
for i, lang_dir in enumerate(lang_dirs):
    files = list(lang_dir.glob("*.mp3"))
    # Get random sample of audio files from each language
    chosen = random.sample(files, min(args.samples, len(files)))
    print(f"fetching {len(chosen)} files for {lang_dir.name}...", flush=True)
    for f in chosen:
        try:
            data, sr = sf.read(str(f))
            if data.ndim > 1:
                data = data.mean(axis=1)
            w = torch.tensor(data, dtype=torch.float32)
            if sr != TARGET_SR:
                w = T.Resample(sr, TARGET_SR)(w.unsqueeze(0)).squeeze(0)
            if len(w) < FIXED_LEN:
                w = F.pad(w, (0, FIXED_LEN - len(w)))
            else:
                w = w[:FIXED_LEN]
            w = (w - w.mean()) / (w.std() + 1e-9)
            X.append(w)
            y.append(i)
        except sf.LibsndfileError as e:
            print(f"Error loading {f}: {e}")
            continue

X = torch.stack(X).unsqueeze(1)
y = torch.tensor(y)
Xi, Xt, yi, yt = train_test_split(X, y, test_size=0.2, stratify=y)
train_loader = DataLoader(TensorDataset(Xi, yi), batch_size=args.batch, shuffle=True)

model = MelSpectrogramCNN(len(label_names))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Train model
best_acc, best_state = 0.0, None
print(f"\nstarting training loop for {args.epochs} epochs...\n")
for ep in range(1, args.epochs + 1):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        loss = loss_fn(model(batch_X), batch_y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total_loss += loss.item()
    sched.step()

    model.eval()
    with torch.no_grad():
        acc = (model(Xt).argmax(1) == yt).float().mean().item()
    print(
        f"ep {ep}/{args.epochs} | loss: {total_loss/max(1, len(train_loader)):.3f} | acc: {acc:.3f}"
    )
    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)

# Model evaluation
model.eval()
with torch.no_grad():
    preds = model(Xt).argmax(1).numpy()
report = classification_report(yt.numpy(), preds, target_names=label_names)
print(f"\nbest val acc: {best_acc:.3f}\n\n" + report)

# Print results to file and save model
out = pathlib.Path(__file__).parent
out.joinpath("results_mel_nn.txt").write_text(
    f"2-D CNN on mel spectrogram | {args.samples} samples/lang | {args.epochs} epochs | best_acc={best_acc:.4f}\n\n"
    + report
)
torch.save(model.state_dict(), out / "model_mel_nn.pt")
