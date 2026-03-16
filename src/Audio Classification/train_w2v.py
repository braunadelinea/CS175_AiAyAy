# train_w2v.py
# This script trains a Random Forest model on Wav2Vec2 features of audio files.
# Author: Adeline Braun

import argparse
import pathlib
import random
import joblib
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=1000)
args = parser.parse_args()

DATA_DIR = pathlib.Path(__file__).parent / "Data"
TARGET_SR = 16_000

W2V_DEVICE = torch.device("cpu")


W2V_ID = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(W2V_ID)
w2v_model = Wav2Vec2Model.from_pretrained(W2V_ID).to(W2V_DEVICE)
w2v_model.eval()


# Extract embeddings from wav2vec2 for each audio file
def extract_features(audio_array, orig_sr):
    if orig_sr != TARGET_SR:
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        audio_array = (
            T.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)(waveform)
            .squeeze()
            .numpy()
        )
    inputs = processor(
        audio_array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(W2V_DEVICE)
    with torch.no_grad():
        hidden = w2v_model(input_values).last_hidden_state
    return hidden.mean(dim=1).squeeze().numpy()


def load_audio(path):
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


lang_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

# Load data
X, y = [], []
for lang_dir in lang_dirs:
    mp3_files = list(lang_dir.glob("*.mp3"))
    selected = random.sample(mp3_files, min(args.samples, len(mp3_files)))
    print(f"grabbing {len(selected)} audio files for {lang_dir.name}...", flush=True)
    for path in selected:
        try:
            audio, sr = load_audio(path)
            feats = extract_features(audio, sr)
            X.append(feats)
            y.append(lang_dir.name)
        except sf.LibsndfileError as e:
            print(f"Error loading {path}: {e}")
            continue

# Train model
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(f"\nFitting the RandomForest...")
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=clf.classes_))

# Print results to file and save model
out = pathlib.Path(__file__).parent / "results_w2v_rf.txt"
out.write_text(
    f"Wav2Vec2 features + Random Forest | {args.samples} samples/lang\n\n"
    + classification_report(y_test, y_pred, target_names=clf.classes_)
)
save_dir = pathlib.Path(__file__).parent
joblib.dump(clf, save_dir / "model_w2v.pkl")

