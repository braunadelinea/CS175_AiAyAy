# train_mfcc.py
# This script trains a Random Forest model on mel-frequency cepstral coefficients (MFCCs) of audio files.
# Author: Adeline Braun

import argparse
import pathlib
import random
import joblib
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=1000)
args = parser.parse_args()

DATA_DIR = pathlib.Path(__file__).parent / "Data"
TARGET_SR = 16_000
N_MFCC = 40


def extract_features(audio_array, orig_sr):
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    if orig_sr != TARGET_SR:
        waveform = T.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)(waveform)
    mfcc = T.MFCC(
        sample_rate=TARGET_SR,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
    )(waveform)
    # grabbing the mean and std dev to use as features
    feat = torch.cat([mfcc.mean(dim=-1), mfcc.std(dim=-1)], dim=-1)
    return feat.squeeze().numpy()


def load_audio(path):
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


lang_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])


X, y = [], []
for lang_dir in lang_dirs:
    mp3_files = list(lang_dir.glob("*.mp3"))
    selected = random.sample(mp3_files, min(args.samples, len(mp3_files)))
    print(f"getting {len(selected)} samples for lang: {lang_dir.name}...", flush=True)
    ok = 0
    for path in selected:
        try:
            audio, sr = load_audio(path)
            feats = extract_features(audio, sr)
            X.append(feats)
            y.append(lang_dir.name)
            ok += 1
        except sf.LibsndfileError as e:
            print(f"Error loading {path}: {e}")
            continue


X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(f"\ntraining rf...")
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=clf.classes_))

# Print results to file and save model
out = pathlib.Path(__file__).parent / "results_mfcc_rf.txt"
out.write_text(
    f"MFCC features + Random Forest | {args.samples} samples/lang\n\n"
    + classification_report(y_test, y_pred, target_names=clf.classes_)
)
save_dir = pathlib.Path(__file__).parent
joblib.dump(clf, save_dir / "model_mfcc.pkl")
