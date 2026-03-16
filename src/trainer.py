from datamanager import load_and_combine, basic_clean_and_filter, stratified_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import time
from pathlib import Path


def train(
    detection_path="Data/Detection.csv",
    identification_path="Data/Identification.csv",
    out_dir="artifacts",
    seed=0,
    min_chars=30,
    max_chars=600,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print("Loading + cleaning data...")
    df = load_and_combine(detection_path, identification_path)
    df = basic_clean_and_filter(df, min_chars=min_chars, max_chars=max_chars)
    train_df, val_df, test_df = stratified_split(df, seed=seed)

    print(f"Train/Val/Test: {len(train_df)} / {len(val_df)} / {len(test_df)}")

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    print("X_train shape:", X_train.shape)

    print("Training model...")
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
    model.fit(X_train, train_df["label"])

    # quick sanity metrics
    val_acc = accuracy_score(val_df["label"], model.predict(X_val))
    test_acc = accuracy_score(test_df["label"], model.predict(X_test))
    print(f"Val acc:  {val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "meta": {
            "seed": seed,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "ngram_range": (3, 5),
            "min_df": 2,
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "trained_seconds": float(time.time() - start),
        }
    }

    bundle_path = out_dir / "language_pipeline.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"Saved: {bundle_path}")

    return bundle


if __name__ == "__main__":
    train()
