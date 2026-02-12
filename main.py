from datamanager import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import time


def main():
    start_time = time.time()

    # 1) Load + clean + split
    print("Loading and combining datasets...")
    df = load_and_combine("Data/Detection.csv", "Data/Identification.csv")

    print("Cleaning and filtering...")
    df = basic_clean_and_filter(df, min_chars=30, max_chars=600)

    print("Creating stratified split...")
    train, val, test = stratified_split(df, seed=0)

    print(f"Train size: {len(train)} | Val size: {len(val)} | Test size: {len(test)}")

    # 2) Vectorize: character n-grams
    print("\nVectorizing text (this may take a while)...")
    vec_start = time.time()

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2
    )

    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(val["text"])
    X_test = vectorizer.transform(test["text"])

    vec_end = time.time()
    print("Vectorization complete.")
    print("X_train shape:", X_train.shape)
    print(f"Vectorization time: {vec_end - vec_start:.2f} seconds")

    # 3) Train classifier
    print("\nTraining Logistic Regression (this may take a while)...")
    train_start = time.time()

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
    model.fit(X_train, train["label"])

    train_end = time.time()
    print("Training complete.")
    print(f"Training time: {train_end - train_start:.2f} seconds")

    # 4) Evaluate
    print("\nEvaluating model...")
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("Validation accuracy:", accuracy_score(val["label"], val_pred))
    print("Test accuracy:", accuracy_score(test["label"], test_pred))

    # Detailed report (precision/recall/F1 per language)
    print("\n=== Classification report (TEST) ===")
    print(classification_report(test["label"], test_pred, digits=4))

    # Confusion matrix (top mistakes)
    labels = sorted(df["label"].unique())
    cm = confusion_matrix(test["label"], test_pred, labels=labels)

    # show top off-diagonal confusions
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat = cm_off.flatten()
    top_idx = flat.argsort()[::-1][:15]

    print("\n=== Top confusions (TEST) ===")
    for idx in top_idx:
        if flat[idx] == 0:
            break
        i, j = divmod(idx, len(labels))
        print(f"{labels[i]} -> {labels[j]} : {flat[idx]}")

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
