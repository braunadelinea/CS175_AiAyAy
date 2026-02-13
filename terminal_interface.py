import joblib

BUNDLE_PATH = "artifacts/language_pipeline.joblib"


def main():
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    meta = bundle.get("meta", {})

    print("Language classifier ready.")
    if meta:
        print(f"(Val acc: {meta.get('val_acc'):.4f} | Test acc: {meta.get('test_acc'):.4f})")
    print("Type text and press Enter. Type /quit to exit.\n")

    while True:
        text = input("> ").strip()
        if not text:
            continue
        if text.lower() in {"/quit", "/q", "quit", "exit"}:
            break

        if len(text) < 10:
            print("Please type a bit more text (10+ chars) for a reliable prediction.\n")
            continue

        X = vectorizer.transform([text])
        probs = model.predict_proba(X)[0]
        best_idx = probs.argmax()
        pred = model.classes_[best_idx]
        conf = probs[best_idx]

        print(f"Predicted: {pred}  (confidence: {conf:.3f})\n")


if __name__ == "__main__":
    main()
