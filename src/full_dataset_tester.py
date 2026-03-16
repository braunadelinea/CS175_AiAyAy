# full_dataset_tester.py: Evaluates the trained language classification model on an external HuggingFace language
# dataset, reporting overall accuracy, per-language performance, and how the model classifies languages it was not
# trained on.

# Author: Wisam Zeidan

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


BUNDLE_PATH = "../artifacts/language_pipeline.joblib"


def get_new_dataset():
    splits = {
        "train": "train.csv",
        "validation": "valid.csv",
        "test": "test.csv"
    }

    dfs = []

    for split in splits.values():
        df = pd.read_csv("hf://datasets/papluca/language-identification/" + split)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    return full_df


def fix_labels(full_df):
    code_to_label = {
        "ar": "arabic",
        "de": "german",
        "el": "greek",
        "en": "english",
        "es": "spanish",
        "fr": "french",
        "hi": "hindi",
        "it": "italian",
        "ja": "japanese",
        "nl": "dutch",
        "pt": "portuguese",
        "ru": "russian",
        "th": "thai",
        "tr": "turkish",
        "ur": "urdu",
        "zh": "chinese"
    }

    full_df["mapped_label"] = full_df["labels"].map(code_to_label)
    full_df = full_df.dropna(subset=["mapped_label"])
    return full_df


def analyze_unseen_languages(full_df, model, vectorizer):
    unseen_codes = {
        "bg": "bulgarian",
        "pl": "polish",
        "sw": "swahili",
        "vi": "vietnamese"
    }

    unseen_df = full_df[full_df["labels"].isin(unseen_codes.keys())].copy()
    unseen_df["true_unseen_label"] = unseen_df["labels"].map(unseen_codes)

    X = vectorizer.transform(unseen_df["text"])
    unseen_df["prediction"] = model.predict(X)

    print("\nPredictions for unseen languages:\n")

    for lang in unseen_df["true_unseen_label"].unique():
        subset = unseen_df[unseen_df["true_unseen_label"] == lang]

        print(f"--- {lang} (n={len(subset)}) ---")
        print(subset["prediction"].value_counts(normalize=True).head(10))
        print()


def main():
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    meta = bundle.get("meta", {})

    new_dataset = get_new_dataset()
    new_dataset = fix_labels(new_dataset)

    texts = new_dataset["text"].tolist()

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    new_dataset["prediction"] = preds

    acc = accuracy_score(new_dataset["mapped_label"], new_dataset["prediction"])
    print("Overall Accuracy:", acc)

    print("\nAccuracy per language:")
    for lang in sorted(new_dataset["mapped_label"].unique()):
        subset = new_dataset[new_dataset["mapped_label"] == lang]
        lang_acc = accuracy_score(subset["mapped_label"], subset["prediction"])
        print(f"{lang:12s} {lang_acc:.4f}  (n={len(subset)})")

    print(classification_report(
        new_dataset["mapped_label"],
        new_dataset["prediction"],
        labels=model.classes_
    ))

    full_df = get_new_dataset()

    new_dataset = fix_labels(full_df.copy())

    texts = new_dataset["text"].tolist()
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    new_dataset["prediction"] = preds

    acc = accuracy_score(new_dataset["mapped_label"], new_dataset["prediction"])
    print("Accuracy:", acc)

    analyze_unseen_languages(full_df, model, vectorizer)


if __name__ == "__main__":
    main()