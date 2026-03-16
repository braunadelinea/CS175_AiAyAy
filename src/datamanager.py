import pandas as pd

# Put all known label typo fixes here (keep this as the single source of truth)
LABEL_FIXES = {
    "portugeese": "portuguese",
    "portugese": "portuguese",
    "sweedish": "swedish",
    "pushto": "pashto",
}


def _normalize_label(s: str) -> str:
    """Lowercase, trim, collapse spaces, then fix known typos."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return LABEL_FIXES.get(s, s)


def standardize_df(df: pd.DataFrame, label_col: str, text_col: str, source: str) -> pd.DataFrame:
    """
    Returns a standardized DF with columns: text, label, source
    """
    out = df.rename(columns={label_col: "label", text_col: "text"}).copy()

    out = out[["text", "label"]]
    out["source"] = source

    out["label"] = out["label"].apply(_normalize_label)
    out["text"] = out["text"].astype(str)

    return out


def load_and_combine(
    detection_csv_path: str,
    identification_csv_path: str,
    detection_label_col: str = "Language",
    identification_label_col: str = "language",
    detection_text_col: str = "Text",
    identification_text_col: str = "Text",
) -> pd.DataFrame:
    d1 = pd.read_csv(detection_csv_path)
    d2 = pd.read_csv(identification_csv_path)

    d1s = standardize_df(d1, detection_label_col, detection_text_col, source="detection")
    d2s = standardize_df(d2, identification_label_col, identification_text_col, source="identification")

    combined = pd.concat([d1s, d2s], ignore_index=True)

    combined["text"] = combined["text"].str.strip()
    combined["label"] = combined["label"].str.strip()

    combined = combined[(combined["text"] != "") & (combined["label"] != "")]
    combined = combined.drop_duplicates(subset=["text", "label"])

    return combined


def basic_clean_and_filter(df: pd.DataFrame, *, min_chars: int = 20, max_chars: int | None = 1000,
                           max_nonalpha_ratio: float = 0.6) -> pd.DataFrame:
    out = df.copy()
    out["text"] = out["text"].astype(str).str.strip()

    # length filter
    out = out[out["text"].str.len() >= min_chars]

    # truncate (optional)
    if max_chars is not None:
        out["text"] = out["text"].str.slice(0, max_chars)

    # remove rows that are mostly non-letters (numbers/symbol spam)
    def nonalpha_ratio(s: str) -> float:
        if not s:
            return 1.0
        nonalpha = sum(1 for ch in s if not ch.isalpha() and not ch.isspace())
        return nonalpha / max(len(s), 1)

    out = out[out["text"].apply(nonalpha_ratio) <= max_nonalpha_ratio]
    return out


def stratified_split(df: pd.DataFrame, *, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=0):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # split per class to keep proportions
    train_parts, val_parts, test_parts = [], [], []
    for _, g in df.groupby("label", sort=False):
        n = len(g)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # remainder goes to test
        n_test = n - n_train - n_val
        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train:n_train+n_val])
        test_parts.append(g.iloc[n_train+n_val:])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val   = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test  = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val, test