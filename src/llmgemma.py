# llmgemma.py: Uses the Gemma language model from HuggingFace to classify the language of text samples,
# evaluates prediction accuracy against the true labels, and saves the results to a CSV file.

# Author: Aryan Joshi

from transformers import pipeline
from huggingface_hub import login
import pandas as pd
import torch

INPUT_CSV = "combined_dataset1.csv"
OUTPUT_CSV = "hf_gemma_predictions.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
MAX_ROWS = 10000

LABELS = [
    "english", "spanish", "french", "german", "italian",
    "portuguese", "russian", "arabic", "turkish", "dutch",
    "swedish", "danish", "greek", "hindi", "tamil",
    "kannada", "malayalam", "indonesian", "japanese", "korean",
    "chinese", "thai", "estonian", "romanian", "persian"
]

login(token= "****TOKEN NEEDED****")

pipe = pipeline(
    "text-generation",
    model="google/gemma-3-1b-it",
    device=0,
    dtype=torch.float16
)

def build_messages(text):
    label_text = ", ".join(LABELS)
    return [
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a language classifier. Reply with only one label from the allowed list."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
                        Classify the language of this text.

                        Allowed labels:
                        {label_text}

                        Rules:
                        - Return only one label
                        - Use lowercase
                        - Do not explain
                        - Do not add punctuation

                        Text:
                        {text}
                        """}
                ]
            }
        ]
    ]

def clean_label(output_text):
    output_text = output_text.strip().lower()
    for label in LABELS:
        if label in output_text:
            return label
    return "invalid"

df = pd.read_csv(INPUT_CSV).sample(n=MAX_ROWS, random_state=42)
df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip().str.lower()

predictions = []

for i, row in df.iterrows():
    text = str(row[TEXT_COLUMN])

    messages = build_messages(text)

    output = pipe(messages, max_new_tokens=10, do_sample=False)
    raw_text = output[0][0]["generated_text"][-1]["content"]

    pred = clean_label(raw_text)
    predictions.append(pred)

    print(f"Done {len(predictions)}/{len(df)} | Pred: {pred} | True: {row[LABEL_COLUMN]}")
    # accuracy every 100
    if len(predictions) % 100 == 0:
        correct_so_far = sum(
            p == t for p, t in zip(predictions, df[LABEL_COLUMN].iloc[:len(predictions)])
        )
        accuracy_so_far = correct_so_far / len(predictions)

        print("-----")
        print(f"Processed: {len(predictions)}/{len(df)}")
        print(f"Running accuracy: {accuracy_so_far:.4f}")
        print("-----")

df["predicted_label"] = predictions
df["correct"] = df[LABEL_COLUMN] == df["predicted_label"]

accuracy = df["correct"].mean()
print("Accuracy:", accuracy)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved to", OUTPUT_CSV)