import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

start = time.time()

def eval_model(model, loader, id2label):
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    acc = accuracy_score(all_true, all_pred)

    y_true_names = [id2label[i] for i in all_true]
    y_pred_names = [id2label[i] for i in all_pred]

    report = classification_report(y_true_names, y_pred_names, digits=4)
    cm = confusion_matrix(y_true_names, y_pred_names, labels=sorted(id2label.values()))

    return acc, report, cm, sorted(id2label.values())

test_ds = LangDataset(test_df, tokenizer, label2id, MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

test_acc, test_report, cm, label_order = eval_model(model, test_loader, id2label)

print("Evaluating model...")
print(f"Test accuracy: {test_acc:.6f}\n")
print("=== Classification report (TEST) ===")
print(test_report)

pairs = []
for i, true_lab in enumerate(label_order):
    for j, pred_lab in enumerate(label_order):
        if i != j and cm[i, j] > 0:
            pairs.append((cm[i, j], true_lab, pred_lab))

pairs.sort(reverse=True, key=lambda x: x[0])

print("\n=== Top confusions (TEST) ===")
for count, t, p in pairs[:15]:
    print(f"{t} -> {p} : {count}")

print(f"\nTotal runtime: {time.time() - start:.2f} seconds")