import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

DEVICE = "cpu"

ckpt = torch.load("bertlang.pt", map_location=DEVICE)
id2label = ckpt["id2label"]
MAX_LEN = 128

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

class LangID(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

model = LangID(len(id2label)).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()


# ---- Test function ----
def predict(text):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        pred = torch.argmax(logits, dim=-1).item()

    return id2label[pred]


# ---- Try some test sentences ----
tests = [
    "Hello world",
    "Hola amigo",
    "Bonjour tout le monde",
    "Olá tudo bem"
]

for t in tests:
    print(t, "->", predict(t))