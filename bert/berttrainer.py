import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

df = pd.read_csv("combined_dataset.csv")
df = df.sample(n=1000, random_state=42) 

#minor adjustments to the file 
df = df.dropna(subset=["Text", "Language"])
df["Language"] = df["Language"].astype(str)

#language label numbering
labels = sorted(df["Language"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
df["label"] = df["Language"].map(label2id)

#split 80 20 
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#load bert
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

#tokenizing, padding
class LangDataset(Dataset):
    def __init__(self, frame):
        self.texts = frame["Text"].tolist()
        self.labels = frame["label"].tolist()
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        #covert into bert tokens
        enc = tokenizer(self.texts[idx], truncation=True, padding="max_length",
                        max_length=128, return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

#making dataloader for training and validation
train_loader = DataLoader(LangDataset(train_df), batch_size=16, shuffle=True)
val_loader   = DataLoader(LangDataset(val_df), batch_size=16, shuffle=False)


#model
class LangID(nn.Module):
    def __init__(self, num_labels):
        #load bert
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        #predicting language
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0,:]
        return self.classifier(cls)


#training setup 
#makes model
model = LangID(len(labels)).to("cpu")
#adjusts weights
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#checks wrong
loss_fn = nn.CrossEntropyLoss()

#run the training (twice) 
for _ in range(2):
    model.train()
    for batch in train_loader:
        batch = {k:v.to("cpu") for k,v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = loss_fn(logits, batch["labels"])
        loss.backward()
        optimizer.step()



torch.save({"state_dict": model.state_dict(), "id2label": id2label}, "bertlang.pt")
print("Saved bertlang.pt")

