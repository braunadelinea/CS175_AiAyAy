
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from transformers import BertTokenizer, BertModel

CKPT_PATH = "bertlang.pt"
MAX_LEN = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
id2label = ckpt["id2label"]

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


class LangID(nn.Module):
    def __init__(self, num_labels: int):
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


def predict_top3(text: str):
    text = (text or "").strip()
    if not text:
        return "Type something.", "", "", ""

    if len(text) < 10:
        return "Please type 10+ characters for a reliable prediction.", "", "", ""

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])  
        probs = F.softmax(logits, dim=-1).squeeze(0)             
    k = min(3, probs.numel())
    top_probs, top_ids = torch.topk(probs, k=k)

    top_probs = top_probs.detach().cpu().tolist()
    top_ids = top_ids.detach().cpu().tolist()

    pred_lang = id2label[top_ids[0]]
    pred_conf = top_probs[0]

    conf_str = f"{pred_conf:.4f} ({pred_conf*100:.2f}%)"

    second = ""
    third = ""
    if k >= 2:
        second = f"{id2label[top_ids[1]]} — {top_probs[1]:.4f} ({top_probs[1]*100:.2f}%)"
    if k >= 3:
        third = f"{id2label[top_ids[2]]} — {top_probs[2]:.4f} ({top_probs[2]*100:.2f}%)"

    return pred_lang, conf_str, second, third


with gr.Blocks(title="Language ID (mBERT)") as demo:
    gr.Markdown("# Language Identification (mBERT)")
    gr.Markdown("Type a sentence and press **Enter** or click **Predict**.")

    inp = gr.Textbox(label="Input text", placeholder="Enter text...", lines=2)
    btn = gr.Button("Predict")

    out_lang = gr.Textbox(label="Predicted language")
    out_conf = gr.Textbox(label="Confidence")
    out_2 = gr.Textbox(label="2nd most likely")
    out_3 = gr.Textbox(label="3rd most likely")

    btn.click(predict_top3, inputs=inp, outputs=[out_lang, out_conf, out_2, out_3])
    inp.submit(predict_top3, inputs=inp, outputs=[out_lang, out_conf, out_2, out_3])

demo.launch()