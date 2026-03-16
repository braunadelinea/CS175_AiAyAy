import tkinter as tk
from tkinter import ttk, messagebox
import joblib

BUNDLE_PATH = "artifacts/language_pipeline.joblib"


class LanguageClassifierUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Language Classifier")
        self.geometry("760x520")
        self.minsize(640, 420)

        # Load bundle (model + vectorizer)
        try:
            bundle = joblib.load(BUNDLE_PATH)
            self.model = bundle["model"]
            self.vectorizer = bundle["vectorizer"]
            self.meta = bundle.get("meta", {})
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load model bundle:\n{e}")
            raise

        # ---------- Header ----------
        header = ttk.Frame(self)
        header.pack(fill="x", padx=12, pady=(12, 6))

        ttk.Label(header, text="Type text to classify:", font=("Helvetica", 12, "bold")).pack(side="left")

        if self.meta:
            acc_text = f"Val: {self.meta.get('val_acc', 0):.4f} | Test: {self.meta.get('test_acc', 0):.4f}"
            ttk.Label(header, text=acc_text).pack(side="right")

        # ---------- Text input ----------
        self.text = tk.Text(self, height=12, wrap="word")
        self.text.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        # Help text / hint
        hint = ttk.Label(self, text="Tip: Press Ctrl+Enter to classify. Minimum 10 characters recommended.")
        hint.pack(anchor="w", padx=12, pady=(0, 8))

        # ---------- Buttons ----------
        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=12, pady=(0, 10))

        ttk.Button(btn_row, text="Classify (Ctrl+Enter)", command=self.classify).pack(side="left")
        ttk.Button(btn_row, text="Clear", command=self.clear).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Quit", command=self.destroy).pack(side="right")

        # ---------- Output ----------
        self.result_var = tk.StringVar(value="Result: —")
        ttk.Label(self, textvariable=self.result_var, font=("Helvetica", 14)).pack(
            anchor="w", padx=12, pady=(6, 4)
        )

        self.topk_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.topk_var).pack(anchor="w", padx=12, pady=(0, 10))

        # Keybind
        self.bind("<Control-Return>", lambda e: self.classify())

    def clear(self):
        self.text.delete("1.0", "end")
        self.result_var.set("Result: —")
        self.topk_var.set("")

    def classify(self):
        raw = self.text.get("1.0", "end").strip()
        if len(raw) < 10:
            self.result_var.set("Result: Please type 10+ characters for a reliable prediction.")
            self.topk_var.set("")
            return

        try:
            X = self.vectorizer.transform([raw])
            probs = self.model.predict_proba(X)[0]

            # Top-1
            best_idx = probs.argmax()
            pred = self.model.classes_[best_idx]
            conf = float(probs[best_idx])
            self.result_var.set(f"Result: {pred.capitalize()}   (confidence: {conf:.3f})")

            # Top-3
            topk = probs.argsort()[::-1][:3]
            lines = []
            for i, idx in enumerate(topk, start=1):
                lines.append(f"{i}) {self.model.classes_[idx].capitalize()}  ({float(probs[idx]):.3f})")
            self.topk_var.set("Top 3: " + "   |   ".join(lines))

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))


if __name__ == "__main__":
    app = LanguageClassifierUI()
    app.mainloop()
