import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

MODEL_DIR = "intent_model"

class IntentNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)

def load_threshold_default() -> float:
    try:
        with open(os.path.join(MODEL_DIR, "threshold.txt"), "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        return 0.45

def load_model():
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    emb_name = joblib.load(os.path.join(MODEL_DIR, "embedder_name.joblib"))
    st = SentenceTransformer(emb_name)

    dim = st.get_sentence_embedding_dimension()
    model = IntentNet(dim, len(le.classes_))
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "intent_head.pt"), map_location="cpu"))
    model.eval()

    thresh = load_threshold_default()
    return st, model, le, thresh

def topk_predict(text: str, st, model, le, k: int = 3):
    emb = st.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    x = torch.tensor(emb, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)                      # shape (1, C)
        probs = torch.softmax(logits, dim=1)   # shape (1, C)

    probs = probs[0].cpu().numpy()
    idxs = probs.argsort()[::-1][:k]
    items = [(le.inverse_transform([int(i)])[0], float(probs[int(i)])) for i in idxs]
    best_label, best_conf = items[0]
    return best_label, best_conf, items

def main():
    st, model, le, THRESH = load_model()
    print(f"Готово. THRESH={THRESH:.2f}")
    print("Вводи фразы по-русски. 'выход' чтобы закончить.\n")

    while True:
        text = input("> ").strip()
        if not text:
            continue
        if text.lower() in ("выход", "quit", "exit"):
            break

        best_label, best_conf, top3 = topk_predict(text, st, model, le, k=3)
        print("TOP3:", ", ".join([f"{lbl}:{p:.2f}" for lbl, p in top3]))

        if best_conf < THRESH:
            print(f"→ НЕУВЕРЕННО (conf={best_conf:.2f})\n")
        else:
            print(f"→ {best_label} (conf={best_conf:.2f})\n")

if __name__ == "__main__":
    main()