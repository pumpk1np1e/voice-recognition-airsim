import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt


DATA_PATH = "data/commands_ru.csv"
OUT_DIR = "intent_model"
RESULTS_DIR = "results"

# Мультиязычный эмбеддер (лучше для русского)
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


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


def softmax_2d(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def choose_threshold(conf_correct: np.ndarray, conf_wrong: np.ndarray) -> float:
    """
    Выбираем порог уверенности по тестовой выборке.
    Эвристика:
      - если есть и верные, и неверные: (max_wrong + min_correct) / 2
      - если неверных нет: max(0.20, min_correct - 0.05)
      - если верных нет: 0.45
    """
    if len(conf_correct) == 0:
        return 0.45
    if len(conf_wrong) == 0:
        return float(max(0.20, float(np.min(conf_correct)) - 0.05))

    max_wrong = float(np.max(conf_wrong))
    min_correct = float(np.min(conf_correct))
    thr = (max_wrong + min_correct) / 2.0
    return float(np.clip(thr, 0.15, 0.80))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Настройка шрифта для поддержки русского
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    # ---- Load data ----
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    print("Метки в CSV:", sorted(df["label"].unique()))

    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str))

    # ---- Embeddings ----
    st = SentenceTransformer(EMB_MODEL_NAME)
    X = st.encode(df["text"].tolist(), convert_to_numpy=True, normalize_embeddings=True)

    # ---- Split (robust) ----
    n_classes = len(np.unique(y))
    min_test = n_classes  # по 1 примеру на класс в тесте минимум
    test_size = max(0.2, min_test / len(y))
    test_size = min(test_size, 0.5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    print(f"Примеров={len(y)} | Классов={n_classes} | test_size={test_size:.2f}")
    print(f"Обучение={len(y_train)} | Тест={len(y_test)}")

    # ---- Torch tensors ----
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ---- Model ----
    model = IntentNet(in_dim=X_train.shape[1], n_classes=len(le.classes_))
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # ---- Training history ----
    loss_history = []
    acc_history = []

    # ---- Train ----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)

        # accuracy на тесте по эпохам
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t)
            pred_test = torch.argmax(logits_test, dim=1)
            acc = float((pred_test == y_test_t).float().mean().item())
        acc_history.append(acc)

        print(f"эпоха {epoch}/{EPOCHS} | loss={avg_loss:.4f} | точность_тест={acc:.3f}")

    # ---- Final eval + report ----
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t).cpu().numpy()

    probs_test = softmax_2d(logits_test)
    pred_test = probs_test.argmax(axis=1)
    conf_test = probs_test[np.arange(len(y_test)), pred_test]

    print("\nОтчет по метрикам (тест):")
    print(classification_report(y_test, pred_test, target_names=le.classes_, zero_division=0))

    # ---- Threshold selection ----
    correct_mask = (pred_test == y_test)
    conf_correct = conf_test[correct_mask]
    conf_wrong = conf_test[~correct_mask]

    print("\nСтатистика уверенности (confidence):")
    if len(conf_correct):
        print(f" верные: n={len(conf_correct)} среднее={conf_correct.mean():.2f} минимум={conf_correct.min():.2f}")
    if len(conf_wrong):
        print(f" ошибки: n={len(conf_wrong)} среднее={conf_wrong.mean():.2f} максимум={conf_wrong.max():.2f}")

    thr = choose_threshold(conf_correct, conf_wrong)
    print(f"\nРекомендуемый порог THRESH={thr:.2f} (сохранено в {OUT_DIR}/threshold.txt)")

    # ---- Save artifacts ----
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "intent_head.pt"))
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))
    joblib.dump(EMB_MODEL_NAME, os.path.join(OUT_DIR, "embedder_name.joblib"))
    with open(os.path.join(OUT_DIR, "threshold.txt"), "w", encoding="utf-8") as f:
        f.write(f"{thr:.4f}")

    # ---- Save training history ----
    hist = pd.DataFrame({
        "epoch": list(range(1, EPOCHS + 1)),
        "loss": loss_history,
        "test_accuracy": acc_history
    })
    hist.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)

    # ---- График: Функция потерь (точки + подписи) ----
    plt.figure(figsize=(10, 6))
    x = hist["epoch"].values
    y_loss = hist["loss"].values

    plt.plot(x, y_loss, marker="o", linewidth=2)
    for xi, yi in zip(x, y_loss):
        plt.annotate(
            f"{yi:.2f}", (xi, yi),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=9
        )

    plt.xlabel("Эпоха")
    plt.ylabel("Функция потерь")
    plt.title("График изменения функции потерь")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=300)
    plt.close()

    # ---- График: Точность на тестовой выборке (точки + подписи) ----
    plt.figure(figsize=(10, 6))
    y_acc = hist["test_accuracy"].values

    plt.plot(x, y_acc, marker="o", linewidth=2)
    for xi, yi in zip(x, y_acc):
        plt.annotate(
            f"{yi:.2f}", (xi, yi),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=9
        )

    plt.ylim(0.0, 1.0)
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.title("Изменение точности на тестовой выборке")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=300)
    plt.close()

    print("\nСохранено:")
    print(f" - {OUT_DIR}/intent_head.pt")
    print(f" - {OUT_DIR}/label_encoder.joblib")
    print(f" - {OUT_DIR}/embedder_name.joblib")
    print(f" - {OUT_DIR}/threshold.txt")
    print(f" - {RESULTS_DIR}/training_history.csv")
    print(f" - {RESULTS_DIR}/loss_curve.png")
    print(f" - {RESULTS_DIR}/accuracy_curve.png")


if __name__ == "__main__":
    main()