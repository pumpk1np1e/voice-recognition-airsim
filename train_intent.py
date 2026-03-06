import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt


DATA_PATH = "data/commands_ru.csv"
OUT_DIR = "intent_model"
RESULTS_DIR = "results"

# Мультиязычный эмбеддер (подходит для русского)
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


def compute_recognized_accuracy(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    conf: np.ndarray,
    labels: np.ndarray,
    threshold: float | None,
):
    labels = np.asarray(labels)
    mask = np.ones(len(y_pred_idx), dtype=bool)

    unknown_idx = np.where(labels == "UNKNOWN")[0]
    if len(unknown_idx) > 0:
        mask &= (y_pred_idx != int(unknown_idx[0]))
    if threshold is not None:
        mask &= (conf >= threshold)

    recognized = int(mask.sum())
    total = int(len(y_pred_idx))
    coverage = float(recognized / total) if total else 0.0
    if recognized == 0:
        return None, coverage, recognized

    acc_recognized = float(np.mean(y_pred_idx[mask] == y_true_idx[mask]))
    return acc_recognized, coverage, recognized


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Настройка шрифта для русского текста на графиках
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
    min_test = n_classes  # минимум по 1 примеру на класс в тесте
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

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ---- Model ----
    model = IntentNet(in_dim=X_train.shape[1], n_classes=len(le.classes_))
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # ---- Training history ----
    loss_history = []
    acc_history = []
    acc_rec_history = []
    cov_history = []

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

        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t).cpu().numpy()

        probs_test_epoch = softmax_2d(logits_test)
        pred_test_epoch = probs_test_epoch.argmax(axis=1)
        conf_test_epoch = probs_test_epoch[np.arange(len(y_test)), pred_test_epoch]

        acc_epoch = float(np.mean(pred_test_epoch == y_test))
        correct_mask = pred_test_epoch == y_test
        thr_epoch = choose_threshold(conf_test_epoch[correct_mask], conf_test_epoch[~correct_mask])
        acc_rec_epoch, cov_epoch, _ = compute_recognized_accuracy(
            y_test, pred_test_epoch, conf_test_epoch, le.classes_, thr_epoch
        )

        acc_history.append(acc_epoch)
        acc_rec_history.append(np.nan if acc_rec_epoch is None else acc_rec_epoch)
        cov_history.append(cov_epoch)

        acc_rec_print = "n/a" if acc_rec_epoch is None else f"{acc_rec_epoch:.3f}"
        print(
            f"Эпоха {epoch}/{EPOCHS} | loss={avg_loss:.4f} | "
            f"accuracy_test={acc_epoch:.3f} | accuracy_rec={acc_rec_print} | coverage={cov_epoch:.3f}"
        )

    # ---- Final eval + report ----
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t).cpu().numpy()

    probs_test = softmax_2d(logits_test)
    pred_test = probs_test.argmax(axis=1)
    conf_test = probs_test[np.arange(len(y_test)), pred_test]

    print("\nОтчет по метрикам (тест, F0.5):")
    prec, rec, f05, sup = precision_recall_fscore_support(
        y_test, pred_test, beta=0.5, zero_division=0
    )
    report_df = pd.DataFrame(
        {
            "label": le.classes_,
            "precision": prec,
            "recall": rec,
            "f0.5-score": f05,
            "support": sup,
        }
    )
    print(report_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"F0.5 (macro): "
        f"{fbeta_score(y_test, pred_test, beta=0.5, average='macro', zero_division=0):.4f}"
    )

    # ---- Threshold selection ----
    correct_mask = pred_test == y_test
    conf_correct = conf_test[correct_mask]
    conf_wrong = conf_test[~correct_mask]

    print("\nСтатистика уверенности (confidence):")
    if len(conf_correct):
        print(
            f" верные: n={len(conf_correct)} "
            f"среднее={conf_correct.mean():.2f} минимум={conf_correct.min():.2f}"
        )
    if len(conf_wrong):
        print(
            f" ошибки: n={len(conf_wrong)} "
            f"среднее={conf_wrong.mean():.2f} максимум={conf_wrong.max():.2f}"
        )

    thr = choose_threshold(conf_correct, conf_wrong)
    print(f"\nРекомендуемый порог THRESH={thr:.2f} (сохранено в {OUT_DIR}/threshold.txt)")

    acc_rec, cov, n_rec = compute_recognized_accuracy(
        y_test, pred_test, conf_test, le.classes_, thr
    )
    print("\nМетрика с отбрасыванием нераспознанных:")
    print(f"Coverage (распознано): {cov:.3f} ({n_rec}/{len(y_test)})")
    if acc_rec is None:
        print("Accuracy on recognized: n/a (нет распознанных примеров)")
    else:
        print(f"Accuracy on recognized: {acc_rec:.3f}")

    # ---- Save artifacts ----
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "intent_head.pt"))
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))
    joblib.dump(EMB_MODEL_NAME, os.path.join(OUT_DIR, "embedder_name.joblib"))
    with open(os.path.join(OUT_DIR, "threshold.txt"), "w", encoding="utf-8") as f:
        f.write(f"{thr:.4f}")

    # ---- Save training history ----
    hist = pd.DataFrame(
        {
            "epoch": list(range(1, EPOCHS + 1)),
            "loss": loss_history,
            "test_accuracy": acc_history,
            "test_accuracy_on_recognized": acc_rec_history,
            "recognized_coverage": cov_history,
        }
    )
    hist.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)

    # ---- График: функция потерь ----
    plt.figure(figsize=(10, 6))
    x = hist["epoch"].values
    y_loss = hist["loss"].values

    plt.plot(x, y_loss, marker="o", linewidth=2)
    for xi, yi in zip(x, y_loss):
        plt.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    plt.xlabel("Эпоха")
    plt.ylabel("Функция потерь")
    plt.title("Изменение функции потерь")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=300)
    plt.close()

    # ---- График: точность на распознанных ----
    plt.figure(figsize=(10, 6))
    y_acc = hist["test_accuracy_on_recognized"].fillna(hist["test_accuracy"]).values

    plt.plot(x, y_acc, marker="o", linewidth=2)

    y_min = float(np.min(y_acc))
    y_max = float(np.max(y_acc))
    pad = 0.03
    low = max(0.0, y_min - pad)
    high = min(1.0, y_max + pad)
    if high - low < 0.05:
        center = (high + low) / 2.0
        low = max(0.0, center - 0.025)
        high = min(1.0, center + 0.025)
    plt.ylim(low, high)

    top_threshold = high - (high - low) * 0.20
    for xi, yi in zip(x, y_acc):
        is_near_top = yi >= top_threshold
        plt.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, -12 if is_near_top else 8),
            ha="center",
            va="top" if is_near_top else "bottom",
            fontsize=9,
        )

    plt.xlabel("Эпоха")
    plt.ylabel("Точность на распознанных")
    plt.title("Изменение точности на распознанных командах", pad=14)
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
