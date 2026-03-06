import os
import json
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from sentence_transformers import SentenceTransformer

DATA_PATH = "data/commands_ru.csv"
MODEL_DIR = "intent_model"
OUT_DIR = "results"
SEED = 42

# Если меняли архитектуру "головы" в train_intent.py,
# укажите такие же параметры здесь.
HIDDEN = 256
DROPOUT = 0.2


class IntentNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def softmax_2d(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def read_threshold(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        return None


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_f05_report(y_true, y_pred, labels: list[str]):
    label_ids = np.arange(len(labels))
    prec, rec, f05, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        beta=0.5,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, beta=0.5, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, beta=0.5, average="weighted", zero_division=0
    )

    rows = []
    report = {}
    for i, label in enumerate(labels):
        item = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f0.5-score": float(f05[i]),
            "support": int(sup[i]),
        }
        report[label] = item
        rows.append({"label": label, **item})

    total_support = int(np.sum(sup))
    report["macro avg"] = {
        "precision": float(macro[0]),
        "recall": float(macro[1]),
        "f0.5-score": float(macro[2]),
        "support": total_support,
    }
    report["weighted avg"] = {
        "precision": float(weighted[0]),
        "recall": float(weighted[1]),
        "f0.5-score": float(weighted[2]),
        "support": total_support,
    }

    rows.append({"label": "macro avg", **report["macro avg"]})
    rows.append({"label": "weighted avg", **report["weighted avg"]})

    report_df = pd.DataFrame(rows, columns=["label", "precision", "recall", "f0.5-score", "support"])
    return report, report_df, prec, rec, f05, sup


def compute_recognized_accuracy(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    conf: np.ndarray,
    labels: list[str],
    threshold: float | None,
):
    label_arr = np.asarray(labels)
    mask = np.ones(len(y_pred_idx), dtype=bool)

    unknown_idx = np.where(label_arr == "UNKNOWN")[0]
    if len(unknown_idx) > 0:
        mask &= y_pred_idx != int(unknown_idx[0])
    if threshold is not None:
        mask &= conf >= threshold

    recognized = int(mask.sum())
    total = int(len(y_pred_idx))
    coverage = float(recognized / total) if total else 0.0
    if recognized == 0:
        return None, coverage, recognized

    acc_recognized = float(np.mean(y_pred_idx[mask] == y_true_idx[mask]))
    return acc_recognized, coverage, recognized


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, out_png: str):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Истинный класс",
        xlabel="Предсказанный класс",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Подписи значений в ячейках
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_bar(values: np.ndarray, labels: list[str], title: str, ylabel: str, out_png: str):
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_prf(prec, rec, f05, labels: list[str], out_png: str):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, prec, width=w, label="Точность (Precision)")
    ax.bar(x, rec, width=w, label="Полнота (Recall)")
    ax.bar(x + w, f05, width=w, label="F0.5-мера (F0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Метрики по классам")
    ax.set_ylabel("Значение")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_confidence_hist(conf_correct, conf_wrong, out_png: str, thr):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(conf_correct, bins=bins, alpha=0.6, label="Верные предсказания")
    ax.hist(conf_wrong, bins=bins, alpha=0.6, label="Ошибочные предсказания")

    if thr is not None:
        ax.axvline(thr, linestyle="--", linewidth=2, label=f"Порог={thr:.2f}")

    ax.set_title("Распределение уверенности (confidence)")
    ax.set_xlabel("Уверенность")
    ax.set_ylabel("Количество")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ensure_out_dir()

    # Настройка шрифта для русского
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    # --- Load dataset ---
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    # --- Load model artifacts ---
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    emb_name = joblib.load(os.path.join(MODEL_DIR, "embedder_name.joblib"))
    thr = read_threshold(os.path.join(MODEL_DIR, "threshold.txt"))

    embedder = SentenceTransformer(emb_name)
    X = embedder.encode(df["text"].tolist(), convert_to_numpy=True, normalize_embeddings=True)

    y = le.transform(df["label"].astype(str))
    labels = list(le.classes_)
    n_classes = len(labels)

    # --- Split the same robust way as in train_intent ---
    min_test = n_classes
    test_size = max(0.2, min_test / len(y))
    test_size = min(test_size, 0.5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    # --- Load head ---
    dim = embedder.get_sentence_embedding_dimension()
    model = IntentNet(dim, n_classes)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "intent_head.pt"), map_location="cpu"))
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_test_t).cpu().numpy()

    probs = softmax_2d(logits)
    y_pred = probs.argmax(axis=1)
    conf = probs[np.arange(len(y_pred)), y_pred]

    # --- Metrics ---
    acc = float(accuracy_score(y_test, y_pred))
    acc_rec, coverage, recognized_n = compute_recognized_accuracy(
        y_test, y_pred, conf, labels, thr
    )
    rep, rep_df, prec, rec, f05, sup = build_f05_report(y_test, y_pred, labels)
    rep_text = rep_df.to_string(index=False, float_format=lambda v: f"{v:.4f}")

    # Save report
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep_text)
        f.write("\n")
        f.write(f"\nТочность (accuracy): {acc:.4f}\n")
        if acc_rec is None:
            f.write("Accuracy on recognized: n/a (no recognized samples)\n")
        else:
            f.write(f"Accuracy on recognized: {acc_rec:.4f}\n")
        f.write(f"Coverage (recognized/total): {coverage:.4f} ({recognized_n}/{len(y_test)})\n")
        if thr is not None:
            f.write(f"Порог (из файла): {thr:.4f}\n")
        f.write(f"Эмбеддер: {emb_name}\n")
        f.write(f"test_size: {test_size:.2f} | тестовых примеров: {len(y_test)}\n")

    save_json(
        {
            "accuracy": acc,
            "accuracy_on_recognized": acc_rec,
            "coverage": coverage,
            "recognized_samples": recognized_n,
            "threshold": thr,
            "embedder": emb_name,
            "classes": labels,
            "test_size": float(test_size),
            "test_samples": int(len(y_test)),
            "report": rep,
        },
        os.path.join(OUT_DIR, "metrics.json"),
    )

    # Per-class PRF arrays (F0.5)
    metrics_df = pd.DataFrame(
        {"label": labels, "precision": prec, "recall": rec, "f0_5": f05, "support": sup}
    )
    metrics_df.to_csv(os.path.join(OUT_DIR, "per_class_metrics.csv"), index=False)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
    plot_confusion_matrix(
        cm,
        labels,
        title="Матрица ошибок (тестовая выборка)",
        out_png=os.path.join(OUT_DIR, "confusion_matrix.png"),
    )

    # --- Bars ---
    plot_bar(
        f05,
        labels,
        title="F0.5-мера по классам",
        ylabel="F0.5-мера",
        out_png=os.path.join(OUT_DIR, "f0_5_per_class.png"),
    )
    plot_prf(
        prec, rec, f05, labels, out_png=os.path.join(OUT_DIR, "prf_per_class.png")
    )

    # --- Confidence histogram ---
    correct_mask = y_pred == y_test
    conf_correct = conf[correct_mask]
    conf_wrong = conf[~correct_mask]
    plot_confidence_hist(
        conf_correct,
        conf_wrong,
        out_png=os.path.join(OUT_DIR, "confidence_hist.png"),
        thr=thr,
    )

    # --- Summary print ---
    print(f"Графики и метрики сохранены в: {OUT_DIR}/")
    print(f"Точность (accuracy)={acc:.3f} | тестовых примеров={len(y_test)} | классов={n_classes}")
    if acc_rec is None:
        print("Accuracy on recognized: n/a")
    else:
        print(f"Accuracy on recognized={acc_rec:.3f} | coverage={coverage:.3f}")
    if thr is not None:
        print(f"Порог (из файла)={thr:.2f}")
    print("Созданные файлы:")
    for fn in [
        "classification_report.txt",
        "metrics.json",
        "per_class_metrics.csv",
        "confusion_matrix.png",
        "f0_5_per_class.png",
        "prf_per_class.png",
        "confidence_hist.png",
    ]:
        print(" -", os.path.join(OUT_DIR, fn))


if __name__ == "__main__":
    main()
