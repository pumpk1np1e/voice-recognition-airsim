import airsim
import queue
import json
import time
import os

import sounddevice as sd
from vosk import Model, KaldiRecognizer

import joblib
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# ====== НАСТРОЙКИ ======
VOSK_MODEL_PATH = "models/vosk-model-small-ru-0.22"
SAMPLE_RATE = 16000

INTENT_DIR = "intent_model"   # папка после train_intent.py
DEFAULT_THRESH = 0.35         # если threshold.txt не найден

# параметры движения
VX, VY, VZ = 2.0, 2.0, 1.0
DURATION = 2.0
# =======================

q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("AUDIO:", status)
    q.put(bytes(indata))

def norm(text: str) -> str:
    return text.strip().lower()

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

def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / (e.sum() + 1e-12)

def load_threshold(intent_dir: str, default: float) -> float:
    path = os.path.join(intent_dir, "threshold.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return float(f.read().strip())
        except Exception:
            pass
    return default

def load_intent_model():
    le = joblib.load(os.path.join(INTENT_DIR, "label_encoder.joblib"))
    emb_name = joblib.load(os.path.join(INTENT_DIR, "embedder_name.joblib"))
    embedder = SentenceTransformer(emb_name)

    dim = embedder.get_sentence_embedding_dimension()
    model = IntentNet(dim, len(le.classes_))
    model.load_state_dict(torch.load(os.path.join(INTENT_DIR, "intent_head.pt"), map_location="cpu"))
    model.eval()

    thresh = load_threshold(INTENT_DIR, DEFAULT_THRESH)
    return embedder, model, le, thresh

def predict_intent(text: str, embedder, model, le):
    emb = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    x = torch.tensor(emb, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)[0].cpu().numpy()
    probs = softmax_np(logits)
    idx = int(probs.argmax())
    label = le.inverse_transform([idx])[0]
    conf = float(probs[idx])
    return label, conf

def do_airsim(client: airsim.MultirotorClient, cmd: str):
    if cmd == "takeoff":
        client.takeoffAsync().join()
    elif cmd == "land":
        client.landAsync().join()
    elif cmd == "hover":
        client.hoverAsync().join()
    elif cmd == "forward":
        client.moveByVelocityAsync(VX, 0, 0, DURATION).join()
    elif cmd == "back":
        client.moveByVelocityAsync(-VX, 0, 0, DURATION).join()
    elif cmd == "left":
        client.moveByVelocityAsync(0, -VY, 0, DURATION).join()
    elif cmd == "right":
        client.moveByVelocityAsync(0, VY, 0, DURATION).join()
    elif cmd == "up":
        client.moveByVelocityAsync(0, 0, -VZ, DURATION).join()  # Z вниз -> вверх это -VZ
    elif cmd == "down":
        client.moveByVelocityAsync(0, 0, VZ, DURATION).join()
    else:
        print("Unknown cmd:", cmd)

def main():
    # 1) AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # 2) Vosk
    vosk_model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)

    # 3) Intent model
    embedder, intent_model, le, THRESH = load_intent_model()
    print("Intent labels:", list(le.classes_))
    print(f"THRESH={THRESH:.2f}")
    print("Слушаю... Команды: взлет/посадка/стоп/вперёд/назад/влево/вправо/вверх/вниз. 'выход' для остановки")

    mapping = {
        "TAKEOFF": "takeoff",
        "LAND": "land",
        "HOVER": "hover",
        "FORWARD": "forward",
        "BACK": "back",
        "RIGHT": "right",
        "UP": "up",
        "DOWN": "down",
    }

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = res.get("text", "")
                    if not text:
                        continue

                    t = norm(text)
                    print("ASR:", t)

                    if "выход" in t or "выйти" in t or "закрой" in t:
                        print("Выход.")
                        break

                    label, conf = predict_intent(t, embedder, intent_model, le)
                    print(f"INTENT: {label} conf={conf:.2f}")

                    if conf < THRESH:
                        print("Слишком низкая уверенность — команда игнорируется.\n")
                        continue

                    cmd = mapping.get(label)
                    if cmd is None:
                        print("Нет маппинга для метки:", label, "\n")
                        continue

                    do_airsim(client, cmd)
                    print()

    except KeyboardInterrupt:
        print("Остановлено пользователем.")

    finally:
        try:
            client.hoverAsync().join()
            time.sleep(0.2)
        except Exception:
            pass
        client.armDisarm(False)
        client.enableApiControl(False)
        print("Пока!")

if __name__ == "__main__":
    main()