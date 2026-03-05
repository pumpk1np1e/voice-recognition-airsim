import airsim
import queue
import json
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ====== НАСТРОЙКИ ======
MODEL_PATH = "models/vosk-model-small-ru-0.22"  
SAMPLE_RATE = 16000

VX, VY, VZ = 2.0, 2.0, 1.0
DURATION = 2.0
YAW_RATE = 30
# =======================

q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("AUDIO:", status)
    q.put(bytes(indata))

def norm(text: str) -> str:
    return text.strip().lower()

def text_to_cmd_ru(text: str) -> str | None:
    t = norm(text)

    # Выход
    if "выход" in t or "выйти" in t or "закрой" in t:
        return "quit"

    # Базовые команды
    if "взлет" in t or "взлёт" in t or "взлетай" in t:
        return "takeoff"
    if "посад" in t or "садись" in t:
        return "land"
    if "стоп" in t or "завис" in t or "стой" in t:
        return "hover"

    # Движение
    if "впер" in t or "вперёд" in t or "вперед" in t:
        return "forward"
    if "назад" in t:
        return "back"
    if "влево" in t:
        # если есть "поворот" — крутимся, иначе сдвиг влево
        if "повор" in t or "развер" in t:
            return "yaw_left"
        return "left"
    if "вправо" in t:
        if "повор" in t or "развер" in t:
            return "yaw_right"
        return "right"
    if "вверх" in t:
        return "up"
    if "вниз" in t:
        return "down"

    # Повороты (если отдельно произнесли без "влево/вправо" — редко)
    if "поворот" in t and "лев" in t:
        return "yaw_left"
    if "поворот" in t and "прав" in t:
        return "yaw_right"

    return None

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
        # Z вниз -> вверх это отрицательная скорость по Z
        client.moveByVelocityAsync(0, 0, -VZ, DURATION).join()
    elif cmd == "down":
        client.moveByVelocityAsync(0, 0, VZ, DURATION).join()
    elif cmd == "yaw_left":
        client.rotateByYawRateAsync(-YAW_RATE, DURATION).join()
    elif cmd == "yaw_right":
        client.rotateByYawRateAsync(YAW_RATE, DURATION).join()
    else:
        print("Unknown cmd:", cmd)

def main():
    # 1) AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # 2) Vosk
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    print("Слушаю... Скажи: взлёт, вперёд, влево, поворот влево, стоп, посадка, выход")

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

                    print("ASR:", text)
                    cmd = text_to_cmd_ru(text)

                    if cmd is None:
                        print("Команда не распознана.")
                        continue

                    if cmd == "quit":
                        print("Выход.")
                        break

                    print("CMD:", cmd)
                    do_airsim(client, cmd)

    except KeyboardInterrupt:
        print("Остановлено пользователем.")

    finally:
        try:
            client.hoverAsync().join()
        except Exception:
            pass
        client.armDisarm(False)
        client.enableApiControl(False)
        print("Пока!")

if __name__ == "__main__":
    main()