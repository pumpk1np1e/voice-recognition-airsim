import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Команды: takeoff, land, forward, back, left, right, up, down, yaw_left, yaw_right, hover, quit")

def do(cmd: str):
    cmd = cmd.strip().lower()

    if cmd == "takeoff":
        client.takeoffAsync().join()

    elif cmd == "land":
        client.landAsync().join()

    elif cmd == "hover":
        client.hoverAsync().join()

    elif cmd == "forward":
        client.moveByVelocityAsync(2, 0, 0, 2).join()

    elif cmd == "back":
        client.moveByVelocityAsync(-2, 0, 0, 2).join()

    elif cmd == "left":
        client.moveByVelocityAsync(0, -2, 0, 2).join()

    elif cmd == "right":
        client.moveByVelocityAsync(0, 2, 0, 2).join()

    elif cmd == "up":
        # В AirSim ось Z направлена вниз, поэтому "вверх" = отрицательная скорость по Z
        client.moveByVelocityAsync(0, 0, -1, 2).join()

    elif cmd == "down":
        client.moveByVelocityAsync(0, 0, 1, 2).join()

    elif cmd == "yaw_left":
        client.rotateByYawRateAsync(-30, 2).join()

    elif cmd == "yaw_right":
        client.rotateByYawRateAsync(30, 2).join()

    else:
        print("Неизвестная команда:", cmd)

try:
    while True:
        cmd = input("> ")
        if cmd.strip().lower() == "quit":
            break
        do(cmd)
finally:
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Выход")