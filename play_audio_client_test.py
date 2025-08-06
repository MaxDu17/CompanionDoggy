import socket
import sys

SOCKET_PATH = "/tmp/audio_socket"  # Inside container mount point

def play_audio(file_path):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(SOCKET_PATH)
    client.sendall(file_path.encode())
    client.close()

    print(f"[DOCKER] Sent play request for: {file_path}")

play_audio("/home/max/CompanionDoggy/assets/dog-barking.mp3")