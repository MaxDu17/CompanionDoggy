import socket
import os
import threading
from playsound import playsound

SOCKET_PATH = "/tmp/audio_socket"

# Ensure the socket does not already exist
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

# Create a Unix socket
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
server.listen()

print(f"[HOST] Listening for audio requests on {SOCKET_PATH}...")

def handle_client(conn):
    try:
        data = conn.recv(1024).decode().strip()
        print(f"[HOST] Received request to play: {data}")
        playsound(data)  # Play the given file path (host must have the file)
    except Exception as e:
        print(f"[HOST] Error: {e}")
    finally:
        conn.close()

while True:
    conn, _ = server.accept()
    threading.Thread(target=handle_client, args=(conn,)).start()
