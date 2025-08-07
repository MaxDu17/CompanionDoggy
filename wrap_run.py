import threading
from ui import GUI
from global_state import GlobalState
from dog_controller import DogController
import argparse
from threading import Thread

def main(mode: str, speed: int):
    global_state = GlobalState()

    # Start the GUI in a separate thread
    gui = GUI(global_state)
    gui_thread = Thread(target=gui.demo.queue().launch, daemon=True)
    gui_thread.start()

    # Start dog controller in a separate thread
    dog_controller = DogController(global_state)
    if mode == "warmup":
        dog_controller_thread = Thread(target=dog_controller.run_warmup, daemon=True)
    elif mode == "interval":
        dog_controller_thread = Thread(target=dog_controller.run_interval, args=(speed,), daemon=True)
    elif mode == "run":
        dog_controller_thread = Thread(target=dog_controller.run_fixed_speed, args=(speed,), daemon=True)
    dog_controller_thread.start()


if __name__ == "__main__":
    # arg parse for mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="warmup", choices=["warmup", "interval", "run"])
    parser.add_argument("--speed", type=int, default=4)
    args = parser.parse_args()

    main(args.mode, args.speed)