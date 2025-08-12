import threading
from ui import GUI
from global_state import GlobalState
from dog_controller import DogController
import argparse
from threading import Thread
import signal

LOGS_DIR = "logs/"
def main():
    global_state = GlobalState()

    # Start the GUI in a separate thread
    gui = GUI(global_state)
    gui_thread = Thread(target=gui.demo.queue().launch, daemon=True)
    gui_thread.start()

    # Start dog controller in a separate thread

    dog_controller = DogController(global_state)
    global_state.lock_set("mode", "Warmup_Idle")

    average_speed = dog_controller.run_warmup(duration = 30)
    global_state.lock_set("mode", "Run_Idle")
    global_state.lock_set("speed", average_speed)

    # dog_controller.run_fixed_speed(speed = 4, duration = 30, mode = "run")

    dog_controller.run_interval(speed = average_speed, duration = 30)

if __name__ == "__main__":
    main() # args.mode, args.speed)