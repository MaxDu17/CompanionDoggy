"""
global_state.py

Container class defining a thread-safe Global State --> accessed by all modules in a Vocal Sandbox system.
"""

import inspect
import json
import os
import datetime
import time
from copy import copy
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


class GlobalState:
    def __init__(
        self,
        gui = None,
    ) -> None:
        """Initialize Global State with a mutable `dict` of fields, and pointers to GUI/API Spec."""
        self.data = {
            "speed": 0,
            "person_distance": None, # if no person detected, set to None
        }
        self.locks = {k: RLock() for k in self.data}
        self.gui = gui

    def lock_get(self, key: str) -> Any:
        """Thread-safe getter."""
        with self.locks[key]:
            val = self.data[key]
            return val

    def lock_set(self, key: str, value: Any) -> None:
        """Thread-safe setter; additionally propagates changes to GUI (as side-effect)."""
        with self.locks[key]:
            self.data[key] = value

            # Only update `GUI` if exists...
            if self.gui is not None:
                self._update_gui(key, value)

    def _update_gui(self, key: str, value: Any) -> None:
        # fmt: off
        if key == "person_distance":                                                # String
            self.gui.person_distance = value
        # fmt: on

if __name__ == "__main__":
    global_state = GlobalState()
    global_state.lock_set("speed", 10)
