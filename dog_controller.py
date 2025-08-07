from global_state import GlobalState

class DogController:
    def __init__(self, global_state: GlobalState):
        self.global_state = global_state

    def run_warmup(self):
        self.global_state.lock_set("person_distance", 100)
        pass

    def run_interval(self, speed: int):
        speed = self.global_state.lock_get("speed")
        self.global_state.lock_set("person_distance", 100)
        pass

    def run_fixed_speed(self, speed: int):
        speed = self.global_state.lock_get("speed")
        self.global_state.lock_set("person_distance", 100)
        pass