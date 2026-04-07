class EventEngine:
    def __init__(self):
        self.last_event_time = {}

    def allow(self, event_type, t):
        cooldowns = {
            "shot": 2.0,
            "pass": 0.5,
            "goal": 5.0
        }

        last = self.last_event_time.get(event_type, -999)

        if t - last < cooldowns.get(event_type, 1):
            return False

        self.last_event_time[event_type] = t
        return True

    def score(self, speed, proximity):
        return 0.6 * speed + 0.4 * proximity