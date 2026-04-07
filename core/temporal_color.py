from collections import defaultdict
import numpy as np

class TemporalColorTracker:
    def __init__(self, max_history=30):
        self.history = defaultdict(list)
        self.max_history = max_history

    def smooth(self, player_id, color):
        self.history[player_id].append(color)

        if len(self.history[player_id]) > self.max_history:
            self.history[player_id].pop(0)

        avg = np.mean(self.history[player_id], axis=0)
        return tuple(avg.astype(int))