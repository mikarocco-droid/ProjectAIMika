import math

class PlayerIdentityManager:
    def __init__(self):
        self.players = {}
        self.next_id = 1

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def assign_id(self, position, color):
        for pid, data in self.players.items():
            dist = self._distance(position, data["pos"])
            color_diff = sum(abs(a - b) for a, b in zip(color, data["color"]))

            if dist < 80 and color_diff < 120:
                data["pos"] = position
                data["color"] = color
                return pid

        pid = self.next_id
        self.players[pid] = {
            "pos": position,
            "color": color
        }
        self.next_id += 1
        return pid