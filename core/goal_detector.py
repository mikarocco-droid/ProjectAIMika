class GoalDetector:
    def __init__(self):
        self.candidates = []

    def detect(self, ball_pos, speed, t):
        x, y = ball_pos

        if x > 1800 and 400 < y < 700 and speed > 12:
            self.candidates.append((t, ball_pos, speed))

    def validate(self):
        confirmed = []

        for t, pos, speed in self.candidates:
            # simple validation (peut être améliorée)
            if speed > 15:
                confirmed.append({
                    "time": t,
                    "type": "goal"
                })

        return confirmed