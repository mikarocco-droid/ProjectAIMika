import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter

class TeamClassifier:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, n_init=10)
        self.fitted = False
        self.team_history = defaultdict(list)

    def fit(self, colors):
        self.kmeans.fit(colors)
        self.fitted = True

    def predict(self, player_id, color):
        if not self.fitted:
            return None

        team = int(self.kmeans.predict([color])[0])

        self.team_history[player_id].append(team)

        if len(self.team_history[player_id]) > 20:
            self.team_history[player_id].pop(0)

        # majorité
        final_team = Counter(self.team_history[player_id]).most_common(1)[0][0]

        return final_team