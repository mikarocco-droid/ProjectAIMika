# ai/learning.py

from sklearn.cluster import KMeans


def build_features(events):
    return [[e.get("x", 0), e.get("y", 0)] for e in events]


def cluster_actions(events):
    X = build_features(events)

    if len(X) < 10:
        return events

    model = KMeans(n_clusters=3, n_init=10)
    labels = model.fit_predict(X)

    for i, e in enumerate(events):
        e["cluster"] = int(labels[i])

    return events


def learn_action_importance(events):
    for e in events:
        e["ml_score"] = 1 + e.get("xg", 0) * 5
    return events


def detect_key_moments(events):
    return [e for e in events if e.get("xg", 0) > 0.3]