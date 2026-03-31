# ai/learning.py
# -*- coding: utf-8 -*-

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("sklearn non installe — clustering désactivé")


# ─────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────
def build_features(events):
    """Extrait les features x/y de chaque event."""
    return [[
        float(e.get("x", 0)),
        float(e.get("y", 0))
    ] for e in events]


# ─────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────
def cluster_actions(events, n_clusters=3):
    """
    Assigne un cluster spatial à chaque event.
    Fallback propre si sklearn absent ou trop peu de points.
    """
    if not SKLEARN_AVAILABLE or len(events) < n_clusters * 2:
        for e in events:
            e["cluster"] = 0
        return events

    X = build_features(events)

    try:
        model  = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = model.fit_predict(X)
        for i, e in enumerate(events):
            e["cluster"] = int(labels[i])
    except Exception as ex:
        print(f"  Clustering error : {ex}")
        for e in events:
            e["cluster"] = 0

    return events


# ─────────────────────────────────────────
# IMPORTANCE ML
# ─────────────────────────────────────────
def learn_action_importance(events):
    """
    Calcule un score ML pour chaque event
    basé sur xG, type et danger.
    """
    type_weights = {
        "goal":            10.0,
        "score":           10.0,
        "shot":             5.0,
        "key_pass":         4.0,
        "progressive_run":  3.0,
        "interception":     3.0,
        "dribble":          2.0,
        "pass":             1.0,
        "possession":       0.5,
    }

    for e in events:
        base   = type_weights.get(e.get("type", ""), 1.0)
        xg     = float(e.get("xg",     0))
        danger = float(e.get("danger", 0))

        e["ml_score"] = round(base + xg * 5 + danger * 0.5, 3)

    return events


# ─────────────────────────────────────────
# KEY MOMENTS
# ─────────────────────────────────────────
def detect_key_moments(events, xg_threshold=0.25, ml_threshold=5.0):
    """
    Retourne les events jugés décisifs :
    - xG élevé (tir dangereux)
    - ou ml_score élevé (action importante)
    - ou type but/goal
    """
    key = []

    for e in events:
        is_goal    = e.get("type") in ["goal", "score"]
        high_xg    = float(e.get("xg",       0)) >= xg_threshold
        high_score = float(e.get("ml_score", 0)) >= ml_threshold

        if is_goal or high_xg or high_score:
            key.append(e)

    return key