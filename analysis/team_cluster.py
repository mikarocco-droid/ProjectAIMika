from collections import defaultdict
import numpy as np


def assign_teams_by_color(events):
    """
    Assigne team=0 ou team=1 sur chaque event en utilisant les centroides
    de couleur calibrés par TeamColorDetector (via player_reid).

    Si player_reid n'est pas disponible, fallback sur la logique RGB simple.
    """
    # ── Récupérer les centroides calibrés depuis player_reid ─────────────────
    team_centroids = None
    try:
        from analysis.player_reid import get_team_colors as _gtc
        raw = _gtc()
        if raw and len(raw) >= 2:
            # raw = {0: (B,G,R), 1: (B,G,R)}
            team_centroids = {
                int(k): np.array(v, dtype=np.float32)
                for k, v in raw.items()
                if v is not None and len(v) == 3
            }
            if len(team_centroids) < 2:
                team_centroids = None
    except Exception:
        team_centroids = None

    # ── Construire pid → team depuis les couleurs sur les events ─────────────
    pid_color_votes = defaultdict(lambda: defaultdict(int))

    for e in events:
        color = e.get("color") or e.get("team_color") or e.get("jersey_color")
        pid = e.get("player")
        if not color or pid is None:
            continue
        color = tuple(color[:3])

        if team_centroids:
            # Distance L2 aux deux centroides calibrés
            c = np.array(color, dtype=np.float32)
            dists = {
                team_id: float(np.linalg.norm(c - centroid))
                for team_id, centroid in team_centroids.items()
            }
            team = min(dists, key=dists.get)
        else:
            # Fallback RGB : bleu dominant → team 0, rouge dominant → team 1
            b, g, r = color
            team = 0 if b > r else 1

        pid_color_votes[pid][team] += 1

    # Résoudre le vote majoritaire par joueur
    pid_to_team = {
        pid: max(votes, key=votes.get)
        for pid, votes in pid_color_votes.items()
        if votes
    }

    # ── Propager sur les events ───────────────────────────────────────────────
    for e in events:
        pid = e.get("player")
        # Ne pas écraser si déjà renseigné
        if e.get("team") is not None:
            continue
        if pid in pid_to_team:
            e["team"] = pid_to_team[pid]

    n_assigned = sum(1 for e in events if e.get("team") is not None)
    n_total    = len(events)
    print(f"  [TEAM_CLUSTER] {n_assigned}/{n_total} events avec team "
          f"({'centroides calibrés' if team_centroids else 'fallback RGB'})")

    return events