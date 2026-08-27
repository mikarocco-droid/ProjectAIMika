from collections import defaultdict
import re
import numpy as np


def _team_depuis_prefixe(pid):
    """
    V5.2 : l'identite canonique d'un joueur (ex: "P0_10", "P1_7") encode
    DEJA l'equipe dans son prefixe, quand elle a ete confirmee lors de la
    fusion d'identite (voir analysis/player_identity.py). C'est la source
    la PLUS fiable disponible - deja validee par la reconciliation sure
    (numero confirme sur une seule equipe + pas de chevauchement temporel),
    bien plus fiable que n'importe quel repli couleur.

    "P4" (sans prefixe) -> equipe jamais confirmee -> None (pas de fausse
    certitude fabriquee).
    "P0_10" -> equipe 0, confirmee.

    Retourne l'entier equipe, ou None si pas de prefixe exploitable.
    """
    if not isinstance(pid, str):
        return None
    m = re.match(r"^P(\d+)_", pid)
    if m:
        return int(m.group(1))
    return None


def assign_teams_by_color(events, team_colors=None):
    """
    Assigne team=0 ou team=1 sur chaque event.

    Priorité des sources (V5.2 - réordonnée) :
      0. Préfixe d'équipe déjà présent dans l'identité canonique du joueur
         (le plus fiable - déjà validé par la réconciliation sûre)
      1. team_colors passé en paramètre (depuis pipeline._captured_team_colors)
      2. player_reid.get_team_colors() (si disponible après tracking)
      3. Fallback RGB (bleu > rouge → team 0) - DÉSACTIVÉ (voir note V5.2
         ci-dessous), gardé en dernier recours documenté seulement.

    NOTE V5.2 : le repli RGB (étape 3) suppose une équipe bleue contre une
    équipe rouge - faux dès que les couleurs réelles sont différentes
    (ex: Raeren, rouge vs jaune/noir - confirmé aujourd'hui). Le champ
    "color" lu sur les events est en outre un vecteur 19D (histogramme +
    LAB), pas du BGR - `color[:3]` ne prend que les 3 premiers bins d'un
    histogramme, une valeur sans rapport avec une vraie couleur. Ce repli
    RGB est donc DÉSACTIVÉ par défaut (RGB_FALLBACK_ACTIF=False) plutôt
    que réparé - la priorité 0 (préfixe déjà validé) couvre la grande
    majorité des cas de façon fiable ; pour le reste, team=None honnête
    est préférable à une équipe fabriquée à partir d'un signal cassé.
    """
    RGB_FALLBACK_ACTIF = False  # V5.2 : désactivé, voir note ci-dessus

    # ── Priorité 0 : préfixe d'équipe déjà dans l'identité canonique ─────────
    n_via_prefixe = 0
    for e in events:
        if e.get("team") is not None:
            continue
        team_prefixe = _team_depuis_prefixe(e.get("player"))
        if team_prefixe is not None:
            e["team"] = team_prefixe
            n_via_prefixe += 1
    if n_via_prefixe:
        print(f"  [TEAM_CLUSTER] {n_via_prefixe} events résolus via préfixe d'identité (source la plus fiable)")

    # ── Source 1 : centroides passés explicitement depuis pipeline.py ─────────
    team_centroids = None
    _source = "aucune (repli RGB désactivé)"

    if team_colors and len(team_colors) >= 2:
        try:
            team_centroids = {
                int(k): np.array(v, dtype=np.float32)
                for k, v in team_colors.items()
                if v is not None and len(v) == 3
            }
            if len(team_centroids) < 2:
                team_centroids = None
            else:
                _source = "centroides pipeline"
        except Exception:
            team_centroids = None

    # ── Source 2 : player_reid (fallback si team_colors absent) ──────────────
    if team_centroids is None:
        try:
            from analysis.player_reid import get_team_colors as _gtc
            raw = _gtc()
            if raw and len(raw) >= 2:
                team_centroids = {
                    int(k): np.array(v, dtype=np.float32)
                    for k, v in raw.items()
                    if v is not None and len(v) == 3
                }
                if len(team_centroids) < 2:
                    team_centroids = None
                else:
                    _source = "centroides player_reid"
        except Exception:
            team_centroids = None

    if team_centroids:
        c0 = tuple(int(x) for x in team_centroids[0])
        c1 = tuple(int(x) for x in team_centroids[1])
        print(f"  [TEAM_CLUSTER] centroides : team0={c0} team1={c1} (src={_source})")
    elif RGB_FALLBACK_ACTIF:
        print(f"  [TEAM_CLUSTER] aucun centroide disponible → fallback RGB")
    else:
        print(f"  [TEAM_CLUSTER] aucun centroide disponible, repli RGB désactivé (V5.2) — "
              f"team restera None plutôt qu'une valeur fabriquée")

    # ── Construire pid → team depuis les couleurs sur les events ─────────────
    pid_color_votes = defaultdict(lambda: defaultdict(int))

    for e in events:
        if e.get("team") is not None:
            continue  # déjà résolu (préfixe ou tour précédent)
        color = e.get("color") or e.get("team_color") or e.get("jersey_color")
        pid = e.get("player")
        if not color or pid is None:
            continue
        color = tuple(color[:3])

        if team_centroids:
            c = np.array(color, dtype=np.float32)
            dists = {
                team_id: float(np.linalg.norm(c - centroid))
                for team_id, centroid in team_centroids.items()
            }
            team = min(dists, key=dists.get)
        elif RGB_FALLBACK_ACTIF:
            # Fallback RGB : bleu dominant → team 0, rouge dominant → team 1
            b, g, r = color
            team = 0 if b > r else 1
        else:
            continue  # pas de source fiable - laisser team=None

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
        if e.get("team") is not None:
            continue
        if pid in pid_to_team:
            e["team"] = pid_to_team[pid]

    n_assigned = sum(1 for e in events if e.get("team") is not None)
    n_total    = len(events)
    print(f"  [TEAM_CLUSTER] {n_assigned}/{n_total} events avec team "
          f"({n_via_prefixe} via préfixe, reste via {_source})")

    return events