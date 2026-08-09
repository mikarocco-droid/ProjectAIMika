"""
detect_c_capitaines.py
=========================
Détecteur de l'étape C de la séquence pré-KO (arbitre(s) + 2 capitaines,
choix de camp) — voir JALON_10_MATCHS_ET_CONTRAT_FONCTIONNEL.md
section 10-11 pour la définition et sa vérification visuelle sur 9 matchs.

Ne modifie PAS ko_features.py (gelé). Module séparé, comme
candidates_mining.py.

Principe : chaque joueur détecté par le pipeline a déjà un champ "team"
(0, 1, ou une valeur absente/None si sa couleur ne correspond à aucune
des deux équipes calibrées par TeamColorDetector — typiquement l'arbitre
ou un gardien). On cherche un petit groupe isolé (2-6 personnes)
contenant au moins un joueur "sans équipe", nettement séparé des deux
masses de joueurs de champ.

Définition retenue (validée par vérification visuelle sur 9 matchs) :
  C = petit groupe (2-6 pers.) + au moins 1 personne hors équipe A/B
      + séparation nette des deux masses de joueurs.
"""

import numpy as np


def _players_with_team(frame_data):
    """Extrait les joueurs d'une frame avec leur team (0/1/None) et leur centre."""
    players = []
    for p in (frame_data.get("players") or []):
        center = p.get("center")
        if center is None:
            continue
        team = p.get("team")
        players.append({"team": team, "center": center})
    return players


def _cluster_isolated_group(hors_equipe, team0, team1, max_group_size=6,
                              min_isolation_ratio=2.0):
    """
    Pour chaque joueur "hors équipe", regarde s'il existe un petit groupe
    (lui + ses voisins les plus proches, jusqu'à max_group_size) dont la
    distance moyenne interne est nettement plus petite que sa distance
    aux centroïdes des deux masses d'équipe (ratio >= min_isolation_ratio).
    Retourne True si un tel groupe est trouvé.
    """
    if not hors_equipe or (not team0 and not team1):
        return False, None

    centroid0 = np.mean([p["center"] for p in team0], axis=0) if team0 else None
    centroid1 = np.mean([p["center"] for p in team1], axis=0) if team1 else None

    all_others = team0 + team1
    all_positions = np.array([p["center"] for p in all_others]) if all_others else np.empty((0, 2))

    for ref_player in hors_equipe:
        ref_pos = np.array(ref_player["center"])

        # distance aux centroïdes des deux équipes (mesure d'isolement)
        dists_centroids = []
        if centroid0 is not None:
            dists_centroids.append(np.linalg.norm(ref_pos - centroid0))
        if centroid1 is not None:
            dists_centroids.append(np.linalg.norm(ref_pos - centroid1))
        dist_min_equipe = min(dists_centroids) if dists_centroids else None
        if dist_min_equipe is None:
            continue

        # les N joueurs (toutes couleurs) les plus proches de ref_player
        if len(all_positions) > 0:
            dists_all = np.linalg.norm(all_positions - ref_pos, axis=1)
            order = np.argsort(dists_all)
        else:
            order = []

        # construit un petit groupe : ref_player + voisins proches (hors équipe ou non)
        group_positions = [ref_pos]
        for idx in order:
            if len(group_positions) >= max_group_size:
                break
            # n'ajoute que des voisins nettement plus proches que la distance aux équipes
            if dists_all[idx] < dist_min_equipe / min_isolation_ratio:
                group_positions.append(all_positions[idx])

        if len(group_positions) < 2:
            continue  # groupe trop petit (juste le joueur hors équipe, isolé seul)

        group_arr = np.array(group_positions)
        group_spread = np.mean(np.linalg.norm(group_arr - group_arr.mean(axis=0), axis=1))

        if group_spread > 0 and dist_min_equipe / group_spread >= min_isolation_ratio:
            return True, {
                "taille_groupe": len(group_positions),
                "centre_groupe": group_arr.mean(axis=0).tolist(),
                "isolement_ratio": round(dist_min_equipe / group_spread, 2),
            }

    return False, None


def find_C_candidates(frames_data, fps, max_group_size=6, min_isolation_ratio=2.0):
    """
    Scanne toutes les frames_data et retourne les instants où le motif C
    (petit groupe isolé avec au moins 1 personne hors équipe) est détecté.

    Retourne une liste de dicts : {"t":, "taille_groupe":, "isolement_ratio":, "centre_groupe":}
    """
    results = []
    for fd in frames_data:
        t = fd.get("frame", 0) / fps
        players = _players_with_team(fd)
        if len(players) < 2:
            continue

        team0 = [p for p in players if p["team"] == 0]
        team1 = [p for p in players if p["team"] == 1]
        hors_equipe = [p for p in players if p["team"] not in (0, 1)]

        if not hors_equipe:
            continue
        if len(team0) < 2 or len(team1) < 2:
            continue  # pas assez de joueurs des 2 équipes pour juger l'isolement

        found, info = _cluster_isolated_group(
            hors_equipe, team0, team1, max_group_size, min_isolation_ratio
        )
        if found:
            results.append({"t": t, **info})

    return results


if __name__ == "__main__":
    print(__doc__)
    print("Module à importer dans un notebook Kaggle avec frames_data en mémoire.")
