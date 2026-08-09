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


def _players_with_team(frame_data, frame_w=None, frame_h=None):
    """Extrait les joueurs d'une frame avec leur team (0/1/None) et leur centre.
    Filtre les détections dont la position est hors du cadre réel de l'image
    (traces fantômes du tracker après occlusion prolongée, cf. diagnostic
    Andrimont t=200s : coordonnées jusqu'à x=2774 sur une image de 960px).

    Filtre AUSSI les joueurs non fraîchement détectés cette frame
    (time_since_update != 0 : position purement prédite par Kalman après
    perte de la détection). Important : team=None peut désormais signifier
    soit "arbitre/hors équipe" (ce qu'on veut détecter), soit "couleur pas
    évaluée car détection pas fraîche" (à ne pas confondre avec le premier
    cas, sinon un joueur en dérive pourrait être pris à tort pour
    l'arbitre)."""
    fw = frame_w or frame_data.get("frame_w")
    fh = frame_h or frame_data.get("frame_h")
    players = []
    for p in (frame_data.get("players") or []):
        if p.get("time_since_update", 0) != 0:
            continue  # position pas fraîche -> exclu, pas mis dans "hors_equipe"
        center = p.get("center")
        if center is None:
            continue
        x, y = center
        if fw is not None and fh is not None:
            if not (0 <= x <= fw and 0 <= y <= fh):
                continue  # position impossible -> trace fantôme, on l'ignore
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


def _team_spread(players):
    """Dispersion moyenne d'une équipe autour de son propre centroïde
    (distance moyenne de chaque joueur au centre du groupe)."""
    if len(players) < 3:
        return None
    positions = np.array([p["center"] for p in players])
    centroid = positions.mean(axis=0)
    return float(np.mean(np.linalg.norm(positions - centroid, axis=1)))


def find_C_candidates(frames_data, fps, max_group_size=6, min_isolation_ratio=2.0,
                       compact_percentile=25.0):
    """
    Scanne toutes les frames_data et retourne les instants où le motif C
    est détecté, avec un critère à DEUX contraintes (pas une seule) :
      1. Les deux équipes sont elles-mêmes compactes à cet instant
         (dispersion interne <= compact_percentile de la distribution du
         MATCH LUI-MÊME, pas un seuil fixe en pixels — chaque match a sa
         propre échelle caméra) ;
      2. ET il existe un petit groupe isolé (2-6 pers.) avec au moins une
         personne hors équipe, nettement séparé des deux masses.
    Sans la contrainte 1, un simple sous-groupe compact d'une équipe en
    jeu normal (ex: ligne défensive resserrée) peut être confondu avec C
    (cf. diagnostic Andrimont t=200s : team0=22/team1=4, faux positif).
    """
    # Passe 1 : calcul de la dispersion de chaque équipe à chaque instant
    raw = []
    for fd in frames_data:
        t = fd.get("frame", 0) / fps
        players = _players_with_team(fd, fd.get("frame_w"), fd.get("frame_h"))
        team0 = [p for p in players if p["team"] == 0]
        team1 = [p for p in players if p["team"] == 1]
        hors_equipe = [p for p in players if p["team"] not in (0, 1)]
        raw.append({
            "t": t, "team0": team0, "team1": team1, "hors_equipe": hors_equipe,
            "spread0": _team_spread(team0), "spread1": _team_spread(team1),
        })

    spreads0 = [r["spread0"] for r in raw if r["spread0"] is not None]
    spreads1 = [r["spread1"] for r in raw if r["spread1"] is not None]
    if not spreads0 or not spreads1:
        return []
    seuil0 = np.percentile(spreads0, compact_percentile)
    seuil1 = np.percentile(spreads1, compact_percentile)

    # Passe 2 : application des deux contraintes
    results = []
    for r in raw:
        if r["spread0"] is None or r["spread1"] is None:
            continue
        if r["spread0"] > seuil0 or r["spread1"] > seuil1:
            continue  # au moins une équipe n'est pas compacte -> pas C
        if not r["hors_equipe"]:
            continue
        if len(r["team0"]) < 2 or len(r["team1"]) < 2:
            continue

        found, info = _cluster_isolated_group(
            r["hors_equipe"], r["team0"], r["team1"], max_group_size, min_isolation_ratio
        )
        if found:
            results.append({"t": r["t"], "spread0": round(r["spread0"],1),
                             "spread1": round(r["spread1"],1), **info})

    return results


if __name__ == "__main__":
    print(__doc__)
    print("Module à importer dans un notebook Kaggle avec frames_data en mémoire.")
