"""
player_confidence.py
======================
V5.2 (2.4) - Score de confiance par joueur, decompose par dimension.

Principe : un score global unique ne suffit pas (un joueur peut avoir une
identite fragile mais des tirs individuels parfaitement valides). On
calcule donc plusieurs sous-scores independants, bases sur des signaux
REELS decouverts pendant l'investigation V5.1/V5.2 d'aujourd'hui - pas
des heuristiques inventees.

Signaux utilises :
  1. team_confidence   : l'identite canonique a-t-elle une equipe confirmee
                          (prefixe P{team}_{jersey}) ET les evenements
                          individuels sont-ils coherents avec ce prefixe ?
                          (decouverte du jour : P0_7 avait des evenements
                          team=1 malgre son prefixe team=0 - signal reel
                          d'incoherence, pas suppose)
  2. identity_fragmentation : combien de track_id bruts ont ete fusionnes
                          pour former cette identite (plus de fragments =
                          plus de risque de melange, meme si la fusion a
                          ete jugee "sure" par reconcile_unknown_team_fragments)
  3. stats_confidence   : le joueur a-t-il atteint le seuil minimal de
                          touches pour avoir une entree dans stats ?
                          (demontre aujourd'hui : ce seuil explique 100%
                          des cas, ni plus ni moins - donc un signal fiable,
                          pas un artefact a corriger)

Usage :
    python3 player_confidence.py --frames frames_data.pkl \\
        --jersey-map jersey_map_brut.json --analysis analysis.json
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict


def construire_team_map(frames_data):
    team_votes = defaultdict(list)
    for fd in frames_data:
        for p in fd.get("players", []):
            tid = str(p.get("id", p.get("tracker_id", "")))
            team = p.get("team")
            if tid and team is not None:
                team_votes[tid].append(team)
    return {tid: Counter(v).most_common(1)[0][0] for tid, v in team_votes.items()}


def score_team_confidence(player_id, events_du_joueur):
    """
    🟢 ELEVEE : identite prefixee (equipe connue a la fusion) ET tous les
                evenements individuels sont coherents avec ce prefixe.
    🟠 MOYENNE : identite prefixee MAIS au moins un evenement contredit
                le prefixe (incoherence reelle, comme P0_7 aujourd'hui).
    🔴 FAIBLE : identite NON prefixee - l'equipe n'a jamais ete confirmee
                avec une confiance suffisante a la fusion (numero partage
                entre les deux equipes, ou aucune equipe connue du tout).
    """
    a_prefixe = player_id.startswith("P") and "_" in player_id
    if not a_prefixe:
        return "FAIBLE", "identité jamais confirmée (numéro ambigu entre équipes, ou équipe inconnue)"

    try:
        team_prefixe = int(player_id.split("_")[0][1:])
    except (ValueError, IndexError):
        return "FAIBLE", "format d'identité non reconnu"

    teams_vus_events = set(
        e.get("team") for e in events_du_joueur
        if e.get("team") is not None
    )
    incoherents = teams_vus_events - {team_prefixe}
    if incoherents:
        return "MOYENNE", f"préfixe équipe {team_prefixe}, mais {len(incoherents)} événement(s) contredisent (team={incoherents})"

    return "ELEVEE", f"préfixe équipe {team_prefixe}, cohérent sur tous les événements"


def score_identity_fragmentation(n_events, n_fragments, seuils_percentiles):
    """
    V5.2 CORRECTIF : plus un seuil ABSOLU sur le nombre de fragments (qui
    penalisait a tort les joueurs les plus actifs - decouvert sur Raeren :
    P0_4 avec 53 events/13 fragments tombait a tort en FAIBLE, alors que
    ce ratio (4.08 events/fragment) se situe exactement au 75e percentile
    de la vraie distribution, donc pas du tout un cas extreme).

    Utilise desormais le ratio events/fragment (plus un fragment contribue
    d'evenements en moyenne, plus il represente un suivi stable et non
    un artefact) compare aux percentiles REELS de la distribution du
    match en cours - pas des chiffres fixes devines a l'avance.

    seuils_percentiles : dict {25: x, 75: y} calcule sur l'ensemble des
    joueurs du match (voir calculer_seuils_fragmentation ci-dessous).
    """
    if n_fragments <= 0:
        return "FAIBLE", "aucun fragment identifié"
    ratio = n_events / n_fragments

    p25, p75 = seuils_percentiles[25], seuils_percentiles[75]
    if ratio < p25:
        return "FAIBLE", f"{ratio:.2f} évts/fragment ({n_events}évts/{n_fragments}frag) — sous le 25e percentile ({p25:.2f}) du match"
    elif ratio <= p75:
        return "MOYENNE", f"{ratio:.2f} évts/fragment ({n_events}évts/{n_fragments}frag) — dans la médiane du match"
    else:
        return "ELEVEE", f"{ratio:.2f} évts/fragment ({n_events}évts/{n_fragments}frag) — au-dessus du 75e percentile ({p75:.2f}) du match"


def calculer_seuils_fragmentation(events_par_joueur, fragments_par_canonical):
    """
    Calcule les percentiles 25/75 du ratio events/fragment sur TOUS les
    joueurs du match en cours - auto-calibrant, pas de chiffre fixe
    transposé d'un match a l'autre.
    """
    import numpy as np
    ratios = []
    for pid, n_events in events_par_joueur.items():
        n_frag = len(fragments_par_canonical.get(pid, {pid}))
        if n_frag > 0:
            ratios.append(n_events / n_frag)
    if not ratios:
        return {25: 1.0, 75: 1.0}
    return {25: float(np.percentile(ratios, 25)), 75: float(np.percentile(ratios, 75))}


def score_stats_confidence(player_id, stats_dict):
    """
    Binaire, demontre aujourd'hui comme 100% fiable (pas d'exception
    trouvee sur 49 joueurs) : au-dessus du seuil touches -> ELEVEE,
    en dessous -> le joueur n'a simplement pas de stats du tout
    (comportement voulu de stats.py, pas un defaut a signaler comme "bas").
    """
    if player_id in stats_dict:
        return "ELEVEE", "au-dessus du seuil touches minimal"
    return "N/A", "sous le seuil touches minimal - pas de statistiques calculées"


def calculer_confiance_globale(team_conf, identity_conf):
    """
    Combine team_confidence et identity_fragmentation en un score global
    simple (le plus bas des deux domine - principe de prudence : la
    confiance globale ne peut pas dépasser le maillon le plus faible).
    stats_confidence n'entre pas dans ce calcul car "N/A" ne signifie pas
    "mauvais", juste "pas assez de données pour ce joueur sur ce match".
    """
    ordre = {"ELEVEE": 3, "MOYENNE": 2, "FAIBLE": 1}
    pire = min(ordre.get(team_conf, 1), ordre.get(identity_conf, 1))
    return {3: "ELEVEE", 2: "MOYENNE", 1: "FAIBLE"}[pire]


def analyser(frames_data, jersey_map, analysis):
    events = analysis["events"]
    stats = analysis.get("stats", {})

    events_par_joueur = defaultdict(list)
    for e in events:
        pid = e.get("player")
        if pid:
            events_par_joueur[pid].append(e)

    # Fragmentation : reconstruire combien de track_id bruts par identite
    # canonique, a partir de jersey_map (meme logique que build_identity_map)
    team_map = construire_team_map(frames_data)
    jersey_to_tids = defaultdict(set)
    for tid, jersey in jersey_map.items():
        if jersey is not None:
            jersey_to_tids[str(jersey)].add(str(tid))

    fragments_par_canonical = defaultdict(set)
    for jersey, tids in jersey_to_tids.items():
        for tid in tids:
            team = team_map.get(tid)
            canonical = f"P{team}_{jersey}" if team is not None else f"P{jersey}"
            fragments_par_canonical[canonical].add(tid)

    events_par_joueur_count = {pid: len(evts) for pid, evts in events_par_joueur.items()}
    seuils_fragmentation = calculer_seuils_fragmentation(events_par_joueur_count, fragments_par_canonical)
    print(f"  [SEUILS FRAGMENTATION] calculés sur ce match : "
          f"25e percentile={seuils_fragmentation[25]:.2f}  75e percentile={seuils_fragmentation[75]:.2f}")

    resultats = {}
    for pid, evts in events_par_joueur.items():
        team_conf, team_raison = score_team_confidence(pid, evts)
        n_frag = len(fragments_par_canonical.get(pid, {pid}))
        identity_conf, identity_raison = score_identity_fragmentation(len(evts), n_frag, seuils_fragmentation)
        stats_conf, stats_raison = score_stats_confidence(pid, stats)
        globale = calculer_confiance_globale(team_conf, identity_conf)

        resultats[pid] = {
            "confiance_globale": globale,
            "team": {"niveau": team_conf, "raison": team_raison},
            "identity": {"niveau": identity_conf, "raison": identity_raison, "n_fragments": n_frag},
            "stats": {"niveau": stats_conf, "raison": stats_raison},
            "n_events": len(evts),
        }

    return resultats


def afficher_rapport(resultats):
    ordre_affichage = {"ELEVEE": 0, "MOYENNE": 1, "FAIBLE": 2}
    symboles = {"ELEVEE": "🟢", "MOYENNE": "🟠", "FAIBLE": "🔴", "N/A": "⚪"}

    print(f"{'Joueur':<10} {'Global':<10} {'Équipe':<10} {'Identité':<10} {'Stats':<10} {'Events':>7}")
    print("=" * 70)
    for pid, r in sorted(resultats.items(),
                          key=lambda kv: (ordre_affichage[kv[1]["confiance_globale"]], -kv[1]["n_events"])):
        g = symboles[r["confiance_globale"]]
        t = symboles[r["team"]["niveau"]]
        i = symboles[r["identity"]["niveau"]]
        s = symboles[r["stats"]["niveau"]]
        print(f"{pid:<10} {g} {r['confiance_globale']:<7} {t} {r['team']['niveau']:<7} "
              f"{i} {r['identity']['niveau']:<7} {s} {r['stats']['niveau']:<7} {r['n_events']:>7}")

    dist = Counter(r["confiance_globale"] for r in resultats.values())
    print(f"\nDistribution globale : {dict(dist)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--jersey-map", required=True)
    parser.add_argument("--analysis", required=True)
    args = parser.parse_args()

    with open(args.frames, "rb") as f:
        frames_data = pickle.load(f)
    with open(args.jersey_map) as f:
        jersey_map = json.load(f)
    with open(args.analysis) as f:
        analysis = json.load(f)

    resultats = analyser(frames_data, jersey_map, analysis)
    afficher_rapport(resultats)
