# analysis/player_identity.py
# -*- coding: utf-8 -*-
#
# Résolution d'identité joueur
# Fusionne les IDs fragmentés vers un ID canonique basé sur le numéro de maillot
#
# Problème résolu :
#   ID-12 → #3        ┐
#   ID-257 → #3       ├── fusionnés en P3
#   ID-89  → #3       ┘
#
# Résultat :
#   tous les events → player = "P3"
#   jersey_map["P3"] = 3

from collections import defaultdict


# ─────────────────────────────────────────
# BUILD IDENTITY MAP
# ─────────────────────────────────────────
def build_identity_map(events, jersey_map, team_map=None):
    """
    Construit un mapping {old_id → canonical_id} basé sur le numéro de maillot.

    Logique :
      1. Regrouper tous les IDs qui partagent le même numéro de maillot
      2. Créer un ID canonique "P{jersey}" pour chaque groupe
      3. Les IDs sans maillot gardent leur ID original

    IMPORTANT (V5.1) : si team_map est fourni ({track_id: "team_a"/"team_b"/...}),
    la fusion se fait sur la paire (équipe, maillot) plutôt que sur le maillot
    seul. Un match de foot a deux équipes qui partagent presque toujours les
    mêmes petits numéros (1-11) — fusionner sur le numéro seul mélange à tort
    deux joueurs réels de camps différents sous un seul ID (démontré : 725
    fusions erronées prouvées sur un match de test, concentrées sur les
    numéros 1-9, quasi absentes sur les numéros à deux chiffres rares à une
    seule équipe — cf. audit_identite_joueurs.py). Sans team_map, l'ancien
    comportement (fusion sur maillot seul) est conservé pour compatibilité,
    mais reste vulnérable à ce même problème.

    Retourne : identity_map {old_id → canonical_id}
    """
    team_map = team_map or {}
    jersey_to_ids = defaultdict(set)
    for pid, jersey in jersey_map.items():
        if jersey is not None and str(jersey).strip():
            team = team_map.get(str(pid)) or team_map.get(pid)
            key = (team, str(jersey)) if team else (None, str(jersey))
            jersey_to_ids[key].add(str(pid))

    identity_map = {}
    for key, ids in jersey_to_ids.items():
        team, jersey = key
        canonical_id = f"P{team}_{jersey}" if team else f"P{jersey}"
        for pid in ids:
            if pid != canonical_id:
                identity_map[pid] = canonical_id

    n_merged = sum(1 for ids in jersey_to_ids.values() if len(ids) > 1)
    if identity_map:
        mode = "clé=(équipe,maillot)" if team_map else "clé=maillot seul — team_map absent, risque de fusion inter-équipe"
        print(f"  Identity : {len(identity_map)} IDs → {len(jersey_to_ids)} joueurs "
              f"({n_merged} fusions, {mode})")

    return identity_map


# ─────────────────────────────────────────
# APPLY IDENTITY MAP
# ─────────────────────────────────────────
def apply_identity_map(events, identity_map):
    """Remplace les player IDs dans tous les events."""
    if not identity_map:
        return events

    replaced = 0
    for e in events:
        pid = str(e.get("player", ""))
        if pid in identity_map:
            e["player"] = identity_map[pid]
            replaced += 1

    if replaced:
        print(f"  Identity : {replaced} events remappés")

    return events


# ─────────────────────────────────────────
# BUILD FINAL JERSEY MAP
# ─────────────────────────────────────────
def build_final_jersey_map(identity_map, jersey_map):
    """
    Crée un jersey_map propre avec les IDs canoniques.
    Avant : {"12": 3, "257": 3} → Après : {"P3": 3}
    """
    if not identity_map:
        return jersey_map

    final_map = {}

    for old_pid, jersey in jersey_map.items():
        new_pid = identity_map.get(str(old_pid), str(old_pid))
        if jersey is not None:
            final_map[new_pid] = jersey

    # Garder les entrées non remappées
    for pid, jersey in jersey_map.items():
        if str(pid) not in identity_map and jersey is not None:
            final_map[str(pid)] = jersey

    return final_map


# ─────────────────────────────────────────
# GET PLAYER LABEL — fallback propre
# JAMAIS afficher ID-257 ou ID-12
# ─────────────────────────────────────────
def get_player_label(pid, jersey_map):
    """
    Retourne un label lisible pour un joueur.
    Priorité : #numéro > #canonique > Joueur X
    Jamais : "ID-257", "ID-12"
    """
    pid = str(pid)

    jersey = jersey_map.get(pid)
    if jersey is not None:
        return f"#{jersey}"

    # ID canonique type "P3" — NE PAS convertir en numéro de maillot
    # P4 ne signifie PAS #4 — c'est le 4ème tracker DeepSort
    # Sans lecture Gemini confirmée, le numéro est inconnu
    return "?"


# ─────────────────────────────────────────
# RESOLVE ALL — point d'entrée unique
# ─────────────────────────────────────────
def resolve_player_identities(events, jersey_map, team_map=None):
    """
    Effectue toute la résolution d'identité en une seule fonction.
    Retourne (events, jersey_map) mis à jour.

    team_map optionnel ({track_id: equipe}) - voir build_identity_map()
    pour la raison de son importance (evite les fusions inter-equipes).
    """
    identity_map = build_identity_map(events, jersey_map, team_map=team_map)
    if not identity_map:
        return events, jersey_map

    events     = apply_identity_map(events, identity_map)
    jersey_map = build_final_jersey_map(identity_map, jersey_map)

    return events, jersey_map