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
def build_identity_map(events, jersey_map):
    """
    Construit un mapping {old_id → canonical_id} basé sur le numéro de maillot.

    Logique :
      1. Regrouper tous les IDs qui partagent le même numéro de maillot
      2. Créer un ID canonique "P{jersey}" pour chaque groupe
      3. Les IDs sans maillot gardent leur ID original

    Retourne : identity_map {old_id → canonical_id}
    """
    jersey_to_ids = defaultdict(set)
    for pid, jersey in jersey_map.items():
        if jersey is not None and str(jersey).strip():
            jersey_to_ids[str(jersey)].add(str(pid))

    identity_map = {}
    for jersey, ids in jersey_to_ids.items():
        canonical_id = f"P{jersey}"
        for pid in ids:
            if pid != canonical_id:
                identity_map[pid] = canonical_id

    n_merged = sum(1 for ids in jersey_to_ids.values() if len(ids) > 1)
    if identity_map:
        print(f"  Identity : {len(identity_map)} IDs → {len(jersey_to_ids)} joueurs "
              f"({n_merged} fusions)")

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

    # ID canonique type "P3" → "#3"
    if pid.startswith("P") and pid[1:].isdigit():
        return f"#{pid[1:]}"

    # Fallback court propre
    short = pid[:4] if len(pid) > 4 else pid
    return f"Joueur {short}"


# ─────────────────────────────────────────
# RESOLVE ALL — point d'entrée unique
# ─────────────────────────────────────────
def resolve_player_identities(events, jersey_map):
    """
    Effectue toute la résolution d'identité en une seule fonction.
    Retourne (events, jersey_map) mis à jour.
    """
    identity_map = build_identity_map(events, jersey_map)
    if not identity_map:
        return events, jersey_map

    events     = apply_identity_map(events, identity_map)
    jersey_map = build_final_jersey_map(identity_map, jersey_map)

    return events, jersey_map