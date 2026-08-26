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
            # V5.1 FIX (2e bug du meme type) : `team_map.get(str(pid)) or
            # team_map.get(pid)` echoue silencieusement quand team=0, car
            # `0 or x` evalue x, pas 0 - retombant a tort sur la seconde
            # cle. Et `if team else` traitait ensuite team=0 comme "aucune
            # equipe connue", desactivant la fusion composite pour toute
            # l'equipe 0 (41.1% des classifications reelles sur Raeren).
            team = team_map.get(str(pid))
            if team is None:
                team = team_map.get(pid)
            key = (team, str(jersey)) if team is not None else (None, str(jersey))
            jersey_to_ids[key].add(str(pid))

    identity_map = {}
    for key, ids in jersey_to_ids.items():
        team, jersey = key
        canonical_id = f"P{team}_{jersey}" if team is not None else f"P{jersey}"
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
# RECONCILE UNKNOWN-TEAM FRAGMENTS (V5.2)
# ─────────────────────────────────────────
def reconcile_unknown_team_fragments(identity_map, jersey_map, team_map, frames_data):
    """
    Tente de fusionner les fragments "equipe inconnue" (ex: P4) dans le
    fragment "equipe connue" correspondant (ex: P1_4) - UNIQUEMENT quand
    c'est sûr.

    ATTENTION DE SECURITE (decouverte sur Raeren) : un numero de maillot
    peut être confirmé sur LES DEUX équipes a la fois (amateur : 1-11 des
    deux cotes). Sur Raeren, 8 numeros/17 avec equipe connue etaient dans
    ce cas (2,4,5,6,7,8,9,17) - fusionner aveuglement "meme numero, equipe
    inconnue" dans "meme numero, UNE equipe connue" aurait ete FAUX pour
    la majorite de ces cas (P4 est confirme majoritairement equipe 0, pas
    equipe 1 - une fusion naive dans P1_4 aurait aggrave le probleme).

    Regle stricte : on ne fusionne QUE si le numero de maillot est confirmé
    sur UNE SEULE équipe parmi tous les track_id connus qui le portent.
    Si le numero est confirmé sur les deux équipes (ou si aucune équipe
    n'est connue du tout), le fragment "équipe inconnue" reste séparé et
    marqué incertain - conformement au principe V5.2 : conserver
    l'incertitude plutôt que fabriquer une identité.

    En plus de cette condition, verifie l'absence de chevauchement
    temporel (test de presence simultanee, meme logique que
    audit_identite_joueurs.py) avant de fusionner, par securite
    supplementaire.

    Retourne : identity_map mis à jour (copie), avec un rapport imprimé.
    """
    identity_map = dict(identity_map)

    # 1. Regrouper les track_id par numero de maillot
    jersey_to_tids = {}
    for pid, jersey in jersey_map.items():
        if jersey is None:
            continue
        jersey_to_tids.setdefault(str(jersey), set()).add(str(pid))

    # 2. Pour chaque numero, verifier s'il est confirme sur UNE SEULE equipe
    fusions_appliquees = []
    fusions_refusees_ambigu = []
    for jersey, tids in jersey_to_tids.items():
        equipes_connues = set()
        tids_connus, tids_inconnus = [], []
        for tid in tids:
            team = team_map.get(tid)
            if team is not None:
                equipes_connues.add(team)
                tids_connus.append(tid)
            else:
                tids_inconnus.append(tid)

        if not tids_inconnus or not tids_connus:
            continue  # rien a reconcilier

        if len(equipes_connues) != 1:
            # Numero ambigu (0 ou 2+ equipes confirmees) - ON NE FUSIONNE PAS
            fusions_refusees_ambigu.append((jersey, sorted(equipes_connues), tids_inconnus))
            continue

        equipe_unique = next(iter(equipes_connues))
        canonical_connu = f"P{equipe_unique}_{jersey}"
        canonical_inconnu = f"P{jersey}"

        # 3. Verification de securite supplementaire : pas de chevauchement
        # temporel entre le groupe "connu" et le groupe "inconnu"
        frames_connu, frames_inconnu = set(), set()
        for fd in frames_data:
            frame_id = fd.get("frame")
            for p in fd.get("players", []):
                tid = str(p.get("id", p.get("tracker_id", "")))
                if tid in tids_connus:
                    frames_connu.add(frame_id)
                elif tid in tids_inconnus:
                    frames_inconnu.add(frame_id)

        chevauchement = frames_connu & frames_inconnu
        if chevauchement:
            fusions_refusees_ambigu.append((jersey, [equipe_unique], tids_inconnus,
                                             f"chevauchement sur {len(chevauchement)} frames"))
            continue

        # Sûr : fusionner le fragment "inconnu" dans le fragment "connu"
        for tid in tids_inconnus:
            identity_map[tid] = canonical_connu
        identity_map[canonical_inconnu] = canonical_connu  # au cas ou canonical_inconnu est deja utilise comme cle
        fusions_appliquees.append((jersey, equipe_unique, len(tids_inconnus)))

    print(f"  [RECONCILE] {len(fusions_appliquees)} numéro(s) réconcilié(s) "
          f"(équipe unique confirmée + pas de chevauchement) :")
    for jersey, equipe, n in fusions_appliquees:
        print(f"    #{jersey} → équipe {equipe} ({n} track_id inconnus fusionnés)")
    print(f"  [RECONCILE] {len(fusions_refusees_ambigu)} numéro(s) LAISSÉ(S) SÉPARÉ(S) "
          f"(ambigu ou chevauchement — incertitude conservée) :")
    for item in fusions_refusees_ambigu:
        jersey, equipes = item[0], item[1]
        raison = item[3] if len(item) > 3 else f"{len(equipes)} équipe(s) confirmée(s)"
        print(f"    #{jersey} : {raison}")

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
def resolve_player_identities(events, jersey_map, team_map=None, frames_data=None):
    """
    Effectue toute la résolution d'identité en une seule fonction.
    Retourne (events, jersey_map) mis à jour.

    team_map optionnel ({track_id: equipe}) - voir build_identity_map()
    pour la raison de son importance (evite les fusions inter-equipes).

    frames_data optionnel : si fourni (avec team_map), tente une
    reconciliation SÛRE des fragments "équipe inconnue" (ex: P4) dans
    leur fragment "équipe connue" correspondant (ex: P1_4) - uniquement
    quand le numéro de maillot est confirmé sur une seule équipe et sans
    chevauchement temporel. Voir reconcile_unknown_team_fragments().
    """
    identity_map = build_identity_map(events, jersey_map, team_map=team_map)
    if not identity_map:
        return events, jersey_map

    if frames_data and team_map:
        identity_map = reconcile_unknown_team_fragments(
            identity_map, jersey_map, team_map, frames_data
        )

    events     = apply_identity_map(events, identity_map)
    jersey_map = build_final_jersey_map(identity_map, jersey_map)

    return events, jersey_map