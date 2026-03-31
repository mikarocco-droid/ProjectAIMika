# ai/commentary.py
# -*- coding: utf-8 -*-

import random


# Commentaires par type d'événement
_COMMENTARY = {
    "goal": [
        "INCROYABLE BUT !!!",
        "QUELLE FINITION !",
        "MAGNIFIQUE ACTION COLLECTIVE !",
        "IL NE POUVAIT PAS MIEUX PLACER !",
        "LE GARDIEN N'A PU QUE REGARDER PASSER !"
    ],
    "score": [
        "BUT VALIDÉ !",
        "ÇA RENTRE !",
        "IL FAIT 1-0 !"
    ],
    "shot": [
        "Frappe dangereuse !",
        "Beau tir, repoussé par le gardien !",
        "Juste à côté !",
        "Quelle occasion !"
    ],
    "fast_break": [
        "Contre-attaque fulgurante !",
        "Transition rapide, danger imminent !",
        "Ils partent en contre !"
    ],
    "interception": [
        "Bonne récupération !",
        "Il coupe la trajectoire !",
        "Excellente lecture du jeu !"
    ],
    "dribble": [
        "Super dribble !",
        "Il élimine son adversaire !",
        "Quelle technique !"
    ],
    "progressive_run": [
        "Percée intéressante !",
        "Il progresse balle au pied !"
    ],
    "key_pass": [
        "Passe décisive !",
        "Quelle vision du jeu !"
    ],
    "build_up": [
        "Belle construction collective.",
        "Le jeu se développe proprement."
    ],
}

_DEFAULT = [
    "Action intéressante.",
    "Le jeu se poursuit.",
    "Belle phase de jeu."
]


def generate_commentary(events):
    """
    Génère une ligne de commentaire pour chaque event.
    Retourne une liste de strings.
    """
    lines = []

    for e in events:
        pool = _COMMENTARY.get(e.get("type", ""), _DEFAULT)
        lines.append(random.choice(pool))

    return lines