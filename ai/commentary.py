# ai/commentary.py
# -*- coding: utf-8 -*-

import random

def generate_commentary(events):

    lines = []

    for e in events:

        if e["type"] == "goal":
            lines.append(random.choice([
                "INCROYABLE BUT !!!",
                "QUELLE FINITION !",
                "MAGNIFIQUE ACTION COLLECTIVE !"
            ]))

        elif e["type"] == "shot":
            lines.append("Frappe dangereuse !")

        elif e["type"] == "fast_break":
            lines.append("Contre-attaque fulgurante !")

        elif e["type"] == "interception":
            lines.append("Bonne récupération !")

        elif e["type"] == "dribble":
            lines.append("Super dribble !")

    return lines