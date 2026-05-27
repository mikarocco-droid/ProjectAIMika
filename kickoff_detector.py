# kickoff_detector.py
# -*- coding: utf-8 -*-
#
# Module de correction des timestamps au coup d'envoi.
#
# La DÉTECTION du coup d'envoi se fait dans terminal_events.py pendant
# le tracking YOLO — events de type "kickoff" sont émis quand le ballon
# est au rond central avec joueurs symétriques.
#
# Ce module fournit uniquement les fonctions d'APPLICATION de l'offset
# sur les events, frames_data, jersey_map et ball_tracker.

import logging
logger = logging.getLogger(__name__)


def find_kickoff_offset(events, video_duration_s=None):
    """
    Cherche le premier event de type 'kickoff' dans les events du pipeline.

    Retourne (kickoff_time_s, confidence) ou (0.0, 0.0) si non trouvé.

    Règles :
    - Si kickoff détecté dans les 60 premières secondes → offset=0
      (match qui commence dès le début, pas de pré-match)
    - Si kickoff détecté après 50% de la durée → ignoré
      (probablement une remise en jeu après but, pas le coup d'envoi initial)
    - Sinon → offset = kickoff_time
    """
    kickoff_events = [
        e for e in events
        if e.get("type") == "kickoff" or e.get("terminal_type") == "kickoff"
    ]

    if not kickoff_events:
        return 0.0, 0.0

    # Prendre le premier kickoff chronologiquement
    kickoff_events.sort(key=lambda e: e.get("time", 0))
    first = kickoff_events[0]
    t     = float(first.get("time", 0))
    conf  = float(first.get("confidence", 0.7))

    # Kickoff trop tôt = match commence dès le début
    if t < 60.0:
        print(f"  [KICKOFF] Kickoff détecté à {int(t//60):02d}:{int(t%60):02d} < 60s "
              f"→ match commence dès le début → offset=0s")
        return 0.0, 0.0

    # Kickoff trop tard = remise en jeu après but ou mi-temps
    if video_duration_s and t > video_duration_s * 0.50:
        print(f"  [KICKOFF] Kickoff à {int(t//60):02d}:{int(t%60):02d} > 50% durée vidéo "
              f"→ remise en jeu, ignoré → offset=0s")
        return 0.0, 0.0

    mm = int(t // 60)
    ss = int(t % 60)
    print(f"  [KICKOFF] ✅ Coup d'envoi détecté à {mm:02d}:{ss:02d} "
          f"→ t=0 (conf={conf:.2f})")
    print(f"  [KICKOFF] Timestamps corrigés : t_video - {t:.0f}s")
    return t, conf


def apply_kickoff_offset(events, kickoff_offset_s, fps=25.0):
    """
    Soustrait kickoff_offset_s de tous les timestamps.
    Supprime les events avant le coup d'envoi (temps < -5s).
    """
    if kickoff_offset_s <= 0:
        return events, 0

    adjusted, n_removed = [], 0
    for e in events:
        t_adj = float(e.get("time", 0) or 0) - kickoff_offset_s
        if t_adj < -5.0:
            n_removed += 1
            continue
        e = dict(e)
        e["time"] = max(0.0, t_adj)
        if "frame" in e:
            e["frame"] = max(0, int(e["frame"]) - int(kickoff_offset_s * fps))
        adjusted.append(e)

    if n_removed > 0:
        print(f"  [KICKOFF] {n_removed} events supprimés (échauffement avant coup d'envoi)")
    return adjusted, n_removed


def apply_kickoff_offset_frames(frames_data, kickoff_offset_s, fps=25.0):
    """
    Filtre frames_data pour ne garder que les frames après le coup d'envoi.
    Soustrait l'offset des numéros de frame.
    """
    if kickoff_offset_s <= 0:
        return frames_data

    cutoff_frame = int(kickoff_offset_s * fps)
    filtered     = []
    for fd in frames_data:
        f = int(fd.get("frame", 0) or 0)
        if f < cutoff_frame:
            continue
        fd = dict(fd)
        fd["frame"] = f - cutoff_frame
        filtered.append(fd)

    print(f"  [KICKOFF] frames_data : {len(frames_data)} → {len(filtered)} "
          f"({len(frames_data)-len(filtered)} frames pré-match retirées)")
    return filtered


def reset_pre_kickoff_state(jersey_map, kickoff_offset_s, fps=25.0):
    """
    Nettoie le jersey_map des joueurs vus uniquement pendant l'échauffement.
    Retourne un jersey_map nettoyé.

    Note : le ball_tracker et DeepSort n'ont pas besoin d'être réinitialisés
    explicitement car leurs états historiques sont liés aux frames_data
    qui sont déjà filtrées par apply_kickoff_offset_frames.
    """
    if kickoff_offset_s <= 0 or not jersey_map:
        return jersey_map

    # Pour l'instant on garde tout le jersey_map — les numéros lus pendant
    # l'échauffement sont quand même valides (même joueurs en match).
    # On pourrait filtrer par "dernière vue avant kickoff" mais c'est risqué
    # de perdre des identifications valides.
    print(f"  [KICKOFF] jersey_map conservé ({len(jersey_map)} joueurs) "
          f"— numéros valides même pendant l'échauffement")
    return jersey_map
