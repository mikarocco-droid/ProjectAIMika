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


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION FIN DE MATCH
# ─────────────────────────────────────────────────────────────────────────────

def find_match_end(frames_data, fps=25.0, team_colors=None,
                   silence_threshold_s=1500.0, min_match_duration_s=2700.0):
    """
    Trouve le timestamp de fin du match en analysant les frames_data du tracking.

    Principe :
    - On scanne les frames_data toutes les ~30s
    - On considère qu'il y a du "vrai jeu" quand des joueurs avec vareuses
      d'équipe sont détectés (couleurs team0/team1 calibrées)
    - On garde le dernier timestamp avec du vrai jeu
    - Si pendant 25 min consécutives aucun vrai jeu → fin du match
    - Prolongation et tirs au but sont couverts car le seuil est de 25 min

    Paramètres
    ----------
    frames_data         : liste de dicts du pipeline (players, ball, frame, fps)
    fps                 : FPS de la vidéo
    team_colors         : dict {0: (B,G,R), 1: (B,G,R)} couleurs équipes calibrées
                          Si None, on utilise juste la présence de joueurs
    silence_threshold_s : durée sans jeu pour conclure fin de match (défaut 25 min)
    min_match_duration_s: durée minimale de match avant de chercher la fin (défaut 45 min)

    Retourne
    --------
    match_end_s : float — timestamp de fin du match en secondes
                  None si non détecté (on garde toute la vidéo)
    """
    if not frames_data:
        return None

    import numpy as np

    # Scan toutes les ~30s pour la rapidité
    scan_interval_frames = int(30.0 * fps)

    last_real_game_time  = 0.0
    last_real_game_frame = 0
    found_any_game       = False

    # Construire un index rapide frame → fd
    # On ne garde qu'une frame toutes les 30s
    sampled = []
    prev_frame = -scan_interval_frames
    for fd in frames_data:
        f = int(fd.get("frame", 0) or 0)
        if f - prev_frame >= scan_interval_frames:
            sampled.append(fd)
            prev_frame = f

    for fd in sampled:
        f   = int(fd.get("frame", 0) or 0)
        t   = f / max(fps, 1)
        players = fd.get("players", [])

        # Compter les joueurs avec couleur d'équipe reconnue
        team_players = 0
        for p in players:
            team = p.get("team")
            if team is not None:
                team_players += 1
            elif team_colors and p.get("color") is not None:
                # Vérifier si la couleur correspond à une équipe
                color = np.array(p["color"], dtype=np.float32)
                for tid, tc in team_colors.items():
                    tc_arr = np.array(tc, dtype=np.float32)
                    dist = float(np.linalg.norm(color - tc_arr))
                    if dist < 60:
                        team_players += 1
                        break

        # "Vrai jeu" = au moins 4 joueurs avec vareuses d'équipe détectés
        # (les gamins sans vareuse ne seront pas assignés à une équipe)
        if team_players >= 4:
            last_real_game_time  = t
            last_real_game_frame = f
            found_any_game       = True

            # Vérifier si on a dépassé la durée minimale de match
            # avant de commencer à surveiller la fin
            if t < min_match_duration_s:
                continue

        # Si on a trouvé du jeu ET qu'on dépasse le seuil de silence
        if found_any_game and t > min_match_duration_s:
            silence = t - last_real_game_time
            if silence > silence_threshold_s:
                mm = int(last_real_game_time // 60)
                ss = int(last_real_game_time % 60)
                print(f"  [MATCH_END] ✅ Fin du match détectée à {mm:02d}:{ss:02d} "
                      f"(silence={silence/60:.1f} min > seuil={silence_threshold_s/60:.0f} min)")
                return last_real_game_time

    if found_any_game and last_real_game_time > 0:
        # Fin de vidéo atteinte — vérifier si on a du silence en fin
        total_time = frames_data[-1].get("frame", 0) / max(fps, 1)
        silence    = total_time - last_real_game_time
        if silence > silence_threshold_s and last_real_game_time > min_match_duration_s:
            mm = int(last_real_game_time // 60)
            ss = int(last_real_game_time % 60)
            print(f"  [MATCH_END] ✅ Fin du match détectée à {mm:02d}:{ss:02d} "
                  f"(silence fin vidéo={silence/60:.1f} min)")
            return last_real_game_time

    print(f"  [MATCH_END] Pas de fin de match détectée → vidéo utilisée entièrement")
    return None


def apply_match_end(events, frames_data, match_end_s, fps=25.0):
    """
    Supprime les events et frames après la fin du match.

    Paramètres
    ----------
    events      : liste d'events du pipeline
    frames_data : liste de frames_data du pipeline
    match_end_s : timestamp de fin du match en secondes
    fps         : FPS de la vidéo

    Retourne
    --------
    events_trimmed, frames_data_trimmed
    """
    if match_end_s is None or match_end_s <= 0:
        return events, frames_data

    # Ajouter une marge de 60s après la fin détectée
    # (pour capturer les célébrations et le coup de sifflet final)
    cutoff_s     = match_end_s + 60.0
    cutoff_frame = int(cutoff_s * fps)

    events_trimmed = [
        e for e in events
        if float(e.get("time", 0) or 0) <= cutoff_s
    ]
    frames_trimmed = [
        fd for fd in frames_data
        if int(fd.get("frame", 0) or 0) <= cutoff_frame
    ]

    n_ev  = len(events) - len(events_trimmed)
    n_fd  = len(frames_data) - len(frames_trimmed)
    mm    = int(match_end_s // 60)
    ss    = int(match_end_s % 60)

    if n_ev > 0 or n_fd > 0:
        print(f"  [MATCH_END] Après-match supprimé : {n_ev} events + "
              f"{n_fd} frames (après {mm:02d}:{ss:02d} + 60s marge)")

    return events_trimmed, frames_trimmed