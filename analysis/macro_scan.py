# analysis/macro_scan.py
# -*- coding: utf-8 -*-
#
# V5.2 (12/09/2026) — Chantier A : POC macro-scan / micro-scan
#
# Remplace (à terme) la détection de buts par trajectoire de ballon
# (analysis/terminal_events.py + find_goal_after_shot/_v2 dans
# ai/gemini_validator.py) — voir ANALYSE_NOUVELLE_ARCHITECTURE_DETECTION.md
# pour le contexte complet et les métriques de succès attendues.
#
# Principe : le macro-scan NE cherche PAS l'action elle-même (un tir/
# une passe peut durer 1-2s et passer entre deux échantillons) — il
# repère des ZONES D'INTÉRÊT à examiner plus finement ensuite
# (micro-scan, pas encore implémenté dans ce POC).
#
# Ce module est un PREMIER TEST, pas une intégration dans le pipeline
# principal — objectif : mesurer si le macro-scan détecte correctement
# les reprises au centre (RESTART_KO) sur un vrai match, avant d'aller
# plus loin.

import os
import json
import time
import re

STATES = ["NORMAL", "ATTACK", "CELEBRATION", "RESTART_KO", "DEAD_BALL"]

PROMPT_MACRO_SCAN = """Analyse cette image d'un match de football amateur/semi-pro.
Reponds UNIQUEMENT avec un objet JSON, sans aucun texte avant ou apres, sans balises markdown.

Etats possibles (choisis EXACTEMENT un seul) :
- "NORMAL" : jeu en cours, rien de particulier a signaler
- "ATTACK" : action offensive dangereuse, ballon proche de la surface de reparation adverse
- "CELEBRATION" : joueurs celebrant clairement (bras leves, embrassades, course de joie, groupe de joueurs qui se rassemble en euphorie)
- "RESTART_KO" : UNIQUEMENT un vrai coup d'envoi au centre du terrain (voir criteres stricts ci-dessous)
- "DEAD_BALL" : arret de jeu (touche, corner, coup franc, faute, joueur au sol, arbitre qui intervient) — INCLUT toute remise en jeu qui se joue pres du centre mais qui n'est PAS un vrai coup d'envoi (ex. coup franc central, remise apres une sortie de balle proche du milieu)

═══════════════════════════════════════════════════
CRITERES STRICTS POUR "RESTART_KO" — TOUS obligatoires
═══════════════════════════════════════════════════
Un vrai coup d'envoi (RESTART_KO) exige TOUS ces elements ensemble :

1. BALLON EXACTEMENT au point/rond central (pas juste "proche du centre").
2. LES DEUX EQUIPES clairement separees, CHACUNE sur SA moitie de terrain
   respective — pas un seul joueur isole pres du ballon avec les autres
   disperses sans repartition claire par moitie.
3. Joueurs RELATIVEMENT STATIQUES ou en train de se placer calmement
   (pas en pleine course/action de jeu).
4. CE N'EST PAS une autre remise en jeu localisee qui se joue pres du
   centre par coincidence (coup franc central, remise apres sortie de
   balle) — si un SEUL joueur s'apprete a jouer le ballon sans que les
   DEUX equipes soient visiblement organisees chacune sur leur moitie,
   c'est "DEAD_BALL", PAS "RESTART_KO".

Si le MOINDRE de ces 4 criteres n'est pas clairement rempli, reponds
"DEAD_BALL" ou "NORMAL" plutot que "RESTART_KO" — en cas de doute,
NE CHOISIS PAS "RESTART_KO".

Reponds avec exactement ce format :
{"state": "NORMAL"}
(en remplacant par l'etat detecte parmi les 5 listes ci-dessus)
"""


def _safe_json_load(text):
    """Parseur JSON tolerant (backticks, texte parasite autour)."""
    if not text:
        return None
    try:
        clean = re.sub(r"```json|```", "", text).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)
        return json.loads(clean)
    except Exception:
        return None


def macro_scan(video_path, pas_scan=5.0, t_debut=0.0, t_fin=None,
               model_name="gemini-3.5-flash", max_calls=None,
               jpeg_quality=70):
    """
    Scanne une vidéo à intervalle régulier (pas_scan secondes) et classe
    chaque image échantillonnée dans un état simple via Gemini.

    NE détecte PAS l'action elle-même — repère des zones d'intérêt à
    examiner plus finement ensuite (voir detect_goal_candidates_via_ko
    pour l'usage concret sur les buts).

    Args:
        video_path   : chemin de la vidéo
        pas_scan     : intervalle entre deux échantillons (secondes)
        t_debut      : instant de départ du scan (secondes)
        t_fin        : instant de fin (secondes) — par défaut, fin de la vidéo
        model_name   : modèle Gemini à utiliser
        max_calls    : limite le nombre d'appels (utile pour un premier
                       test rapide/pas cher sur un extrait)
        jpeg_quality : qualité JPEG d'encodage (70 = compromis coût/lisibilité)

    Returns:
        (resultats, n_appels_gemini)
        resultats : liste de dicts {"t": float, "state": str}
        n_appels_gemini : nombre d'appels Gemini réellement effectués
    """
    import cv2
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY manquante")
    client = genai.Client(api_key=api_key)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duree = total_frames / fps if fps else 0.0
    if t_fin is None:
        t_fin = duree
    t_fin = min(t_fin, duree)

    timestamps = []
    t = t_debut
    while t < t_fin:
        timestamps.append(t)
        t += pas_scan
    if max_calls is not None:
        timestamps = timestamps[:max_calls]

    print(f"  [MACRO_SCAN] Démarrage : {len(timestamps)} échantillon(s) prévu(s) "
          f"(pas={pas_scan}s, fenêtre=[{t_debut:.0f}s, {t_fin:.0f}s])")

    resultats = []
    n_appels = 0
    _t0 = time.time()

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            print(f"  [MACRO_SCAN] ⚠️ lecture échouée à t={t:.0f}s — ignoré")
            continue

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        img_bytes = buf.tobytes()

        state = "NORMAL"  # repli par défaut si l'appel échoue
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    {"parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_bytes}},
                        {"text": PROMPT_MACRO_SCAN},
                    ]}
                ],
            )
            n_appels += 1
            data = _safe_json_load(response.text.strip())
            if data and data.get("state") in STATES:
                state = data["state"]
            else:
                print(f"  [MACRO_SCAN] ⚠️ réponse invalide à t={t:.0f}s : {response.text[:100]!r}")
        except Exception as e:
            print(f"  [MACRO_SCAN] ⚠️ erreur Gemini à t={t:.0f}s : {e}")

        resultats.append({"t": t, "state": state})
        _mm, _ss = int(t // 60), int(t % 60)
        marker = "◀━━" if state != "NORMAL" else ""
        print(f"  [MACRO_SCAN] t={t:.0f}s ({_mm:02d}:{_ss:02d}) → {state} {marker}")

    cap.release()
    _duree_totale = time.time() - _t0
    print(f"  [MACRO_SCAN] ✅ Terminé : {len(resultats)} échantillon(s) analysé(s), "
          f"{n_appels} appel(s) Gemini réel(s), {_duree_totale:.0f}s")

    return resultats, n_appels


def detect_goal_candidates_via_ko(macro_scan_results, ko1_s, ko2_s,
                                    marge_avant_goal=30.0,
                                    marge_exclusion_ko=120.0):
    """
    À partir des résultats d'un macro_scan(), identifie les RESTART_KO
    qui ne correspondent ni à KO1 ni à KO2 (déjà connus) — chacun de ces
    "autres" RESTART_KO est un candidat de but : le but a probablement
    eu lieu juste avant cette reprise.

    Args:
        macro_scan_results : sortie de macro_scan() (liste de dicts)
        ko1_s, ko2_s        : timestamps absolus connus de KO1 et KO2
        marge_avant_goal    : taille de la fenêtre à examiner avant chaque
                              RESTART_KO candidat, pour y chercher le but
                              (micro-scan, non implémenté dans ce POC)
        marge_exclusion_ko  : tolérance (secondes) pour exclure les
                              RESTART_KO trop proches de KO1/KO2 connus

    Returns:
        liste de tuples (t_debut_fenetre, t_restart_ko) à analyser finement
    """
    fenetres = []
    for r in macro_scan_results:
        if r["state"] != "RESTART_KO":
            continue
        t = r["t"]
        if abs(t - ko1_s) < marge_exclusion_ko:
            continue
        if abs(t - ko2_s) < marge_exclusion_ko:
            continue
        fenetres.append((max(0.0, t - marge_avant_goal), t))

    # Fusionne les fenêtres qui se chevauchent (plusieurs échantillons
    # consécutifs peuvent être classés RESTART_KO pour le même but)
    if not fenetres:
        return []
    fenetres.sort()
    fusionnees = [fenetres[0]]
    for debut, fin in fenetres[1:]:
        _d_prec, _f_prec = fusionnees[-1]
        if debut <= _f_prec:
            fusionnees[-1] = (_d_prec, max(_f_prec, fin))
        else:
            fusionnees.append((debut, fin))

    return fusionnees
