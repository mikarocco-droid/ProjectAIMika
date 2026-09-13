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


def _safe_seek_frame(cap, target_frame, max_jump=30):
    """
    Seek précis vers une frame en avançant depuis un keyframe antérieur.
    cap.set(CAP_PROP_POS_FRAMES) est approximatif sur MP4 (arrondit au
    keyframe le plus proche, potentiellement plusieurs secondes
    d'écart) — repris tel quel de ai/gemini_validator.py, où ce
    problème est déjà documenté et corrigé pour le reste du pipeline.
    Sans cette correction, un échantillon visé à t=382.35s peut en
    réalité lire une frame à plusieurs secondes de distance — constaté
    empiriquement le 12/09/2026 (but confirmé par l'utilisateur à
    6:22-6:23, manqué par le micro-scan malgré un échantillon calculé
    à 382.35s).
    """
    import cv2
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = max(0, min(target_frame, total_frames - 1)) if total_frames > 0 else max(0, target_frame)

    start_frame = max(0, target_frame - max_jump)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current = start_frame
    last_frame = None
    while current <= target_frame:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        current += 1

    return last_frame

STATES = ["NORMAL", "ATTACK", "SHOT", "CELEBRATION", "RESTART_KO", "DEAD_BALL"]

PROMPT_MACRO_SCAN = """Analyse cette image d'un match de football amateur/semi-pro.
Reponds UNIQUEMENT avec un objet JSON, sans aucun texte avant ou apres, sans balises markdown.

Etats possibles (choisis EXACTEMENT un seul) :
- "NORMAL" : jeu en cours, rien de particulier a signaler
- "ATTACK" : le BALLON est visiblement DANS LE GRAND RECTANGLE (surface de reparation) de l'une des deux equipes — peu importe l'action precise en cours (centre, une-deux, remise, dribble) ; c'est la PRESENCE du ballon dans cette zone qui compte, pas un jugement sur si l'action parait "dangereuse". PAS un tir en train de se produire (voir "SHOT" ci-dessous).
- "SHOT" : un tir est visiblement en train de se produire — voir criteres ci-dessous
- "CELEBRATION" : joueurs celebrant clairement (bras leves, embrassades, course de joie, groupe de joueurs qui se rassemble en euphorie)
- "RESTART_KO" : UNIQUEMENT un vrai coup d'envoi au centre du terrain (voir criteres stricts ci-dessous)
- "DEAD_BALL" : arret de jeu (touche, corner, coup franc, faute, joueur au sol, arbitre qui intervient) — INCLUT toute remise en jeu qui se joue pres du centre mais qui n'est PAS un vrai coup d'envoi (ex. coup franc central, remise apres une sortie de balle proche du milieu)

═══════════════════════════════════════════════════
CRITERES POUR "SHOT" — au moins UN signal fort suffit
═══════════════════════════════════════════════════
Un tir est difficile a capter sur une seule image (action tres breve) —
cherche ACTIVEMENT ces indices, meme partiels :

1. POSTURE DE FRAPPE : un joueur en pleine extension de jambe vers le
   ballon, jambe d'appui plantee, corps penche — position caracteristique
   d'un tir au moment de l'impact ou juste apres.
2. BALLON EN TRAJECTOIRE VERS LE BUT : ballon visiblement en l'air ou en
   mouvement rapide, dirige vers l'un des deux buts (pas juste au sol
   pres d'un joueur qui marche).
3. REACTION DU GARDIEN : gardien en train de plonger, sauter, ou tendre
   les bras vers un ballon qui arrive — signal tres fiable meme si le
   tir lui-meme n'est pas visible dans le cadre.
4. JOUEURS FIGES REGARDANT VERS LE BUT : plusieurs joueurs immobiles,
   tous tournes vers le meme but, dans une posture d'attente du resultat
   d'un tir qui vient d'avoir lieu.

Si un SEUL de ces 4 signaux est present, meme sans les autres, reponds
"SHOT" plutot que "ATTACK" ou "NORMAL".

═══════════════════════════════════════════════════
CRITERES STRICTS POUR "RESTART_KO" — TOUS obligatoires
═══════════════════════════════════════════════════
Un vrai coup d'envoi (RESTART_KO) exige TOUS ces elements ensemble :

1. BALLON IMMOBILE, exactement au point/rond central (pas en mouvement,
   pas juste "quelque part pres du centre").
2. JOUEURS NON MELANGES PAR EQUIPE, avec tolerance pour les traineurs —
   c'est le critere qui separe un vrai coup d'envoi d'un COUP FRANC
   CENTRAL : sur un coup franc (meme loin du but), les joueurs des deux
   equipes sont generalement MELANGES ensemble pres du ballon
   (contestation, discussion, positionnement defensif improvise) ; sur
   un coup d'envoi, LA MAJORITE des joueurs d'une equipe sont d'UN cote
   et LA MAJORITE des joueurs de l'autre equipe sont de L'AUTRE cote —
   quelques joueurs encore en transition (pas totalement revenus dans
   leur moitie, arbitre qui n'attend pas que tout le monde soit range)
   sont NORMAUX et NE DOIVENT PAS faire rejeter un vrai coup d'envoi.
   Seul un melange GENERALISE des deux equipes au meme endroit doit
   faire rejeter "RESTART_KO".
3. Joueurs RELATIVEMENT STATIQUES ou en train de se placer calmement
   (pas en pleine course/action de jeu).

Si le MOINDRE de ces 3 criteres n'est pas clairement rempli, reponds
"DEAD_BALL" (ballon immobile mais joueurs melanges ou hors du point
central) ou "NORMAL" (ballon en mouvement) plutot que "RESTART_KO" — en
cas de doute, NE CHOISIS PAS "RESTART_KO".

Reponds avec exactement ce format :
{"state": "NORMAL"}
(en remplacant par l'etat detecte parmi les 6 listes ci-dessus)
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
        frame = _safe_seek_frame(cap, int(t * fps))
        if frame is None:
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
                                    marge_exclusion_ko=60.0):
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
                              RESTART_KO trop proches de KO1/KO2 connus.
                              Défaut 60s : compromis entre tolérer le
                              bruit de détection autour de KO1/KO2 (qui
                              peut s'étaler sur plusieurs échantillons)
                              et ne pas exclure à tort un but précoce
                              (ex. un but 90s après KO1 a été observé
                              sur Andrimont — une marge de 120s
                              l'excluait à tort, cf. test du 12/09/2026)

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


# ─────────────────────────────────────────────────────────────────────────
# MICRO-SCAN — confirmation fine d'un candidat de but (V5.2, 12/09/2026)
# ─────────────────────────────────────────────────────────────────────────
# Contrairement au macro-scan (1 appel Gemini par échantillon, 5s d'écart),
# le micro-scan analyse UNE SEULE fenêtre candidate (issue de
# detect_goal_candidates_via_ko) avec PLUSIEURS frames envoyées ensemble
# en UN SEUL appel Gemini — permet une décision beaucoup mieux étayée
# qu'une classification isolée, avec un coût qui reste faible car il n'y
# a que quelques fenêtres candidates par match (pas 134 comme l'ancien
# système par trajectoire de ballon).
#
# Les critères ci-dessous reprennent volontairement le langage déjà
# éprouvé de find_goal_after_shot() (ai/gemini_validator.py, prompt
# "early-stop") — distinction but/touche/dégagement/corner déjà validée
# en production, pas réinventée ici.

PROMPT_MICRO_SCAN_GOAL = """Analyse football match — {n} images extraites de la fenetre {debut} a {fin} (juste avant une reprise au centre du terrain detectee).

Question : Un BUT a-t-il ete marque dans cette fenetre ? Si oui, a quel instant approximatif (en secondes depuis le debut de la fenetre, 0 = premiere image) ?

Regles — a lire attentivement :
- OUI : le ballon est clairement A L'INTERIEUR du but (derriere la ligne, dans les filets), les FILETS SONT VISIBLEMENT DEFORMES/GONFLES par le ballon
- OUI : le gardien est accroupi ou plonge pour recuperer le ballon DEPUIS L'INTERIEUR des filets
- NON : le ballon est pres des filets ou devant, mais les filets sont plats/non deformes
- NON : le ballon est a cote du poteau ou hors du cadre du but
- NON : le gardien tient/attrape le ballon dans ses mains ou ses bras (meme dans la surface)
- NON : le gardien est debout, avec ou sans ballon
- NON : le ballon est DERRIERE le but (hors des filets, de l'autre cote de la structure) → corner ou but de gardien, PAS un but
- NON : situation de remise en touche — joueur sur la ligne de touche tenant ou lancant le ballon
- NON : joueurs regroupes pres de la LIGNE DE TOUCHE → remise en touche, PAS un but
- NON : degagement au but — gardien ou defenseur qui tape un ballon immobile depuis la surface de 6 metres, vers l'exterieur
- NON : gardien qui degage le ballon (au pied ou aux poings) depuis l'interieur de son but
- NON : degagement defensif — defenseur qui tete ou tape le ballon loin du but
- NON : contenu hors-match — enfants qui jouent, terrain vide, activite informelle

CRITIQUE : si le ballon et les joueurs sont pres de la LIGNE DE TOUCHE, c'est presque certainement une remise en touche, PAS un but. Reponds NON immediatement dans ce cas.

Reponds UNIQUEMENT en JSON valide, sans texte avant/apres, sans balises markdown :
{{"is_goal": true ou false, "timestamp_dans_fenetre": <secondes ou null>, "confidence": <0.0-1.0>, "evidence": "<decris precisement : position du ballon par rapport aux filets/ligne, et tout signal negatif observe>"}}
confidence=0.90+ uniquement si le ballon est sans ambiguite a l'interieur des filets avec deformation visible.
Si le moindre doute, is_goal=false."""


def micro_scan_confirm_goal(video_path, window_start, window_end,
                             pas_echantillon=2.0, max_frames=25,
                             model_name="gemini-3.5-flash",
                             jpeg_quality=75):
    """
    Analyse finement une fenêtre candidate (typiquement produite par
    detect_goal_candidates_via_ko) pour confirmer si un but a réellement
    eu lieu à l'intérieur, et à quel instant précis.

    Envoie des frames ESPACÉES DE pas_echantillon SECONDES (pas un
    nombre fixe réparti sur la fenêtre) — un but est un événement bref
    (1-2s) : avec un nombre de frames fixe, une fenêtre plus longue
    dilue l'espacement et peut manquer le but exactement comme le
    macro-scan (constaté empiriquement le 12/09/2026 : 8 frames sur 35s
    donnait un espacement de ~5s, identique au macro-scan, et a manqué
    un but confirmé). TOUTES les frames sont envoyées dans un SEUL appel
    Gemini — le coût reste 1 appel par fenêtre, seule la quantité de
    frames dans cet appel augmente.

    Args:
        video_path      : chemin de la vidéo
        window_start    : début de la fenêtre à analyser (secondes, absolu)
        window_end      : fin de la fenêtre (secondes, absolu)
        pas_echantillon : espacement cible entre deux frames (secondes) —
                          défaut 2.0s, dense mais raisonnable pour capter
                          un événement de 1-2s
        max_frames      : plafond de sécurité (évite un appel démesuré
                          sur une fenêtre anormalement longue)
        model_name      : modèle Gemini à utiliser
        jpeg_quality    : qualité JPEG d'encodage

    Returns:
        dict {"is_goal": bool, "timestamp_absolu": float|None,
              "confidence": float, "evidence": str, "n_appels_gemini": int}
        ou None si l'extraction vidéo échoue
    """
    import cv2
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY manquante")
    client = genai.Client(api_key=api_key)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    duree_fenetre = window_end - window_start
    if duree_fenetre <= 0:
        cap.release()
        return None

    n_frames = max(2, min(max_frames, int(duree_fenetre / pas_echantillon) + 1))
    offsets = [i * duree_fenetre / (n_frames - 1) for i in range(n_frames)]

    parts = []
    offsets_valides = []
    for off in offsets:
        t_abs = window_start + off
        frame = _safe_seek_frame(cap, int(t_abs * fps))
        if frame is None:
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": buf.tobytes()}})
        offsets_valides.append(off)
    cap.release()

    if not parts:
        print(f"  [MICRO_SCAN] ⚠️ aucune frame extraite pour la fenêtre [{window_start:.0f}s, {window_end:.0f}s]")
        return None

    _mm_d, _ss_d = int(window_start // 60), int(window_start % 60)
    _mm_f, _ss_f = int(window_end // 60), int(window_end % 60)
    prompt = PROMPT_MICRO_SCAN_GOAL.format(
        n=len(parts), debut=f"{_mm_d:02d}:{_ss_d:02d}", fin=f"{_mm_f:02d}:{_ss_f:02d}",
    )

    print(f"  [MICRO_SCAN] Analyse fenêtre [{window_start:.0f}s, {window_end:.0f}s] "
          f"({len(parts)} frame(s), 1 appel Gemini)...")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[{"parts": [{"text": prompt}] + parts}],
        )
        data = _safe_json_load(response.text.strip())
    except Exception as e:
        print(f"  [MICRO_SCAN] ⚠️ erreur Gemini : {e}")
        return {"is_goal": False, "timestamp_absolu": None, "confidence": 0.0,
                "evidence": f"erreur: {e}", "n_appels_gemini": 1}

    if not data:
        print(f"  [MICRO_SCAN] ⚠️ réponse invalide : {response.text[:150]!r}")
        return {"is_goal": False, "timestamp_absolu": None, "confidence": 0.0,
                "evidence": "réponse Gemini invalide", "n_appels_gemini": 1}

    is_goal = bool(data.get("is_goal", False))
    ts_relatif = data.get("timestamp_dans_fenetre")
    ts_absolu = (window_start + float(ts_relatif)) if (is_goal and ts_relatif is not None) else None
    confidence = float(data.get("confidence", 0.0) or 0.0)
    evidence = data.get("evidence", "")

    _verdict = "BUT CONFIRMÉ" if is_goal else "pas de but"
    print(f"  [MICRO_SCAN] → {_verdict} (confidence={confidence:.2f})"
          + (f", t≈{ts_absolu:.0f}s" if ts_absolu is not None else ""))

    return {
        "is_goal": is_goal,
        "timestamp_absolu": ts_absolu,
        "confidence": confidence,
        "evidence": evidence,
        "n_appels_gemini": 1,
    }
