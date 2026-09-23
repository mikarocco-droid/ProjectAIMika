# vision/detector.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ultralytics import YOLO
from vision.ball import BallDetector
import config


PLAY_ZONES = {
    "football":         {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "mini-foot":        {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "basketball":       {"x_min": 0.02, "x_max": 0.98, "y_min": 0.10, "y_max": 0.92},
    "handball":         {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "rugby":            {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "hockey sur glace": {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "hockey sur gazon": {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "tennis":           {"x_min": 0.05, "x_max": 0.95, "y_min": 0.10, "y_max": 0.90},
    "tennis de table":  {"x_min": 0.05, "x_max": 0.95, "y_min": 0.10, "y_max": 0.90},
    "padel":            {"x_min": 0.05, "x_max": 0.95, "y_min": 0.10, "y_max": 0.90},
}

MIN_PLAYER_W = 0.015
MIN_PLAYER_H = 0.04
MAX_PLAYER_W = 0.25
MAX_PLAYER_H = 0.70
MIN_RATIO    = 1.2
MAX_RATIO    = 4.0

# ─────────────────────────────────────────
# MODÈLES — séparés joueurs / ballon
# yolo11m  pour les joueurs  : rapide, suffisant
# yolo11x  en fallback si m  indisponible
# ─────────────────────────────────────────
PLAYER_MODELS = [
    "yolo11m.pt",   # priorité — medium, ~3x plus rapide que x
    "yolo11l.pt",   # fallback grand
    "yolo11x.pt",   # fallback max
    "yolov8m.pt",   # fallback YOLOv8
    "yolov8n.pt",   # fallback minimal
]


def load_player_model(sport):
    """Charge le modèle joueurs — priorité yolo11m."""
    import os

    specialized = {
        "football":   "models/yolov8_football.pt",
        "mini-foot":  "models/yolov8_football.pt",
        "basketball": "models/yolov8_basketball.pt",
        "handball":   "models/yolov8_handball.pt",
    }
    sport_model = specialized.get(sport)
    if sport_model and os.path.exists(sport_model):
        print(f"  Modele specialise joueurs : {sport_model}")
        return YOLO(sport_model), sport_model

    for name in PLAYER_MODELS:
        try:
            print(f"  Chargement {name}...")
            m = YOLO(name)
            print(f"  Modele joueurs : {name}")
            return m, name
        except Exception as e:
            print(f"  {name} indisponible : {e}")

    raise RuntimeError("Aucun modèle YOLO disponible")


# ─────────────────────────────────────────
# DÉTECTEUR BALLON HSV — indépendant de YOLO
# Détecte les objets ronds et blancs/clairs
# Beaucoup plus fiable que yolo11m sur les petits objets
# ─────────────────────────────────────────
class BallHSVDetector:
    """
    Détection ballon par couleur HSV + circularité.
    Fonctionne bien pour football (ballon blanc/noir),
    basketball (ballon orange), handball (ballon jaune/orange).
    """

    # Plages HSV par sport
    HSV_RANGES = {
        "football":   [
            # blanc
            ((0,   0, 180), (180,  40, 255)),
            # noir/sombre (partie noire du ballon)
            ((0,   0,   0), (180,  50,  80)),
        ],
        "basketball": [
            # orange
            ((5,  100, 100), (25, 255, 255)),
        ],
        "handball":   [
            # jaune/orange
            ((15,  80, 100), (35, 255, 255)),
        ],
        "default":    [
            ((0,   0, 160), (180,  50, 255)),   # blanc/clair
            ((5,  100, 100), (25, 255, 255)),   # orange
        ],
    }

    # V5.2 (20/09/2026) : bornes de surface (aire bbox, px²) par
    # camera_type - MEMES VALEURS que le systeme camera_type existant
    # (build_camera_profile(), vision/camera_profile.py). "low_side"/
    # "low_side_zoom" = valeurs D'ORIGINE, INCHANGEES (30-3000) - deja
    # validees sur des matchs reels filmes a la barriere, ballon
    # proche/gros a l'ecran. "high_side" = PREMIERE ESTIMATION,
    # PROVISOIRE, basee sur une seule mesure ce soir (match Andrimont,
    # ~5490 candidats HSV analyses : aire mediane 54px2, p25=40,
    # p75=76) - resserree pour exclure les gros faux positifs
    # (structure de but, vetement clair) tout en gardant une marge
    # autour du ballon reel mesure. A RAFFINER avec plus de
    # donnees/matchs avant de considerer ces valeurs comme definitives.
    AIRE_BORNES = {
        "low_side":      (30, 3000),
        "low_side_zoom": (30, 3000),
        "high_side":     (15, 200),
    }

    def __init__(self, sport="football", camera_type="low_side", proximite_poids=0.5,
                 seuil_gap_protection=None, intervalle_recherche_globale=None):
        """
        proximite_poids : V5.2 (20/09/2026) - poids du bonus de
        proximite a last_pos dans le score final. Defaut 0.5 = valeur
        D'ORIGINE, INCHANGEE. Ajoute pour un test cible (voir
        ANALYSE_NOUVELLE_ARCHITECTURE_DETECTION.md section 3.29) suite
        a la decouverte que ce bonus change le gagnant dans 46,6% des
        frames a candidats multiples (L/M/N, camera high_side) - les
        candidats HSV intrinseques etant presque toujours proches/a
        egalite, ce bonus devient le vrai arbitre. AUCUN changement de
        la logique elle-meme, uniquement ce poids rendu configurable
        pour mesurer son effet reel avant de decider s'il faut le
        modifier en profondeur.

        seuil_gap_protection : V5.2 (20/09/2026) - si fourni (float),
        la proximite ne peut PAS renverser un candidat dont l'ecart
        intrinseque (score_forme) avec le 2e meilleur depasse ce
        seuil. Defaut None = DESACTIVE, comportement IDENTIQUE a avant
        cette modification (proximite toujours appliquee, peut
        toujours renverser). PREMIERE ESTIMATION PROPOSEE (non un
        defaut actif) : 0.08, milieu de la zone ou un renversement
        nuisible a ete confirme visuellement (gap=0.05-0.09, section
        3.31) - PAS une valeur validee de facon exhaustive, un point
        de depart pour un test cible avant toute generalisation.

        intervalle_recherche_globale : V5.2 (20/09/2026) - MECANISME
        DE REPRISE, propose par l'utilisateur suite a la decouverte
        (section 3.33-3.34 de l'analyse) que search_radius (recherche
        recadree autour de last_pos) EXCLUT STRUCTURELLEMENT la vraie
        position du ballon des qu'il est ailleurs sur le terrain -
        confirme avec des donnees precises : sur 4 cas verifies
        visuellement (zone P, jamais exploree), 4/4 montrent le vrai
        ballon HORS de la zone de recherche (ex. last_pos_x=72,
        zone=[-178,324], vrai ballon a x=446). Ce n'est PAS un probleme
        de couleur/forme/taille - HSV ne regarde simplement jamais la
        ou se trouve le ballon une fois last_pos deja errone.

        IMPORTANT (raffinement par rapport a la proposition initiale) :
        la recherche locale trouve presque TOUJOURS un candidat (pas
        "aucun candidat") - juste souvent le mauvais, puisque quelque
        chose de rond/clair existe generalement quelque part dans la
        zone recadree (structure de but, etc.). Un compteur base sur
        "N frames SANS aucun candidat" ne se declencherait donc quasi
        jamais dans la pratique. Mecanisme retenu a la place :
        recherche globale PERIODIQUE (toutes les N frames,
        INDEPENDAMMENT du succes local) plutot que conditionnee a un
        echec local explicite - casse le verrouillage meme quand la
        recherche locale "reussit" a tort.

        Defaut None = DESACTIVE, comportement IDENTIQUE a avant cette
        modification (recherche toujours recadree si last_pos connu).
        Si fourni (int, ex. 10), toutes les N frames la recherche
        ignore last_pos/search_radius et scanne l'image entiere - si
        un candidat de meilleur score y est trouve, il devient le
        nouveau point d'ancrage. PREMIERE IMPLEMENTATION, NON VALIDEE -
        valeur et frequence a calibrer par mesure, pas par supposition.
        """
        self.sport       = sport
        self.camera_type = camera_type
        self.ranges      = self.HSV_RANGES.get(sport, self.HSV_RANGES["default"])
        self.aire_min, self.aire_max = self.AIRE_BORNES.get(
            camera_type, self.AIRE_BORNES["low_side"]
        )
        self.proximite_poids      = proximite_poids
        self.seuil_gap_protection = seuil_gap_protection
        self.intervalle_recherche_globale = intervalle_recherche_globale
        self._compteur_frames_recherche   = 0  # etat interne, incremente a chaque appel

    def detect(self, frame, last_pos=None, search_radius=200, debug_t=None):
        """
        Cherche le ballon dans la frame.
        Si last_pos connu, cherche dans un rayon réduit (plus rapide).
        Retourne dict {bbox, center, conf} ou None.

        V5.2 (20/09/2026) : ajout de debug_t (optionnel, timestamp pour
        le log de diagnostic) - AUCUNE modification de la logique de
        detection/scoring elle-meme, uniquement instrumentation pour
        l'audit de la couche perception (voir
        ANALYSE_NOUVELLE_ARCHITECTURE_DETECTION.md, decouverte du
        mecanisme de derive HSV pres du but gauche).
        """
        h, w = frame.shape[:2]

        # V5.2 (20/09/2026) : MECANISME DE REPRISE - recherche globale
        # periodique, independante du succes/echec local (voir
        # docstring de __init__ pour la justification complete et les
        # donnees ayant motive ce mecanisme). Desactive par defaut
        # (intervalle_recherche_globale=None) - comportement
        # STRICTEMENT inchange dans ce cas.
        self._compteur_frames_recherche += 1
        _force_recherche_globale = False
        if self.intervalle_recherche_globale is not None:
            if last_pos is None or self._compteur_frames_recherche >= self.intervalle_recherche_globale:
                _force_recherche_globale = True
                self._compteur_frames_recherche = 0

        # Zone de recherche réduite si position précédente connue
        # (SAUF si recherche globale forcee ce tour-ci)
        if last_pos is not None and not _force_recherche_globale:
            h_f, w_f = frame.shape[:2]
            cx, cy  = last_pos
            x1 = max(0, int(cx - search_radius))
            y1 = max(0, int(cy - search_radius))
            x2 = min(w_f, int(cx + search_radius))
            y2 = min(h_f, int(cy + search_radius))
            # FIX — si le crop est vide (last_pos hors frame) → fallback frame entière
            if x2 <= x1 or y2 <= y1:
                roi    = frame
                offset = (0, 0)
            else:
                roi    = frame[y1:y2, x1:x2]
                offset = (x1, y1)
        else:
            roi    = frame
            offset = (0, 0)

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for (lo, hi) in self.ranges:
            m    = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.bitwise_or(mask, m)

        # Morphologie pour nettoyer
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # V5.2 (20/09/2026) : diagnostic - active seulement si debug_t
        # fourni ET DEBUG=true, pour ne jamais impacter les appels
        # normaux (aucun cout de calcul supplementaire hors diagnostic).
        _diag_actif = False
        if debug_t is not None:
            try:
                from config import DEBUG as _DBG_HSV
                _diag_actif = _DBG_HSV
            except ImportError:
                _diag_actif = False
        _diag_candidats = []

        if _diag_actif and _force_recherche_globale:
            print(f"  [RECHERCHE_GLOBALE] t={debug_t:.2f}s — recherche élargie à l'image "
                  f"entière déclenchée (compteur atteint {self.intervalle_recherche_globale}"
                  f" ou last_pos absent), proximité désactivée pour cet appel")

        # V5.2 (20/09/2026) : RESTRUCTURE pour proximite CONDITIONNELLE.
        # Avant : selection en un seul passage (max du score final au fur
        # et a mesure) - la proximite pouvait renverser un candidat
        # intrinsequement bien meilleur des qu'elle etait appliquee.
        # Preuve mecanique du probleme sur M-2/M-3 (voir
        # ANALYSE_NOUVELLE_ARCHITECTURE_DETECTION.md section 3.29-3.31) :
        # audit sur 1281 frames, 28.9% de renversements, concentres sur
        # les petits intrinsic_gap (mediane 0.030 quand renverse, vs
        # 0.090 sinon). Verification visuelle confirmant un renversement
        # nuisible net a gap=0.05-0.09 (frame M-1 5/10, section 3.31).
        # Nouvelle logique : collecter TOUS les candidats valides, puis
        # decider APRES coup si le meilleur intrinseque est deja assez
        # net pour que la proximite ne doive pas intervenir.
        _candidats_valides = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Taille min/max ballon en pixels - depend de self.camera_type
            # (voir AIRE_BORNES en tete de classe)
            if area < self.aire_min or area > self.aire_max:
                continue

            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue

            # Circularité : 1.0 = cercle parfait
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < 0.5:   # filtre les formes non rondes
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Rapport largeur/hauteur proche de 1 (carré = rond)
            if bw == 0 or bh == 0:
                continue
            ratio = bw / bh
            if ratio < 0.5 or ratio > 2.0:
                continue

            # Score = circularité × taille normalisée
            score_forme = circularity * min(area / 200, 1.0)
            score = score_forme

            # Bonus si proche de la dernière position
            # V5.2 (20/09/2026) : SUPPRIME si recherche globale forcee -
            # last_pos est justement connu comme non fiable dans ce cas
            # (c'est pourquoi on force une recherche globale). Le laisser
            # influencer le score reproduirait le meme piege qu'on
            # cherche a casser : favoriser un candidat pres de last_pos
            # au lieu du meilleur candidat intrinseque trouve sur toute
            # l'image.
            score_proximite = 0.0
            dist = None
            if last_pos is not None and not _force_recherche_globale:
                cx_b = x + bw // 2 + offset[0]
                cy_b = y + bh // 2 + offset[1]
                dist = np.hypot(cx_b - last_pos[0], cy_b - last_pos[1])
                score_proximite = max(0, 1.0 - dist / search_radius) * self.proximite_poids
                score += score_proximite

            if _diag_actif:
                _diag_candidats.append({
                    "cx": x + bw // 2 + offset[0], "cy": y + bh // 2 + offset[1],
                    "w": bw, "h": bh, "area": area, "circularity": circularity,
                    "ratio": ratio, "dist": dist,
                    "score_forme": score_forme, "score_proximite": score_proximite,
                    "score_final": score,
                })

            _candidats_valides.append({
                "bbox": (x, y, bw, bh, offset),
                "score_forme": score_forme,
                "score_final": score,
            })

        best = None
        best_score = -1
        if _candidats_valides:
            if len(_candidats_valides) == 1:
                best = _candidats_valides[0]["bbox"]
                best_score = _candidats_valides[0]["score_final"]
            else:
                par_intrinseque = sorted(_candidats_valides, key=lambda c: -c["score_forme"])
                meilleur_intrinseque = par_intrinseque[0]
                second_intrinseque   = par_intrinseque[1]
                intrinsic_gap = meilleur_intrinseque["score_forme"] - second_intrinseque["score_forme"]

                if self.seuil_gap_protection is not None and intrinsic_gap > self.seuil_gap_protection:
                    # Ecart intrinseque deja net - la proximite ne doit
                    # pas pouvoir renverser ce candidat.
                    best = meilleur_intrinseque["bbox"]
                    best_score = meilleur_intrinseque["score_final"]
                else:
                    # Ecart faible - proximite autorisee a departager,
                    # comportement identique a avant cette restructuration.
                    gagnant = max(_candidats_valides, key=lambda c: c["score_final"])
                    best = gagnant["bbox"]
                    best_score = gagnant["score_final"]

        if _diag_actif:
            _cands_str = " ".join(
                f"[cx={c['cx']},cy={c['cy']},w={c['w']},h={c['h']},area={c['area']:.0f},"
                f"circ={c['circularity']:.2f},ratio={c['ratio']:.2f},"
                f"dist={'%.0f' % c['dist'] if c['dist'] is not None else -1},"
                f"score_forme={c['score_forme']:.2f},score_prox={c['score_proximite']:.2f},"
                f"score={c['score_final']:.2f}]"
                for c in _diag_candidats
            )
            _lp_str = f"({last_pos[0]:.0f},{last_pos[1]:.0f})" if last_pos is not None else "None"
            print(f"  [HSV_DIAG] t={debug_t:.2f}s last_pos={_lp_str} search_radius={search_radius} "
                  f"n_contours_bruts={len(contours)} n_candidats_valides={len(_diag_candidats)} "
                  f"{_cands_str}")

        if best is None:
            return None

        x, y, bw, bh, (ox, oy) = best
        cx = x + bw // 2 + ox
        cy = y + bh // 2 + oy
        x1b = x + ox
        y1b = y + oy

        return {
            "bbox":   [x1b, y1b, x1b + bw, y1b + bh],
            "center": [cx, cy],
            "conf":   round(best_score, 2),
            "method": "hsv"
        }


# ─────────────────────────────────────────
# DÉTECTEUR PRINCIPAL
# ─────────────────────────────────────────
class Detector:

    def __init__(self, sport="football", camera_type="low_side", proximite_poids=0.5,
                 seuil_gap_protection=None, intervalle_recherche_globale=None):
        self.sport        = sport
        # V5.2 (20/09/2026) : camera_type - PARAMETRE NORMAL, pas de
        # flag config.py separe (retire suite a une remarque justifiee
        # de l'utilisateur : on veut UN SEUL systeme de classification
        # camera, pas deux qui pourraient diverger). Defaut "low_side"
        # preserve le comportement existant. En production, ce
        # parametre devra etre alimente par la meme source que
        # camera_type ailleurs dans pipeline.py
        # (_camera_profile.get("camera_type")) - actuellement calcule
        # APRES un passage complet sur frames_data (build_camera_profile()),
        # donc pas encore disponible a la creation de ce Detector pour
        # un run normal (dependance circulaire : camera_type depend des
        # positions ballon, qui dependent de ce meme detecteur). Non
        # resolu ce soir - voir
        # ANALYSE_NOUVELLE_ARCHITECTURE_DETECTION.md.
        self.camera_type      = camera_type
        self.proximite_poids  = proximite_poids  # V5.2 (20/09/2026) : defaut 0.5 = inchange
        self.seuil_gap_protection = seuil_gap_protection  # V5.2 (20/09/2026) : defaut None = inchange
        self.intervalle_recherche_globale = intervalle_recherche_globale  # V5.2 (20/09/2026) : defaut None = inchange
        self.zone         = PLAY_ZONES.get(sport, PLAY_ZONES["football"])
        self.model, self.model_name = load_player_model(sport)

        # Détecteur ballon — HSV en priorité + BallDetector en fallback
        self.hsv_ball    = BallHSVDetector(sport=sport, camera_type=self.camera_type,
                                            proximite_poids=self.proximite_poids,
                                            seuil_gap_protection=self.seuil_gap_protection,
                                            intervalle_recherche_globale=self.intervalle_recherche_globale)
        self.ball_backup = BallDetector(method=config.BALL_METHOD)
        self._last_ball_pos = None   # mémorise dernière position ballon

        self.player_cls = 0    # COCO : person
        self.ball_cls   = 32   # COCO : sports ball

        print(f"  Detector pret : {self.model_name} | sport={sport} | camera_type={self.camera_type} | proximite_poids={self.proximite_poids} | seuil_gap_protection={self.seuil_gap_protection} | intervalle_recherche_globale={self.intervalle_recherche_globale}")

    def set_sport(self, sport):
        if sport == self.sport:
            return
        self.sport    = sport
        self.zone     = PLAY_ZONES.get(sport, PLAY_ZONES["football"])
        self.hsv_ball = BallHSVDetector(sport=sport, camera_type=self.camera_type,
                                         proximite_poids=self.proximite_poids,
                                         seuil_gap_protection=self.seuil_gap_protection,
                                         intervalle_recherche_globale=self.intervalle_recherche_globale)
        new_model, new_name = load_player_model(sport)
        if new_name != self.model_name:
            self.model      = new_model
            self.model_name = new_name
            print(f"  Modele mis a jour : {new_name}")

    def _in_play_zone(self, center, frame_w, frame_h):
        cx, cy = center
        z      = self.zone
        return (z["x_min"] * frame_w <= cx <= z["x_max"] * frame_w and
                z["y_min"] * frame_h <= cy <= z["y_max"] * frame_h)

    def _valid_size(self, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:            return False
        if w < MIN_PLAYER_W * frame_w:  return False
        if h < MIN_PLAYER_H * frame_h:  return False
        if w > MAX_PLAYER_W * frame_w:  return False
        if h > MAX_PLAYER_H * frame_h:  return False
        ratio = h / w
        return MIN_RATIO <= ratio <= MAX_RATIO

    def _detect_ball(self, frame, yolo_ball, last_pos_override=None, debug_t=None):
        """
        Stratégie ballon à 3 niveaux :
        1. YOLO a trouvé le ballon → on l'utilise + on met à jour last_pos
        2. HSV trouve quelque chose de circulaire → on l'utilise
        3. Fallback BallDetector (méthode config)

        last_pos_override : position en coordonnées de `frame` (pas originales)
                            Utilisé quand frame est une version réduite.
        debug_t : V5.2 (20/09/2026) - timestamp optionnel, propage a
                  hsv_ball.detect() pour l'audit [HSV_DIAG]. N'affecte
                  aucune logique.
        """
        # 1. YOLO
        if yolo_ball is not None:
            self._last_ball_pos = yolo_ball["center"]
            return yolo_ball

        # Position de recherche HSV — override si fourni (frame réduite)
        search_pos = last_pos_override if last_pos_override is not None \
                     else self._last_ball_pos

        # 2. HSV avec recherche guidée
        hsv_result = self.hsv_ball.detect(
            frame,
            last_pos      = search_pos,
            search_radius = 250,
            debug_t       = debug_t
        )
        if hsv_result is not None:
            self._last_ball_pos = hsv_result["center"]
            return hsv_result

        # 3. Fallback BallDetector
        fallback = self.ball_backup.get_position(frame, None)
        if fallback is not None:
            self._last_ball_pos = fallback.get("center")
        return fallback

    def detect(self, frame):
        """
        Détecte joueurs et ballon sur une frame.
        Utilisé hors batch (calibration, sport_detector...).
        Retourne :
            players : list de dicts {bbox, center, conf}
            ball    : dict {bbox, center, conf} ou None
        """
        h_frame, w_frame = frame.shape[:2]

        results = self.model(
            frame,
            conf    = config.YOLO_CONFIDENCE,
            verbose = False,
            imgsz   = 960    # FIX: 1280 → 960 cohérent avec batch
        )[0]

        players   = []
        # V5.2 (19/09/2026) FIX CRITIQUE : meme motif que dans main.py
        # (process_batch) - "yolo_ball = {...}" ecrasait silencieusement
        # a chaque iteration, sans comparer confiance ni position, si
        # YOLO detectait 2+ objets classes "ballon". Corrige de la meme
        # facon : collecte tous les candidats, selectionne le plus
        # confiant (cette methode n'a pas de last_pos a ce stade, geree
        # separement par _detect_ball() juste apres).
        _candidats_ball = []

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox   = [x1, y1, x2, y2]

            if cls == self.player_cls:
                if not self._in_play_zone(center, w_frame, h_frame):
                    continue
                if not self._valid_size(bbox, w_frame, h_frame):
                    continue
                players.append({
                    "bbox":   bbox,
                    "center": [center[0], center[1]],
                    "conf":   conf
                })
            elif cls == self.ball_cls:
                _candidats_ball.append({
                    "bbox":   bbox,
                    "center": [center[0], center[1]],
                    "conf":   conf
                })

        yolo_ball = max(_candidats_ball, key=lambda b: b["conf"]) if _candidats_ball else None

        ball = self._detect_ball(frame, yolo_ball)
        return players, ball