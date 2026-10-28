# analysis/player_reid.py
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
from collections import defaultdict

# V5.2 (13/09/2026) : profilage fin de process() - couleur/inference
# equipe/embedding/assignation - pour confirmer precisement si OSNet
# (embedding) domine le temps, suite au diagnostic montrant
# tracker_update = 92,3% du temps total dans main.py.
_PROFILING_REID = defaultdict(float)
_PROFILING_REID_N = defaultdict(int)


def print_profiling_reid_summary():
    print()
    print("=" * 80)
    print("PROFILAGE FIN — PlayerReID.process() décomposé")
    print("=" * 80)
    _total = sum(_PROFILING_REID.values())
    for cle, secondes in sorted(_PROFILING_REID.items(), key=lambda x: -x[1]):
        _pct = 100 * secondes / _total if _total > 0 else 0
        _n = _PROFILING_REID_N.get(cle)
        _detail = f", {secondes/_n*1000:.2f}ms/appel" if _n else ""
        print(f"  {cle:15s} : {secondes:8.1f}s ({_pct:5.1f}%){_detail}")
    print(f"  {'TOTAL':15s} : {_total:8.1f}s")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────
# EMBEDDING D'APPARENCE PROFOND (V5.2, 13/09/2026)
# ─────────────────────────────────────────────────────────────────────────
# Remplace l'ancien "_extract_embedding" qui n'était qu'un histogramme de
# couleur (cv2.calcHist) malgré son nom — incapable de distinguer deux
# joueurs du même maillot, confirmé par lecture de code le 12/09/2026
# (voir ANALYSE_NOUVELLE_ARCHITECTURE_DETECTION.md section 3.2). Sur un
# vrai export (frames_data_complet.pkl), ce défaut était corrélé à 389
# tracker_id distincts pour ~22 personnes réelles sur le terrain.
#
# Chargement paresseux (1 seule fois, mis en cache globalement) — le
# modèle est lourd à charger, pas à faire à chaque frame. Repli
# automatique et silencieux sur l'ancien histogramme de couleur si
# torch/torchreid indisponibles ou le téléchargement des poids échoue
# (ex. environnement sans accès réseau complet) — ne doit jamais faire
# planter le pipeline de production.

_OSNET_EXTRACTOR = None
_OSNET_DISPONIBLE = None  # None = pas encore testé, True/False après 1ère tentative


def _charger_osnet():
    """Charge (une seule fois) l'extracteur d'embedding OSNet. Retourne
    None si indisponible - l'appelant doit alors utiliser le repli
    histogramme de couleur."""
    global _OSNET_EXTRACTOR, _OSNET_DISPONIBLE
    if _OSNET_DISPONIBLE is not None:
        return _OSNET_EXTRACTOR
    try:
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError:
            # Certaines versions de torchreid n'exposent pas le raccourci
            # top-level torchreid.utils — chemin réel du sous-module
            from torchreid.reid.utils import FeatureExtractor
        # V5.2 (13/09/2026) FIX CRITIQUE DE PERFORMANCE : device="cpu"
        # etait code en dur depuis la premiere integration d'OSNet,
        # jamais revu lors du passage au pipeline complet. Sur un
        # test reel (15 min de match, Andrimont), le traitement a pris
        # 89,8 min (extrapolation ~10h/match) alors que le GPU est
        # disponible et deja utilise par DeepSort ("FP16 GPU"). OSNet
        # tournait sur CPU pour CHAQUE joueur detecte a CHAQUE frame
        # traitee - tres probablement la cause dominante du
        # ralentissement, plus que le bug frame_skip deja corrige.
        import torch as _torch_reid
        _device = "cuda" if _torch_reid.cuda.is_available() else "cpu"
        _OSNET_EXTRACTOR = FeatureExtractor(model_name="osnet_x0_25", device=_device)
        _OSNET_DISPONIBLE = True
        print(f"  [REID] OSNet (osnet_x0_25) chargé sur {_device} — embedding d'apparence profond actif")
    except Exception as e:
        _OSNET_DISPONIBLE = False
        _OSNET_EXTRACTOR = None
        import traceback
        print(f"  [REID] ⚠️ OSNet indisponible ({e}) — repli sur histogramme de couleur "
              f"(insuffisant pour distinguer 2 joueurs du même maillot)")
        print(f"  [REID] Détail de l'erreur (diagnostic) :")
        traceback.print_exc()
    return _OSNET_EXTRACTOR


def _extraire_crop(frame, bbox):
    """Extrait et borne un crop depuis bbox, retourne None si invalide.
    Même logique de bornage que l'ancienne _extract_embedding (évite
    l'indexation négative NumPy sur bbox partiellement hors cadre)."""
    h_f, w_f = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w_f))
    x2 = max(0, min(x2, w_f))
    y1 = max(0, min(y1, h_f))
    y2 = max(0, min(y2, h_f))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _extraire_embeddings_batch(crops, extractor):
    """Extrait les embeddings de PLUSIEURS crops en UN SEUL appel OSNet
    (V5.2, 13/09/2026) — bien plus efficace sur GPU qu'un appel par
    crop. Constaté empiriquement : 13,18ms/appel individuel, 95,6% du
    temps total de PlayerReID.process() (838s sur 876,6s, segment de
    5 min) — dominé par l'overhead fixe par appel (lancement GPU,
    transfert mémoire) plutôt que le calcul lui-même sur ce petit
    réseau (203k paramètres). Le traitement par lot amortit cet
    overhead sur tous les joueurs d'une frame en un seul appel."""
    if not crops:
        return []
    embeddings = extractor(crops)
    resultats = []
    for emb in embeddings:
        emb = emb.cpu().numpy() if hasattr(emb, "cpu") else np.asarray(emb)
        norme = np.linalg.norm(emb)
        if norme > 0:
            emb = emb / norme
        resultats.append(emb.astype(np.float32))
    return resultats


class PlayerReID:
    """
    ReID hybride calibré :
    - position + couleur maillot + embedding histogramme
    - contrainte spatiale (SPATIAL_MAX_DIST appris par MatchLearner)
    - contrainte inter-équipes (jamais fusion A/B)
    - TTL adaptatif (seuil plus strict pour joueurs en veille)
    - MAX_PLAYERS = 25
    - Calibration dynamique des couleurs équipes (KMeans)
    - jersey_map intégré : stabilise les IDs via numéro maillot
    """

    TTL_ACTIVE_SECONDS = 5.0    # durée réelle avant passage en veille
    TTL_SLEEP_SECONDS  = 20.0   # durée réelle avant oubli définitif
    MAX_PLAYERS = 25

    SPATIAL_MAX_DIST = 150.0  # V5.2 (13/09/2026) : 80px (ancienne valeur
    # de production, via vision/tracker.py) causait 77,5% des blocages
    # d'assignation par MAX_PLAYERS_STEAL (vol d'identite) sur un
    # diagnostic instrumente. Teste empiriquement sur 11 matchs de
    # reference (80/100/120/150/200px) : 80->100px ne change RIEN
    # (aucun candidat ne bascule) ; 150px reduit la fragmentation de
    # ~32% (103->70 ID median) et les vols de ~80% (427->84 median),
    # pour une hausse mesuree du risque de fusion a tort de seulement
    # ~7% (metrique quasi-ex-aequo) et une derive de couleur des
    # appariements acceptes (col_d_P99) stable. Voir
    # ANALYSE_NOUVELLE_ARCHITECTURE_DETECTION.md section 3.6.
    THRESHOLD_ACTIVE = 90.0
    THRESHOLD_SLEEP  = 120.0

    CALIB_FRAMES     = 50
    CALIB_MIN_SAMPLE = 8

    def __init__(self, fps=25):
        # V5.2 (13/09/2026) : TTL calculés en fonction du fps EFFECTIF
        # (rythme réel des appels à process(), pas le fps natif de la
        # vidéo) — corrige un bug où TTL_ACTIVE/TTL_SLEEP étaient des
        # constantes fixes en nombre d'appels, implicitement calibrées
        # pour un appel par frame à 25fps. Si process() est appelé à un
        # rythme réduit (frame_skip en production, ou échantillonnage
        # dans un test), les fenêtres de mémoire étaient en réalité
        # étirées d'un facteur egal au frame_skip - potentiellement
        # ~4-5x trop longues en production (frame_skip=5, ~6fps
        # effectif). Passer ici le fps EFFECTIF (appels/seconde réels,
        # pas le fps natif de la vidéo) pour un calcul correct.
        self.TTL_ACTIVE = max(1, int(round(self.TTL_ACTIVE_SECONDS * fps)))
        self.TTL_SLEEP  = max(1, int(round(self.TTL_SLEEP_SECONDS * fps)))
        self.fps         = fps
        self.memory      = {}
        self.next_id     = 0
        self.frame_count = 0

        # Calibration équipes
        self._team_colors_calibrated = False
        self._team_color_samples     = []
        self._team_centroids         = None

        # V5.1 DIAGNOSTIC : distribution reelle de abs(d0-d1), pour calibrer
        # le seuil d'ambiguite (actuellement 15.0, suspecte trop large -
        # seulement 25/665 track_id ont recu une equipe sur un match test).
        # Retirer une fois le bon seuil determine.
        self._diag_gaps = []
        self._diag_calibration = None
        self._diag_decisions = {"team0": 0, "team1": 0, "none_ambigu": 0, "gk": 0}

        # jersey_map intégré : {reid_id → jersey_number}
        # Permet de stabiliser les IDs via les numéros détectés
        self._jersey_map             = {}
        # Mapping inverse : jersey_number → reid_id canonique
        self._jersey_to_canonical    = {}

    def set_spatial_max_dist(self, dist):
        self.SPATIAL_MAX_DIST = max(100.0, min(400.0, float(dist)))

    def update_jersey_map(self, jersey_map):
        """
        Intègre le jersey_map externe (depuis Gemini OCR) dans le ReID.
        Permet de lier les IDs numériques aux numéros de maillots.
        """
        for pid, jersey in jersey_map.items():
            pid_str    = str(pid)
            jersey_str = str(jersey) if jersey is not None else None
            if jersey_str:
                self._jersey_map[pid_str] = jersey_str
                # Premier ID avec ce jersey = canonique
                if jersey_str not in self._jersey_to_canonical:
                    self._jersey_to_canonical[jersey_str] = pid_str

    # ─────────────────────────────────────────
    # COULEUR MAILLOT (torse)
    # ─────────────────────────────────────────
    def _extract_color(self, frame, bbox):
        """
        Feature couleur 19D = [hist_H_maillot(16D), LAB_mean(3D)×4].
        Torse uniquement (20-55%), masque elliptique central.
        LAB canal A discrimine vert vs rouge/bordeaux indépendamment lumière.

        FIX : x1/x2/y1/y2 bornés à [0, largeur/hauteur] AVANT le slicing.
        Sans ça, une bbox partiellement hors cadre (trace fantôme du
        tracker, ex: x2=-56) déclenche l'indexation négative de NumPy
        (frame[:, 0:-56] compte depuis la fin du tableau), produisant un
        crop d'environ 1864px de large au lieu de quelques pixels — assez
        large pour passer le contrôle `w < 8` sans être détecté, corrompant
        le vecteur couleur utilisé pour la réidentification. Même bug
        diagnostiqué et corrigé dans rendering/overlay.py.
        """
        import cv2 as _cv2
        try:
            h_f, w_f = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w_f))
            x2 = max(0, min(x2, w_f))
            y1 = max(0, min(y1, h_f))
            y2 = max(0, min(y2, h_f))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1:
                return np.zeros(19, dtype=np.float32)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return np.zeros(19, dtype=np.float32)
            h, w = crop.shape[:2]
            if h < 20 or w < 8:
                return np.zeros(19, dtype=np.float32)

            # Torse uniquement : 20-55% hauteur, 20-80% largeur
            torso = crop[int(h*0.20):int(h*0.55), int(w*0.20):int(w*0.80)]
            if torso.size == 0:
                return np.zeros(19, dtype=np.float32)

            # Histogramme teinte H (16 bins)
            hsv  = _cv2.cvtColor(torso, _cv2.COLOR_BGR2HSV)
            S, V = hsv[:,:,1], hsv[:,:,2]
            H    = hsv[:,:,0]
            mask = (S > 40) & (V > 20) & (V < 240)
            if mask.sum() < 5:
                mask = V < 230
            if mask.sum() >= 5:
                h_vals = H[mask].astype(np.float32)
                hist   = np.histogram(h_vals, bins=16, range=(0,180))[0].astype(np.float32)
                if hist.sum() > 0: hist /= hist.sum()
            else:
                hist = np.zeros(16, dtype=np.float32)

            # Mean LAB normalisé (canal A = discriminant vert/rouge)
            lab      = _cv2.cvtColor(torso, _cv2.COLOR_BGR2LAB).astype(np.float32)
            mean_lab = lab.mean(axis=(0,1))
            lab_feat = np.array([
                mean_lab[0] / 255.0,
                (mean_lab[1] - 128.0) / 128.0,
                (mean_lab[2] - 128.0) / 128.0,
            ], dtype=np.float32) * 4.0  # pondérer

            return np.concatenate([hist, lab_feat])  # 19D

        except Exception:
            return np.zeros(19, dtype=np.float32)

    # ─────────────────────────────────────────
    # CALIBRATION ÉQUIPES (KMeans dynamique)
    # ─────────────────────────────────────────
    def _calibrate_teams(self):
        if len(self._team_color_samples) < self.CALIB_MIN_SAMPLE:
            return

        samples  = np.array(self._team_color_samples, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        compacite, labels, centroids = cv2.kmeans(
            samples, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )

        self._team_centroids         = centroids
        self._team_colors_calibrated = True

        # V5.1 DIAGNOSTIC - le print precedent affichait des valeurs
        # CODEES EN DUR (BGR 128/128/128 et 64/64/64), jamais les vraies
        # centroides - corrige ici avec les vraies stats de calibration,
        # demandees explicitement pour diagnostiquer le desequilibre
        # 41.1% team=0 vs 3.6% team=1 observe sur Raeren.
        labels_flat = labels.flatten()
        n_cluster0  = int((labels_flat == 0).sum())
        n_cluster1  = int((labels_flat == 1).sum())
        dist_entre_centroides = float(np.linalg.norm(centroids[0] - centroids[1]))

        # Sauvegarde pour restitution via stats() apres le match complet
        self._diag_calibration = {
            "n_samples_total":       len(samples),
            "n_cluster0":            n_cluster0,
            "n_cluster1":            n_cluster1,
            "ratio_cluster0_pct":    round(100 * n_cluster0 / len(samples), 1),
            "ratio_cluster1_pct":    round(100 * n_cluster1 / len(samples), 1),
            "distance_entre_centroides": round(dist_entre_centroides, 3),
            "compacite_kmeans":      float(compacite),
        }

        print(f"  ReID équipes calibrées : {len(samples)} samples au total → "
              f"cluster0={n_cluster0} ({100*n_cluster0/len(samples):.1f}%)  "
              f"cluster1={n_cluster1} ({100*n_cluster1/len(samples):.1f}%)  "
              f"distance_centroides={dist_entre_centroides:.3f}")

    def _infer_team(self, color):
        c = color.astype(np.float32)

        if not self._team_colors_calibrated:
            # FIX V5.1 : l'ancien filtre `np.any(c > 10)` supposait un vecteur
            # BGR brut (0-255). Le vecteur couleur actuel est un histogramme
            # NORMALISE (somme=1 si valide, tout-zero si _extract_color a
            # echoue) + LAB pondere (~[-4,4]) - AUCUNE composante ne depasse
            # jamais 10 dans ce nouveau format, donc l'ancien filtre etait
            # TOUJOURS faux, `_team_color_samples` ne se remplissait jamais,
            # et la calibration ne se declenchait JAMAIS de tout un match
            # (confirme : teams_calibrated=False apres 1800s sur Raeren,
            # cf. reid_diag.json). Nouveau test : le vecteur est valide si
            # sa portion histogramme (16 premieres valeurs) somme a ~1.0,
            # ce qui echoue proprement sur le sentinel tout-zero.
            if c[:16].sum() > 0.9:
                self._team_color_samples.append(c)
            if (len(self._team_color_samples) >= self.CALIB_MIN_SAMPLE
                    and self.frame_count >= self.CALIB_FRAMES):
                self._calibrate_teams()
            return None

        d0 = np.linalg.norm(c - self._team_centroids[0])
        d1 = np.linalg.norm(c - self._team_centroids[1])

        # Couleur très éloignée des deux équipes → probable gardien
        # FIX V5.1 : multiplicateur 0.6 -> 1.5. Avec 0.6 et une distance
        # entre centroides typiquement petite (~1.7 mesure sur Raeren),
        # le seuil (~1.0) etait deja depasse par 44.2% des joueurs NORMAUX
        # (statistiquement absurde - 2 gardiens sur 22+ joueurs attendus).
        # 1.5 reste une estimation raisonnee, pas une valeur validee -
        # a reconsiderer si le pourcentage "gk" du prochain run reste
        # trop eloigne d'un ordre de grandeur de ~5-10%.
        dist_threshold = np.linalg.norm(
            self._team_centroids[0] - self._team_centroids[1]
        ) * 1.5
        if d0 > dist_threshold and d1 > dist_threshold:
            self._diag_decisions["gk"] += 1  # V5.1 DIAGNOSTIC
            return "gk"

        gap = abs(d0 - d1)
        self._diag_gaps.append(gap)  # V5.1 DIAGNOSTIC - a retirer une fois stabilise
        # FIX V5.1 (2e correction) : seuil recalibre sur donnees reelles
        # (Raeren, apres correction du filtre de collecte) - percentiles
        # mesures (10/25/50/75/90) = [0.68, 1.15, 1.25, 1.48, 1.59].
        # L'ANCIEN seuil (15.0) etait ~10x plus grand que le 90e percentile
        # reel -> 100% des comparaisons tombaient dessous -> aucune equipe
        # jamais assignee. Nouveau seuil (0.3) : nettement sous le 10e
        # percentile (0.68), ne rejette que les cas vraiment ambigus.
        # A RE-VALIDER sur 2-3 matchs supplementaires avant de considerer
        # ce chiffre comme definitif.
        if gap < 0.3:
            self._diag_decisions["none_ambigu"] += 1  # V5.1 DIAGNOSTIC
            return None

        decision = 0 if d0 < d1 else 1
        self._diag_decisions["team0" if decision == 0 else "team1"] += 1  # V5.1 DIAGNOSTIC
        return decision

    # ─────────────────────────────────────────
    # EMBEDDING HISTOGRAMME
    # ─────────────────────────────────────────
    def _extract_embedding(self, frame, bbox):
        """FIX : même bornage correct que _extract_color (voir plus haut) —
        x1/x2/y1/y2 bornés à [0, largeur/hauteur] avant slicing, pour éviter
        l'indexation négative de NumPy sur les bbox partiellement hors cadre.

        V5.2 (13/09/2026) : utilise OSNet (embedding profond, distingue
        les joueurs du même maillot) si disponible, sinon repli sur
        l'histogramme de couleur (comportement historique, insuffisant
        pour maillots identiques — voir commentaire en tête de fichier)."""
        h_f, w_f = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w_f))
        x2 = max(0, min(x2, w_f))
        y1 = max(0, min(y1, h_f))
        y2 = max(0, min(y2, h_f))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(512)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512)

        extractor = _charger_osnet()
        if extractor is not None:
            try:
                return _extraire_embeddings_batch([crop], extractor)[0]
            except Exception as e:
                print(f"  [REID] ⚠️ erreur extraction OSNet, repli histogramme : {e}")

        # Repli historique : histogramme de couleur
        crop_resized = cv2.resize(crop, (32, 64))
        hist = cv2.calcHist(
            [crop_resized], [0, 1, 2], None,
            [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        return cv2.normalize(hist, hist).flatten()

    # ─────────────────────────────────────────
    # CONTRAINTE SPATIALE
    # ─────────────────────────────────────────
    def _spatial_gate(self, c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2)) < self.SPATIAL_MAX_DIST

    # ─────────────────────────────────────────
    # SCORE GLOBAL
    # ─────────────────────────────────────────
    def _compute_score(self, det, mem):
        pos_d = np.linalg.norm(np.array(det["center"]) - np.array(mem["center"]))
        col_d = np.linalg.norm(det["color"]     - mem["color"])
        emb_d = np.linalg.norm(det["embedding"] - mem["embedding"])

        frames_absent = self.frame_count - mem["last_seen"]
        penalty       = 1.0 + (frames_absent / self.TTL_SLEEP) * 0.5

        return (pos_d * 0.5 + col_d * 0.3 + emb_d * 0.2) * penalty

    # ─────────────────────────────────────────
    # CLEANUP TTL
    # ─────────────────────────────────────────
    def _cleanup_memory(self):
        to_del = [
            pid for pid, mem in self.memory.items()
            if self.frame_count - mem["last_seen"] > self.TTL_SLEEP
        ]
        for pid in to_del:
            del self.memory[pid]

    # ─────────────────────────────────────────
    # ASSIGNATION ID
    # Amélioration : si deux IDs ont le même jersey → retourner le canonique
    # ─────────────────────────────────────────
    def _assign_id(self, det):
        active_count = sum(
            1 for m in self.memory.values()
            if self.frame_count - m["last_seen"] <= self.TTL_ACTIVE
        )

        best_id    = None
        best_score = float("inf")

        for pid, mem in self.memory.items():
            if not self._spatial_gate(det["center"], mem["center"]):
                continue

            dt = det.get("team")
            mt = mem.get("team")
            if dt is not None and mt is not None and dt != mt:
                continue

            frames_absent = self.frame_count - mem["last_seen"]
            threshold = (self.THRESHOLD_ACTIVE
                         if frames_absent <= self.TTL_ACTIVE
                         else self.THRESHOLD_SLEEP)

            score = self._compute_score(det, mem)
            if score < best_score and score < threshold:
                best_score = score
                best_id    = pid

        if best_id is not None:
            # Mise à jour mémoire avec lissage exponentiel sur la couleur
            alpha = 0.3
            mem   = self.memory[best_id]
            mem["center"]    = det["center"]
            mem["color"]     = alpha * det["color"] + (1 - alpha) * mem["color"]
            mem["embedding"] = alpha * det["embedding"] + (1 - alpha) * mem["embedding"]
            # V5.2 (17/09/2026) FIX : "det.get('team') or mem.get('team')"
            # traite a tort une classification FRAICHE team=0 comme
            # absente (0 est faux en Python), retombant sur l'ANCIENNE
            # valeur en memoire au lieu de la mettre a jour - perdait
            # silencieusement des classifications team=0 fraiches a
            # chaque mise a jour. Meme motif que le bug deja documente
            # et corrige ailleurs dans ce projet (team_cluster.py,
            # "if _team:"). Fix : verification explicite avec "is not None".
            mem["team"]      = det.get("team") if det.get("team") is not None else mem.get("team")
            mem["last_seen"] = self.frame_count
            return best_id

        if active_count >= self.MAX_PLAYERS:
            if self.memory:
                return min(
                    self.memory.keys(),
                    key=lambda p: np.linalg.norm(
                        np.array(det["center"]) - np.array(self.memory[p]["center"])
                    )
                )
            return 0

        pid = self.next_id
        self.memory[pid] = {**det, "last_seen": self.frame_count}
        self.next_id += 1
        return pid

    # ─────────────────────────────────────────
    # PROCESS FRAME
    # ─────────────────────────────────────────
    def process(self, frame, detections):
        self.frame_count += 1
        self._cleanup_memory()

        # Étape 1 (rapide, par détection) : centre, couleur, équipe, crop
        partiels = []
        crops_valides = []
        indices_avec_crop = []
        for i, det in enumerate(detections):
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            _t0 = time.perf_counter()
            color = self._extract_color(frame, bbox)
            _PROFILING_REID["color"] += time.perf_counter() - _t0

            existing_team = det.get("team")
            _t0 = time.perf_counter()
            inferred = self._infer_team(color)
            _PROFILING_REID["infer_team"] += time.perf_counter() - _t0
            team = existing_team if existing_team is not None else inferred
            # V5.2 (17/09/2026) FIX : "not existing_team" traite a tort
            # existing_team=0 (equipe 0, une classification VALIDE) comme
            # "aucune equipe" (0 est faux en Python) - meme motif que le
            # bug deja trouve/corrige ligne ~500 (mem["team"] = det.get(...)
            # or mem.get(...)). Un joueur deja classe equipe 0 pouvait donc
            # se faire marquer is_goalkeeper=True a tort si un seul
            # echantillon de couleur ambigu tombait sur "gk" ce tour-ci.
            if inferred == "gk" and existing_team is None:
                det["is_goalkeeper"] = True

            crop = _extraire_crop(frame, bbox)
            if crop is not None:
                crops_valides.append(crop)
                indices_avec_crop.append(i)

            partiels.append({"det": det, "center": center, "color": color, "team": team})

        # Étape 2 (LE CORRECTIF) : TOUS les embeddings OSNet de cette
        # frame en UN SEUL appel groupé, au lieu d'un appel par joueur
        # (V5.2, 13/09/2026 — voir _extraire_embeddings_batch pour la
        # justification empirique : 95,6% du temps total auparavant).
        _t0 = time.perf_counter()
        extractor = _charger_osnet()
        embeddings_batch = None
        if extractor is not None and crops_valides:
            try:
                embeddings_batch = _extraire_embeddings_batch(crops_valides, extractor)
            except Exception as e:
                print(f"  [REID] ⚠️ erreur extraction OSNet par lot, repli histogramme : {e}")

        embeddings_par_indice = {}
        if embeddings_batch is not None:
            for idx, emb in zip(indices_avec_crop, embeddings_batch):
                embeddings_par_indice[idx] = emb
        else:
            # Repli histogramme (ou OSNet indisponible) : par crop,
            # comme le comportement historique — pas de gain de lot
            # possible ici (cv2.calcHist est déjà peu coûteux par appel).
            for idx in indices_avec_crop:
                det_repli = partiels[idx]["det"]
                embeddings_par_indice[idx] = self._extract_embedding(
                    frame, det_repli.get("bbox", [0, 0, 0, 0]))
        _PROFILING_REID["embedding"] += time.perf_counter() - _t0
        _PROFILING_REID_N["embedding"] += 1

        # Étape 3 (rapide, par détection) : assignation d'identité
        results = []
        for i, partiel in enumerate(partiels):
            embedding = embeddings_par_indice.get(i, np.zeros(512))
            enriched = {
                "center":    partiel["center"],
                "color":     partiel["color"],
                "embedding": embedding,
                "team":      partiel["team"],
            }

            _t0 = time.perf_counter()
            reid_id = self._assign_id(enriched)
            _PROFILING_REID["assign_id"] += time.perf_counter() - _t0

            det = partiel["det"]
            results.append({
                **det,
                "id":         reid_id,
                "player_id":  reid_id,
                "tracker_id": det.get("id"),
                "center":     list(partiel["center"]),
                "team":       partiel["team"],
            })

        return results

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────
    def stats(self):
        active   = sum(1 for m in self.memory.values()
                       if self.frame_count - m["last_seen"] <= self.TTL_ACTIVE)
        sleeping = len(self.memory) - active
        result = {
            "total_ids":        self.next_id,
            "in_memory":        len(self.memory),
            "active":           active,
            "sleeping":         sleeping,
            "spatial_max_dist": self.SPATIAL_MAX_DIST,
            "teams_calibrated": self._team_colors_calibrated,
        }
        # V5.1 DIAGNOSTIC - a retirer apres calibration du seuil d'ambiguite
        if self._diag_gaps:
            import numpy as _np_diag
            arr = _np_diag.array(self._diag_gaps)
            result["diag_gap_n"]      = len(arr)
            result["diag_gap_pct_below_15"] = float((arr < 15.0).mean() * 100)
            result["diag_gap_percentiles"] = {
                p: float(_np_diag.percentile(arr, p)) for p in [10, 25, 50, 75, 90]
            }
            print(f"  [DIAG team gap] n={len(arr)}  "
                  f"%<15.0={(arr < 15.0).mean()*100:.1f}%  "
                  f"percentiles(10/25/50/75/90)="
                  f"{[round(float(_np_diag.percentile(arr, p)), 1) for p in [10,25,50,75,90]]}")

        # V5.1 DIAGNOSTIC - rapport complet calibration + classification,
        # pour comprendre le desequilibre team0=41.1%/team1=3.6% observe
        # sur Raeren. A retirer une fois la cause identifiee et corrigee.
        if self._diag_calibration is not None:
            result["diag_calibration"] = self._diag_calibration
        total_decisions = sum(self._diag_decisions.values())
        if total_decisions > 0:
            result["diag_decisions"] = dict(self._diag_decisions)
            result["diag_decisions_pct"] = {
                k: round(100 * v / total_decisions, 1)
                for k, v in self._diag_decisions.items()
            }
            calib = self._diag_calibration or {}
            print(f"\n  [DIAG CALIBRATION+CLASSIFICATION]")
            print(f"  Calibration")
            print(f"  -----------")
            print(f"  samples utilisés       : {calib.get('n_samples_total', 'N/A')}")
            print(f"  cluster 0              : {calib.get('n_cluster0', 'N/A')}")
            print(f"  cluster 1              : {calib.get('n_cluster1', 'N/A')}")
            print(f"  ratio                  : {calib.get('ratio_cluster0_pct', '?')}% / "
                  f"{calib.get('ratio_cluster1_pct', '?')}%")
            print(f"  distance centroides    : {calib.get('distance_entre_centroides', 'N/A')}")
            print(f"\n  Classification (sur {total_decisions} decisions post-calibration)")
            print(f"  --------------")
            print(f"  team 0                 : {result['diag_decisions_pct']['team0']}%  "
                  f"({self._diag_decisions['team0']})")
            print(f"  team 1                 : {result['diag_decisions_pct']['team1']}%  "
                  f"({self._diag_decisions['team1']})")
            print(f"  unknown (ambigu)       : {result['diag_decisions_pct']['none_ambigu']}%  "
                  f"({self._diag_decisions['none_ambigu']})")
            print(f"  gk                     : {result['diag_decisions_pct']['gk']}%  "
                  f"({self._diag_decisions['gk']})")
            if arr_present := self._diag_gaps:
                import numpy as _np2
                _a = _np2.array(arr_present)
                print(f"\n  distance d0/d1 (gap) percentiles(10/25/50/75/90) : "
                      f"{[round(float(_np2.percentile(_a, p)), 2) for p in [10,25,50,75,90]]}")

        return result

    # ─────────────────────────────────────────
    # TEAM DISTRIBUTION
    # ─────────────────────────────────────────
    def get_team_distribution(self):
        dist = defaultdict(int)
        for mem in self.memory.values():
            t = mem.get("team")
            if t is not None:
                dist[t] += 1
        return dict(dist)

    def get_team_centroids(self):
        """V5.2 : expose les 2 centroides calibres (vecteurs 19D
        histogramme-teinte + LAB, cf. _extract_color/_calibrate_teams)
        pour un usage externe - notamment l'appariement avec les
        couleurs nommees confirmees par Gemini pendant la pre-analyse
        (voir analysis/team_color_matching.py). Retourne None si la
        calibration n'a pas encore eu lieu (pas assez d'echantillons
        ou de frames vues)."""
        if not self._team_colors_calibrated or self._team_centroids is None:
            return None
        return [self._team_centroids[0].copy(), self._team_centroids[1].copy()]

    def reset(self):
        self.memory                  = {}
        self.next_id                 = 0
        self.frame_count             = 0
        self._team_colors_calibrated = False
        self._team_color_samples     = []
        self._team_centroids         = None
        self._jersey_map             = {}
        self._jersey_to_canonical    = {}

# ─────────────────────────────────────────
# WRAPPER — appelé depuis pipeline.py
# ─────────────────────────────────────────
def reidentify_players(events):
    """
    Wrapper stateless pour le pipeline.
    Reconstruit les IDs joueurs depuis les events en utilisant
    position + couleur + numéro de maillot.

    Note : sans frame vidéo, seule la cohérence temporelle
    des positions est utilisée (pas d'embedding visuel).
    """
    if not events:
        return events

    # Index par player_id existant → canonical via jersey si dispo
    jersey_groups = {}  # jersey_number → premier player_id vu
    remaps        = {}  # old_pid → canonical_pid

    for e in events:
        pid    = str(e.get("player", "")) if e.get("player") is not None else None
        jersey = e.get("jersey") or e.get("jersey_number")
        if not pid:
            continue
        if jersey:
            jersey_str = str(jersey)
            if jersey_str not in jersey_groups:
                jersey_groups[jersey_str] = pid
            canonical = jersey_groups[jersey_str]
            if pid != canonical:
                remaps[pid] = canonical

    if remaps:
        n = 0
        for e in events:
            pid = str(e.get("player", "")) if e.get("player") is not None else None
            if pid and pid in remaps:
                e["player"] = remaps[pid]
                n += 1
        print(f"  ReID : {n} events remappés ({len(remaps)} fusions jersey)")

    return events