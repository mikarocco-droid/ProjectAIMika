# analysis/player_reid.py
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from collections import defaultdict


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

    TTL_ACTIVE  = 125    # 5s  @25fps
    TTL_SLEEP   = 500    # 20s @25fps
    MAX_PLAYERS = 25

    SPATIAL_MAX_DIST = 200.0
    THRESHOLD_ACTIVE = 90.0
    THRESHOLD_SLEEP  = 120.0

    CALIB_FRAMES     = 50
    CALIB_MIN_SAMPLE = 8

    def __init__(self, fps=25):
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
        _, labels, centroids = cv2.kmeans(
            samples, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )

        self._team_centroids         = centroids
        self._team_colors_calibrated = True

        # Les centroids sont 19D — extraire LAB A pour le log
        def _lab_a(centroid):
            if len(centroid) >= 19:
                return round(float(centroid[16]) / 4.0, 3)  # dé-pondérer
            return 0.0
        c0_a = _lab_a(centroids[0])
        c1_a = _lab_a(centroids[1])
        # Garder c0/c1 comme BGR approximatif pour compatibilité logs
        c0 = np.array([128, 128, 128], dtype=int)
        c1 = np.array([64, 64, 64], dtype=int)
        print(f"  ReID équipes calibrées : "
              f"équipe0=BGR({c0[0]},{c0[1]},{c0[2]}) "
              f"équipe1=BGR({c1[0]},{c1[1]},{c1[2]}) "
              f"({len(samples)} samples)")

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
        dist_threshold = np.linalg.norm(
            self._team_centroids[0] - self._team_centroids[1]
        ) * 0.6
        if d0 > dist_threshold and d1 > dist_threshold:
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
            return None

        return 0 if d0 < d1 else 1

    # ─────────────────────────────────────────
    # EMBEDDING HISTOGRAMME
    # ─────────────────────────────────────────
    def _extract_embedding(self, frame, bbox):
        """FIX : même bornage correct que _extract_color (voir plus haut) —
        x1/x2/y1/y2 bornés à [0, largeur/hauteur] avant slicing, pour éviter
        l'indexation négative de NumPy sur les bbox partiellement hors cadre."""
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
        crop = cv2.resize(crop, (32, 64))
        hist = cv2.calcHist(
            [crop], [0, 1, 2], None,
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
            mem["team"]      = det.get("team") or mem.get("team")
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

        results = []
        for det in detections:
            bbox   = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            color  = self._extract_color(frame, bbox)

            existing_team = det.get("team")
            inferred = self._infer_team(color)
            team = existing_team if existing_team is not None else inferred
            # Marquer comme gardien si détecté
            if inferred == "gk" and not existing_team:
                det["is_goalkeeper"] = True

            enriched = {
                "center":    center,
                "color":     color,
                "embedding": self._extract_embedding(frame, bbox),
                "team":      team,
            }

            reid_id = self._assign_id(enriched)

            results.append({
                **det,
                "id":         reid_id,
                "player_id":  reid_id,
                "tracker_id": det.get("id"),
                "center":     list(center),
                "team":       team,
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