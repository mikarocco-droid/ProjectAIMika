# vision/tracker.py
# -*- coding: utf-8 -*-
# ByteTrack natif — zéro dépendance externe, Python pur + numpy + scipy
# Utilisé sur tous les environnements (Kaggle, RunPod, prod)

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from analysis.player_reid import PlayerReID


# ─────────────────────────────────────────────────────────────────────────────
# KALMAN FILTER (simplifié pour tracking 2D boîtes)
# ─────────────────────────────────────────────────────────────────────────────
class KalmanBox:
    """Filtre de Kalman pour une boîte [cx, cy, w, h]."""

    def __init__(self, bbox):
        # État : [cx, cy, w, h, vx, vy, vw, vh]
        self.x  = np.array([
            (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2,
            bbox[2]-bbox[0],     bbox[3]-bbox[1],
            0., 0., 0., 0.
        ], dtype=float)
        # Matrice de transition
        self.F  = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1.0
        # Matrice d'observation (cx,cy,w,h)
        self.H  = np.eye(4, 8)
        # Covariance état
        self.P  = np.diag([10.,10.,10.,10., 1e4,1e4,1e4,1e4])
        # Bruit processus
        self.Q  = np.diag([1.,1.,1.,1., 0.01,0.01,0.01,0.01])
        # Bruit mesure
        self.R  = np.diag([1.,1.,10.,10.])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_ltrb()

    def update(self, bbox):
        z = np.array([
            (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2,
            bbox[2]-bbox[0],     bbox[3]-bbox[1],
        ])
        y  = z - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def _to_ltrb(self):
        cx, cy, w, h = self.x[:4]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    def get_ltrb(self):
        return self._to_ltrb()


# ─────────────────────────────────────────────────────────────────────────────
# TRACK
# ─────────────────────────────────────────────────────────────────────────────
class _Track:
    _next_id = 1

    def __init__(self, bbox, conf):
        self.id        = _Track._next_id
        _Track._next_id += 1
        self.kalman    = KalmanBox(bbox)
        self.conf      = conf
        self.hits      = 1
        self.age       = 0       # frames depuis dernière mise à jour
        self.state     = "tentative"  # tentative → confirmed → lost

    def predict(self):
        self.kalman.predict()
        self.age += 1

    def update(self, bbox, conf):
        self.kalman.update(bbox)
        self.conf  = conf
        self.hits += 1
        self.age   = 0
        if self.hits >= 3:
            self.state = "confirmed"

    def get_ltrb(self):
        return self.kalman.get_ltrb()


# ─────────────────────────────────────────────────────────────────────────────
# BYTETRACK NATIF
# ─────────────────────────────────────────────────────────────────────────────
def _iou(b1, b2):
    """IoU entre deux boîtes [x1,y1,x2,y2]."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


def _iou_matrix(tracks, dets):
    M = np.zeros((len(tracks), len(dets)))
    for i, t in enumerate(tracks):
        tb = t.get_ltrb()
        for j, d in enumerate(dets):
            M[i, j] = _iou(tb, d["bbox"])
    return M


def _hungarian(cost):
    """Matching par algorithme hongrois. Retourne (matched_t, matched_d, unmatched_t, unmatched_d)."""
    if cost.size == 0:
        return [], [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_ind, col_ind = linear_sum_assignment(-cost)
    matched_t, matched_d = [], []
    unmatched_t = list(range(cost.shape[0]))
    unmatched_d = list(range(cost.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= 0.3:
            matched_t.append(r); matched_d.append(c)
            if r in unmatched_t: unmatched_t.remove(r)
            if c in unmatched_d: unmatched_d.remove(c)
    return matched_t, matched_d, unmatched_t, unmatched_d


class _ByteTracker:
    """ByteTrack natif — IoU + Kalman, deux queues high/low confidence."""

    def __init__(self, track_thresh=0.35, track_buffer=150, match_thresh=0.85):
        self.track_thresh  = track_thresh
        self.track_buffer  = track_buffer   # max age avant suppression
        self.match_thresh  = match_thresh
        self.tracks        = []             # tracks actifs
        _Track._next_id    = 1

    def update(self, detections):
        """
        detections : liste de {"bbox":[x1,y1,x2,y2], "conf":float}
        retourne   : liste de {"id", "bbox", "center", "conf"}
        """
        # Séparer high/low confidence
        high = [d for d in detections if d["conf"] >= self.track_thresh]
        low  = [d for d in detections if d["conf"] <  self.track_thresh]

        # Prédire tous les tracks
        for t in self.tracks:
            t.predict()

        active = [t for t in self.tracks if t.age <= 1]
        lost   = [t for t in self.tracks if t.age >  1]

        # 1. Associer high confidence ↔ tracks actifs
        if active and high:
            iou = _iou_matrix(active, high)
            mt, md, umt, umd = _hungarian(iou)
            for ti, di in zip(mt, md):
                active[ti].update(high[di]["bbox"], high[di]["conf"])
            unmatched_active = [active[i] for i in umt]
            unmatched_high   = [high[i]   for i in umd]
        else:
            unmatched_active = list(active)
            unmatched_high   = list(high)

        # 2. Associer low confidence ↔ tracks non matchés (récupère occlusions)
        if unmatched_active and low:
            iou2 = _iou_matrix(unmatched_active, low)
            mt2, md2, umt2, _ = _hungarian(iou2)
            for ti, di in zip(mt2, md2):
                unmatched_active[ti].update(low[di]["bbox"], low[di]["conf"])
            unmatched_active = [unmatched_active[i] for i in umt2]

        # 3. Associer high confidence non matché ↔ lost tracks
        if lost and unmatched_high:
            iou3 = _iou_matrix(lost, unmatched_high)
            mt3, md3, _, umd3 = _hungarian(iou3)
            for ti, di in zip(mt3, md3):
                lost[ti].update(unmatched_high[di]["bbox"], unmatched_high[di]["conf"])
                self.tracks.append(lost[ti])
            unmatched_high = [unmatched_high[i] for i in umd3]

        # 4. Créer nouveaux tracks pour détections non appariées
        for d in unmatched_high:
            self.tracks.append(_Track(d["bbox"], d["conf"]))

        # 5. Supprimer tracks trop vieux
        self.tracks = [t for t in self.tracks if t.age <= self.track_buffer]

        # 6. Retourner tracks confirmés
        results = []
        for t in self.tracks:
            if t.state == "confirmed":
                x1, y1, x2, y2 = t.get_ltrb()
                results.append({
                    "id":     t.id,
                    "bbox":   [x1, y1, x2, y2],
                    "center": [(x1+x2)/2, (y1+y2)/2],
                    "conf":   t.conf,
                })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# TRACKER PUBLIC
# ─────────────────────────────────────────────────────────────────────────────
class Tracker:
    def __init__(self):
        self.tracker = _ByteTracker(
            track_thresh = 0.35,
            track_buffer = 150,
            match_thresh = 0.85,
        )
        self.mode = "bytetrack_native"
        print("  Tracker : ByteTrack natif (IoU+Kalman, zéro réseau)")

        # ReID hybride couleur maillot en aval
        self.reid = PlayerReID(fps=config.FPS)
        self.reid.set_spatial_max_dist(80)

    def update(self, players, frame):
        if not players:
            return []

        results = self.tracker.update(players)

        # ReID hybride avec la frame courante
        results = self.reid.process(frame, results)

        # Normaliser ids pour cohérence pipeline
        for r in results:
            if "player_id" in r:
                r["tracker_id"] = r.get("id", r["player_id"])
                r["id"]         = r["player_id"]

        return results

    def reset(self):
        self.tracker = _ByteTracker(
            track_thresh = 0.35,
            track_buffer = 150,
            match_thresh = 0.85,
        )
        _Track._next_id = 1