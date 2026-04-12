# ai/learning_model.py
# -*- coding: utf-8 -*-

import os
import json
import math
import random
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────
# SEUILS PAR DÉFAUT
# ─────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    "football": {
        "goal_frames_min":   8,
        "shot_cooldown":     3.0,
        "goal_cooldown":   150.0,
        "ball_speed_min":    0.02,
        "player_near_goal":  0.15,
        "spatial_max_dist": 200.0,
    },
    "basketball": {
        "goal_frames_min":    5,
        "shot_cooldown":      1.5,
        "goal_cooldown":      5.0,
        "ball_speed_min":     0.03,
        "player_near_goal":   0.10,
        "spatial_max_dist": 150.0,
    },
    "handball": {
        "goal_frames_min":    8,
        "shot_cooldown":      2.0,
        "goal_cooldown":     20.0,
        "ball_speed_min":     0.025,
        "player_near_goal":   0.12,
        "spatial_max_dist": 180.0,
    },
    "rugby": {
        "goal_frames_min":   10,
        "shot_cooldown":      3.0,
        "goal_cooldown":     60.0,
        "ball_speed_min":     0.02,
        "player_near_goal":   0.20,
        "spatial_max_dist": 250.0,
    },
}

THRESHOLD_BOUNDS = {
    "football": {
        "goal_frames_min":   (4,    20),
        "shot_cooldown":     (1.0,  8.0),
        "goal_cooldown":     (60.0, 300.0),
        "ball_speed_min":    (0.01, 0.05),
        "player_near_goal":  (0.08, 0.25),
        "spatial_max_dist":  (100,  300),
    },
    "basketball": {
        "goal_frames_min":   (3,    10),
        "shot_cooldown":     (0.5,  4.0),
        "goal_cooldown":     (2.0,  15.0),
        "ball_speed_min":    (0.01, 0.06),
        "player_near_goal":  (0.05, 0.20),
        "spatial_max_dist":  (80,   200),
    },
    "handball": {
        "goal_frames_min":   (4,    15),
        "shot_cooldown":     (0.5,  5.0),
        "goal_cooldown":     (10.0, 60.0),
        "ball_speed_min":    (0.01, 0.06),
        "player_near_goal":  (0.06, 0.20),
        "spatial_max_dist":  (80,   250),
    },
    "rugby": {
        "goal_frames_min":   (5,    20),
        "shot_cooldown":     (1.0,  8.0),
        "goal_cooldown":     (30.0, 180.0),
        "ball_speed_min":    (0.01, 0.05),
        "player_near_goal":  (0.10, 0.35),
        "spatial_max_dist":  (100,  350),
    },
}

DEFAULT_TYPE_WEIGHTS = {
    "goal":            10.0,
    "shot":             5.0,
    "key_pass":         4.0,
    "progressive_run":  3.0,
    "interception":     3.0,
    "dribble":          2.0,
    "pass":             1.0,
    "possession":       0.5,
}


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class MatchLearner:

    def __init__(self, sport="football", base_dir="outputs/learning"):
        self.sport    = sport
        self.base_dir = os.path.join(base_dir, sport)
        os.makedirs(self.base_dir, exist_ok=True)

        self._paths = {
            "events":       os.path.join(self.base_dir, "events.json"),
            "xg":           os.path.join(self.base_dir, "xg_model.json"),
            "thresholds":   os.path.join(self.base_dir, "thresholds.json"),
            "fp_zones":     os.path.join(self.base_dir, "false_positive_zones.json"),
            "players":      os.path.join(self.base_dir, "player_profiles.json"),
            "teams":        os.path.join(self.base_dir, "team_patterns.json"),
            "reid":         os.path.join(self.base_dir, "reid_calibration.json"),
            "ocr":          os.path.join(self.base_dir, "ocr_corrections.json"),
            "highlights":   os.path.join(self.base_dir, "highlight_scores.json"),
            "history":      os.path.join(self.base_dir, "match_history.json"),
            "type_weights": os.path.join(self.base_dir, "type_weights.json"),
            "xg_training":  os.path.join(self.base_dir, "xg_training_data.json"),
            # Données features avancées pour sklearn
            "xg_advanced":  os.path.join(self.base_dir, "xg_advanced_data.json"),
        }
        self._sklearn_model = None
        self._load()
        # Entraîner le modèle sklearn si assez de données
        self._train_advanced_if_possible()

    # ─────────────────────────────────────────
    # LOAD / SAVE
    # ─────────────────────────────────────────
    def _load(self):
        def _read(key, default):
            p = self._paths[key]
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        return json.load(f)
                except Exception:
                    return default
            return default

        self.events_db     = _read("events",       [])
        self.xg_model      = _read("xg",            {"w0": 0.0, "w1": -2.0, "w2": 1.0, "n_samples": 0})
        self.thresholds    = _read("thresholds",    DEFAULT_THRESHOLDS.get(self.sport, DEFAULT_THRESHOLDS["football"]).copy())
        self.fp_zones      = _read("fp_zones",      [])
        self.players_db    = _read("players",       {})
        self.teams_db      = _read("teams",         {})
        self.reid_cal      = _read("reid",          {"spatial_max_dist": 200.0, "n_matches": 0})
        self.ocr_db        = _read("ocr",           {})
        self.hl_scores     = _read("highlights",    {"by_type": {}, "by_position": []})
        self.history       = _read("history",       [])
        self.type_weights  = _read("type_weights",  DEFAULT_TYPE_WEIGHTS.copy())
        self.xg_training   = _read("xg_training",  [])
        self.xg_advanced   = _read("xg_advanced",  [])

        defaults = DEFAULT_THRESHOLDS.get(self.sport, DEFAULT_THRESHOLDS["football"])
        for k, v in defaults.items():
            if k not in self.thresholds:
                self.thresholds[k] = v
        if self.thresholds.get("goal_frames_min", 12) > 8:
            self.thresholds["goal_frames_min"] = 8

        self._clamp_all_thresholds()

    def _clamp_all_thresholds(self):
        bounds = THRESHOLD_BOUNDS.get(self.sport, {})
        for k, (lo, hi) in bounds.items():
            if k in self.thresholds:
                self.thresholds[k] = _clamp(self.thresholds[k], lo, hi)

    def _save(self):
        data = {
            "events":       self.events_db,
            "xg":           self.xg_model,
            "thresholds":   self.thresholds,
            "fp_zones":     self.fp_zones,
            "players":      self.players_db,
            "teams":        self.teams_db,
            "reid":         self.reid_cal,
            "ocr":          self.ocr_db,
            "highlights":   self.hl_scores,
            "history":      self.history,
            "type_weights": self.type_weights,
            "xg_training":  self.xg_training,
            "xg_advanced":  self.xg_advanced,
        }
        for key, obj in data.items():
            try:
                with open(self._paths[key], "w") as f:
                    json.dump(obj, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Learning warning : impossible de sauver {key} — {e}")

    # ─────────────────────────────────────────
    # FILTRE QUALITÉ EVENTS
    # ─────────────────────────────────────────
    def _is_quality_event(self, e):
        etype = e.get("type")
        if etype == "goal":
            return e.get("gemini_validated", False) and e.get("gemini_conf", 0) >= 0.85
        if etype == "shot":
            return e.get("gemini_validated", False) and e.get("gemini_conf", 0) >= 0.70
        if etype == "dribble":
            conf = e.get("confidence", e.get("gemini_conf", 1.0))
            if conf < 0.7:
                return False
            return random.random() < 0.30
        if etype == "interception":
            return True
        return False

    # ─────────────────────────────────────────
    # FEATURE ENGINEERING — xG avancé
    # Utilisé pour sklearn LogisticRegression
    # Features : distance, angle, pressure, séquence, phase
    # ─────────────────────────────────────────
    def _compute_advanced_features(self, e, frame_w=1920, frame_h=1080,
                                   pitch_l_m=105.0, goal_w_m=7.32):
        x = float(e.get("x", 0))
        y = float(e.get("y", frame_h / 2))

        # Centre du but le plus proche
        goal_y_px  = frame_h / 2.0
        goal_left  = (0.0,     goal_y_px)
        goal_right = (frame_w, goal_y_px)

        dist_left  = math.hypot(x - goal_left[0],  y - goal_left[1])
        dist_right = math.hypot(x - goal_right[0], y - goal_right[1])
        goal_cx, goal_cy = goal_left if dist_left <= dist_right else goal_right

        # Distance pixels → mètres
        dist_px  = math.hypot(x - goal_cx, y - goal_cy)
        px_per_m = frame_w / max(pitch_l_m, 1e-6)
        dist_m   = dist_px / max(px_per_m, 1e-6)

        # Angle entre les deux poteaux
        half_goal_px = (goal_w_m / pitch_l_m) * frame_w * 0.5
        post1 = (goal_cx, goal_cy - half_goal_px)
        post2 = (goal_cx, goal_cy + half_goal_px)

        v1x, v1y = post1[0] - x, post1[1] - y
        v2x, v2y = post2[0] - x, post2[1] - y

        dot   = v1x * v2x + v1y * v2y
        norm1 = math.hypot(v1x, v1y) + 1e-6
        norm2 = math.hypot(v2x, v2y) + 1e-6
        cos_a = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        angle = math.acos(cos_a)

        # Normalisation + non-linéarités calibrées
        distance_norm   = min(dist_m / pitch_l_m, 1.0)
        angle_norm      = angle / math.pi
        distance_effect = (1.0 - distance_norm) ** 1.3
        angle_effect    = angle_norm ** 1.7

        # Features contextuelles
        pressure    = float(e.get("pressure", 0.0) or 0.0)
        seq_len     = float(e.get("sequence_length", 1) or 1)
        seq_norm    = min(seq_len / 10.0, 1.0)
        phase       = str(e.get("shot_context", e.get("phase", "open_play")) or "open_play")
        is_counter  = 1.0 if "counter" in phase.lower() else 0.0
        is_on_target = 1.0 if e.get("on_target", False) else 0.0

        return [
            distance_effect,   # distance non-linéaire
            angle_effect,      # angle non-linéaire
            pressure,          # pression défensive
            seq_norm,          # longueur séquence normalisée
            is_counter,        # contre-attaque
            is_on_target,      # tir cadré
        ]

    # ─────────────────────────────────────────
    # ENTRAÎNEMENT SKLEARN
    # Actif dès 50 samples, très fiable à 500+
    # ─────────────────────────────────────────
    def _train_advanced_if_possible(self):
        if len(self.xg_advanced) < 50:
            self._sklearn_model = None
            return

        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            X = np.array([d["features"] for d in self.xg_advanced])
            y = np.array([d["is_goal"]  for d in self.xg_advanced])

            # Vérifier qu'on a les deux classes
            if len(set(y)) < 2:
                self._sklearn_model = None
                return

            model = LogisticRegression(max_iter=500, C=1.0)
            model.fit(X, y)
            self._sklearn_model = model
            print(f"  Learning xG avancé : modèle sklearn actif "
                  f"({len(self.xg_advanced)} samples, "
                  f"{int(sum(y))} buts / {int(len(y)-sum(y))} non-buts)")
        except Exception as ex:
            self._sklearn_model = None
            print(f"  Learning xG avancé ignoré : {ex}")

    # ─────────────────────────────────────────
    # HAS ADVANCED XG
    # Retourne True si le modèle sklearn est prêt
    # ─────────────────────────────────────────
    def has_advanced_xg(self):
        return self._sklearn_model is not None

    # ─────────────────────────────────────────
    # PREDICT ADVANCED XG
    # Utilise sklearn LogisticRegression
    # Fallback automatique si modèle pas prêt
    # ─────────────────────────────────────────
    def predict_advanced_xg(self, e, frame_w=1920, frame_h=1080):
        if self._sklearn_model is None:
            return None

        try:
            import numpy as np
            features = self._compute_advanced_features(e, frame_w, frame_h)
            proba    = self._sklearn_model.predict_proba([features])[0][1]
            return float(max(0.01, min(0.99, proba)))
        except Exception:
            return None

    # ─────────────────────────────────────────
    # 1. ENREGISTRER UN MATCH
    # ─────────────────────────────────────────
    def record_match(self, events, summary, fps=25, jersey_map=None,
                     highlights=None, goals_real=None):
        match_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_matches = len(self.history) + 1
        print(f"  Learning : enregistrement match #{n_matches} ({match_id})")

        added = skipped = 0
        for e in events:
            if e.get("type") not in ["goal", "shot", "interception", "dribble"]:
                continue
            if not self._is_quality_event(e):
                skipped += 1
                continue
            self.events_db.append({
                "match_id":         match_id,
                "type":             e.get("type"),
                "x":                e.get("x", 0),
                "y":                e.get("y", 0),
                "xg":               e.get("xg", 0),
                "gemini_validated": e.get("gemini_validated", False),
                "gemini_type":      e.get("gemini_type", ""),
                "gemini_conf":      e.get("gemini_conf", 0),
                "time":             e.get("time", 0),
                "player":           str(e.get("player", "")),
                "team":             e.get("team"),
            })
            added += 1

        if skipped:
            print(f"  Learning qualité : {added} events retenus | {skipped} bruyants ignorés")

        self._update_xg_model(match_id=match_id)
        self._collect_advanced_features(events, match_id)
        self._recalibrate_thresholds(events, summary, fps, goals_real=goals_real)
        self._update_fp_zones(events)
        self._update_player_profiles(events, fps)
        self._update_team_patterns(events, summary)
        self._update_reid_calibration(events)
        self._update_ocr_corrections(jersey_map or {})
        self._update_highlight_scores(highlights or [])
        self._update_type_weights(events)
        self._record_history(match_id, events, summary, highlights or [],
                             goals_real=goals_real)

        # Ré-entraîner le modèle sklearn avec les nouvelles données
        self._train_advanced_if_possible()

        self._save()

        result = {
            "match_id":       match_id,
            "match_number":   n_matches,
            "events_added":   added,
            "events_skipped": skipped,
            "total_events":   len(self.events_db),
            "xg_samples":     self.xg_model["n_samples"],
            "xg_advanced_samples": len(self.xg_advanced),
            "xg_advanced_ready":   self.has_advanced_xg(),
            "thresholds":     self.thresholds,
        }
        print(f"  Learning OK : match #{n_matches} | {added} events | "
              f"total={len(self.events_db)} | "
              f"xG_samples={self.xg_model['n_samples']} | "
              f"xG_avancé={'✅' if self.has_advanced_xg() else f'({len(self.xg_advanced)}/50)'}")
        return result

    # ─────────────────────────────────────────
    # COLLECTE FEATURES AVANCÉES
    # Stocke features + label (is_goal) pour sklearn
    # ─────────────────────────────────────────
    def _collect_advanced_features(self, events, match_id,
                                   frame_w=1920, frame_h=1080):
        """
        Collecte les features avancées pour chaque tir/but.
        is_goal = 1 si le tir est rentré, 0 sinon.
        """
        # Timestamps des buts pour labelliser les tirs proches
        goal_times = [
            e.get("time", 0) for e in events
            if e.get("type") == "goal"
        ]

        new_samples = 0
        for e in events:
            if e.get("type") not in ["shot", "goal"]:
                continue
            if e.get("x", 0) <= 0:
                continue

            features = self._compute_advanced_features(e, frame_w, frame_h)

            # Label : 1 si but, sinon vérifier si un but arrive dans les 5s
            if e.get("type") == "goal":
                is_goal = 1
            else:
                t       = e.get("time", 0)
                is_goal = 1 if any(0 <= gt - t <= 5.0 for gt in goal_times) else 0

            self.xg_advanced.append({
                "match_id": match_id,
                "features": features,
                "is_goal":  is_goal,
                "x":        round(float(e.get("x", 0)), 1),
                "y":        round(float(e.get("y", frame_h/2)), 1),
            })
            new_samples += 1

        # Limite dataset à 5000 samples récents
        self.xg_advanced = self.xg_advanced[-5000:]

        n_goals = sum(1 for d in self.xg_advanced if d.get("is_goal"))
        n_total = len(self.xg_advanced)
        if new_samples > 0:
            print(f"  Learning xG avancé : {n_total} samples "
                  f"({n_goals} buts, taux={n_goals/n_total:.1%})"
                  + (" → modèle actif ✅" if n_total >= 50
                     else f" → encore {50 - n_total} avant activation"))

    # ─────────────────────────────────────────
    # HISTORIQUE
    # ─────────────────────────────────────────
    def _record_history(self, match_id, events, summary, highlights,
                        goals_real=None):
        shots     = sum(1 for e in events if e.get("type") == "shot")
        goals_det = sum(1 for e in events if e.get("type") == "goal")
        passes    = sum(1 for e in events if e.get("type") == "pass")
        real      = goals_real if goals_real is not None else summary.get("goals", goals_det)

        entry = {
            "match_id":        match_id,
            "date":            datetime.now().isoformat(),
            "sport":           self.sport,
            "goals_detected":  goals_det,
            "goals_real":      real,
            "shots_detected":  shots,
            "passes":          passes,
            "players":         summary.get("players", 0),
            "formation":       summary.get("formation", ""),
            "n_highlights":    len(highlights),
            "top_highlight":   highlights[0].get("title", "") if highlights else "",
            "thresholds_used": self.thresholds.copy(),
            "spatial_max_dist": self.reid_cal.get("spatial_max_dist", 200.0),
            "fp_zones":        len(self.fp_zones),
            "xg_samples":      self.xg_model["n_samples"],
            "xg_advanced_samples": len(self.xg_advanced),
        }
        self.history.append(entry)

        n = len(self.history)
        if n >= 2:
            prev      = self.history[-2]
            prev_real = max(prev.get("goals_real", 1), 1)
            cur_acc   = goals_det / max(real, 1) if real > 0 else 0
            prev_acc  = prev["goals_detected"] / prev_real
            delta     = cur_acc - prev_acc
            print(f"  Learning progression : match {n-1}→{n} | "
                  f"précision buts {'+' if delta >= 0 else ''}{delta:.0%}")

    # ─────────────────────────────────────────
    # 2. MODÈLE xG SGD
    # ─────────────────────────────────────────
    def _update_xg_model(self, match_id="", frame_w=1920, frame_h=1080,
                         pitch_l_m=105.0, goal_w_m=7.32):
        shots_this_match = [
            e for e in self.events_db
            if e.get("type") in ["shot", "goal"]
            and e.get("x", 0) > 0
            and e.get("match_id") == match_id
        ]

        for s in shots_this_match:
            x  = float(s.get("x", 0))
            y  = float(s.get("y", frame_h / 2))

            goal_y_px  = frame_h / 2.0
            goal_left  = (0.0,     goal_y_px)
            goal_right = (frame_w, goal_y_px)

            dist_left  = math.hypot(x - goal_left[0],  y - goal_left[1])
            dist_right = math.hypot(x - goal_right[0], y - goal_right[1])
            goal_cx, goal_cy = goal_left if dist_left <= dist_right else goal_right

            dist_px  = math.hypot(x - goal_cx, y - goal_cy)
            px_per_m = frame_w / max(pitch_l_m, 1e-6)
            dist_m   = dist_px / max(px_per_m, 1e-6)

            half_goal_px = (goal_w_m / pitch_l_m) * frame_w * 0.5
            post1 = (goal_cx, goal_cy - half_goal_px)
            post2 = (goal_cx, goal_cy + half_goal_px)

            v1x, v1y = post1[0] - x, post1[1] - y
            v2x, v2y = post2[0] - x, post2[1] - y

            dot   = v1x * v2x + v1y * v2y
            norm1 = math.hypot(v1x, v1y) + 1e-6
            norm2 = math.hypot(v2x, v2y) + 1e-6
            cos_a = max(-1.0, min(1.0, dot / (norm1 * norm2)))
            angle = math.acos(cos_a)

            is_goal = 1 if s.get("type") == "goal" else 0

            self.xg_training.append({
                "match_id": match_id,
                "x":        round(x, 1),
                "y":        round(y, 1),
                "dist_m":   round(dist_m, 2),
                "angle":    round(angle, 4),
                "is_goal":  is_goal,
                "xg_model": round(s.get("xg", 0), 3),
                "pressure": round(s.get("pressure", 0.0), 3),
            })

        n_total = len(self.xg_training)
        n_goals = sum(1 for d in self.xg_training if d.get("is_goal"))
        self.xg_model["n_samples"] = n_total

        if n_total > 0:
            print(f"  Learning xG : {n_total} tirs collectés "
                  f"({n_goals} buts, taux={n_goals/n_total:.1%})"
                  + (" → lance calibrate_xg.py !" if n_total >= 200 else
                     f" → encore {200 - n_total} tirs avant calibration externe"))

        shots_validated = [
            e for e in self.events_db
            if e.get("type") == "shot"
            and e.get("gemini_validated")
            and e.get("x", 0) > 0
            and e.get("gemini_conf", 0) >= 0.70
        ]

        if len(shots_validated) < 20:
            return

        X, y = [], []
        for s in shots_validated:
            x_n  = s["x"] / frame_w
            y_n  = s["y"] / frame_h
            dist = math.sqrt((1.0 - x_n) ** 2 + (0.5 - y_n) ** 2)
            ang  = abs(math.atan2(0.5 - y_n, 1.0 - x_n))
            lbl  = 1 if (s.get("gemini_type") == "shot"
                         and s.get("gemini_conf", 0) > 0.8) else 0
            X.append([dist, ang])
            y.append(lbl)

        w0 = self.xg_model.get("w0", 0.0)
        w1 = self.xg_model.get("w1", -2.0)
        w2 = self.xg_model.get("w2", 1.0)
        lr = 0.01

        for _ in range(50):
            dw0 = dw1 = dw2 = 0.0
            for (d, a), lbl in zip(X, y):
                z    = _clamp(w0 + w1 * d + w2 * a, -100, 100)
                pred = 1 / (1 + math.exp(-z))
                err  = pred - lbl
                dw0 += err
                dw1 += err * d
                dw2 += err * a
            n   = len(X)
            w0 -= lr * dw0 / n
            w1 -= lr * dw1 / n
            w2 -= lr * dw2 / n

        self.xg_model.update({
            "w0": round(_clamp(w0, -10, 10), 4),
            "w1": round(_clamp(w1, -10,  0), 4),
            "w2": round(_clamp(w2,   0, 10), 4),
            "n_samples": n_total,
        })

    # ─────────────────────────────────────────
    # 3. RECALIBRATION SEUILS
    # ─────────────────────────────────────────
    def _recalibrate_thresholds(self, events, summary, fps, goals_real=None):
        goals_det  = sum(1 for e in events if e.get("type") == "goal")
        real       = goals_real if goals_real is not None else summary.get("goals", goals_det)
        shots_det  = sum(1 for e in events if e.get("type") == "shot")
        dur_min    = max(1, summary.get("total_frames", 15000) / fps / 60)
        bounds     = THRESHOLD_BOUNDS.get(self.sport, {})
        changed    = []

        def _update(key, new_val):
            lo, hi = bounds.get(key, (-1e9, 1e9))
            clamped = _clamp(new_val, lo, hi)
            self.thresholds[key] = clamped
            changed.append(f"{key}={clamped:.3g}")

        if real > 0 and goals_det > real * 1.5:
            _update("goal_cooldown", self.thresholds["goal_cooldown"] * 1.1)
        elif real > 0 and goals_det < real:
            _update("goal_cooldown", self.thresholds["goal_cooldown"] * 0.92)

        spm = shots_det / dur_min
        if spm > 5:
            _update("shot_cooldown", self.thresholds["shot_cooldown"] * 1.1)
        elif spm < 0.5 and shots_det == 0 and real > 0:
            _update("ball_speed_min", self.thresholds["ball_speed_min"] * 0.9)

        if changed:
            print(f"  Learning seuils : {' | '.join(changed)}")

    # ─────────────────────────────────────────
    # 4. ZONES DE FAUX POSITIFS
    # ─────────────────────────────────────────
    def _update_fp_zones(self, events, frame_w=1920, frame_h=1080, grid=10):
        zone_stats = defaultdict(lambda: {"total": 0, "fp": 0})

        for e in events:
            if e.get("type") not in ["shot", "goal"]:
                continue
            if not e.get("gemini_validated") and not e.get("_geo_rejected"):
                continue

            gx  = int(e.get("x", 0) / frame_w * grid)
            gy  = int(e.get("y", 0) / frame_h * grid)
            key = f"{gx}_{gy}"
            zone_stats[key]["total"] += 1

            is_fp = (
                e.get("_geo_rejected", False) or
                e.get("gemini_type") in [
                    "touche", "corner", "none",
                    "defensive_clearance",
                    "goalkeeper_hold",
                    "goalkeeper_throw"
                ]
            )
            if is_fp:
                zone_stats[key]["fp"] += 1

        for key, s in zone_stats.items():
            existing = next((z for z in self.fp_zones if z["key"] == key), None)
            if existing:
                existing["total"] += s["total"]
                existing["fp"]    += s["fp"]
            else:
                self.fp_zones.append({"key": key, **s})

        self.fp_zones = [z for z in self.fp_zones if z["total"] >= 5]

        fp_count = sum(
            1 for z in self.fp_zones
            if z["fp"] / max(z["total"], 1) > 0.6
        )
        if fp_count > 0:
            print(f"  Learning : {fp_count} zones FP détectées")

    # ─────────────────────────────────────────
    # 5. PROFILS JOUEURS
    # ─────────────────────────────────────────
    def _update_player_profiles(self, events, fps):
        player_events = defaultdict(list)
        for e in events:
            pid = str(e.get("player", ""))
            if pid and e.get("x"):
                player_events[pid].append(e)

        updated = 0
        for pid, evts in player_events.items():
            xs    = [e["x"] for e in evts if e.get("x")]
            ys    = [e["y"] for e in evts if e.get("y")]
            types = [e["type"] for e in evts]
            n     = len(evts)

            if n < 3:
                continue

            profile = self.players_db.get(pid, {
                "touches": 0, "zone_x": 0, "zone_y": 0,
                "n_shots": 0, "n_goals": 0, "matches": 0, "avg_xg": 0.0,
            })

            alpha = 0.3
            avg_x = sum(xs) / len(xs)
            avg_y = sum(ys) / len(ys)
            profile["zone_x"]  = round(alpha * avg_x + (1 - alpha) * profile.get("zone_x", avg_x), 1)
            profile["zone_y"]  = round(alpha * avg_y + (1 - alpha) * profile.get("zone_y", avg_y), 1)
            profile["touches"] = profile.get("touches", 0) + n
            profile["n_shots"] = profile.get("n_shots", 0) + types.count("shot")
            profile["n_goals"] = profile.get("n_goals", 0) + types.count("goal")
            profile["matches"] = profile.get("matches", 0) + 1

            shot_xgs = [e.get("xg", 0) for e in evts
                        if e.get("type") == "shot" and e.get("xg")]
            if shot_xgs:
                profile["avg_xg"] = round(
                    alpha * (sum(shot_xgs) / len(shot_xgs))
                    + (1 - alpha) * profile.get("avg_xg", 0), 3
                )

            self.players_db[pid] = profile
            updated += 1

        if updated:
            print(f"  Learning : {updated} profils joueurs mis à jour "
                  f"(total={len(self.players_db)})")

    # ─────────────────────────────────────────
    # 6. PATTERNS ÉQUIPE
    # ─────────────────────────────────────────
    def _update_team_patterns(self, events, summary):
        for team_id in [0, 1]:
            key    = str(team_id)
            t_evts = [e for e in events if e.get("team") == team_id]
            if not t_evts:
                continue

            xs      = [e.get("x", 0) for e in t_evts if e.get("x")]
            avg_x   = sum(xs) / len(xs) if xs else 960
            side    = "left" if avg_x < 640 else ("right" if avg_x > 1280 else "center")

            intercepts  = [e for e in t_evts if e.get("type") == "interception"]
            high_press  = sum(1 for e in intercepts if e.get("x", 0) > 1200)
            press_ratio = high_press / max(len(intercepts), 1)

            pattern = self.teams_db.get(key, {
                "matches": 0, "dominant_side": side, "press_ratio": press_ratio
            })
            alpha = 0.4
            pattern["press_ratio"]   = round(
                alpha * press_ratio + (1 - alpha) * pattern.get("press_ratio", press_ratio), 3
            )
            pattern["dominant_side"] = side
            pattern["matches"]       = pattern.get("matches", 0) + 1

            if summary.get("formation") and team_id == 0:
                pattern["last_formation"] = summary.get("formation", "")

            self.teams_db[key] = pattern

    # ─────────────────────────────────────────
    # 7. CALIBRATION ReID
    # ─────────────────────────────────────────
    def _update_reid_calibration(self, events):
        player_moves = defaultdict(list)
        prev_pos     = {}

        for e in sorted(events, key=lambda x: x.get("time", 0)):
            pid = str(e.get("player", ""))
            if not pid or not e.get("x"):
                continue
            pos = (e["x"], e["y"])
            if pid in prev_pos:
                dx   = pos[0] - prev_pos[pid][0]
                dy   = pos[1] - prev_pos[pid][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 500:
                    player_moves[pid].append(dist)
            prev_pos[pid] = pos

        all_moves = [d for moves in player_moves.values() for d in moves]
        if len(all_moves) < 50:
            return

        sorted_moves = sorted(all_moves)
        p95_dist     = sorted_moves[int(len(sorted_moves) * 0.95)]

        n        = self.reid_cal.get("n_matches", 0)
        old_dist = self.reid_cal.get("spatial_max_dist", 200.0)
        new_dist = _clamp((old_dist * n + p95_dist) / (n + 1), 100.0, 300.0)

        self.reid_cal["spatial_max_dist"] = round(new_dist, 1)
        self.reid_cal["n_matches"]        = n + 1
        print(f"  Learning ReID : SPATIAL_MAX_DIST → {new_dist:.0f}px "
              f"(après {n+1} match(s))")

    # ─────────────────────────────────────────
    # 8. OCR
    # ─────────────────────────────────────────
    def _update_ocr_corrections(self, jersey_map):
        for pid, number in jersey_map.items():
            key = str(number)
            self.ocr_db[key] = self.ocr_db.get(key, 0) + 1

    # ─────────────────────────────────────────
    # 9. SCORES HIGHLIGHTS
    # ─────────────────────────────────────────
    def _update_highlight_scores(self, highlights):
        by_type = self.hl_scores.get("by_type", {})

        for h in highlights:
            htype = h.get("main_type", "action")
            score = h.get("score", 0)
            if not htype or not score:
                continue

            entry = by_type.get(htype, {"sum": 0, "count": 0, "avg": 0})
            entry["sum"]   += score
            entry["count"] += 1
            entry["avg"]    = round(entry["sum"] / entry["count"], 2)
            by_type[htype]  = entry

        self.hl_scores["by_type"] = by_type

        if by_type:
            top = sorted(by_type.items(), key=lambda x: x[1]["avg"], reverse=True)[:3]
            print(f"  Learning highlights : top → "
                  f"{', '.join(f'{t}={v[chr(97)+chr(118)+chr(103)]:.1f}' for t, v in top)}")

    # ─────────────────────────────────────────
    # 10. POIDS TYPES D'EVENTS
    # ─────────────────────────────────────────
    def _update_type_weights(self, events):
        goal_times = [e.get("time", 0) for e in events if e.get("type") == "goal"]
        if not goal_times:
            return

        type_pre_goal  = defaultdict(int)
        type_total     = defaultdict(int)

        for e in events:
            etype = e.get("type")
            if not etype or etype == "goal":
                continue
            t = e.get("time", 0)
            type_total[etype] += 1
            if any(0 < (gt - t) <= 20 for gt in goal_times):
                type_pre_goal[etype] += 1

        alpha = 0.2
        for etype, total in type_total.items():
            if total < 5:
                continue
            pre_goal_rate = type_pre_goal[etype] / total
            base    = DEFAULT_TYPE_WEIGHTS.get(etype, 1.0)
            learned = base * (1 + pre_goal_rate * 2)
            current = self.type_weights.get(etype, base)
            new_w   = _clamp(alpha * learned + (1 - alpha) * current, 0.1, 20.0)
            self.type_weights[etype] = round(new_w, 3)

    # ─────────────────────────────────────────
    # PRÉDICTIONS / GETTERS
    # ─────────────────────────────────────────
    def predict_xg(self, x, y, frame_w=1920, frame_h=1080):
        """Modèle SGD simple — utilisé quand sklearn pas encore actif."""
        if self.xg_model.get("n_samples", 0) < 20:
            x_norm = x / frame_w
            z      = -4 * (abs(1.0 - x_norm) - 0.5)
        else:
            x_n  = x / frame_w
            y_n  = y / frame_h
            dist = math.sqrt((1.0 - x_n) ** 2 + (0.5 - y_n) ** 2)
            ang  = abs(math.atan2(0.5 - y_n, 1.0 - x_n))
            z    = (self.xg_model["w0"]
                    + self.xg_model["w1"] * dist
                    + self.xg_model["w2"] * ang)
        z  = _clamp(z, -100, 100)
        xg = 1 / (1 + math.exp(-z))
        return round(_clamp(xg, 0.01, 0.99), 3)

    def get_thresholds(self):
        return self.thresholds.copy()

    def get_spatial_max_dist(self):
        return self.reid_cal.get("spatial_max_dist", 200.0)

    def is_fp_zone(self, x, y, frame_w=1920, frame_h=1080, grid=10):
        gx  = int(x / frame_w * grid)
        gy  = int(y / frame_h * grid)
        key = f"{gx}_{gy}"
        zone = next((z for z in self.fp_zones if z["key"] == key), None)
        if zone and zone["total"] >= 5:
            return (zone["fp"] / zone["total"]) > 0.6
        return False

    def get_player_profile(self, player_id):
        return self.players_db.get(str(player_id), {})

    def get_event_weight(self, event_type):
        return self.type_weights.get(event_type, DEFAULT_TYPE_WEIGHTS.get(event_type, 1.0))

    def get_min_highlight_score(self, event_type):
        entry = self.hl_scores.get("by_type", {}).get(event_type, {})
        avg   = entry.get("avg", 5.0)
        return max(1.0, avg * 0.7)

    def stats(self):
        matches = list({e["match_id"] for e in self.events_db})
        types   = defaultdict(int)
        for e in self.events_db:
            types[e["type"]] += 1

        return {
            "sport":                 self.sport,
            "n_matches":             len(matches),
            "n_events":              len(self.events_db),
            "event_types":           dict(types),
            "xg_samples":            self.xg_model["n_samples"],
            "xg_advanced_samples":   len(self.xg_advanced),
            "xg_advanced_ready":     self.has_advanced_xg(),
            "fp_zones":              len(self.fp_zones),
            "player_profiles":       len(self.players_db),
            "spatial_max_dist":      self.reid_cal.get("spatial_max_dist", 200.0),
            "thresholds":            self.thresholds,
            "type_weights":          self.type_weights,
            "xg_training_total":     len(self.xg_training),
        }