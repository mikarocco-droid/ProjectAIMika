# ai/learning_model.py
# -*- coding: utf-8 -*-
"""
Modèle d'apprentissage cumulatif — s'améliore match après match.

Ce qui est appris :
1. Modèle xG (régression logistique position → probabilité but)
2. Seuils temporels (goal_cooldown, shot_cooldown)
3. Zones de faux positifs (zones qui génèrent toujours des faux tirs)
4. Profils joueurs (vitesse, zone d'action, style)
5. Patterns équipe (formation, côté dominant, pressing)
6. Seuils ReID (SPATIAL_MAX_DIST optimal par terrain)
7. Calibration OCR maillots (corrections récurrentes)
8. Scores highlights (pré-filtrage avant Gemini)

Structure :
outputs/learning/{sport}/
    events.json
    xg_model.json
    thresholds.json
    false_positive_zones.json
    player_profiles.json
    team_patterns.json
    reid_calibration.json
    ocr_corrections.json
    highlight_scores.json
"""

import os
import json
import math
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────
# SEUILS PAR DÉFAUT
# ─────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    "football": {
        "goal_frames_min":    12,
        "shot_cooldown":       3.0,
        "goal_cooldown":     150.0,
        "ball_speed_min":      0.03,
        "player_near_goal":    0.15,
        "spatial_max_dist":  200.0,
    },
    "basketball": {
        "goal_frames_min":    5,
        "shot_cooldown":      1.5,
        "goal_cooldown":      5.0,
        "ball_speed_min":     0.04,
        "player_near_goal":   0.10,
        "spatial_max_dist": 150.0,
    },
    "handball": {
        "goal_frames_min":    8,
        "shot_cooldown":      2.0,
        "goal_cooldown":     20.0,
        "ball_speed_min":     0.03,
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
        }
        self._load()

    # ─────────────────────────────────────────
    # LOAD / SAVE
    # ─────────────────────────────────────────
    def _load(self):
        def _read(key, default):
            p = self._paths[key]
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)
            return default

        self.events_db   = _read("events",     [])
        self.xg_model    = _read("xg",         {"w0": 0.0, "w1": -2.0, "w2": 1.0, "n_samples": 0})
        self.thresholds  = _read("thresholds", DEFAULT_THRESHOLDS.get(self.sport, DEFAULT_THRESHOLDS["football"]).copy())
        self.fp_zones    = _read("fp_zones",   [])           # zones de faux positifs
        self.players_db  = _read("players",    {})           # profils joueurs
        self.teams_db    = _read("teams",      {})           # patterns équipes
        self.reid_cal    = _read("reid",        {"spatial_max_dist": 200.0, "n_matches": 0})
        self.ocr_db      = _read("ocr",         {})           # corrections OCR
        self.hl_scores   = _read("highlights",  {"by_type": {}, "by_position": []})

    def _save(self):
        data = {
            "events":     self.events_db,
            "xg":         self.xg_model,
            "thresholds": self.thresholds,
            "fp_zones":   self.fp_zones,
            "players":    self.players_db,
            "teams":      self.teams_db,
            "reid":       self.reid_cal,
            "ocr":        self.ocr_db,
            "highlights": self.hl_scores,
        }
        for key, obj in data.items():
            with open(self._paths[key], "w") as f:
                json.dump(obj, f, indent=2)

    # ─────────────────────────────────────────
    # 1. ENREGISTRER UN MATCH
    # ─────────────────────────────────────────
    def record_match(self, events, summary, fps=25, jersey_map=None, highlights=None):
        match_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"  Learning : enregistrement match {match_id}")

        # Extraire events pertinents
        for e in events:
            if e.get("type") not in ["goal", "shot", "interception", "dribble"]:
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

        # Mettre à jour tous les modèles
        self._update_xg_model()
        self._recalibrate_thresholds(events, summary, fps)
        self._update_fp_zones(events)
        self._update_player_profiles(events, fps)
        self._update_team_patterns(events, summary)
        self._update_reid_calibration(events)
        self._update_ocr_corrections(jersey_map or {})
        self._update_highlight_scores(highlights or [])

        self._save()

        result = {
            "match_id":     match_id,
            "events_added": len([e for e in events if e.get("type") in
                                 ["goal", "shot", "interception", "dribble"]]),
            "total_events": len(self.events_db),
            "xg_samples":   self.xg_model["n_samples"],
        }
        print(f"  Learning OK : {result}")
        return result

    # ─────────────────────────────────────────
    # 2. MODÈLE xG
    # ─────────────────────────────────────────
    def _update_xg_model(self):
        shots = [e for e in self.events_db
                 if e.get("type") == "shot" and e.get("gemini_validated")
                 and e.get("x", 0) > 0]

        if len(shots) < 10:
            return

        X, y = [], []
        for s in shots:
            x_n  = s["x"] / 1920.0
            y_n  = s["y"] / 1080.0
            dist = math.sqrt((1.0 - x_n)**2 + (0.5 - y_n)**2)
            ang  = abs(math.atan2(0.5 - y_n, 1.0 - x_n))
            lbl  = 1 if s.get("gemini_type") == "shot" and s.get("gemini_conf", 0) > 0.8 else 0
            X.append([dist, ang])
            y.append(lbl)

        w0, w1, w2 = (self.xg_model.get(k, v)
                      for k, v in [("w0", 0.0), ("w1", -2.0), ("w2", 1.0)])
        lr = 0.01

        for _ in range(20):
            dw0 = dw1 = dw2 = 0.0
            for (d, a), lbl in zip(X, y):
                z    = max(-100, min(100, w0 + w1*d + w2*a))
                pred = 1 / (1 + math.exp(-z))
                err  = pred - lbl
                dw0 += err; dw1 += err*d; dw2 += err*a
            n   = len(X)
            w0 -= lr * dw0/n
            w1 -= lr * dw1/n
            w2 -= lr * dw2/n

        self.xg_model = {
            "w0": round(w0, 4), "w1": round(w1, 4),
            "w2": round(w2, 4), "n_samples": len(shots)
        }

    # ─────────────────────────────────────────
    # 3. SEUILS TEMPORELS
    # ─────────────────────────────────────────
    def _recalibrate_thresholds(self, events, summary, fps):
        goals_det  = sum(1 for e in events if e.get("type") == "goal")
        goals_real = summary.get("goals", 0)
        shots_det  = sum(1 for e in events if e.get("type") == "shot")
        dur_min    = max(1, summary.get("total_frames", 15000) / fps / 60)

        # goal_cooldown
        if goals_real > 0 and goals_det > goals_real * 1.5:
            self.thresholds["goal_cooldown"] = min(300.0,
                self.thresholds["goal_cooldown"] * 1.1)
            print(f"  Learning : goal_cooldown ↑ {self.thresholds['goal_cooldown']:.0f}s")
        elif goals_det <= goals_real:
            self.thresholds["goal_cooldown"] = max(60.0,
                self.thresholds["goal_cooldown"] * 0.95)

        # shot_cooldown
        spm = shots_det / dur_min
        if spm > 4:
            self.thresholds["shot_cooldown"] = min(6.0,
                self.thresholds["shot_cooldown"] * 1.1)
        elif spm < 1 and shots_det > 0:
            self.thresholds["shot_cooldown"] = max(1.5,
                self.thresholds["shot_cooldown"] * 0.95)

    # ─────────────────────────────────────────
    # 4. ZONES DE FAUX POSITIFS
    # Zones qui génèrent systématiquement des faux tirs
    # ─────────────────────────────────────────
    def _update_fp_zones(self, events, frame_w=1920, frame_h=1080, grid=10):
        """
        Divise le terrain en grille 10x10.
        Si une zone a >50% de faux positifs Gemini → zone FP.
        """
        zone_stats = defaultdict(lambda: {"total": 0, "fp": 0})

        for e in events:
            if e.get("type") != "shot" or not e.get("gemini_validated"):
                continue
            gx = int(e.get("x", 0) / frame_w * grid)
            gy = int(e.get("y", 0) / frame_h * grid)
            key = f"{gx}_{gy}"
            zone_stats[key]["total"] += 1
            if e.get("gemini_type") in ["touche", "corner", "none"]:
                zone_stats[key]["fp"] += 1

        # Accumuler dans la DB globale
        for key, s in zone_stats.items():
            existing = next((z for z in self.fp_zones if z["key"] == key), None)
            if existing:
                existing["total"] += s["total"]
                existing["fp"]    += s["fp"]
            else:
                self.fp_zones.append({"key": key, **s})

        # Recalculer les zones FP (>60% de FP sur min 5 events)
        self.fp_zones = [
            z for z in self.fp_zones
            if z["total"] >= 5
        ]

        fp_count = sum(1 for z in self.fp_zones
                       if z["fp"] / max(z["total"], 1) > 0.6)
        if fp_count > 0:
            print(f"  Learning : {fp_count} zones FP détectées")

    # ─────────────────────────────────────────
    # 5. PROFILS JOUEURS
    # ─────────────────────────────────────────
    def _update_player_profiles(self, events, fps):
        """
        Apprend : zone préférée, vitesse typique, fréquence d'actions.
        Clé = numéro maillot ou ID stable.
        """
        player_events = defaultdict(list)
        for e in events:
            pid = str(e.get("player", ""))
            if pid and e.get("x"):
                player_events[pid].append(e)

        for pid, evts in player_events.items():
            xs     = [e["x"] for e in evts if e.get("x")]
            ys     = [e["y"] for e in evts if e.get("y")]
            types  = [e["type"] for e in evts]
            n      = len(evts)

            if n < 3:
                continue

            profile = self.players_db.get(pid, {
                "touches": 0, "zone_x": 0, "zone_y": 0,
                "n_shots": 0, "n_goals": 0, "matches": 0
            })

            # Moyenne mobile
            alpha = 0.3
            profile["zone_x"]  = round(alpha * (sum(xs)/len(xs)) +
                                        (1-alpha) * profile.get("zone_x", sum(xs)/len(xs)), 1)
            profile["zone_y"]  = round(alpha * (sum(ys)/len(ys)) +
                                        (1-alpha) * profile.get("zone_y", sum(ys)/len(ys)), 1)
            profile["touches"] = profile.get("touches", 0) + n
            profile["n_shots"] = profile.get("n_shots", 0) + types.count("shot")
            profile["n_goals"] = profile.get("n_goals", 0) + types.count("goal")
            profile["matches"] = profile.get("matches", 0) + 1

            self.players_db[pid] = profile

        print(f"  Learning : {len(self.players_db)} profils joueurs")

    # ─────────────────────────────────────────
    # 6. PATTERNS ÉQUIPE
    # ─────────────────────────────────────────
    def _update_team_patterns(self, events, summary):
        """
        Apprend : formation préférée, côté dominant, intensité pressing.
        """
        for team_id in [0, 1]:
            key    = str(team_id)
            t_evts = [e for e in events if e.get("team") == team_id]
            if not t_evts:
                continue

            # Côté dominant (gauche/droite/centre)
            xs         = [e.get("x", 0) for e in t_evts if e.get("x")]
            avg_x      = sum(xs) / len(xs) if xs else 960
            if avg_x < 640:
                side = "left"
            elif avg_x > 1280:
                side = "right"
            else:
                side = "center"

            # Intensité pressing (interceptions en zone adverse)
            intercepts  = [e for e in t_evts if e.get("type") == "interception"]
            high_press  = sum(1 for e in intercepts if e.get("x", 0) > 1200)
            press_ratio = high_press / max(len(intercepts), 1)

            pattern = self.teams_db.get(key, {
                "matches": 0, "dominant_side": side, "press_ratio": press_ratio
            })
            alpha = 0.4
            pattern["press_ratio"]   = round(alpha * press_ratio +
                                             (1-alpha) * pattern.get("press_ratio", press_ratio), 3)
            pattern["dominant_side"] = side
            pattern["matches"]       = pattern.get("matches", 0) + 1
            self.teams_db[key]       = pattern

    # ─────────────────────────────────────────
    # 7. CALIBRATION ReID
    # Apprend le SPATIAL_MAX_DIST optimal par terrain
    # ─────────────────────────────────────────
    def _update_reid_calibration(self, events):
        """
        Estime la vitesse max des joueurs pour calibrer SPATIAL_MAX_DIST.
        Un joueur ne peut pas se déplacer plus de X pixels par frame.
        """
        # Calculer les déplacements frame à frame
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
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 500:  # ignore les téléportations
                    player_moves[pid].append(dist)
            prev_pos[pid] = pos

        all_moves = [d for moves in player_moves.values() for d in moves]
        if len(all_moves) < 50:
            return

        # Percentile 95 des déplacements → SPATIAL_MAX_DIST
        sorted_moves  = sorted(all_moves)
        p95_idx       = int(len(sorted_moves) * 0.95)
        p95_dist      = sorted_moves[p95_idx]

        # Moyenne mobile avec historique
        n  = self.reid_cal.get("n_matches", 0)
        old_dist = self.reid_cal.get("spatial_max_dist", 200.0)
        new_dist = (old_dist * n + p95_dist) / (n + 1)
        new_dist = max(100.0, min(400.0, new_dist))

        self.reid_cal["spatial_max_dist"] = round(new_dist, 1)
        self.reid_cal["n_matches"]        = n + 1
        print(f"  Learning : SPATIAL_MAX_DIST → {new_dist:.0f}px")

    # ─────────────────────────────────────────
    # 8. CORRECTIONS OCR MAILLOTS
    # ─────────────────────────────────────────
    def _update_ocr_corrections(self, jersey_map):
        """
        Si Gemini corrige systématiquement un numéro OCR,
        on apprend la correction pour les prochains matchs.
        Format : {"ocr_read": "4", "real": "14", "count": 3}
        """
        # Pour l'instant stocke les numéros vus
        # (enrichissable avec feedback utilisateur)
        for pid, number in jersey_map.items():
            key = str(number)
            self.ocr_db[key] = self.ocr_db.get(key, 0) + 1

    # ─────────────────────────────────────────
    # 9. SCORES HIGHLIGHTS
    # Apprend quels types d'actions Gemini score haut
    # ─────────────────────────────────────────
    def _update_highlight_scores(self, highlights):
        """
        Apprend les scores moyens par type d'action.
        Permet de pré-filtrer avant d'appeler Gemini.
        """
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
            top = sorted(by_type.items(),
                         key=lambda x: x[1]["avg"], reverse=True)[:3]
            print(f"  Learning : top highlights → "
                  f"{', '.join(f'{t}={v[\"avg\"]:.1f}' for t, v in top)}")

    # ─────────────────────────────────────────
    # PRÉDICTIONS / GETTERS
    # ─────────────────────────────────────────
    def predict_xg(self, x, y, frame_w=1920, frame_h=1080):
        if self.xg_model.get("n_samples", 0) < 10:
            x_norm = x / frame_w
            z      = -4 * (abs(1.0 - x_norm) - 0.5)
        else:
            x_n  = x / frame_w
            y_n  = y / frame_h
            dist = math.sqrt((1.0 - x_n)**2 + (0.5 - y_n)**2)
            ang  = abs(math.atan2(0.5 - y_n, 1.0 - x_n))
            z    = (self.xg_model["w0"] +
                    self.xg_model["w1"] * dist +
                    self.xg_model["w2"] * ang)
        z  = max(-100, min(100, z))
        xg = 1 / (1 + math.exp(-z))
        return round(max(0.01, min(0.5, xg)), 3)

    def get_thresholds(self):
        return self.thresholds.copy()

    def get_spatial_max_dist(self):
        return self.reid_cal.get("spatial_max_dist", 200.0)

    def is_fp_zone(self, x, y, frame_w=1920, frame_h=1080, grid=10):
        """Retourne True si la position est dans une zone de faux positifs."""
        gx  = int(x / frame_w * grid)
        gy  = int(y / frame_h * grid)
        key = f"{gx}_{gy}"
        zone = next((z for z in self.fp_zones if z["key"] == key), None)
        if zone and zone["total"] >= 5:
            return (zone["fp"] / zone["total"]) > 0.6
        return False

    def get_player_profile(self, player_id):
        return self.players_db.get(str(player_id), {})

    def get_min_highlight_score(self, event_type):
        """Score minimum appris pour inclure un highlight de ce type."""
        entry = self.hl_scores.get("by_type", {}).get(event_type, {})
        avg   = entry.get("avg", 5.0)
        # Seuil = moyenne - 1 écart type estimé (conservateur)
        return max(1.0, avg * 0.7)

    def stats(self):
        matches = list({e["match_id"] for e in self.events_db})
        types   = defaultdict(int)
        for e in self.events_db:
            types[e["type"]] += 1

        return {
            "sport":            self.sport,
            "n_matches":        len(matches),
            "n_events":         len(self.events_db),
            "event_types":      dict(types),
            "xg_samples":       self.xg_model["n_samples"],
            "fp_zones":         len(self.fp_zones),
            "player_profiles":  len(self.players_db),
            "spatial_max_dist": self.reid_cal.get("spatial_max_dist", 200.0),
            "thresholds":       self.thresholds,
        }