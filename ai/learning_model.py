# ai/learning_model.py
# -*- coding: utf-8 -*-

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import os
import json
import math
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────
# SEUILS PAR DÉFAUT
# FIX : goal_frames_min 12 → 8
# ─────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    "football": {
        "goal_frames_min":   8,      # FIX : était 12
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


class MatchLearner:

    def __init__(self, sport="football", base_dir="outputs/learning"):
        self.sport    = sport
        self.base_dir = os.path.join(base_dir, sport)
        os.makedirs(self.base_dir, exist_ok=True)

        self._paths = {
            "events":     os.path.join(self.base_dir, "events.json"),
            "xg":         os.path.join(self.base_dir, "xg_model.json"),
            "thresholds": os.path.join(self.base_dir, "thresholds.json"),
            "fp_zones":   os.path.join(self.base_dir, "false_positive_zones.json"),
            "players":    os.path.join(self.base_dir, "player_profiles.json"),
            "teams":      os.path.join(self.base_dir, "team_patterns.json"),
            "reid":       os.path.join(self.base_dir, "reid_calibration.json"),
            "ocr":        os.path.join(self.base_dir, "ocr_corrections.json"),
            "highlights": os.path.join(self.base_dir, "highlight_scores.json"),
            "history":    os.path.join(self.base_dir, "match_history.json"),
        }
        self._load()

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

        self.events_db  = _read("events",     [])
        self.xg_model   = _read("xg",         {"w0": 0.0, "w1": -2.0, "w2": 1.0, "n_samples": 0})
        self.thresholds = _read("thresholds", DEFAULT_THRESHOLDS.get(self.sport, DEFAULT_THRESHOLDS["football"]).copy())
        self.fp_zones   = _read("fp_zones",   [])
        self.players_db = _read("players",    {})
        self.teams_db   = _read("teams",      {})
        self.reid_cal   = _read("reid",       {"spatial_max_dist": 200.0, "n_matches": 0})
        self.ocr_db     = _read("ocr",        {})
        self.hl_scores  = _read("highlights", {"by_type": {}, "by_position": []})
        self.history    = _read("history",    [])

        # FIX : forcer goal_frames_min à 8 si le fichier sauvegardé contient encore 12
        defaults = DEFAULT_THRESHOLDS.get(self.sport, DEFAULT_THRESHOLDS["football"])
        for k, v in defaults.items():
            if k not in self.thresholds:
                self.thresholds[k] = v
        if self.thresholds.get("goal_frames_min", 12) > 8:
            self.thresholds["goal_frames_min"] = 8

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
            "history":    self.history,
        }
        for key, obj in data.items():
            try:
                with open(self._paths[key], "w") as f:
                    json.dump(obj, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Learning warning : impossible de sauver {key} — {e}")

    # ─────────────────────────────────────────
    # 1. ENREGISTRER UN MATCH
    # FIX : goals_real passé explicitement depuis le pipeline
    # ─────────────────────────────────────────
    def record_match(self, events, summary, fps=25, jersey_map=None,
                     highlights=None, goals_real=None):
        match_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_matches = len(self.history) + 1
        print(f"  Learning : enregistrement match #{n_matches} ({match_id})")

        added = 0
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
            added += 1

        self._update_xg_model()
        # FIX : on passe goals_real explicitement
        self._recalibrate_thresholds(events, summary, fps, goals_real=goals_real)
        self._update_fp_zones(events)
        self._update_player_profiles(events, fps)
        self._update_team_patterns(events, summary)
        self._update_reid_calibration(events)
        self._update_ocr_corrections(jersey_map or {})
        self._update_highlight_scores(highlights or [])
        self._record_history(match_id, events, summary, highlights or [],
                             goals_real=goals_real)

        self._save()

        result = {
            "match_id":     match_id,
            "match_number": n_matches,
            "events_added": added,
            "total_events": len(self.events_db),
            "xg_samples":   self.xg_model["n_samples"],
            "thresholds":   self.thresholds,
        }
        print(f"  Learning OK : match #{n_matches} | {added} events | "
              f"total={len(self.events_db)} | "
              f"xG_samples={self.xg_model['n_samples']}")
        return result

    # ─────────────────────────────────────────
    # HISTORIQUE
    # FIX : goals_real séparé de goals_detected
    # ─────────────────────────────────────────
    def _record_history(self, match_id, events, summary, highlights,
                        goals_real=None):
        shots        = sum(1 for e in events if e.get("type") == "shot")
        goals_det    = sum(1 for e in events if e.get("type") == "goal")
        passes       = sum(1 for e in events if e.get("type") == "pass")

        # FIX : goals_real = valeur passée explicitement OU fallback summary
        # Le pipeline doit passer le vrai nombre de buts
        real = goals_real if goals_real is not None else summary.get("goals", goals_det)

        entry = {
            "match_id":        match_id,
            "date":            datetime.now().isoformat(),
            "sport":           self.sport,
            "goals_detected":  goals_det,
            "goals_real":      real,           # FIX : maintenant distinct
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
        }
        self.history.append(entry)

        n = len(self.history)
        if n >= 2:
            prev      = self.history[-2]
            prev_real = max(prev.get("goals_real", 1), 1)
            prev_acc  = prev["goals_detected"] / prev_real
            cur_acc   = goals_det / max(real, 1) if real > 0 else 0
            delta_acc = cur_acc - prev_acc
            print(f"  Learning progression : match {n-1}→{n} | "
                  f"précision buts {'+' if delta_acc >= 0 else ''}{delta_acc:.0%}")

    # ─────────────────────────────────────────
    # 3. RECALIBRATION SEUILS
    # FIX : utilise goals_real passé en paramètre
    # ─────────────────────────────────────────
    def _recalibrate_thresholds(self, events, summary, fps, goals_real=None):
        goals_det  = sum(1 for e in events if e.get("type") == "goal")
        real       = goals_real if goals_real is not None else summary.get("goals", goals_det)
        shots_det  = sum(1 for e in events if e.get("type") == "shot")
        dur_min    = max(1, summary.get("total_frames", 15000) / fps / 60)

        changed = []

        # Trop de faux positifs buts → augmenter cooldown
        if real > 0 and goals_det > real * 1.5:
            self.thresholds["goal_cooldown"] = min(
                300.0, self.thresholds["goal_cooldown"] * 1.1
            )
            changed.append(f"goal_cooldown↑={self.thresholds['goal_cooldown']:.0f}s")

        # Buts manqués → réduire le cooldown
        elif real > 0 and goals_det < real:
            self.thresholds["goal_cooldown"] = max(
                60.0, self.thresholds["goal_cooldown"] * 0.92
            )
            changed.append(f"goal_cooldown↓={self.thresholds['goal_cooldown']:.0f}s")

        # Trop de tirs → augmenter cooldown tir
        spm = shots_det / dur_min
        if spm > 5:
            self.thresholds["shot_cooldown"] = min(
                6.0, self.thresholds["shot_cooldown"] * 1.1
            )
            changed.append(f"shot_cooldown↑={self.thresholds['shot_cooldown']:.1f}s")
        elif spm < 0.5 and shots_det == 0 and real > 0:
            self.thresholds["ball_speed_min"] = max(
                0.01, self.thresholds.get("ball_speed_min", 0.02) * 0.9
            )
            changed.append(f"ball_speed_min↓={self.thresholds['ball_speed_min']:.3f}")

        if changed:
            print(f"  Learning seuils : {' | '.join(changed)}")

    # ─────────────────────────────────────────
    # 4. ZONES DE FAUX POSITIFS
    # FIX : inclut aussi les buts rejetés géométriquement
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

            # FIX : compter comme FP si rejeté géo OU si Gemini dit non
            is_fp = (
                e.get("_geo_rejected", False) or
                e.get("gemini_type") in ["touche", "corner", "none",
                                          "defensive_clearance",
                                          "goalkeeper_hold",
                                          "goalkeeper_throw"]
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
            pattern["press_ratio"]   = round(alpha * press_ratio + (1 - alpha) * pattern.get("press_ratio", press_ratio), 3)
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
        new_dist = max(100.0, min(200.0, (old_dist * n + p95_dist) / (n + 1)))

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
    # PRÉDICTIONS / GETTERS
    # ─────────────────────────────────────────
    def predict_xg(self, x, y, frame_w=1920, frame_h=1080):
        if self.xg_model.get("n_samples", 0) < 10:
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
        z  = max(-100, min(100, z))
        xg = 1 / (1 + math.exp(-z))
        return round(max(0.01, min(0.5, xg)), 3)

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

    def get_min_highlight_score(self, event_type):
        entry = self.hl_scores.get("by_type", {}).get(event_type, {})
        avg   = entry.get("avg", 5.0)
        return max(1.0, avg * 0.7)

    def progression_report(self):
        n = len(self.history)
        if n == 0:
            return {"n_matches": 0, "message": "Aucun match enregistré"}

        recent       = self.history[-5:]
        avg_goal_acc = []
        for h in recent:
            real = h.get("goals_real", 0)
            det  = h.get("goals_detected", 0)
            if real > 0:
                avg_goal_acc.append(min(1.0, det / real))

        return {
            "n_matches":         n,
            "sport":             self.sport,
            "xg_samples":        self.xg_model["n_samples"],
            "fp_zones":          len(self.fp_zones),
            "player_profiles":   len(self.players_db),
            "spatial_max_dist":  self.reid_cal.get("spatial_max_dist", 200.0),
            "thresholds":        self.thresholds,
            "recent_matches":    recent,
            "avg_goal_accuracy": round(sum(avg_goal_acc) / len(avg_goal_acc), 2)
                                  if avg_goal_acc else None,
            "last_match":        self.history[-1] if self.history else None,
        }

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