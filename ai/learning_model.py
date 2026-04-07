# ai/learning_model.py
# -*- coding: utf-8 -*-
"""
Modèle d'apprentissage cumulatif — s'améliore match après match.

Structure des données :
outputs/learning/
    football/
        events.json      ← tous les events validés par Gemini
        xg_model.json    ← paramètres du modèle xG appris
        thresholds.json  ← seuils recalibrés automatiquement
    basketball/
        ...

Utilisation dans pipeline.py :
    from ai.learning_model import MatchLearner
    learner = MatchLearner(sport)
    learner.record_match(events, summary)   # après chaque match
    xg       = learner.predict_xg(x, y)    # xG amélioré
    thresholds = learner.get_thresholds()  # seuils recalibrés
"""

import os
import json
import math
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────
# SEUILS PAR DÉFAUT (avant apprentissage)
# ─────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    "football": {
        "goal_frames_min":   12,     # frames consécutives pour valider un but
        "shot_cooldown":     3.0,    # secondes entre deux tirs
        "goal_cooldown":   150.0,    # secondes entre deux buts
        "ball_speed_min":    0.03,   # vitesse min balle pour un tir (normalisée)
        "player_near_goal":  0.15,   # distance max joueur/but (% frame_w)
    },
    "basketball": {
        "goal_frames_min":   5,
        "shot_cooldown":     1.5,
        "goal_cooldown":     5.0,
        "ball_speed_min":    0.04,
        "player_near_goal":  0.10,
    },
    "handball": {
        "goal_frames_min":   8,
        "shot_cooldown":     2.0,
        "goal_cooldown":    20.0,
        "ball_speed_min":    0.03,
        "player_near_goal":  0.12,
    },
}


class MatchLearner:
    """
    Apprend des patterns de chaque match pour améliorer les analyses suivantes.
    Un fichier JSON par sport, cumulatif.
    """

    def __init__(self, sport="football", base_dir="outputs/learning"):
        self.sport    = sport
        self.base_dir = os.path.join(base_dir, sport)
        os.makedirs(self.base_dir, exist_ok=True)

        self.events_path     = os.path.join(self.base_dir, "events.json")
        self.xg_path         = os.path.join(self.base_dir, "xg_model.json")
        self.thresholds_path = os.path.join(self.base_dir, "thresholds.json")

        self._load()

    # ─────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────
    def _load(self):
        # Events cumulés
        if os.path.exists(self.events_path):
            with open(self.events_path) as f:
                self.events_db = json.load(f)
        else:
            self.events_db = []

        # Modèle xG
        if os.path.exists(self.xg_path):
            with open(self.xg_path) as f:
                self.xg_model = json.load(f)
        else:
            self.xg_model = {"w0": 0.0, "w1": -2.0, "w2": 1.0, "n_samples": 0}

        # Seuils
        if os.path.exists(self.thresholds_path):
            with open(self.thresholds_path) as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = DEFAULT_THRESHOLDS.get(
                self.sport,
                DEFAULT_THRESHOLDS["football"]
            ).copy()

    # ─────────────────────────────────────────
    # SAUVEGARDE
    # ─────────────────────────────────────────
    def _save(self):
        with open(self.events_path,     "w") as f:
            json.dump(self.events_db,   f, indent=2)
        with open(self.xg_path,         "w") as f:
            json.dump(self.xg_model,    f, indent=2)
        with open(self.thresholds_path, "w") as f:
            json.dump(self.thresholds,  f, indent=2)

    # ─────────────────────────────────────────
    # ENREGISTRER UN MATCH
    # ─────────────────────────────────────────
    def record_match(self, events, summary, fps=25):
        """
        Enregistre les events validés d'un match et met à jour les modèles.
        À appeler à la fin de chaque pipeline.
        """
        match_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_events   = len(events)
        n_goals    = summary.get("goals",  0)
        n_shots    = summary.get("shots",  0)
        duration   = summary.get("duration", "00:00")

        print(f"  Learning : enregistrement match {match_id} "
              f"({n_goals} buts, {n_shots} tirs, {n_events} events)")

        # ── Extraire les events pertinents pour l'apprentissage ──
        learning_events = []
        for e in events:
            if e.get("type") not in ["goal", "shot", "interception", "dribble"]:
                continue
            learning_events.append({
                "match_id":         match_id,
                "type":             e.get("type"),
                "x":                e.get("x", 0),
                "y":                e.get("y", 0),
                "xg":               e.get("xg", 0),
                "gemini_validated": e.get("gemini_validated", False),
                "gemini_type":      e.get("gemini_type", ""),
                "gemini_conf":      e.get("gemini_conf",  0),
                "frame":            e.get("frame", 0),
                "time":             e.get("time",  0),
            })

        self.events_db.extend(learning_events)

        # ── Mettre à jour les modèles ──
        self._update_xg_model()
        self._recalibrate_thresholds(events, summary, fps)

        # ── Sauvegarder ──
        self._save()

        n_total = len(self.events_db)
        print(f"  Learning : {n_total} events cumulés | "
              f"xG model n={self.xg_model['n_samples']} | "
              f"thresholds recalibrés")

        return {
            "match_id":       match_id,
            "events_added":   len(learning_events),
            "total_events":   n_total,
            "xg_model":       self.xg_model,
            "thresholds":     self.thresholds,
        }

    # ─────────────────────────────────────────
    # MODÈLE xG — RÉGRESSION LOGISTIQUE SIMPLE
    # xG = sigmoid(w0 + w1*distance + w2*angle)
    # ─────────────────────────────────────────
    def _update_xg_model(self):
        """
        Met à jour le modèle xG avec les tirs validés par Gemini.
        Utilise une descente de gradient simple (pas de sklearn requis).
        """
        # Récupérer les tirs avec validation Gemini
        shots = [
            e for e in self.events_db
            if e.get("type") == "shot"
            and e.get("gemini_validated")
            and e.get("x", 0) > 0
        ]

        if len(shots) < 10:
            return  # pas assez de données

        # Labels : 1 si Gemini confirme "shot" avec conf > 0.8, 0 sinon
        X, y = [], []
        for s in shots:
            x_norm = s["x"] / 1920.0
            y_norm = s["y"] / 1080.0
            dist   = math.sqrt((1.0 - x_norm)**2 + (0.5 - y_norm)**2)
            angle  = abs(math.atan2(0.5 - y_norm, 1.0 - x_norm))
            label  = 1 if s.get("gemini_type") == "shot" and s.get("gemini_conf", 0) > 0.8 else 0
            X.append([dist, angle])
            y.append(label)

        # Descente de gradient (10 itérations, lr=0.01)
        w0 = self.xg_model.get("w0", 0.0)
        w1 = self.xg_model.get("w1", -2.0)
        w2 = self.xg_model.get("w2",  1.0)
        lr = 0.01

        for _ in range(10):
            dw0 = dw1 = dw2 = 0.0
            for (dist, angle), label in zip(X, y):
                z    = w0 + w1 * dist + w2 * angle
                pred = 1 / (1 + math.exp(-max(-100, min(100, z))))
                err  = pred - label
                dw0 += err
                dw1 += err * dist
                dw2 += err * angle
            n    = len(X)
            w0  -= lr * dw0 / n
            w1  -= lr * dw1 / n
            w2  -= lr * dw2 / n

        self.xg_model = {
            "w0":       round(w0, 4),
            "w1":       round(w1, 4),
            "w2":       round(w2, 4),
            "n_samples": len(shots),
        }

    # ─────────────────────────────────────────
    # RECALIBRATION DES SEUILS
    # ─────────────────────────────────────────
    def _recalibrate_thresholds(self, events, summary, fps):
        """
        Recalibre les seuils basés sur les stats du match validé.
        Approche conservative : ajustement de ±10% max par match.
        """
        goals_detected = sum(1 for e in events if e.get("type") == "goal")
        goals_real     = summary.get("goals", 0)
        shots_detected = sum(1 for e in events if e.get("type") == "shot")

        # ── Recalibrer goal_cooldown ──
        # Si trop de buts détectés → augmenter cooldown
        if goals_real > 0 and goals_detected > goals_real * 1.5:
            self.thresholds["goal_cooldown"] = min(
                300.0,
                self.thresholds["goal_cooldown"] * 1.1
            )
            print(f"  Learning : goal_cooldown ↑ → {self.thresholds['goal_cooldown']:.0f}s")
        elif goals_detected <= goals_real:
            self.thresholds["goal_cooldown"] = max(
                60.0,
                self.thresholds["goal_cooldown"] * 0.95
            )

        # ── Recalibrer shot_cooldown ──
        # Heuristique : max 3 tirs/min en football amateur
        duration_min = max(1, summary.get("total_frames", 15000) / fps / 60)
        shots_per_min = shots_detected / duration_min

        if shots_per_min > 4:
            self.thresholds["shot_cooldown"] = min(
                6.0,
                self.thresholds["shot_cooldown"] * 1.1
            )
            print(f"  Learning : shot_cooldown ↑ → {self.thresholds['shot_cooldown']:.1f}s")
        elif shots_per_min < 1 and shots_detected > 0:
            self.thresholds["shot_cooldown"] = max(
                1.5,
                self.thresholds["shot_cooldown"] * 0.95
            )

    # ─────────────────────────────────────────
    # PRÉDIRE xG (utilise le modèle appris)
    # ─────────────────────────────────────────
    def predict_xg(self, x, y, frame_w=1920, frame_h=1080):
        """
        Prédit le xG d'un tir avec le modèle appris.
        Fallback sur le modèle par défaut si pas assez de données.
        """
        if self.xg_model.get("n_samples", 0) < 10:
            # Pas assez de données — modèle par défaut
            x_norm = x / frame_w
            dist   = abs(1.0 - x_norm)
            z      = -4 * (dist - 0.5)
        else:
            x_norm = x / frame_w
            y_norm = y / frame_h
            dist   = math.sqrt((1.0 - x_norm)**2 + (0.5 - y_norm)**2)
            angle  = abs(math.atan2(0.5 - y_norm, 1.0 - x_norm))
            w0     = self.xg_model["w0"]
            w1     = self.xg_model["w1"]
            w2     = self.xg_model["w2"]
            z      = w0 + w1 * dist + w2 * angle

        z   = max(-100, min(100, z))
        xg  = 1 / (1 + math.exp(-z))
        return round(max(0.01, min(0.5, xg)), 3)

    # ─────────────────────────────────────────
    # RÉCUPÉRER LES SEUILS APPRIS
    # ─────────────────────────────────────────
    def get_thresholds(self):
        return self.thresholds.copy()

    # ─────────────────────────────────────────
    # STATS GLOBALES
    # ─────────────────────────────────────────
    def stats(self):
        matches = list({e["match_id"] for e in self.events_db})
        types   = defaultdict(int)
        for e in self.events_db:
            types[e["type"]] += 1

        return {
            "sport":      self.sport,
            "n_matches":  len(matches),
            "n_events":   len(self.events_db),
            "event_types": dict(types),
            "xg_model":   self.xg_model,
            "thresholds": self.thresholds,
        }