"""
GeometryDiagnostics — Sprint BC
=================================
Outils de visualisation et métriques pour valider la reconstruction terrain.

Usage :
    from geometry.diagnostics import GeometryDiagnostics
    diag = GeometryDiagnostics(frame_w=1920, frame_h=1080)
    annotated = diag.draw_state(frame, field_state)
    diag.print_quality_report(observations)
"""

import cv2
import numpy as np
from typing import Optional


class GeometryDiagnostics:
    """Visualisation et métriques de qualité géométrique."""

    def __init__(self, frame_w: int = 1920, frame_h: int = 1080):
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.colors = {
            "goal":    (0, 255, 0),     # vert
            "penalty": (255, 165, 0),   # orange
            "sideline":(0, 165, 255),   # bleu
            "ball":    (0, 0, 255),     # rouge
            "text":    (255, 255, 255), # blanc
        }

    def draw_state(self, frame: np.ndarray, field_state: dict,
                   ball_x: Optional[float] = None,
                   ball_y: Optional[float] = None) -> np.ndarray:
        """
        Dessine l'état terrain sur la frame.
        Retourne une copie annotée.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        if field_state.get("status") != "ok":
            cv2.putText(out, f"GEOMETRY: {field_state.get('status', 'unknown')}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 200), 2)
            return out

        # Lignes de touche
        y_top = field_state.get("sideline_y_top")
        y_bot = field_state.get("sideline_y_bot")
        if y_top is not None:
            y_px = int(y_top * h)
            cv2.line(out, (0, y_px), (w, y_px), self.colors["sideline"], 2)
        if y_bot is not None:
            y_px = int(y_bot * h)
            cv2.line(out, (0, y_px), (w, y_px), self.colors["sideline"], 2)

        # But gauche
        gl = field_state.get("goal_left_x")
        gr = field_state.get("goal_right_x")
        if gl is not None and gr is not None:
            x1 = int(gl * w)
            x2 = int(gr * w)
            y1_ = int((y_top or 0.25) * h)
            y2_ = int((y_bot or 0.90) * h)
            cv2.rectangle(out, (x1, y1_), (x2, y2_), self.colors["goal"], 2)
            cv2.putText(out, f"GOAL [{gl:.2f},{gr:.2f}]",
                        (x1, y1_ - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, self.colors["goal"], 1)

        # Surface de réparation
        pl = field_state.get("penalty_left_x")
        pr = field_state.get("penalty_right_x")
        if pl is not None and pr is not None:
            x1 = int(pl * w)
            x2 = int(pr * w)
            y1_ = int((y_top or 0.25) * h)
            y2_ = int((y_bot or 0.90) * h)
            cv2.rectangle(out, (x1, y1_), (x2, y2_),
                          self.colors["penalty"], 1)

        # Ballon
        if ball_x is not None and ball_y is not None:
            bx_px = int(ball_x * w)
            by_px = int(ball_y * h)
            cv2.circle(out, (bx_px, by_px), 8, self.colors["ball"], -1)

        # Info confiance
        conf = field_state.get("confidence", 0)
        n = field_state.get("n_observations", 0)
        cv2.putText(out, f"GEOM conf={conf:.2f} n={n}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, self.colors["text"], 2)

        return out

    def print_quality_report(self, observations: list):
        """
        Affiche un rapport de qualité sur une liste d'observations.
        """
        if not observations:
            print("  [GEOM DIAG] Aucune observation")
            return

        n = len(observations)
        n_goal     = sum(1 for o in observations if o.get("goal_left_x") is not None)
        n_penalty  = sum(1 for o in observations if o.get("penalty_left_x") is not None)
        n_sideline = sum(1 for o in observations if o.get("sideline_y_bot") is not None)
        confs      = [o.get("confidence", 0) for o in observations]
        avg_conf   = sum(confs) / max(1, len(confs))

        print(f"  [GEOM DIAG] ── Rapport qualité ─────────────────────────")
        print(f"  [GEOM DIAG]   Frames analysées : {n}")
        print(f"  [GEOM DIAG]   But détecté      : {n_goal}/{n} ({100*n_goal//max(1,n)}%)")
        print(f"  [GEOM DIAG]   Surface détectée : {n_penalty}/{n} ({100*n_penalty//max(1,n)}%)")
        print(f"  [GEOM DIAG]   Sideline détecté : {n_sideline}/{n} ({100*n_sideline//max(1,n)}%)")
        print(f"  [GEOM DIAG]   Confiance moy.   : {avg_conf:.3f}")

        # Distribution des valeurs de but gauche
        goal_x_vals = [o["goal_left_x"] for o in observations
                       if o.get("goal_left_x") is not None]
        if goal_x_vals:
            mean_x = sum(goal_x_vals) / len(goal_x_vals)
            std_x  = np.std(goal_x_vals)
            print(f"  [GEOM DIAG]   But gauche x     : μ={mean_x:.3f} σ={std_x:.3f}")

        print(f"  [GEOM DIAG] ──────────────────────────────────────────────")

    @staticmethod
    def evaluate_against_ground_truth(field_state: dict,
                                       gt: dict) -> dict:
        """
        Compare l'état détecté avec une vérité terrain manuelle.
        gt = {"goal_left_x": 0.04, "goal_right_x": 0.15, ...}
        """
        if field_state.get("status") != "ok":
            return {"error": "état terrain non disponible"}

        results = {}
        for key in ["goal_left_x", "goal_right_x",
                    "penalty_left_x", "penalty_right_x"]:
            detected = field_state.get(key)
            expected = gt.get(key)
            if detected is not None and expected is not None:
                err = abs(detected - expected)
                results[key] = {
                    "detected": detected,
                    "expected": expected,
                    "error":    round(err, 4),
                    "ok":       err < 0.03,  # tolérance 3%
                }

        n_ok = sum(1 for r in results.values() if r.get("ok"))
        n_total = len(results)
        results["summary"] = {
            "n_ok": n_ok,
            "n_total": n_total,
            "accuracy": n_ok / max(1, n_total),
        }
        return results
