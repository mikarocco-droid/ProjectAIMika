# shot_profiler.py
# ─────────────────────────────────────────────────────────────────────────────
# Observateur passif — ne modifie JAMAIS le pipeline.
# Reçoit le contexte, calcule les métriques, écrit le CSV.
#
# Usage :
#   profiler = ShotProfiler(output_dir, video_path)
#   profiler.observe(shot, frames_data, fps, camera_type, ...)
#   profiler.update_posthoc(shot_t, posthoc_score, stuck_frames, ...)
#   profiler.update_gemini(shot_t, gemini_called=True, gemini_yes=True, ...)
#   profiler.mark_validated(shot_t)
#   profiler.finish()
#
# Produit :
#   outputs/shot_profiles/{video_name}_shots.csv
#
# shot_id      = {video_hash}_{frame_index}  — joinable avec BC4.csv, Gemini.csv
# candidate_id = {video_hash}_{int(shot_t*10)}_{index}  — stable dans les logs
# REAL_LABEL   = absent du CSV — enrichissement offline via ground_truth.csv
#
# candidate_stage lifecycle (sens unique) :
#   raw → posthoc → pre_gemini → gemini → validated
#   raw → posthoc_rejected   (mort au filtre posthoc)
#
# reject_reason enum :
#   None | "missing_terminal" | "cooldown" | "low_score" | "camera_gate" | "duplicate"
# ─────────────────────────────────────────────────────────────────────────────

import os
import csv
import math
import hashlib
import time


# Valeurs autorisées pour reject_reason — facilite df.groupby("reject_reason")
REJECT_REASONS = frozenset({
    "missing_terminal",
    "cooldown",
    "low_score",
    "camera_gate",
    "duplicate",
    None,
})

# Version du pipeline — à incrémenter à chaque changement de logique métier.
# Permet de retrouver avec quel pipeline un CSV a été produit.
# Format : YYYY-MM-DD-description
PIPELINE_VERSION = "2026-07-02-posthoc-instrumentation"


def _ball_positions(frames_data, t_start, t_end, fps):
    positions = []
    for fd in frames_data:
        frame_idx = fd.get("frame", 0)
        t = frame_idx / max(fps, 1)
        if t_start - 0.1 <= t <= t_end + 0.1:
            ball = fd.get("ball")
            if ball and ball.get("center"):
                cx, cy = ball["center"]
                fw = fd.get("frame_w", 1920)
                fh = fd.get("frame_h", 1080)
                positions.append((t, cx / max(fw, 1), cy / max(fh, 1), frame_idx))
    return positions


def _speeds(positions):
    speeds = []
    for i in range(1, len(positions)):
        dt = positions[i][0] - positions[i - 1][0]
        if dt <= 0:
            continue
        dx = positions[i][1] - positions[i - 1][1]
        dy = positions[i][2] - positions[i - 1][2]
        speeds.append(math.sqrt(dx * dx + dy * dy) / dt)
    return speeds


def _goal_angle_deg(bx, by, goal_x=0.02):
    dx = goal_x - bx
    dy = 0.5 - by
    return round(math.degrees(math.atan2(abs(dy), abs(dx))), 1) if dx != 0 else 90.0


def _keeper_metrics(frames_data, shot_t, fps, window=2.0, goal_x=0.02):
    keeper_pos = []
    for fd in frames_data:
        t = fd.get("frame", 0) / max(fps, 1)
        if abs(t - shot_t) > window:
            continue
        fw = fd.get("frame_w", 1920)
        players = fd.get("players", [])
        gk = [p for p in players
              if p.get("role") == "gk"
              or str(p.get("jersey", "")).lower() in ("gk", "0", "00")]
        if not gk and players:
            gk = [min(players, key=lambda p: p.get("center", [fw, 0])[0])]
        if gk:
            cx = gk[0].get("center", [0, 0])[0] / max(fw, 1)
            keeper_pos.append((t, cx))
    if not keeper_pos:
        return None, None, None
    closest = min(keeper_pos, key=lambda p: abs(p[0] - shot_t))
    dist = round(abs(closest[1] - goal_x), 4)
    if len(keeper_pos) >= 2:
        kp = sorted(keeper_pos)
        dx = kp[-1][1] - kp[0][1]
        dt = kp[-1][0] - kp[0][0]
        speed = round(abs(dx / dt), 4) if dt > 0 else 0.0
        direction = "toward_goal" if dx < 0 else "away_from_goal"
    else:
        speed, direction = None, None
    return dist, speed, direction


def _density_box(frames_data, shot_t, fps, box_x2=0.2):
    best = min(frames_data,
               key=lambda fd: abs(fd.get("frame", 0) / max(fps, 1) - shot_t),
               default=None)
    if not best:
        return None
    fw = best.get("frame_w", 1920)
    return sum(1 for p in best.get("players", [])
               if p.get("center", [fw, 0])[0] / max(fw, 1) <= box_x2)


class ShotProfiler:
    """Observateur passif branché sur le pipeline shot_to_goal."""

    WINDOW_BEFORE = 2.0
    WINDOW_AFTER  = 7.0
    MAX_FRAME_REFS = 10

    def __init__(self, output_dir, video_path="unknown"):
        self.output_dir   = output_dir
        self.video_path   = video_path
        self.video_hash   = hashlib.md5(
            os.path.basename(video_path).encode()
        ).hexdigest()[:8]
        self.rows         = []
        self._started     = time.time()
        self._cand_index  = 0   # compteur pour candidate_id unique
        os.makedirs(os.path.join(output_dir, "shot_profiles"), exist_ok=True)

    def _shot_id(self, shot_t, fps):
        return f"{self.video_hash}_{int(round(shot_t * fps)):06d}"

    def _candidate_id(self, shot_t):
        """ID stable et lisible dans les logs, le CSV et les sorties Gemini."""
        cid = f"{self.video_hash}_{int(shot_t * 10):05d}_{self._cand_index:03d}"
        self._cand_index += 1
        return cid

    def observe(self, shot, frames_data, fps,
                camera_type="unknown",
                gemini_called=False, gemini_yes=None, goal_score=None,
                posthoc_score=None, candidate_source=None,
                world_score=None, in_goal=None, bx_at_goal=None):
        """Non bloquant — toute exception est silencieusement ignorée."""
        try:
            self._observe_inner(
                shot, frames_data, fps, camera_type,
                gemini_called, gemini_yes, goal_score,
                posthoc_score, candidate_source,
                world_score, in_goal, bx_at_goal,
            )
        except Exception:
            pass

    def _observe_inner(self, shot, frames_data, fps, camera_type,
                       gemini_called, gemini_yes, goal_score,
                       posthoc_score, candidate_source,
                       world_score, in_goal, bx_at_goal):
        st  = float(shot.get("time", 0))
        xg  = float(shot.get("xg", 0) or 0)

        pos_b = _ball_positions(frames_data, st - self.WINDOW_BEFORE, st, fps)
        pos_a = _ball_positions(frames_data, st, st + self.WINDOW_AFTER, fps)

        # Position au tir
        shot_x = round(pos_a[0][1], 4) if pos_a else (
                 round(pos_b[-1][1], 4) if pos_b else None)
        shot_y = round(pos_a[0][2], 4) if pos_a else (
                 round(pos_b[-1][2], 4) if pos_b else None)

        # Cinématique
        sp_b = _speeds(pos_b)
        sp_a = _speeds(pos_a)

        speed_before = round(sp_b[-1], 4) if sp_b else None
        speed_at     = round(sp_a[0],  4) if sp_a else None
        speed_peak   = round(max(sp_a), 4) if sp_a else None
        speed_after  = round(sp_a[-1],  4) if sp_a else None
        accel        = round(sp_a[1] - sp_a[0], 4) if len(sp_a) >= 2 else None
        jerk         = round(sp_a[2] - sp_a[0], 4) if len(sp_a) >= 3 else None

        travel_dist = travel_angle = None
        if len(pos_a) >= 2:
            dx = pos_a[-1][1] - pos_a[0][1]
            dy = pos_a[-1][2] - pos_a[0][2]
            travel_dist  = round(math.sqrt(dx * dx + dy * dy), 4)
            travel_angle = round(math.degrees(math.atan2(dy, dx)), 1)

        # Temps avant disparition
        time_to_disappear = None
        for i in range(1, len(pos_a)):
            if pos_a[i][0] - pos_a[i - 1][0] > 1.0:
                time_to_disappear = round(pos_a[i - 1][0] - st, 2)
                break

        frames_visible  = len(pos_a)
        frames_expected = max(1, int(self.WINDOW_AFTER * fps / 4))
        frames_lost     = max(0, frames_expected - frames_visible)

        # Géométrie
        goal_angle = goal_dist = None
        if shot_x is not None:
            goal_angle = _goal_angle_deg(shot_x, shot_y or 0.5)
            goal_dist  = round(math.sqrt((shot_x - 0.02)**2 +
                                         ((shot_y or 0.5) - 0.5)**2), 4)

        # Gardien + densité
        keeper_dist, keeper_speed, keeper_dir = _keeper_metrics(
            frames_data, st, fps)
        density = _density_box(frames_data, st, fps)

        # Références frames
        all_pos = pos_b + pos_a
        step = max(1, len(all_pos) // self.MAX_FRAME_REFS)
        frame_refs = [str(p[3]) for p in all_pos[::step]][:self.MAX_FRAME_REFS]

        self.rows.append({
            "candidate_id":      self._candidate_id(st),   # stable dans logs+CSV+Gemini
            "shot_id":           self._shot_id(st, fps),
            "video":             os.path.basename(self.video_path),
            "pipeline_version":  PIPELINE_VERSION,
            "shot_t_s":          round(st, 2),
            "shot_t_fmt":        f"{int(st//60):02d}:{int(st%60):02d}",
            "camera_type":       camera_type,
            "shot_zone_x":       shot_x,
            "shot_zone_y":       shot_y,
            "speed_before":      speed_before,
            "speed_at_shot":     speed_at,
            "speed_peak":        speed_peak,
            "speed_after":       speed_after,
            "accel":             accel,
            "jerk":              jerk,
            "travel_distance":   travel_dist,
            "travel_angle_deg":  travel_angle,
            "time_to_disappear": time_to_disappear,
            "frames_visible_7s": frames_visible,
            "frames_lost_7s":    frames_lost,
            "goal_angle_deg":    goal_angle,
            "goal_dist_norm":    goal_dist,
            "world_score":       world_score,
            "in_goal":           in_goal,
            "bx_at_goal":        bx_at_goal,
            "keeper_dist":       keeper_dist,
            "keeper_speed":      keeper_speed,
            "keeper_direction":  keeper_dir,
            "density_box":       density,
            "candidate_source":  candidate_source,
            "posthoc_score":     posthoc_score,
            "stuck_frames":      None,       # alimenté par update_posthoc()
            "rebound":           None,       # alimenté par update_posthoc()
            "n_terminal":        None,       # nb terminaux dans voisinage (0/1/2/3...)
            "has_terminal":      None,       # True/False raccourci de n_terminal > 0
            "posthoc_rejected":  None,       # True si filtré avant Gemini
            "reject_reason":     None,       # voir REJECT_REASONS enum
            "candidate_stage":   "raw",      # raw→posthoc→pre_gemini→gemini→validated
            "xG":                round(xg, 4),
            "on_target":         shot.get("on_target", False),
            "gemini_called":     gemini_called,
            "gemini_yes":        gemini_yes,
            "goal_score":        goal_score,
            "frame_refs":        "|".join(frame_refs),
        })

    def update_gemini(self, shot_t, gemini_called=True,
                      gemini_yes=None, goal_score=None):
        """Mise à jour post-Gemini — cherche le profil par shot_t.

        Machine d'états : posthoc_rejected est un état terminal.
        Un candidat mort au filtre posthoc ne peut pas transiter vers gemini.
        """
        for row in self.rows:
            if abs(row["shot_t_s"] - shot_t) < 1.0:
                if row["candidate_stage"] == "posthoc_rejected":
                    return  # état terminal — aucune modification
                row["gemini_called"] = gemini_called
                if gemini_yes  is not None: row["gemini_yes"]  = gemini_yes
                if goal_score  is not None: row["goal_score"]  = goal_score
                if gemini_called and row["candidate_stage"] == "posthoc":
                    row["candidate_stage"] = "gemini"
                break

    def mark_validated(self, shot_t):
        """Marque le candidat comme validé (but confirmé). Étape finale.

        Machine d'états : validated n'est accessible que depuis gemini.
        posthoc_rejected → validated est impossible.
        """
        for row in self.rows:
            if abs(row["shot_t_s"] - shot_t) < 1.0:
                if row["candidate_stage"] == "posthoc_rejected":
                    return  # état terminal — aucune modification
                if row["candidate_stage"] == "gemini":
                    row["candidate_stage"] = "validated"
                break

    def update_posthoc(self, shot_t, posthoc_score=None, stuck_frames=None,
                       rebound=None, n_terminal=None,
                       rejected=None, reject_reason=None):
        """Mise à jour filtre posthoc — appelé depuis pipeline.py après [FILTRE POSTHOC].

        Permet de savoir exactement pourquoi un candidat a été rejeté avant Gemini.
        Clé de jointure : shot_t (linked_shot_t du posthoc → shot_t_s du tir).

        Args:
            n_terminal   : nombre de terminaux dans le voisinage (int, pas bool)
            reject_reason: doit être dans REJECT_REASONS (None si accepté)
        """
        # Validation enum
        if reject_reason not in REJECT_REASONS:
            reject_reason = "missing_terminal"   # fallback sûr

        matched = False
        for row in self.rows:
            # Comparaison exacte à 3 décimales (linked_shot_t est round(..., 3))
            # Pas de fenêtre temporelle — shot_t provient directement de goal_posthoc.linked_shot_t
            if round(row["shot_t_s"], 3) == round(shot_t, 3):
                matched = True
                if posthoc_score is not None: row["posthoc_score"]    = posthoc_score
                if stuck_frames  is not None: row["stuck_frames"]     = stuck_frames
                if rebound       is not None: row["rebound"]          = rebound
                if n_terminal    is not None:
                    row["n_terminal"]  = int(n_terminal)
                    row["has_terminal"] = int(n_terminal) > 0
                if rejected      is not None: row["posthoc_rejected"] = rejected
                # reject_reason : toujours écrire (None pour les acceptés)
                row["reject_reason"] = reject_reason
                # candidate_stage : sens unique seulement
                if rejected is False and row["candidate_stage"] == "raw":
                    row["candidate_stage"] = "posthoc"
                elif rejected is True and row["candidate_stage"] == "raw":
                    row["candidate_stage"] = "posthoc_rejected"
                print(
                    f"  [PROFILER UPDATE] shot_t={shot_t:.2f}s matched row={row['shot_t_s']:.2f}s "
                    f"delta={abs(row['shot_t_s']-shot_t):.3f}s "
                    f"stage={row['candidate_stage']}"
                )
                break
        if not matched:
            print(
                f"  [PROFILER UPDATE] shot_t={shot_t:.2f}s → NO MATCH "
                f"(rows={[round(r['shot_t_s'],2) for r in self.rows]})"
            )

    def finish(self):
        """Sauvegarde le CSV final. Appelé une seule fois en fin de pipeline."""
        if not self.rows:
            return None
        name     = os.path.splitext(os.path.basename(self.video_path))[0]
        csv_path = os.path.join(self.output_dir, "shot_profiles",
                                f"{name}_shots.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        elapsed = round(time.time() - self._started, 1)
        print(f"  [SHOT_PROFILER] {len(self.rows)} tirs → {csv_path} ({elapsed}s)")
        return csv_path