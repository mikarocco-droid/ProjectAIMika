# vision/ball_tracker.py
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
import math


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_valid_ball(ball, frame_w, frame_h):
    if ball is None:
        return False
    x, y, w, h = ball
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
        return False
    if w <= 2 or h <= 2:
        return False
    if w > frame_w * 0.2:
        return False
    cx = x + w // 2
    cy = y + h // 2
    if cx < 0 or cx >= frame_w or cy < 0 or cy >= frame_h:
        return False
    return True


# ─────────────────────────────────────────
# BALL BUFFER TIMESTAMPÉ
# ─────────────────────────────────────────
class BallBuffer:
    def __init__(self, size=30):
        self._buf = deque(maxlen=size)

    def add(self, x, y, t, frame_w=None, frame_h=None):
        # V9.7+ : ignorer les positions hors frame (Kalman/interpolation ratée)
        if frame_w is not None and frame_h is not None:
            if x < 0 or x > frame_w or y < 0 or y > frame_h:
                print(f"  [BALL FILTERED] ({x:.1f},{y:.1f}) t={t:.1f}s")
                return
        self._buf.append((float(x), float(y), float(t)))

    def get(self):
        return list(self._buf)

    def last_pos(self):
        if not self._buf:
            return None
        x, y, _ = self._buf[-1]
        return (x, y)

    def speed_px_per_sec(self):
        """Vitesse en px/seconde — robuste frame skip."""
        pts = list(self._buf)
        if len(pts) < 2:
            return 0.0
        n      = min(len(pts), 4)
        recent = pts[-n:]
        speeds = []
        for i in range(1, len(recent)):
            x1, y1, t1 = recent[i-1]
            x2, y2, t2 = recent[i]
            dt = max(t2 - t1, 1e-4)
            d  = math.hypot(x2 - x1, y2 - y1)
            speeds.append(d / dt)
        return sum(speeds) / len(speeds) if speeds else 0.0

    def speed_px_per_frame(self):
        """Compatibilité arrière."""
        pts = list(self._buf)
        if len(pts) < 2:
            return 0.0
        x1, y1, _ = pts[-2]
        x2, y2, _ = pts[-1]
        return math.hypot(x2 - x1, y2 - y1)

    def direction(self):
        """Vecteur direction normalisé sur les 4 derniers points."""
        pts = list(self._buf)
        if len(pts) < 2:
            return (0.0, 0.0)
        recent = pts[-4:] if len(pts) >= 4 else pts
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        norm = math.hypot(dx, dy) + 1e-6
        return (dx / norm, dy / norm)

    def direction_stability(self, n_frames=3):
        """
        Mesure la cohérence de direction sur les n dernières frames.
        Retourne 0.0 (chaotique) à 1.0 (trajectoire droite).
        Évite de confondre rebond ou passe rapide avec un tir.
        """
        pts = list(self._buf)
        if len(pts) < n_frames + 1:
            return 0.0
        recent = pts[-(n_frames + 1):]
        dirs   = []
        for i in range(1, len(recent)):
            x1, y1, _ = recent[i-1]
            x2, y2, _ = recent[i]
            dx = x2 - x1
            dy = y2 - y1
            norm = math.hypot(dx, dy) + 1e-6
            dirs.append((dx / norm, dy / norm))
        if len(dirs) < 2:
            return 1.0
        dots = [
            dirs[i][0]*dirs[i+1][0] + dirs[i][1]*dirs[i+1][1]
            for i in range(len(dirs) - 1)
        ]
        return max(0.0, sum(dots) / len(dots))

    def toward_goal(self, frame_w, frame_h, threshold=0.55):
        """Vérifie les deux buts en vue latérale."""
        if not self._buf:
            return False, None
        bx, by, _ = self._buf[-1]
        dx, dy    = self.direction()
        goal_centers = [
            (0.0,     frame_h * 0.5),
            (frame_w, frame_h * 0.5),
        ]
        for i, (gx, gy) in enumerate(goal_centers):
            to_goal_x = gx - bx
            to_goal_y = gy - by
            norm      = math.hypot(to_goal_x, to_goal_y) + 1e-6
            dot       = dx * (to_goal_x / norm) + dy * (to_goal_y / norm)
            if dot > threshold:
                return True, i
        return False, None

    def clear(self):
        self._buf.clear()

    def __len__(self):
        return len(self._buf)


# ─────────────────────────────────────────
# KALMAN SIMPLIFIÉ
# ─────────────────────────────────────────
class SimpleKalman:
    def __init__(self):
        self.state    = None
        self.velocity = np.array([0.0, 0.0])

    def update(self, measurement):
        if measurement is None:
            if self.state is not None:
                self.state    = self.state + self.velocity
                self.velocity = self.velocity * 0.7
                # Limiter vélocité max — évite divergence Kalman
                speed = float(np.hypot(self.velocity[0], self.velocity[1]))
                if speed > 300:
                    scale = 300 / speed
                    self.velocity = self.velocity * scale
                # Limiter la vélocité max — évite divergence Kalman
                speed = float(np.hypot(self.velocity[0], self.velocity[1]))
                if speed > 300:  # max 300px/frame = très rapide
                    scale = 300 / speed
                    self.velocity = self.velocity * scale
            return self.state
        m = np.array(measurement, dtype=float)
        if self.state is None:
            self.state    = m
            self.velocity = np.array([0.0, 0.0])
            return self.state
        self.velocity = (m - self.state) * 0.9
        self.state    = m
        return self.state

    def reset(self):
        self.state    = None
        self.velocity = np.array([0.0, 0.0])


# ─────────────────────────────────────────
# SHOT CANDIDATE — lien tir → but
# ─────────────────────────────────────────
class ShotCandidate:
    """
    Lie un tir détecté à un éventuel but.
    Si le ballon reste en zone de but pendant > min_frames
    après un tir candidate → confirme le lien tir → but.
    Boost énorme pour le xG et la qualité des highlights.
    """
    def __init__(self, x, y, t, xg=0.0, player=None, team=None):
        self.x      = x
        self.y      = y
        self.t      = t
        self.xg     = xg
        self.player = player
        self.team   = team
        self.frames = 0
        self.active = True

    def tick_in_goal_zone(self):
        self.frames += 1

    def is_confirmed_goal(self, min_frames=3):
        return self.active and self.frames >= min_frames

    def expire(self, current_t, max_age=2.0):
        """Expire après 2s sans but — évite les faux liens."""
        if current_t - self.t > max_age:
            self.active = False

    def distance_to_goal(self, frame_w, frame_h):
        """
        Distance réelle du tir au but le plus proche.
        Utilisée pour affiner le xG : tir de loin = xG bas.
        """
        dist_right = math.hypot(self.x - frame_w, self.y - frame_h / 2)
        dist_left  = math.hypot(self.x,            self.y - frame_h / 2)
        return min(dist_right, dist_left)


# ─────────────────────────────────────────
# ANTI-SAUT — filtre faux positifs tracking
# ─────────────────────────────────────────
def is_valid_jump(prev, new_pos, max_dist=250, velocity=(0.0, 0.0)):
    """
    Rejette les sauts impossibles du ballon.
    Un ballon ne peut pas téléporter à > 150px entre 2 frames.
    """
    if prev is None:
        return True
    dx = prev[0] - new_pos[0]
    dy = prev[1] - new_pos[1]
    dyn = min(350, max_dist + math.hypot(velocity[0], velocity[1]) * 1.5)
    return (dx * dx + dy * dy) < dyn * dyn


# ─────────────────────────────────────────
# BALL TRACKER PRINCIPAL
# ─────────────────────────────────────────
class BallTracker:

    def __init__(self, max_history=30, fps=25):
        self.fps           = fps
        self.ball_buffer   = BallBuffer(size=max_history)
        self.kalman        = SimpleKalman()
        self.last_seen     = 0
        self.frame_id      = 0
        self.lost_frames   = 0
        self.max_lost      = 12   # V9.6 : tolérance élargie pour tirs rapides
        self.shot_candidate: ShotCandidate | None = None

        # V9.6 — mémoire position + vélocité prédictive
        self.last_valid_ball  = None   # dernière position fiable
        self.last_valid_frame = -1     # frame correspondante
        self.velocity         = (0.0, 0.0)
        self._filtered_streak     = 0
        self._MAX_FILTERED_STREAK = 25  # vélocité estimée
        # SHORT_RANGE_SHOT — flag déclenché quand pic vitesse + disparition immédiate
        self.short_range_shot_pending = False
        self._srs_speed               = 0.0
        self._srs_pos                 = None

    def select_best_ball(self, balls, last_pos):
        if not balls:
            return None
        if last_pos is None:
            return balls[0]
        best       = None
        best_score = 1e9
        for b in balls:
            x, y, w, h = b
            cx = x + w // 2
            cy = y + h // 2
            d            = distance((cx, cy), last_pos)
            size_penalty = w * h * 0.001
            score        = d + size_penalty
            if score < best_score:
                best_score = score
                best       = b
        return best

    def update(self, detected_balls, frame_w, frame_h, timestamp=None):
        self.frame_id += 1
        t = timestamp if timestamp is not None else self.frame_id / self.fps

        # FIX G — reset periodique toutes les 5000 frames
        if self.frame_id > 0 and self.frame_id % 5000 == 0:
            print(f"  [BALL TRACKER] reset periodique frame={self.frame_id}")
            self.ball_buffer.clear(); self.kalman.reset()
            self.last_valid_ball=None; self.last_valid_frame=-1
            self.velocity=(0.0,0.0); self._filtered_streak=0
        # FIX B — expiration last_valid_ball
        if (self.last_valid_ball is not None and self.last_valid_frame>=0
                and self.frame_id - self.last_valid_frame > 30):
            self.last_valid_ball=None; self.last_valid_frame=-1; self.velocity=(0.0,0.0)
        # FIX C+E — streak reset / Kalman reset
        if self._filtered_streak >= self._MAX_FILTERED_STREAK:
            print(f"  [BALL TRACKER RESET] streak={self._filtered_streak}")
            self.ball_buffer.clear(); self.kalman.reset()
            self.last_valid_ball=None; self.last_valid_frame=-1
            self.velocity=(0.0,0.0); self._filtered_streak=0
        elif self._filtered_streak > 10:
            self.kalman.reset(); self.velocity=(0.0,0.0)
        detected_balls = [
            b for b in detected_balls
            if is_valid_ball(b, frame_w, frame_h)
        ]

        last_pos = self.ball_buffer.last_pos()
        best     = self.select_best_ball(detected_balls, last_pos)

        if best is not None:
            x, y, w, h = best
            cx = x + w // 2
            cy = y + h // 2

            # MODIF 3 — anti-saut : rejeter faux positifs
            if is_valid_jump(self.last_valid_ball, (cx, cy), velocity=self.velocity):
                # MODIF 4 — lissage exponentiel vélocité (alpha=0.6)
                if self.last_valid_ball is not None:
                    vx = cx - self.last_valid_ball[0]
                    vy = cy - self.last_valid_ball[1]
                    alpha = 0.6
                    self.velocity = (
                        alpha * vx + (1 - alpha) * self.velocity[0],
                        alpha * vy + (1 - alpha) * self.velocity[1],
                    )
                # MODIF 2 — mémoriser dernière position fiable
                self.last_valid_ball  = (cx, cy)
                self.last_valid_frame = self.frame_id
                self._filtered_streak = 0

                self.ball_buffer.add(cx, cy, t, frame_w, frame_h)
                self.last_seen   = self.frame_id
                self.lost_frames = 0
                pos = self.kalman.update((cx, cy))
                return self.get_ball_bbox(pos), False
            else:
                # Saut invalide → traiter comme perte
                best = None

        if best is None:
            self._filtered_streak += 1
            self.lost_frames += 1
            # ── SHORT_RANGE_SHOT — détection à la première frame de perte ────
            # C'est ici qu'on a lost_frames=1, APRÈS avoir vu le pic de vitesse.
            # last_valid_ball contient la position du dernier ballon visible (le pic).
            if self.lost_frames == 1:
                _srs_ok, _srs_spd = self.is_short_range_shot(frame_w, frame_h)
                if _srs_ok:
                    self.short_range_shot_pending = True
                    self._srs_speed = _srs_spd
                    self._srs_pos   = self.last_valid_ball
                    print(f"  [SHORT_RANGE_SHOT] speed={_srs_spd:.0f}px/f "
                          f"x={self.last_valid_ball[0]/frame_w:.2f} "
                          f"→ tir à bout portant détecté ✅")
                else:
                    self.short_range_shot_pending = False
            # ──────────────────────────────────────────────────────────────────
            if self.lost_frames <= 2:
                # Frames 1-2 : prédiction courte (micro-occlusion / tirs rapides)
                pos = self.kalman.update(None)
                # FIX F — invalider prediction hors frame
                if pos is not None:
                    _px,_py=int(pos[0]),int(pos[1])
                    if _px<0 or _px>=frame_w or _py<0 or _py>=frame_h:
                        pos=None; self.kalman.reset()
                if pos is not None:
                    cx, cy = int(pos[0]), int(pos[1])
                    self.ball_buffer.add(cx, cy, t, frame_w, frame_h)
                    return self.get_ball_bbox(pos), True
                elif self.last_valid_ball is not None:
                    px = int(self.last_valid_ball[0] + self.velocity[0])
                    py = int(self.last_valid_ball[1] + self.velocity[1])
                    px = max(0, min(frame_w - 1, px))
                    py = max(0, min(frame_h - 1, py))
                    self.velocity = (self.velocity[0] * 0.5, self.velocity[1] * 0.5)
                    self.ball_buffer.add(px, py, t, frame_w, frame_h)
                    return self.get_ball_bbox(np.array([px, py])), True
            elif self.lost_frames <= self.max_lost:
                # Frames 3+ : vraie absence → couper le signal
                # Permet à ball_appears_in_goal de détecter la réapparition
                self.velocity = (0.0, 0.0)
                self.kalman.update(None)  # maintenir l'état interne
                return None, False
            else:
                # Après max_lost → reset complet
                self.ball_buffer.clear()
                self.kalman.reset()
                self.last_valid_ball  = None
                self.velocity         = (0.0, 0.0)
                self.last_valid_frame = -1
                self.velocity         = (0.0, 0.0)
            return None, True

    def get_ball_bbox(self, pos):
        if pos is None:
            return None
        x, y = int(pos[0]), int(pos[1])
        size = 10
        return (x - size, y - size, size * 2, size * 2)

    # ─────────────────────────────────────────
    # GETTERS
    # ─────────────────────────────────────────
    def get_trajectory(self):
        """Compatibilité arrière — retourne liste de (x, y)."""
        return [(x, y) for x, y, _ in self.ball_buffer.get()]

    def get_trajectory_with_time(self):
        """Retourne liste de (x, y, t)."""
        return self.ball_buffer.get()

    def get_speed(self):
        """Compatibilité arrière — px/frame."""
        return self.ball_buffer.speed_px_per_frame()

    def get_speed_per_second(self):
        """Vitesse en px/seconde, robuste frame skip."""
        return self.ball_buffer.speed_px_per_sec()

    def get_direction(self):
        return self.ball_buffer.direction()

    def get_direction_stability(self, n=3):
        return self.ball_buffer.direction_stability(n)

    def _dynamic_shot_threshold(self, frame_w):
        # V9.7 — relevé 1.0 → 1.5 : élimine les passes longues interpolées
        # Un vrai tir sur 1920px = 2880+ px/s, une passe = 800-1500 px/s
        return frame_w * 1.5

    def is_shot_candidate(self, frame_w, frame_h,
                          speed_threshold_px_per_sec=None,
                          alignment_threshold=0.55,
                          stability_threshold=0.55):
        # V9.7+ — seuils assouplis pour imgsz=640 (positions moins précises)
        # alignment 0.75→0.55, stability 0.75→0.55 : ballon moins précis à 640
        """
        Combine vitesse px/s + alignement vers but + stabilité direction.
        Évite les faux tirs sur rebonds et passes rapides.
        """
        if speed_threshold_px_per_sec is None:
            speed_threshold_px_per_sec = self._dynamic_shot_threshold(frame_w)

        # Rejeter si position interpolée (Kalman/vélocité)
        # Tolérance 2 frames pour imgsz=640 qui perd plus souvent le ballon
        if self.lost_frames > 2:
            return False

        # Filtre zone offensive — tirs partent des 25% proches des buts
        # (pas du milieu de terrain)
        last = self.ball_buffer.last_pos()
        if last is not None:
            bx = last[0]
            if not (bx < frame_w * 0.25 or bx > frame_w * 0.75):
                return False

        spd       = self.get_speed_per_second()
        toward, _ = self.ball_buffer.toward_goal(frame_w, frame_h,
                                                  threshold=alignment_threshold)
        stability = self.ball_buffer.direction_stability(3)

        # Filtre accélération — ratio robuste (indépendant résolution)
        # tir = accélération brutale, passe = vitesse constante
        pts = self.ball_buffer.get()
        accel_ok = True  # défaut permissif si pas assez de points
        if len(pts) >= 4:
            def seg_speed(p1, p2):
                dt = max(p2[2] - p1[2], 1e-4)
                return math.hypot(p2[0]-p1[0], p2[1]-p1[1]) / dt
            spd_recent = seg_speed(pts[-2], pts[-1])
            spd_before = seg_speed(pts[-4], pts[-3])
            accel_ratio = spd_recent / (spd_before + 1e-6)
            accel_ok = accel_ratio >= 1.2  # vitesse 20% plus haute (frame_skip=4 = timestamps espacés)

        # ── Filtre direction verticale v9.8 ──────────────────────────
        # Un vrai tir se dirige vers le cadre du but (30%-70% de frame_h)
        # Un dégagement/centre se dirige vers le haut ou vers le bas du terrain
        # dx, dy = direction normalisée du ballon
        _dir = self.ball_buffer.direction()
        _dx, _dy = _dir if _dir else (0, 0)
        # La destination projetée à partir de la position actuelle
        _dest_y = (last[1] + _dy * frame_w * 0.3) if last is not None else frame_h / 2
        _goal_y_min = frame_h * 0.25   # 25% du haut
        _goal_y_max = frame_h * 0.75   # 75% du haut
        _toward_goal_height = (_goal_y_min <= _dest_y <= _goal_y_max)

        # Relever la stabilité pour exclure les dégagements en arc
        _stability_min = max(stability_threshold, 0.60)

        result = (spd > speed_threshold_px_per_sec
                  and toward
                  and stability > _stability_min
                  and accel_ok
                  and _toward_goal_height)

        # Log DEBUG — activé via config.DEBUG
        try:
            from config import DEBUG
        except ImportError:
            DEBUG = False
        if DEBUG and last is not None:
            ratio_str = f"{spd_recent/(spd_before+1e-6):.2f}" if len(pts) >= 4 else "n/a"
            print(f"  [SHOT] spd={spd:.0f} stab={stability:.2f} "
                  f"accel={ratio_str} x={last[0]/frame_w:.2f} → {'✅' if result else '❌'}")

        return result

    # ─────────────────────────────────────────
    # SHOT CANDIDATE — gestion lien tir → but
    # ─────────────────────────────────────────
    def register_shot_candidate(self, x, y, t, xg=0.0,
                                 player=None, team=None):
        """Appelé quand un tir est détecté."""
        self.shot_candidate = ShotCandidate(
            x=x, y=y, t=t, xg=xg,
            player=player, team=team
        )

    def tick_shot_candidate(self, in_goal_zone, current_t):
        """
        Appelé à chaque frame.
        Timeout systématique à 2s — évite les faux liens.
        Retourne True si le lien tir→but est confirmé.
        """
        if self.shot_candidate is None:
            return False

        # Timeout systématique à chaque frame
        self.shot_candidate.expire(current_t, max_age=2.0)
        if not self.shot_candidate.active:
            self.shot_candidate = None
            return False

        if in_goal_zone:
            self.shot_candidate.tick_in_goal_zone()
            if self.shot_candidate.is_confirmed_goal(min_frames=3):
                return True

        return False

    def get_shot_candidate(self):
        return self.shot_candidate

    def clear_shot_candidate(self):
        self.shot_candidate = None

    def is_short_range_shot(self, frame_w, frame_h,
                             speed_threshold_px=150,
                             reappear_goal_pct=0.12):
        """
        Détecte les frappes ultra-courtes (reprises à bout portant, tap-ins)
        que is_shot_candidate() rate parce qu'elles durent 1-2 frames.

        Signature caractéristique :
          1. Vitesse brutale sur 1 frame (≥ 150 px/frame)
          2. Ballon perdu immédiatement après (lost_frames >= 1)
          3. Dernière position connue dans la zone offensive (x < 12% ou x > 88%)
          4. Vitesse AVANT le pic était basse (ballon contrôlé, pas un dégagement)

        Cette combinaison est quasi-impossible à réunir autrement que sur
        une reprise à bout portant depuis le petit rectangle.

        Retourne (bool, speed_px) pour logging.
        """
        # Condition 1 : ballon perdu en ce moment (frappe rapide vient de se produire)
        if self.lost_frames < 1:
            return False, 0.0

        # Condition 2 : dernière position connue dans zone offensive
        last = self.last_valid_ball
        if last is None:
            return False, 0.0
        bx = last[0]
        if not (bx < frame_w * reappear_goal_pct or bx > frame_w * (1 - reappear_goal_pct)):
            return False, 0.0

        # Condition 3 : vitesse du dernier déplacement très élevée
        pts = self.ball_buffer.get()
        if len(pts) < 2:
            return False, 0.0
        p_prev = pts[-2]
        p_last = pts[-1]
        dt = max(p_last[2] - p_prev[2], 1e-4)
        speed_px = math.hypot(p_last[0] - p_prev[0], p_last[1] - p_prev[1]) / dt * (1.0 / max(self.fps, 1))
        # speed_px en px/frame (normalisé par fps)
        speed_px_frame = math.hypot(p_last[0] - p_prev[0], p_last[1] - p_prev[1])

        if speed_px_frame < speed_threshold_px:
            return False, speed_px_frame

        # Condition 4 : vitesse AVANT le pic était basse (ballon posé, pas un dégagement)
        # On compare avec le segment précédent
        if len(pts) >= 3:
            p_before = pts[-3]
            dt_before = max(p_prev[2] - p_before[2], 1e-4)
            speed_before = math.hypot(p_prev[0] - p_before[0], p_prev[1] - p_before[1]) / dt_before * (1.0 / max(self.fps, 1))
            speed_before_frame = math.hypot(p_prev[0] - p_before[0], p_prev[1] - p_before[1])
            # Le pic doit être au moins 3x la vitesse précédente
            if speed_px_frame < speed_before_frame * 2.5:
                return False, speed_px_frame

        return True, speed_px_frame

    def closest_player(self, players):
        last = self.ball_buffer.last_pos()
        if last is None:
            return None
        best      = None
        best_dist = 9999
        for p in players:
            x1, y1, x2, y2 = p["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            d  = distance((cx, cy), last)
            if d < best_dist:
                best_dist = d
                best      = p
        if best_dist < 80:
            return best
        return None

    def reset(self):
        self.ball_buffer.clear()
        self.kalman.reset()
        self.last_seen        = 0
        self.frame_id         = 0
        self.lost_frames      = 0
        self.shot_candidate   = None
        self.last_valid_ball  = None
        self.last_valid_frame = -1
        self.velocity         = (0.0, 0.0)
        self._filtered_streak = 0
        self.short_range_shot_pending = False
        self._srs_speed = 0.0
        self._srs_pos   = None