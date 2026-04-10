# analysis/context_engine.py
# -*- coding: utf-8 -*-
"""
Context Engine — enrichit les events avec :
- shot_context  : pression défensive + longueur séquence
- momentum      : fenêtre glissante 20s par équipe (0 et 1)

Ne duplique PAS :
- possession (déjà dans events.py + compute_possession_from_stats)
- xG          (déjà dans events.py + learning_model)
- ReID        (déjà dans pipeline V19)

Plug & play : appelé en post-processing sur la liste d'events,
pas dans la boucle frame par frame.
"""

import math
from collections import deque, defaultdict


# ─────────────────────────────────────────
# POIDS PAR TYPE D'EVENT POUR LE MOMENTUM
# ─────────────────────────────────────────
MOMENTUM_WEIGHTS = {
    "goal":            10.0,
    "shot":             5.0,
    "shot_blocked":     3.0,
    "interception":     2.0,
    "fast_break":       3.0,
    "progressive_run":  2.0,
    "dribble":          1.0,
    "pass":             0.5,
    "key_pass":         3.0,
    "build_up":         1.0,
}

PRESSURE_RADIUS_PCT = 0.06   # ~115px sur 1920


class ContextEngine:

    def __init__(self, fps=25, frame_w=1920, frame_h=1080):
        self.fps     = fps
        self.frame_w = frame_w
        self.frame_h = frame_h

        self._momentum_window = deque()
        self._momentum_ttl    = 20.0

        self._current_seq   = {0: [], 1: []}
        self._seq_last_team = None

    # ─────────────────────────────────────────
    # MOMENTUM
    # ─────────────────────────────────────────
    def _update_momentum(self, event):
        team  = event.get("team")
        etype = event.get("type", "")
        t     = float(event.get("time", 0))

        if team not in (0, 1):
            return

        w = MOMENTUM_WEIGHTS.get(etype, 0)
        if w == 0:
            return

        self._momentum_window.append((team, w, t))

    def _prune_momentum(self, current_t):
        cutoff = current_t - self._momentum_ttl
        while self._momentum_window and self._momentum_window[0][2] < cutoff:
            self._momentum_window.popleft()

    def get_momentum(self, current_t):
        self._prune_momentum(current_t)
        scores = {0: 0.0, 1: 0.0}
        for team, w, _ in self._momentum_window:
            scores[team] += w
        total = scores[0] + scores[1]
        if total > 0:
            return {
                0: round(scores[0] / total, 3),
                1: round(scores[1] / total, 3),
            }
        return {0: 0.5, 1: 0.5}

    def get_dominant_team(self, current_t):
        m = self.get_momentum(current_t)
        if abs(m[0] - m[1]) < 0.1:
            return None
        return 0 if m[0] > m[1] else 1

    # ─────────────────────────────────────────
    # SÉQUENCES
    # ─────────────────────────────────────────
    def _update_sequence(self, event):
        team  = event.get("team")
        etype = event.get("type", "")

        if etype in ("possession", "under_pressure", "build_up",
                     "long_pass", "progressive_run"):
            return

        if team not in (0, 1):
            return

        if self._seq_last_team is not None and team != self._seq_last_team:
            self._current_seq[self._seq_last_team] = []

        self._current_seq[team].append(etype)
        self._seq_last_team = team

    def get_sequence_length(self, team):
        if team not in (0, 1):
            return 0
        return len(self._current_seq[team])

    # ─────────────────────────────────────────
    # PRESSION DÉFENSIVE
    # ─────────────────────────────────────────
    def compute_pressure(self, ball_pos, players, attacking_team):
        if ball_pos is None or not players:
            return 0
        bx, by = ball_pos
        radius = self.frame_w * PRESSURE_RADIUS_PCT
        count  = 0
        for p in players:
            if p.get("team") == attacking_team:
                continue
            cx, cy = p["center"][0], p["center"][1]
            if math.hypot(cx - bx, cy - by) < radius:
                count += 1
        return count

    # ─────────────────────────────────────────
    # ENRICHISSEMENT SHOT
    # ─────────────────────────────────────────
    def enrich_shot(self, event, pressure, sequence_len, momentum):
        event["pressure"]        = pressure
        event["sequence_length"] = sequence_len
        event["momentum_team"]   = momentum.get(event.get("team"), 0.5)

        if pressure >= 3:
            ctx = "under_pressure"
        elif sequence_len <= 2 and pressure <= 1:
            ctx = "counter_attack"
        else:
            ctx = "open_play"
        event["shot_context"] = ctx

        event["xg_context_mult"] = {
            "under_pressure": 0.70,
            "counter_attack": 1.10,
            "open_play":      1.00,
        }.get(ctx, 1.0)

        return event

    # ─────────────────────────────────────────
    # UPDATE PRINCIPAL
    # ─────────────────────────────────────────
    def update(self, event, ball_pos=None, players=None):
        etype = event.get("type", "")
        t     = float(event.get("time", 0))
        team  = event.get("team")

        self._update_momentum(event)
        self._update_sequence(event)

        momentum = self.get_momentum(t)

        if team in (0, 1):
            event["momentum"] = momentum[team]

        if etype == "shot":
            pressure = self.compute_pressure(ball_pos, players or [], team)
            seq_len  = self.get_sequence_length(team)
            event    = self.enrich_shot(event, pressure, seq_len, momentum)

        if etype == "goal":
            event["dominant_team_at_goal"] = self.get_dominant_team(t)

        return event

    # ─────────────────────────────────────────
    # POST-PROCESSING BATCH
    # ─────────────────────────────────────────
    def process_events(self, events, frames_data=None):
        """
        Enrichit toute la liste d'events en une passe.

        Usage dans pipeline.py :
            from analysis.context_engine import ContextEngine
            ctx = ContextEngine(fps=fps, frame_w=_frame_w, frame_h=_frame_h)
            events = ctx.process_events(events, frames_data)
        """
        frame_index = {}
        if frames_data:
            for fd in frames_data:
                fid = fd.get("frame")
                if fid is not None:
                    frame_index[fid] = fd

        enriched = []
        for e in events:
            ball_pos = None
            players  = None

            fid = e.get("frame")
            if fid is not None and fid in frame_index:
                fd      = frame_index[fid]
                ball    = fd.get("ball")
                players = fd.get("players", [])
                if ball and ball.get("center"):
                    ball_pos = tuple(ball["center"][:2])

            enriched.append(self.update(e, ball_pos=ball_pos, players=players))

        return enriched

    # ─────────────────────────────────────────
    # STATS GLOBALES
    # ─────────────────────────────────────────
    def get_match_stats(self, events):
        shots_ctx    = defaultdict(int)
        avg_pressure = []
        avg_seq_len  = []

        for e in events:
            if e.get("type") == "shot":
                shots_ctx[e.get("shot_context", "open_play")] += 1
                if "pressure" in e:
                    avg_pressure.append(e["pressure"])
                if "sequence_length" in e:
                    avg_seq_len.append(e["sequence_length"])

        return {
            "shots_by_context":     dict(shots_ctx),
            "avg_pressure_on_shot": round(
                sum(avg_pressure) / len(avg_pressure), 2
            ) if avg_pressure else 0,
            "avg_sequence_length":  round(
                sum(avg_seq_len) / len(avg_seq_len), 2
            ) if avg_seq_len else 0,
        }