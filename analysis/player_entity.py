# analysis/player_entity.py
# -*- coding: utf-8 -*-
#
# Phase 2 — Identité joueur persistante sur le match
#
# Améliorations V2 :
#   P1 — Logging détaillé [ENTITY MATCH] pour debug
#   P2 — Clé entity_key = "{team_color}_{jersey}" anti-collision
#   P3 — Temporal smoothing — inertie temporelle sur l'attribution
#   P4 — Team color robustness — HSV + nom approximatif
#   P5 — Ownership propagation — events proches héritent de l'owner

import math
import uuid
from collections import defaultdict

try:
    from sports.config import (
        get_player_identity_config,
        get_visual_match_weights,
        get_visual_fields,
        get_owner_confidence_min,
    )
except ImportError:
    def get_player_identity_config(sport): return {}
    def get_visual_match_weights(sport):
        return {"boots": 0.30, "socks": 0.25, "hair": 0.20,
                "skin": 0.10, "body": 0.08, "role": 0.07}
    def get_visual_fields(sport):
        return ["boots", "socks", "hair", "body", "role", "skin"]
    def get_owner_confidence_min(sport): return 0.85


# ─────────────────────────────────────────
# P4 — TEAM COLOR ROBUSTNESS
# Stocke nom + HSV pour comparaison robuste
# ─────────────────────────────────────────
def bgr_to_hsv_name(bgr):
    """
    Convertit BGR en nom couleur + valeurs HSV approximatives.
    Plus robuste que le nom seul face aux variations d'éclairage.

    Retourne dict : {"name": "bleu", "hsv": (h, s, v)}
    """
    if not bgr:
        return {"name": "inconnu", "hsv": (0, 0, 0)}
    try:
        b_raw, g_raw, r_raw = int(bgr[0]), int(bgr[1]), int(bgr[2])
    except Exception:
        return {"name": "inconnu", "hsv": (0, 0, 0)}

    # BGR → RGB → HSV (simplifié, sans OpenCV)
    # Note : OpenCV stocke en BGR, on convertit en RGB pour le calcul HSV
    r, g, b = r_raw, g_raw, b_raw  # déjà dans le bon ordre pour HSV
    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r_n, g_n, b_n)
    cmin = min(r_n, g_n, b_n)
    delta = cmax - cmin

    # Hue
    if delta == 0:
        h = 0
    elif cmax == r_n:
        h = 60 * (((g_n - b_n) / delta) % 6)
    elif cmax == g_n:
        h = 60 * ((b_n - r_n) / delta + 2)
    else:
        h = 60 * ((r_n - g_n) / delta + 4)

    s = 0 if cmax == 0 else delta / cmax
    v = cmax

    h = int(h) % 360
    s = int(s * 100)
    v = int(v * 100)

    # Nom selon H
    if s < 15:
        name = "blanc" if v > 70 else ("gris" if v > 30 else "noir")
    elif v < 20:
        name = "noir"
    elif 0 <= h < 15 or 345 <= h <= 360:
        name = "rouge"
    elif 15 <= h < 30:
        name = "orange"
    elif 30 <= h < 70:
        name = "jaune"
    elif 70 <= h < 150:
        name = "vert"
    elif 150 <= h < 190:
        name = "cyan"
    elif 190 <= h < 260:
        name = "bleu"
    elif 260 <= h < 300:
        name = "violet"
    elif 300 <= h < 345:
        name = "bordeaux" if v < 50 else "rose"
    else:
        name = "inconnu"

    # Raffiner bleu vs bleu marine
    if name == "bleu" and v < 45:
        name = "bleu marine"
    # Bordeaux = rouge sombre
    if name == "rouge" and v < 50 and s > 40:
        name = "bordeaux"

    return {"name": name, "hsv": (h, s, v)}


def color_similarity(c1, c2):
    """
    Similarité entre deux couleurs HSV [0..1].
    Robuste aux variations d'éclairage (tolérance sur V).
    """
    if not c1 or not c2:
        return 0.0

    n1 = c1.get("name", "") if isinstance(c1, dict) else str(c1)
    n2 = c2.get("name", "") if isinstance(c2, dict) else str(c2)

    # Correspondance exacte sur le nom
    if n1 and n2 and n1 == n2:
        return 1.0

    # Correspondance partielle (bleu == bleu marine → 0.7)
    if n1 and n2 and (n1 in n2 or n2 in n1):
        return 0.7

    # Comparaison HSV si disponible
    if isinstance(c1, dict) and isinstance(c2, dict):
        h1, s1, v1 = c1.get("hsv", (0, 0, 0))
        h2, s2, v2 = c2.get("hsv", (0, 0, 0))

        # Distance teinte (circulaire)
        dh = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180.0
        ds = abs(s1 - s2) / 100.0
        dv = abs(v1 - v2) / 100.0

        # Tolérance élevée sur V (éclairage) et S (compression)
        dist = dh * 0.6 + ds * 0.25 + dv * 0.15
        return round(max(0.0, 1.0 - dist), 3)

    return 0.0


# ─────────────────────────────────────────
# VISUAL MATCH SCORE (inchangé mais utilise sport)
# ─────────────────────────────────────────
def visual_match_score(v1, v2, sport="football"):
    """Score de similarité [0..1] entre deux descriptions visuelles."""
    if not v1 or not v2:
        return 0.0

    weights = get_visual_match_weights(sport)
    if not weights:
        return 0.0

    total_weight = 0.0
    total_score  = 0.0

    for field, weight in weights.items():
        if weight <= 0:
            continue
        val1 = str(v1.get(field) or "").lower().strip()
        val2 = str(v2.get(field) or "").lower().strip()
        if not val1 or not val2 or val1 == "none" or val2 == "none":
            continue

        if val1 == val2:
            field_score = 1.0
        elif val1 in val2 or val2 in val1:
            field_score = 0.7
        else:
            words1 = set(val1.split())
            words2 = set(val2.split())
            common = words1 & words2
            field_score = len(common) / max(len(words1), len(words2)) if common else 0.0

        total_weight += weight
        total_score  += weight * field_score

    if total_weight == 0:
        return 0.0
    return round(total_score / total_weight, 3)


# ─────────────────────────────────────────
# PLAYER ENTITY
# ─────────────────────────────────────────
class PlayerEntity:
    """
    Entité joueur stable sur un match.

    P2 — Clé entity_key = "{team_color}_{jersey}" pour éviter les collisions
         entre #11 bleu et #11 rouge par exemple.
    """

    def __init__(self, entity_id, sport="football"):
        self.entity_id           = entity_id
        self.sport               = sport
        self.jersey              = None
        self.team                = None
        self.team_color          = None      # nom couleur (str)
        self.team_color_hsv      = None      # dict {name, hsv} — P4
        self.pids                = set()
        self.visual              = {}
        self.visual_conf         = "low"
        self.events              = []
        self.highlights          = []
        self.identity_confidence = 0.0
        self._observations       = []
        # P3 — dernière attribution temporelle
        self._last_event_time    = None
        self._last_event_pid     = None
        # Temporel — first/last seen + plages actives
        self.first_seen          = None   # timestamp premier event
        self.last_seen           = None   # timestamp dernier event
        self.active_time_ranges  = []     # [(t_start, t_end)] plages d'activité

    @property
    def entity_key(self):
        """
        P2 — Clé unique anti-collision.
        "{team_color}_{jersey}" si les deux connus, sinon uuid partiel.
        """
        if self.team_color and self.jersey is not None:
            return f"{self.team_color}_{self.jersey}"
        if self.jersey is not None:
            return f"unknown_{self.jersey}"
        return f"anon_{self.entity_id}"

    def add_pid(self, pid):
        if pid is not None:
            self.pids.add(str(pid))

    def add_visual_observation(self, visual_dict, source="gemini"):
        if not visual_dict:
            return
        self._observations.append({"visual": visual_dict, "source": source})
        conf = visual_dict.get("confidence", "low")
        if conf == "high" or (conf == "medium" and self.visual_conf != "high"):
            self.visual      = visual_dict
            self.visual_conf = conf

    def set_team_color(self, color_name=None, bgr=None):
        """
        P4 — Stocke couleur avec HSV pour robustesse.
        Peut recevoir un nom str OU des valeurs BGR.
        """
        if bgr:
            result = bgr_to_hsv_name(bgr)
            self.team_color     = result["name"]
            self.team_color_hsv = result
        elif color_name:
            self.team_color     = str(color_name).lower()
            self.team_color_hsv = {"name": self.team_color, "hsv": (0, 0, 0)}

    def add_event(self, event, owner_confidence=1.0):
        """P3 — Stocke le temps + met à jour first/last seen."""
        t = float(event.get("time", 0) or 0)
        self.events.append({
            "event":            event,
            "owner_confidence": round(float(owner_confidence), 3),
        })
        if t > 0:
            self._last_event_time = t
            self._last_event_pid  = str(event.get("player") or "")
            # Mettre à jour first/last seen
            if self.first_seen is None or t < self.first_seen:
                self.first_seen = t
            if self.last_seen is None or t > self.last_seen:
                self.last_seen = t

    def add_highlight(self, highlight, owner_confidence=1.0):
        self.highlights.append({
            "highlight":        highlight,
            "owner_confidence": round(float(owner_confidence), 3),
        })

    def get_events(self, min_confidence=None, event_types=None):
        if min_confidence is None:
            min_confidence = get_owner_confidence_min(self.sport)
        result = []
        for item in self.events:
            if item["owner_confidence"] < min_confidence:
                continue
            e = item["event"]
            if event_types and e.get("type") not in event_types:
                continue
            result.append(e)
        return result

    def get_highlights(self, min_confidence=None):
        if min_confidence is None:
            min_confidence = get_owner_confidence_min(self.sport)
        return [
            item["highlight"]
            for item in self.highlights
            if item["owner_confidence"] >= min_confidence
        ]

    def compute_identity_confidence(self):
        score = 0.0
        if self.jersey is not None:         score += 0.40
        if self.team_color:                 score += 0.20
        if self.visual_conf == "high":      score += 0.25
        elif self.visual_conf == "medium":  score += 0.10
        if len(self._observations) >= 3:    score += 0.10
        if len(self.pids) == 1:             score += 0.05
        self.identity_confidence = round(min(1.0, score), 3)
        return self.identity_confidence

    def compute_active_ranges(self, gap_threshold=120.0):
        """
        Calcule les plages d'activité continues.
        Détecte remplacements et deux joueurs #11 différents.
        gap_threshold : pause > N secondes = nouvelle plage.
        """
        if not self.events:
            self.active_time_ranges = []
            return []
        times = sorted(
            float(item["event"].get("time", 0) or 0)
            for item in self.events
            if item["event"].get("time")
        )
        if not times:
            self.active_time_ranges = []
            return []
        ranges      = []
        range_start = times[0]
        range_end   = times[0]
        for t in times[1:]:
            if t - range_end > gap_threshold:
                ranges.append((round(range_start, 1), round(range_end, 1)))
                range_start = t
            range_end = t
        ranges.append((round(range_start, 1), round(range_end, 1)))
        self.active_time_ranges = ranges
        return ranges

    def _quick_stats(self):
        """Stats rapides sans filtrage (pour debug_view)."""
        s = {"buts": 0, "tirs": 0, "xg_total": 0.0}
        for item in self.events:
            e = item["event"]
            if e.get("type") in ("goal", "score"):
                s["buts"]     += 1
                s["xg_total"] += float(e.get("xg", 0) or 0)
            elif e.get("type") == "shot":
                s["tirs"]     += 1
                s["xg_total"] += float(e.get("xg", 0) or 0)
        s["xg_total"] = round(s["xg_total"], 3)
        return s

    def debug_view(self):
        """Log structuré pour debugging — [ENTITY_DEBUG]."""
        def fmt(t):
            if t is None: return "?"
            return f"{int(t//60):02d}:{int(t%60):02d}"

        stats = self._quick_stats()
        visual_summary = " | ".join(
            f"{k}={v}" for k, v in self.visual.items()
            if v and k not in ("confidence", "_source", "_goal_time")
        ) if self.visual else "aucune"

        ranges_str = (
            ", ".join(f"[{fmt(s)}→{fmt(e)}]" for s, e in self.active_time_ranges)
            if self.active_time_ranges else "?"
        )

        lines = [
            f"ENTITY {self.entity_key}",
            f"  pids          : {self.pids}",
            f"  jersey        : #{self.jersey}" + (f" ({self.team_color})" if self.team_color else ""),
            f"  confidence    : {self.identity_confidence:.2f}",
            f"  first_seen    : {fmt(self.first_seen)}",
            f"  last_seen     : {fmt(self.last_seen)}",
            f"  active_ranges : {ranges_str}",
            f"  buts          : {stats['buts']}",
            f"  tirs          : {stats['tirs']}",
            f"  xg_total      : {stats['xg_total']:.3f}",
            f"  n_events      : {len(self.events)}",
            f"  highlights    : {len(self.highlights)}",
            f"  visual        : {visual_summary}",
        ]
        return "\n".join(lines)

    def to_dict(self):
        return {
            "entity_id":           self.entity_id,
            "entity_key":          self.entity_key,
            "sport":               self.sport,
            "jersey":              self.jersey,
            "team":                self.team,
            "team_color":          self.team_color,
            "team_color_hsv":      self.team_color_hsv,
            "pids":                list(self.pids),
            "visual":              self.visual,
            "visual_conf":         self.visual_conf,
            "identity_confidence": self.identity_confidence,
            "first_seen":          self.first_seen,
            "last_seen":           self.last_seen,
            "active_time_ranges":  self.active_time_ranges,
            "n_events":            len(self.events),
            "n_highlights":        len(self.highlights),
            "n_observations":      len(self._observations),
        }

    def __repr__(self):
        return (f"PlayerEntity(key={self.entity_key} | "
                f"conf={self.identity_confidence:.2f} | "
                f"pids={self.pids} | events={len(self.events)})")


# ─────────────────────────────────────────
# PLAYER ENTITY MANAGER
# ─────────────────────────────────────────
class PlayerEntityManager:

    # P3 — fenêtre temporelle pour l'inertie (secondes)
    TEMPORAL_INERTIA_WINDOW = 3.0

    def __init__(self, sport="football"):
        self.sport      = sport
        self.entities   = {}   # entity_id → PlayerEntity
        self._next_id   = 0
        self._pid_index = {}   # pid_str → entity_id
        # P2 — index par clé anti-collision
        self._key_index = {}   # entity_key → entity_id
        self._cfg       = get_player_identity_config(sport)
        self._debug     = True  # P1 — activer les logs [ENTITY MATCH]

    # ─────────────────────────────────────
    # P1 — LOGGING
    # ─────────────────────────────────────
    def _log_match(self, event, entity, reason, confidence):
        """P1 — Log détaillé de chaque attribution."""
        if not self._debug:
            return
        e_type = event.get("type", "?")
        t      = event.get("time", 0)
        mm     = f"{int(t//60):02d}:{int(t%60):02d}"
        pid    = event.get("player", "?")
        print(f"  [ENTITY MATCH] t={mm} type={e_type} pid={pid} "
              f"→ {entity.entity_key} conf={confidence:.2f} reason={reason}")

    # ─────────────────────────────────────
    # CRÉATION / FUSION
    # ─────────────────────────────────────
    def _new_entity(self):
        eid    = self._next_id
        entity = PlayerEntity(eid, sport=self.sport)
        self.entities[eid] = entity
        self._next_id += 1
        return entity

    def _find_by_pid(self, pid):
        eid = self._pid_index.get(str(pid))
        return self.entities.get(eid) if eid is not None else None

    def _find_by_key(self, entity_key):
        """P2 — Recherche par clé anti-collision."""
        eid = self._key_index.get(entity_key)
        return self.entities.get(eid) if eid is not None else None

    def _find_by_jersey_team(self, jersey, team):
        """Recherche par (jersey, team) — compatible avec tous les sports."""
        # D'abord par clé directe si team_color connu
        for entity in self.entities.values():
            if entity.jersey == int(jersey) and entity.team == team:
                return entity
        return None

    def _find_by_visual(self, visual, team=None, min_score=None):
        if not visual:
            return None, 0.0
        if min_score is None:
            min_score = self._cfg.get("min_match_score", 0.50)

        best_entity, best_score = None, 0.0
        for entity in self.entities.values():
            if not entity.visual:
                continue
            if team is not None and entity.team != team:
                continue
            sc = visual_match_score(visual, entity.visual, self.sport)
            if sc > best_score:
                best_score  = sc
                best_entity = entity

        return (best_entity, best_score) if best_score >= min_score else (None, 0.0)

    # P3 — TEMPORAL SMOOTHING
    def _find_by_temporal_inertia(self, pid, event_time):
        """
        P3 — Si un pid a été vu très récemment sur une entité,
        on maintient l'attribution pour éviter les oscillations.
        """
        if not pid or not event_time:
            return None
        pid_str = str(pid)
        for entity in self.entities.values():
            if pid_str not in entity.pids:
                continue
            if entity._last_event_time is None:
                continue
            dt = abs(float(event_time) - entity._last_event_time)
            if dt <= self.TEMPORAL_INERTIA_WINDOW:
                return entity
        return None

    def get_or_create_entity(self, pid=None, jersey=None, team=None,
                              team_color=None, team_color_bgr=None,
                              visual=None, event_time=None):
        """
        Retourne une entité existante ou en crée une nouvelle.

        Hiérarchie (P1 log à chaque étape) :
        1. jersey + team (le plus fiable)
        2. pid DeepSort  (stable localement)
        3. temporal inertia (P3 — anti-oscillation)
        4. visual match  (fallback bruité)
        5. création      (nouvelle entité)
        """
        entity = None
        reason = "new"

        # P2 — construire entity_key si possible
        _tc_name = team_color or (bgr_to_hsv_name(team_color_bgr)["name"]
                                   if team_color_bgr else None)
        _entity_key = f"{_tc_name}_{jersey}" if (_tc_name and jersey is not None) else None

        # 1. jersey + team
        if jersey is not None and team is not None:
            entity = self._find_by_jersey_team(jersey, team)
            if entity:
                reason = "jersey+team"

        # 1b. entity_key direct (P2)
        if entity is None and _entity_key:
            entity = self._find_by_key(_entity_key)
            if entity:
                reason = "entity_key"

        # 2. pid DeepSort
        if entity is None and pid is not None:
            entity = self._find_by_pid(pid)
            if entity:
                reason = "pid"

        # 3. temporal inertia (P3)
        if entity is None and pid is not None and event_time is not None:
            entity = self._find_by_temporal_inertia(pid, event_time)
            if entity:
                reason = "temporal_inertia"

        # 4. visual match
        if entity is None and visual:
            entity, vs = self._find_by_visual(visual, team=team)
            if entity:
                reason = f"visual(score={vs:.2f})"

        # 5. création
        if entity is None:
            entity = self._new_entity()
            reason = "new"

        # Enrichir l'entité
        if pid is not None:
            entity.add_pid(pid)
            self._pid_index[str(pid)] = entity.entity_id

        if jersey is not None and entity.jersey is None:
            entity.jersey = int(jersey)

        if team is not None and entity.team is None:
            entity.team = team

        # P4 — stocker couleur avec HSV
        if team_color_bgr and not entity.team_color_hsv:
            entity.set_team_color(bgr=team_color_bgr)
        elif _tc_name and not entity.team_color:
            entity.set_team_color(color_name=_tc_name)

        if visual:
            entity.add_visual_observation(visual)

        # Mettre à jour les index (P2)
        new_key = entity.entity_key
        if new_key not in self._key_index:
            self._key_index[new_key] = entity.entity_id

        return entity, reason

    # ─────────────────────────────────────
    # BUILD DEPUIS RÉSULTAT PIPELINE
    # ─────────────────────────────────────
    def build_from_pipeline_result(self, result):
        """
        Construit les entités depuis le résultat du pipeline.
        Inclut la propagation d'ownership (P5).
        """
        events     = result.get("events",     [])
        highlights = result.get("highlights", [])
        jersey_map = result.get("jersey_map", {})
        teams_data = result.get("teams",      {})

        # Construire index couleur équipe (P4 — avec BGR)
        _team_colors_bgr  = {}
        _team_colors_name = {}
        for tid, tdata in teams_data.items():
            bgr = tdata.get("color_bgr")
            if bgr:
                _team_colors_bgr[tid]  = bgr
                result_c = bgr_to_hsv_name(bgr)
                _team_colors_name[tid] = result_c["name"]

        # Phase 1 — events
        for e in events:
            pid        = str(e.get("player") or "")
            team       = e.get("team")
            jersey     = (jersey_map.get(pid) or e.get("jersey")
                          or e.get("player_jersey"))
            visual     = e.get("scorer_visual")
            t          = float(e.get("time", 0) or 0)
            team_color_bgr  = _team_colors_bgr.get(team) or _team_colors_bgr.get(str(team))
            team_color_name = _team_colors_name.get(team) or _team_colors_name.get(str(team))

            entity, reason = self.get_or_create_entity(
                pid            = pid if pid else None,
                jersey         = jersey,
                team           = team,
                team_color     = team_color_name,
                team_color_bgr = team_color_bgr,
                visual         = visual,
                event_time     = t,
            )

            owner_conf = self._compute_event_confidence(e, entity)
            entity.add_event(e, owner_confidence=owner_conf)

            # P1 — log pour les events importants
            if e.get("type") in ("goal", "score", "shot") or self._debug:
                self._log_match(e, entity, reason, owner_conf)

        # Phase 2 — highlights
        for h in highlights:
            pid        = str(h.get("player") or "")
            team       = h.get("team")
            jersey     = jersey_map.get(pid) or h.get("player_jersey")
            team_color_bgr  = _team_colors_bgr.get(team) or _team_colors_bgr.get(str(team))
            team_color_name = _team_colors_name.get(team) or _team_colors_name.get(str(team))

            entity, _ = self.get_or_create_entity(
                pid            = pid if pid else None,
                jersey         = jersey,
                team           = team,
                team_color     = team_color_name,
                team_color_bgr = team_color_bgr,
            )
            hl_conf = 0.95 if h.get("gemini_scored") else 0.70
            entity.add_highlight(h, owner_confidence=hl_conf)

        # P5 — Ownership propagation
        self._propagate_ownership(events, jersey_map)

        # Calculer confiance finale + plages d'activité
        for entity in self.entities.values():
            entity.compute_identity_confidence()
            entity.compute_active_ranges(gap_threshold=120.0)

        n_with_jersey = sum(1 for e in self.entities.values() if e.jersey)
        n_high_conf   = sum(1 for e in self.entities.values()
                            if e.identity_confidence >= 0.85)
        print(f"  [PLAYER_ENTITY] {len(self.entities)} entités | "
              f"{n_with_jersey} avec jersey | {n_high_conf} high-conf (>=0.85)")

        # Debug view pour les entités high-conf si debug activé
        if self._debug:
            for entity in self.get_all_entities(min_confidence=0.60, has_jersey=True):
                print(entity.debug_view())

        return self

    # P5 — OWNERSHIP PROPAGATION
    def _propagate_ownership(self, events, jersey_map, window=3.0):
        """
        P5 — Si un event de type goal/shot a un owner fiable,
        les events dans la fenêtre ±window secondes et même pid
        héritent de cet owner si leur confiance actuelle est faible.
        """
        # Index des events forts (goal/shot avec jersey connu)
        anchors = []
        for item_list in [e.events for e in self.entities.values()]:
            for item in item_list:
                e = item["event"]
                if (e.get("type") in ("goal", "score", "shot")
                        and item["owner_confidence"] >= 0.70
                        and e.get("player")):
                    jersey = jersey_map.get(str(e.get("player")))
                    if jersey:
                        anchors.append({
                            "time":       float(e.get("time", 0) or 0),
                            "pid":        str(e.get("player")),
                            "jersey":     jersey,
                            "team":       e.get("team"),
                            "confidence": item["owner_confidence"],
                        })

        if not anchors:
            return

        n_propagated = 0
        for entity in self.entities.values():
            for item in entity.events:
                e = item["event"]
                if item["owner_confidence"] >= 0.70:
                    continue  # déjà bien attribué
                t   = float(e.get("time", 0) or 0)
                pid = str(e.get("player") or "")

                for anchor in anchors:
                    if abs(anchor["time"] - t) > window:
                        continue
                    if anchor["pid"] != pid and anchor["team"] != e.get("team"):
                        continue
                    # Propager : boost de confiance
                    new_conf = min(0.80, item["owner_confidence"] + 0.25)
                    if new_conf > item["owner_confidence"]:
                        item["owner_confidence"] = new_conf
                        n_propagated += 1
                    break

        if n_propagated > 0:
            print(f"  [ENTITY P5] {n_propagated} events enrichis "
                  f"par ownership propagation (fenêtre ±{window}s)")

    def _compute_event_confidence(self, event, entity):
        """Confiance d'attribution d'un event à une entité."""
        score = 0.0
        if entity.jersey is not None:           score += 0.40
        if (entity.team is not None
                and event.get("team") == entity.team): score += 0.20
        if event.get("gemini_validated"):        score += 0.20
        tracker_conf = float(event.get("confidence", 0) or 0)
        score += tracker_conf * 0.10
        ev_visual = event.get("scorer_visual")
        if ev_visual and entity.visual:
            vs = visual_match_score(ev_visual, entity.visual, self.sport)
            score += vs * 0.10
        return round(min(1.0, score), 3)

    # ─────────────────────────────────────
    # ACCESSEURS
    # ─────────────────────────────────────
    def get_entity(self, jersey=None, team=None, team_color=None, pid=None):
        """Retourne l'entité correspondante."""
        if jersey is not None:
            # P2 — chercher par entity_key d'abord
            if team_color:
                key = f"{team_color}_{jersey}"
                entity = self._find_by_key(key)
                if entity:
                    return entity
            if team is not None:
                entity = self._find_by_jersey_team(jersey, team)
                if entity:
                    return entity
            # Chercher par jersey seul
            for entity in self.entities.values():
                if entity.jersey == int(jersey):
                    return entity

        if pid is not None:
            return self._find_by_pid(pid)

        return None

    def get_all_entities(self, min_confidence=0.0, has_jersey=False):
        """Retourne toutes les entités triées par confiance décroissante."""
        result = [
            e for e in self.entities.values()
            if e.identity_confidence >= min_confidence
            and (not has_jersey or e.jersey is not None)
        ]
        return sorted(result, key=lambda e: e.identity_confidence, reverse=True)

    def get_player_highlights(self, jersey, team_color=None, min_confidence=None):
        """Raccourci : highlights d'un joueur."""
        entity = self.get_entity(jersey=jersey, team_color=team_color)
        if entity is None:
            return []
        return entity.get_highlights(min_confidence=min_confidence)

    def get_player_stats(self, jersey, team_color=None, min_confidence=None):
        """Stats agrégées d'un joueur."""
        entity = self.get_entity(jersey=jersey, team_color=team_color)
        if entity is None:
            return {}

        events = entity.get_events(min_confidence=min_confidence)
        stats = {
            "jersey":              entity.jersey,
            "team":                entity.team,
            "team_color":          entity.team_color,
            "team_color_hsv":      entity.team_color_hsv,
            "entity_key":          entity.entity_key,
            "buts":                0,
            "tirs":                0,
            "xg_total":            0.0,
            "xa_total":            0.0,
            "key_passes":          0,
            "dribbles":            0,
            "interceptions":       0,
            "touches":             0,
            "identity_confidence": entity.identity_confidence,
            "n_events":            len(events),
            "pids":                list(entity.pids),
        }
        for e in events:
            etype = e.get("type", "")
            if etype in ("goal", "score"):
                stats["buts"] += 1
                stats["xg_total"] += float(e.get("xg", 0) or 0)
            elif etype == "shot":
                stats["tirs"]     += 1
                stats["xg_total"] += float(e.get("xg", 0) or 0)
            elif etype == "key_pass":
                stats["key_passes"] += 1
                stats["xa_total"]   += float(e.get("xa", 0) or 0)
            elif etype == "dribble":
                stats["dribbles"] += 1
            elif etype == "interception":
                stats["interceptions"] += 1
            else:
                stats["touches"] += 1

        stats["xg_total"] = round(stats["xg_total"], 3)
        stats["xa_total"] = round(stats["xa_total"], 3)
        return stats

    def summary(self):
        """Résumé de toutes les entités."""
        entities = self.get_all_entities()
        return {
            "sport":         self.sport,
            "n_entities":    len(entities),
            "n_with_jersey": sum(1 for e in entities if e.jersey),
            "n_high_conf":   sum(1 for e in entities
                                 if e.identity_confidence >= 0.85),
            "entities":      [e.to_dict() for e in entities],
        }
