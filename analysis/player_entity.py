# analysis/player_entity.py
# -*- coding: utf-8 -*-
#
# Phase 2 — Identité joueur persistante sur le match
#
# Une PlayerEntity fusionne :
#   - les pids DeepSort (instables, peuvent changer)
#   - le numéro de maillot (OCR + Gemini)
#   - la couleur équipe
#   - la signature visuelle Gemini (boots, socks, hair...)
#   - les events associés avec owner_confidence
#   - les highlights filtrés
#
# Architecture multi-sport :
#   Le sport est un paramètre de premier ordre.
#   Les poids visuels et champs pertinents viennent de
#   sports/config.py → PLAYER_IDENTITY_CONFIG[sport]
#
# Utilisation :
#   from analysis.player_entity import PlayerEntityManager
#   mgr = PlayerEntityManager(sport="football")
#   mgr.build_from_pipeline_result(result)
#   entity = mgr.get_entity(jersey=11, team_color="bleu")
#   highlights = mgr.get_highlights(jersey=11, min_confidence=0.85)

import math
from collections import defaultdict

try:
    from sports.config import (
        get_player_identity_config,
        get_visual_match_weights,
        get_visual_fields,
        get_owner_confidence_min,
    )
except ImportError:
    # Fallback si sports/config non disponible
    def get_player_identity_config(sport):
        return {}
    def get_visual_match_weights(sport):
        return {"boots": 0.30, "socks": 0.25, "hair": 0.20,
                "skin": 0.10, "body": 0.08, "role": 0.07}
    def get_visual_fields(sport):
        return ["boots", "socks", "hair", "body", "role", "skin"]
    def get_owner_confidence_min(sport):
        return 0.85


# ─────────────────────────────────────────
# VISUAL MATCH SCORE
# Score de similarité entre deux descriptions visuelles
# Utilise les poids du sport concerné
# ─────────────────────────────────────────
def visual_match_score(v1, v2, sport="football"):
    """
    Calcule un score de similarité [0..1] entre deux descriptions visuelles.

    v1, v2 : dict avec les champs visuels (boots, socks, hair, etc.)
    sport   : détermine les poids utilisés

    Retourne un float entre 0.0 (aucune correspondance) et 1.0 (identique).
    """
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

        # Correspondance exacte
        if val1 == val2:
            field_score = 1.0
        # Correspondance partielle — un des deux contient l'autre
        elif val1 in val2 or val2 in val1:
            field_score = 0.7
        # Mots communs
        else:
            words1 = set(val1.split())
            words2 = set(val2.split())
            common = words1 & words2
            if common:
                field_score = len(common) / max(len(words1), len(words2))
            else:
                field_score = 0.0

        total_weight += weight
        total_score  += weight * field_score

    if total_weight == 0:
        return 0.0

    return round(total_score / total_weight, 3)


# ─────────────────────────────────────────
# PLAYER ENTITY
# Représente un joueur identifiable sur le match
# ─────────────────────────────────────────
class PlayerEntity:
    """
    Entité joueur stable sur un match.

    Attributs :
        entity_id      : identifiant interne (auto-incrémenté)
        sport          : sport concerné
        jersey         : numéro de maillot (int ou None)
        team           : id équipe (0 ou 1)
        team_color     : nom couleur équipe ("bleu", "rouge"...)
        pids           : set de pids DeepSort fusionnés
        visual         : dict signature visuelle Gemini
        visual_conf    : confiance sur la signature visuelle
        events         : liste d'events avec owner_confidence
        highlights     : liste de highlights filtrés
        identity_confidence : confiance globale sur l'identité [0..1]
    """

    def __init__(self, entity_id, sport="football"):
        self.entity_id           = entity_id
        self.sport               = sport
        self.jersey              = None
        self.team                = None
        self.team_color          = None
        self.is_goalkeeper       = False   # détecté par couleur outlier
        self.pids                = set()
        self.visual              = {}
        self.visual_conf         = "low"
        self.events              = []
        self.highlights          = []
        self.identity_confidence = 0.0
        self._observations       = []  # historique des observations visuelles

    def add_pid(self, pid):
        """Fusionne un pid DeepSort dans cette entité."""
        if pid is not None:
            self.pids.add(str(pid))

    def add_visual_observation(self, visual_dict, source="gemini"):
        """
        Accumule une observation visuelle.
        La signature finale est le consensus des observations.
        """
        if not visual_dict:
            return
        self._observations.append({"visual": visual_dict, "source": source})
        # Mettre à jour la signature avec la dernière observation high-conf
        conf = visual_dict.get("confidence", "low")
        if conf == "high" or (conf == "medium" and self.visual_conf != "high"):
            self.visual      = visual_dict
            self.visual_conf = conf

    def add_event(self, event, owner_confidence=1.0):
        """
        Ajoute un event à cette entité avec sa confiance d'attribution.
        En mode player, seuls les events >= owner_confidence_min sont ajoutés.
        """
        self.events.append({
            "event":            event,
            "owner_confidence": round(float(owner_confidence), 3),
        })

    def add_highlight(self, highlight, owner_confidence=1.0):
        """Ajoute un highlight à cette entité."""
        self.highlights.append({
            "highlight":        highlight,
            "owner_confidence": round(float(owner_confidence), 3),
        })

    def get_events(self, min_confidence=None, event_types=None):
        """
        Retourne les events filtrés par confiance et/ou type.

        min_confidence : seuil de confiance (défaut = owner_confidence_min du sport)
        event_types    : liste de types à inclure (None = tous)
        """
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
        """Retourne les highlights filtrés par confiance."""
        if min_confidence is None:
            min_confidence = get_owner_confidence_min(self.sport)
        return [
            item["highlight"]
            for item in self.highlights
            if item["owner_confidence"] >= min_confidence
        ]

    def compute_identity_confidence(self):
        """
        Calcule la confiance globale sur l'identité du joueur.
        Basée sur : jersey OCR, couleur équipe, observations visuelles, pids stables.
        """
        score = 0.0

        # Jersey confirmé = +0.40
        if self.jersey is not None:
            score += 0.40

        # Couleur équipe confirmée = +0.20
        if self.team_color:
            score += 0.20

        # Signature visuelle high-conf = +0.25, medium = +0.10
        if self.visual_conf == "high":
            score += 0.25
        elif self.visual_conf == "medium":
            score += 0.10

        # Plusieurs observations cohérentes = +0.10
        if len(self._observations) >= 3:
            score += 0.10

        # Pid stable (un seul pid = tracking cohérent) = +0.05
        if len(self.pids) == 1:
            score += 0.05

        self.identity_confidence = round(min(1.0, score), 3)
        return self.identity_confidence

    def to_dict(self):
        """Sérialise l'entité en dict JSON-compatible."""
        return {
            "entity_id":           self.entity_id,
            "sport":               self.sport,
            "jersey":              self.jersey,
            "team":                self.team,
            "team_color":          self.team_color,
            "pids":                list(self.pids),
            "visual":              self.visual,
            "visual_conf":         self.visual_conf,
            "identity_confidence": self.identity_confidence,
            "n_events":            len(self.events),
            "n_highlights":        len(self.highlights),
            "n_observations":      len(self._observations),
        }

    def __repr__(self):
        return (f"PlayerEntity(#{self.jersey} | {self.team_color} | "
                f"conf={self.identity_confidence:.2f} | "
                f"pids={self.pids} | events={len(self.events)})")


# ─────────────────────────────────────────
# PLAYER ENTITY MANAGER
# Construit et gère toutes les entités d'un match
# ─────────────────────────────────────────
class PlayerEntityManager:
    """
    Gestionnaire d'entités joueurs pour un match.

    Usage typique :
        mgr = PlayerEntityManager(sport="football")
        mgr.build_from_pipeline_result(result)
        entity = mgr.get_entity(jersey=11, team_color="bleu")
    """

    def __init__(self, sport="football"):
        self.sport       = sport
        self.entities    = {}   # entity_id → PlayerEntity
        self._next_id    = 0
        self._pid_index  = {}   # pid_str → entity_id
        self._jersey_index = {} # (jersey, team) → entity_id
        self._cfg        = get_player_identity_config(sport)

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
        """Retourne l'entité associée à un pid DeepSort."""
        eid = self._pid_index.get(str(pid))
        return self.entities.get(eid) if eid is not None else None

    def _find_by_jersey_team(self, jersey, team):
        """Retourne l'entité associée à (jersey, team)."""
        key = (int(jersey) if jersey is not None else None, team)
        eid = self._jersey_index.get(key)
        return self.entities.get(eid) if eid is not None else None

    def _find_by_visual(self, visual, team=None, min_score=None):
        """
        Retourne l'entité avec la meilleure correspondance visuelle.
        """
        if not visual:
            return None, 0.0

        if min_score is None:
            min_score = self._cfg.get("min_match_score", 0.50)

        best_entity = None
        best_score  = 0.0

        for entity in self.entities.values():
            if not entity.visual:
                continue
            if team is not None and entity.team != team:
                continue
            sc = visual_match_score(visual, entity.visual, self.sport)
            if sc > best_score:
                best_score  = sc
                best_entity = entity

        if best_score >= min_score:
            return best_entity, best_score
        return None, 0.0

    def get_or_create_entity(self, pid=None, jersey=None, team=None,
                              team_color=None, visual=None):
        """
        Retourne une entité existante ou en crée une nouvelle.

        Hiérarchie de matching :
        1. jersey + team    (le plus fiable)
        2. pid DeepSort     (stable dans une session)
        3. visual match     (fallback si jersey inconnu)
        4. création         (nouvelle entité)
        """
        entity = None

        # 1. jersey + team
        if jersey is not None and team is not None:
            entity = self._find_by_jersey_team(jersey, team)

        # 2. pid DeepSort
        if entity is None and pid is not None:
            entity = self._find_by_pid(pid)

        # 3. visual match
        if entity is None and visual:
            entity, sc = self._find_by_visual(visual, team=team)

        # 4. création
        if entity is None:
            entity = self._new_entity()

        # Enrichir l'entité avec les nouvelles informations
        if pid is not None:
            entity.add_pid(pid)
            self._pid_index[str(pid)] = entity.entity_id

        if jersey is not None and entity.jersey is None:
            entity.jersey = int(jersey)

        if team is not None and entity.team is None:
            entity.team = team

        if team_color and not entity.team_color:
            entity.team_color = team_color

        if visual:
            entity.add_visual_observation(visual)

        # Mettre à jour l'index jersey
        if entity.jersey is not None and entity.team is not None:
            key = (entity.jersey, entity.team)
            self._jersey_index[key] = entity.entity_id

        return entity

    # ─────────────────────────────────────
    # BUILD DEPUIS RÉSULTAT PIPELINE
    # ─────────────────────────────────────
    def build_from_pipeline_result(self, result):
        """
        Construit les entités depuis le résultat du pipeline.

        result : dict retourné par run_pipeline()
            - events        : liste d'events
            - highlights    : liste de highlights
            - jersey_map    : {pid → jersey_number}
            - teams         : {team_id → {color_bgr, ...}}
        """
        events     = result.get("events",     [])
        highlights = result.get("highlights", [])
        jersey_map = result.get("jersey_map", {})
        teams_data = result.get("teams",      {})

        # Construire un index couleur équipe
        _team_colors = {}
        try:
            from export.pdf import bgr_to_name
            for tid, tdata in teams_data.items():
                bgr = tdata.get("color_bgr")
                if bgr:
                    name = bgr_to_name(bgr)
                    if name:
                        _team_colors[tid] = name.lower()
        except Exception:
            pass

        # Phase 1 — parcourir les events
        for e in events:
            pid    = str(e.get("player") or "")
            team   = e.get("team")
            jersey = jersey_map.get(pid) or e.get("jersey") or e.get("player_jersey")
            visual = e.get("scorer_visual")
            team_color = _team_colors.get(team) or _team_colors.get(str(team))

            entity = self.get_or_create_entity(
                pid        = pid if pid else None,
                jersey     = jersey,
                team       = team,
                team_color = team_color,
                visual     = visual,
            )

            # Attribution de l'event avec confiance
            # Confiance = combinaison jersey_conf + visual_conf + tracker_conf
            owner_conf = self._compute_event_confidence(e, entity)
            entity.add_event(e, owner_confidence=owner_conf)

        # Phase 1b — marquer les gardiens détectés par player_reid (team="gk")
        for e in events:
            pid  = str(e.get("player") or "")
            team = e.get("team")
            if team == "gk" and pid:
                entity = self.entities.get(pid)
                if entity:
                    entity.is_goalkeeper = True

        # Phase 1c — marquer les gardiens par événements défensifs dominants
        _gk_types = {"goalkeeper_save", "shot_saved", "clearance"}
        for entity in self.entities.values():
            evts = entity.get_events()
            if not evts: continue
            n_gk = sum(1 for ev in evts if ev.get("type") in _gk_types)
            if n_gk >= 2 and n_gk / len(evts) > 0.4:
                entity.is_goalkeeper = True

        # Phase 2 — parcourir les highlights
        for h in highlights:
            pid    = str(h.get("player") or "")
            team   = h.get("team")
            jersey = jersey_map.get(pid) or h.get("player_jersey")
            team_color = _team_colors.get(team) or _team_colors.get(str(team))

            entity = self.get_or_create_entity(
                pid        = pid if pid else None,
                jersey     = jersey,
                team       = team,
                team_color = team_color,
            )

            # Confiance highlight = Gemini validé > fallback
            hl_conf = 0.95 if h.get("gemini_scored") else 0.70
            entity.add_highlight(h, owner_confidence=hl_conf)

        # Calculer la confiance d'identité pour toutes les entités
        for entity in self.entities.values():
            entity.compute_identity_confidence()

        n = len(self.entities)
        print(f"  [PLAYER_ENTITY] {n} entités construites "
              f"({sum(1 for e in self.entities.values() if e.jersey)} avec jersey)")
        return self

    def _compute_event_confidence(self, event, entity):
        """
        Calcule la confiance d'attribution d'un event à une entité.

        Sources de confiance :
        - jersey OCR/Gemini confirmé    → +0.40
        - couleur équipe correcte       → +0.20
        - Gemini validé (but/tir)       → +0.20
        - tracker confiance             → +0.10
        - visual match score            → +0.10
        """
        score = 0.0

        # Jersey confirmé
        if entity.jersey is not None:
            score += 0.40

        # Couleur équipe
        if entity.team is not None and event.get("team") == entity.team:
            score += 0.20

        # Gemini validé
        if event.get("gemini_validated"):
            score += 0.20

        # Confiance tracker
        tracker_conf = float(event.get("confidence", 0) or 0)
        score += tracker_conf * 0.10

        # Visual match si disponible
        ev_visual = event.get("scorer_visual")
        if ev_visual and entity.visual:
            vs = visual_match_score(ev_visual, entity.visual, self.sport)
            score += vs * 0.10

        return round(min(1.0, score), 3)

    # ─────────────────────────────────────
    # ACCESSEURS
    # ─────────────────────────────────────
    def get_entity(self, jersey=None, team=None, team_color=None, pid=None):
        """
        Retourne l'entité correspondante.

        Exemples :
            mgr.get_entity(jersey=11, team_color="bleu")
            mgr.get_entity(jersey=7)
            mgr.get_entity(pid="P11")
        """
        if jersey is not None:
            # Chercher par jersey + team si team connu
            if team is not None:
                entity = self._find_by_jersey_team(jersey, team)
                if entity:
                    return entity
            # Chercher par jersey seul (toutes équipes)
            for entity in self.entities.values():
                if entity.jersey == int(jersey):
                    if team_color is None or entity.team_color == team_color:
                        return entity

        if pid is not None:
            return self._find_by_pid(pid)

        return None

    def get_all_entities(self, min_confidence=0.0, has_jersey=False):
        """
        Retourne toutes les entités triées par confiance décroissante.

        min_confidence : filtrer par confiance minimale
        has_jersey     : ne retourner que les entités avec jersey identifié
        """
        result = [
            e for e in self.entities.values()
            if e.identity_confidence >= min_confidence
            and (not has_jersey or e.jersey is not None)
        ]
        return sorted(result, key=lambda e: e.identity_confidence, reverse=True)

    def get_goalkeeper_highlights(self, team_id, min_confidence=None):
        """
        Retourne les highlights du gardien d'une équipe.
        Le gardien est identifié par is_goalkeeper=True + team correspondante.

        team_id : 0 ou 1
        """
        gk_entities = [
            e for e in self.entities.values()
            if e.is_goalkeeper and e.team == team_id
        ]

        # Si aucun gardien détecté par couleur outlier, fallback :
        # prendre les entités avec événements défensifs (arrêts, dégagements)
        if not gk_entities:
            defensive_types = {"goalkeeper_save", "clearance", "shot_saved"}
            gk_entities = [
                e for e in self.entities.values()
                if e.team == team_id
                and any(ev.get("type") in defensive_types
                        for ev in e.get_events())
            ]

        if not gk_entities:
            return []

        # Prendre le gardien avec le plus d'événements
        gk = max(gk_entities, key=lambda e: len(e.get_events()))
        return gk.get_highlights(min_confidence=min_confidence)

    def get_goalkeeper_stats(self, team_id, min_confidence=None):
        """Stats du gardien d'une équipe."""
        gk_entities = [
            e for e in self.entities.values()
            if e.is_goalkeeper and e.team == team_id
        ]
        if not gk_entities:
            return {}
        gk = max(gk_entities, key=lambda e: len(e.get_events()))
        return self.get_player_stats(jersey=gk.jersey, team_color=gk.team_color,
                                      min_confidence=min_confidence)

    def get_player_highlights(self, jersey, team_color=None, min_confidence=None):
        """
        Raccourci : retourne tous les highlights d'un joueur.

        Exemple :
            clips = mgr.get_player_highlights(jersey=11, team_color="bleu")
        """
        entity = self.get_entity(jersey=jersey, team_color=team_color)
        if entity is None:
            return []
        return entity.get_highlights(min_confidence=min_confidence)

    def get_player_stats(self, jersey, team_color=None, min_confidence=None):
        """
        Retourne les stats agrégées d'un joueur (buts, tirs, xG, passes...).

        Exemple :
            stats = mgr.get_player_stats(jersey=11)
            # → {"buts": 1, "tirs": 3, "xg_total": 0.72, "key_passes": 2, ...}
        """
        entity = self.get_entity(jersey=jersey, team_color=team_color)
        if entity is None:
            return {}

        events = entity.get_events(min_confidence=min_confidence)

        stats = {
            "jersey":       entity.jersey,
            "team":         entity.team,
            "team_color":   entity.team_color,
            "buts":         0,
            "tirs":         0,
            "xg_total":     0.0,
            "xa_total":     0.0,
            "key_passes":   0,
            "dribbles":     0,
            "interceptions": 0,
            "touches":      0,
            "identity_confidence": entity.identity_confidence,
            "n_events":     len(events),
        }

        for e in events:
            etype = e.get("type", "")
            if etype in ("goal", "score"):
                stats["buts"] += 1
            elif etype == "shot":
                stats["tirs"]    += 1
                stats["xg_total"] += float(e.get("xg", 0) or 0)
            elif etype == "key_pass":
                stats["key_passes"] += 1
                stats["xa_total"]   += float(e.get("xa", 0) or 0)
            elif etype == "dribble":
                stats["dribbles"] += 1
            elif etype == "interception":
                stats["interceptions"] += 1
            elif etype in ("pass", "possession"):
                stats["touches"] += 1

        stats["xg_total"] = round(stats["xg_total"], 3)
        stats["xa_total"] = round(stats["xa_total"], 3)
        return stats

    def summary(self):
        """Retourne un résumé de toutes les entités."""
        entities = self.get_all_entities()
        return {
            "sport":         self.sport,
            "n_entities":    len(entities),
            "n_with_jersey": sum(1 for e in entities if e.jersey),
            "n_high_conf":   sum(1 for e in entities
                                 if e.identity_confidence >= 0.85),
            "entities":      [e.to_dict() for e in entities],
        }
