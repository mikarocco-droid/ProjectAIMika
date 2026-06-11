"""
PlayerIdentityMemory — Sprint A (v2)
=====================================
Deux niveaux d'identité :

  Niveau 1 — MatchMemory (temporaire, par run)
    track_id → jersey_number
    Survit le temps d'un run. Évite de rappeler Gemini sur les mêmes tracks.

  Niveau 2 — PlayerRegistry (persistant, cross-match)
    (jersey_number, team_side) → PlayerRecord
    Survit entre matchs. Accumule les statistiques saison.
    La clé primaire est le NUMÉRO, pas le track_id.

Philosophie :
  - track_id = référence locale temporaire (meurt à chaque run)
  - jersey_number = identité réelle du joueur (persiste à vie)
  - Une lecture fiable une fois = vérité pour toute la saison

Usage dans pipeline.py :
  from analysis.player_identity_memory import PlayerIdentityMemory

  pim = PlayerIdentityMemory(match_date="2026-06-10", team_id="andrimont_p4")

  # Début de run — enrichir jersey_map avec la mémoire
  jersey_map = pim.enrich_jersey_map(jersey_map_from_pipeline)

  # Enregistrer une détection
  pim.register_detection(track_id="P27", jersey=9, team="home",
                         source="gemini_visual", confidence=0.94)

  # Fin de run — consolidation et sauvegarde
  pim.finalize(jersey_map, events)
  pim.save()

  # Cross-match : consulter l'historique d'un joueur
  info = pim.get_player_info(jersey=9, team="home")
  # → {"jersey": 9, "name": None, "seen_matches": 3, "confidence": 0.91}
"""

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────

MEMORY_DIR = Path(os.environ.get("SCOUTIA_MEMORY_DIR", "/tmp/scoutia_memory"))
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# Seuil minimum pour considérer un joueur comme "connu"
MIN_CONFIDENCE = 0.60

# Lissage exponentiel pour la confiance cross-match
CONF_EMA_ALPHA = 0.30   # conf_new = 0.30 * detection + 0.70 * historique

# Confiance initiale par source de détection
SOURCE_CONFIDENCE = {
    "gemini_visual":  0.90,   # Gemini sur frame de but — le plus fiable
    "gemini_batch":   0.72,   # Gemini lecture batch numéros
    "ocr_crop":       0.65,   # OCR local sur crop joueur
    "gemini_general": 0.55,   # Gemini lecture générale
    "inferred":       0.35,   # Inféré par contexte
}

# Au-delà de N détections concordantes dans le même match, confiance max
DETECTIONS_FOR_SATURATION = 4


# ── PlayerRecord : identité persistante ────────────────────────────────────────

class PlayerRecord:
    """
    Identité persistante d'un joueur, indexée par (jersey, team).
    Survit entre matchs, entre saisons.
    """

    def __init__(self, jersey: int, team: str):
        self.jersey       = jersey
        self.team         = team         # "home" | "away" | "0" | "1"
        self.name: Optional[str] = None
        self.confidence   = 0.0
        self.seen_matches = 0
        self.last_seen    = ""           # ISO date YYYY-MM-DD
        self.sources      = []           # historique des sources de détection
        self.locked       = False        # True = ne jamais écraser

    def update(self, confidence_new: float, source: str, match_date: str):
        """
        Mise à jour EMA de la confiance.
        conf = 0.30 * nouveau + 0.70 * historique
        → une lecture ratée ne détruit pas l'identité
        → répétition progressive renforce sans saturer trop vite
        """
        if self.confidence == 0.0:
            self.confidence = confidence_new
        else:
            self.confidence = round(
                CONF_EMA_ALPHA * confidence_new + (1 - CONF_EMA_ALPHA) * self.confidence,
                3
            )
        # seen_matches est incrémenté UNIQUEMENT dans new_match() (appelé par finalize)
        # pour éviter le double comptage (register_detection + finalize)
        self.last_seen    = match_date or self.last_seen
        if source not in self.sources:
            self.sources.append(source)
        self.confidence = min(1.0, self.confidence)

    def new_match(self, match_date: str):
        """Incrémenter le compteur de matchs."""
        self.seen_matches += 1
        self.last_seen = match_date or self.last_seen

    def to_dict(self) -> dict:
        return {
            "jersey":       self.jersey,
            "team":         self.team,
            "name":         self.name,
            "confidence":   self.confidence,
            "seen_matches": self.seen_matches,
            "last_seen":    self.last_seen,
            "sources":      self.sources,
            "locked":       self.locked,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerRecord":
        r = cls(d["jersey"], d["team"])
        r.name         = d.get("name")
        r.confidence   = d.get("confidence", 0.0)
        r.seen_matches = d.get("seen_matches", 0)
        r.last_seen    = d.get("last_seen", "")
        r.sources      = d.get("sources", [])
        r.locked       = d.get("locked", False)
        return r


# ── Classe principale ───────────────────────────────────────────────────────────

class PlayerIdentityMemory:
    """
    Mémoire d'identité joueurs à deux niveaux :
      - match_registry  : track_id → jersey (temporaire, ce run)
      - player_registry : (jersey, team) → PlayerRecord (persistant, cross-match)
    """

    def __init__(self, match_date: str = "", team_id: str = ""):
        self.match_date = match_date or time.strftime("%Y-%m-%d")
        self.team_id    = team_id

        # Niveau 1 : track_id → jersey (temporaire)
        self._match_registry: dict[str, int] = {}

        # Niveau 2 : "jersey_team" → PlayerRecord (persistant)
        self._player_registry: dict[str, PlayerRecord] = {}

        # Détections dans ce run (pour update_ema en fin de run)
        self._run_detections: list[dict] = []

        self._dirty = False
        self._path  = MEMORY_DIR / f"registry_{self._registry_key()}.json"
        self._load()

    def _registry_key(self) -> str:
        """Clé du fichier de registre = équipe ou global."""
        if self.team_id:
            return hashlib.md5(self.team_id.encode()).hexdigest()[:10]
        return "global"

    def _player_key(self, jersey: int, team: str) -> str:
        return f"{jersey}_{team}"

    # ── Persistance ─────────────────────────────────────────────────────────────

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.get("players", {}).items():
                    self._player_registry[k] = PlayerRecord.from_dict(v)
                total = len(self._player_registry)
                known = sum(1 for r in self._player_registry.values()
                            if r.confidence >= MIN_CONFIDENCE)
                if total:
                    print(f"  [PIM] Registre chargé : {known}/{total} joueurs "
                          f"(conf≥{MIN_CONFIDENCE})")
            except Exception as e:
                print(f"  [PIM] Erreur chargement : {e}")

    def save(self):
        if not self._dirty:
            return
        try:
            data = {
                "team_id":    self.team_id,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "players":    {k: r.to_dict()
                               for k, r in self._player_registry.items()},
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  [PIM] Registre sauvegardé : {len(self._player_registry)} joueurs")
        except Exception as e:
            print(f"  [PIM] Erreur sauvegarde : {e}")

    # ── Enregistrement ──────────────────────────────────────────────────────────

    def register_detection(self, track_id: str, jersey: int, team: str = "home",
                           source: str = "gemini_general", confidence: Optional[float] = None):
        """
        Enregistre une détection jersey pour ce run.
        Met à jour le registre niveau 1 (track→jersey) ET niveau 2 (persistant).
        """
        tid  = str(track_id)
        conf = confidence or SOURCE_CONFIDENCE.get(source, 0.50)

        # Niveau 1 : track → jersey (temporaire)
        existing = self._match_registry.get(tid)
        if existing is not None and existing != jersey:
            # Discordance dans le même run
            existing_conf = max(
                (d["confidence"] for d in self._run_detections
                 if d["track_id"] == tid and d["jersey"] == existing),
                default=0.0
            )
            if conf > existing_conf:
                self._match_registry[tid] = jersey
                print(f"  [PIM] Mise à jour {tid}: #{existing}→#{jersey} "
                      f"(nouvelle conf={conf:.2f} > {existing_conf:.2f})")
        else:
            self._match_registry[tid] = jersey

        # Niveau 2 : (jersey, team) → PlayerRecord
        key = self._player_key(jersey, team)
        if key not in self._player_registry:
            self._player_registry[key] = PlayerRecord(jersey, team)

        r = self._player_registry[key]
        if not r.locked:
            r.update(conf, source, self.match_date)
            self._dirty = True

        # Historique du run
        self._run_detections.append({
            "track_id": tid, "jersey": jersey, "team": team,
            "source": source, "confidence": conf,
        })

    def register_goal_scorer(self, track_id: str, jersey: int,
                             team: str = "home", team_color: Optional[str] = None):
        """Source la plus fiable : Gemini a vu le buteur sur frame de but."""
        self.register_detection(
            track_id=track_id, jersey=jersey, team=team,
            source="gemini_visual",
            confidence=SOURCE_CONFIDENCE["gemini_visual"],
        )
        print(f"  [PIM] 🎯 Buteur : #{jersey} ({team_color or team}) "
              f"conf={SOURCE_CONFIDENCE['gemini_visual']:.2f}")

    # ── Lecture ──────────────────────────────────────────────────────────────────

    def get_jersey_map(self) -> dict:
        """
        jersey_map compatible pipeline : { "P27": 9, ... }
        Combine : mémoire niveau 1 (ce run) + niveau 2 (cross-match fiable).
        """
        result = dict(self._match_registry)  # niveau 1 en priorité

        # Compléter avec le niveau 2 si track→jersey pas encore dans ce run
        # (cas où on a l'historique mais pas encore vu le joueur dans ce run)
        # → non applicable ici car track_id change entre runs
        # Le niveau 2 sert uniquement pour get_player_info()

        return result

    def get_player_info(self, jersey: int, team: str = "home") -> Optional[dict]:
        """
        Informations cross-match sur un joueur.
        Retourne None si inconnu.
        """
        key = self._player_key(jersey, team)
        r = self._player_registry.get(key)
        if r is None:
            return None
        return {
            "jersey":       r.jersey,
            "team":         r.team,
            "name":         r.name,
            "confidence":   r.confidence,
            "seen_matches": r.seen_matches,
            "last_seen":    r.last_seen,
            "sources":      r.sources,
        }

    def known_jerseys(self, team: str = "home") -> dict:
        """Tous les numéros connus pour une équipe. { jersey: PlayerRecord }"""
        return {
            r.jersey: r
            for k, r in self._player_registry.items()
            if r.team == team and r.confidence >= MIN_CONFIDENCE
        }

    def tracks_to_skip(self, tracks: list) -> tuple[list, dict]:
        """
        Filtre une liste de tracks pour Gemini.
        Retourne (à_envoyer_à_Gemini, déjà_connus_ce_run).
        Note : cross-match, on ne peut pas skip un track inconnu.
        Seule la mémoire du run courant (niveau 1) permet de skipper.
        """
        to_gemini = []
        known_this_run = {}

        for t in tracks:
            tid = str(t.get("id", ""))
            if tid in self._match_registry:
                known_this_run[tid] = self._match_registry[tid]
            else:
                to_gemini.append(t)

        if known_this_run:
            print(f"  [PIM] ⚡ {len(known_this_run)} tracks déjà identifiés ce run → skip Gemini")
            for tid, num in sorted(known_this_run.items()):
                print(f"         {tid} → #{num}")

        return to_gemini, known_this_run

    # ── Finalisation du run ─────────────────────────────────────────────────────

    def finalize(self, jersey_map: dict, events: Optional[list] = None):
        """
        Appel en fin de run.
        1. Intègre le jersey_map final du pipeline
        2. Déduit les équipes depuis les events
        3. Met à jour le niveau 2 (cross-match)
        4. Marque le match comme "vu" pour chaque joueur
        """
        # Index équipes depuis events
        team_index: dict[str, str] = {}
        if events:
            for e in events:
                pid = str(e.get("player", ""))
                if pid and "team" in e:
                    team_index[pid] = str(e["team"])

        # Intégrer jersey_map dans le niveau 1 et 2
        new_in_map = 0
        for tid, jersey in jersey_map.items():
            if str(tid) not in self._match_registry:
                team = team_index.get(str(tid), "home")
                self.register_detection(
                    track_id=str(tid), jersey=jersey, team=team,
                    source="gemini_batch",
                )
                new_in_map += 1

        # Incrémenter seen_matches pour tous les joueurs vus ce run
        jerseys_seen_this_run = set(self._match_registry.values())
        for key, r in self._player_registry.items():
            if r.jersey in jerseys_seen_this_run:
                r.new_match(self.match_date)

        self._dirty = True
        print(f"  [PIM] Finalisé : {len(self._match_registry)} tracks ce run, "
              f"{new_in_map} nouveaux depuis jersey_map, "
              f"{len(self._player_registry)} joueurs au registre")

    # ── Diagnostic ──────────────────────────────────────────────────────────────

    def summary(self):
        total = len(self._player_registry)
        known = sum(1 for r in self._player_registry.values()
                    if r.confidence >= MIN_CONFIDENCE)
        multi = sum(1 for r in self._player_registry.values()
                    if r.seen_matches >= 2)

        print(f"  [PIM] ── Résumé registre ──────────────────────────────")
        print(f"  [PIM]   Joueurs registrés : {total}")
        print(f"  [PIM]   Conf≥{MIN_CONFIDENCE}           : {known}")
        print(f"  [PIM]   Vus ≥2 matchs     : {multi}")
        print(f"  [PIM]   Ce run (tracks)   : {len(self._match_registry)}")

        by_team: dict[str, list] = {}
        for r in sorted(self._player_registry.values(), key=lambda x: x.jersey):
            if r.confidence < 0.30:
                continue
            by_team.setdefault(r.team, []).append(r)

        for team, records in sorted(by_team.items()):
            print(f"  [PIM]   ── Équipe '{team}' ──")
            for r in records:
                name_str = f" ({r.name})" if r.name else ""
                src = "+".join(set(r.sources[-3:]))  # 3 sources les plus récentes
                print(f"  [PIM]     #{r.jersey:>2}{name_str:<15} "
                      f"conf={r.confidence:.2f} "
                      f"matchs={r.seen_matches} "
                      f"src={src}")
        print(f"  [PIM] ──────────────────────────────────────────────────")


    # Alias de compatibilite pipeline

    def update_from_goal_scorer(self, track_id, jersey,
                                 team='home',
                                 team_color=None,
                                 visual_desc=None):
        self.register_goal_scorer(
            track_id=track_id, jersey=jersey,
            team=team, team_color=team_color,
        )

    def update_from_pipeline(self, jersey_map,
                              events=None,
                              visual_pool=None,
                              teams=None):
        self.finalize(jersey_map=jersey_map, events=events)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pim(video_path: str = "", team_id: str = "", match_date: str = "") -> PlayerIdentityMemory:
    """Point d'entrée unique : créer/charger la mémoire."""
    date = match_date or time.strftime("%Y-%m-%d")
    # Extraire une date depuis le chemin vidéo si possible
    if not match_date and video_path:
        import re
        m = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', video_path)
        if m:
            date = m.group(1).replace("_", "-")
    return PlayerIdentityMemory(match_date=date, team_id=team_id)


def load_memory_for_video(video_path: str) -> PlayerIdentityMemory:
    """Compatibilité avec l'ancienne API."""
    return make_pim(video_path=video_path)


def filter_unknown_players(pim: PlayerIdentityMemory, players: list) -> tuple[list, dict]:
    """Compatibilité avec l'ancienne API."""
    return pim.tracks_to_skip(players)