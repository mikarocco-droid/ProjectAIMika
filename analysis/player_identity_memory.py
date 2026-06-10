"""
PlayerIdentityMemory — Sprint A
================================
Mémoire persistante des identités joueurs entre runs.

Philosophie :
  - Un numéro de maillot identifié une fois = vérité du match
  - On ne redemande plus à Gemini ce qu'on sait déjà
  - L'identité survit aux pertes de tracking, rotations, flou

Architecture :
  - Stockage JSON par match_id (hash vidéo ou timestamp)
  - Chaque joueur = track_id + numéro + équipe + descripteurs visuels
  - Confiance cumulée : chaque détection supplémentaire augmente le score

Usage dans pipeline.py :
  from analysis.player_identity_memory import PlayerIdentityMemory

  # Début de run — charger la mémoire existante
  pim = PlayerIdentityMemory(match_id=video_hash)
  jersey_map = pim.get_jersey_map()   # injecter dans le pipeline

  # Fin de run — sauvegarder les nouvelles identités
  pim.update_from_pipeline(jersey_map, events, visual_pool)
  pim.save()

  # Stats
  pim.summary()
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

# Confiance minimum pour considérer un numéro comme fiable
MIN_CONFIDENCE_THRESHOLD = 0.60

# Confiance accordée à chaque source de détection
SOURCE_CONFIDENCE = {
    "gemini_visual":   0.85,   # Gemini a vu le numéro sur une frame de but
    "gemini_batch":    0.70,   # Gemini batch lecture numéros
    "ocr_crop":        0.65,   # OCR local sur crop joueur
    "gemini_general":  0.55,   # Gemini lecture générale
    "inferred":        0.40,   # Inféré par contexte
}

# Au-delà de N observations concordantes, confiance = 1.0
OBSERVATIONS_FOR_MAX_CONF = 5


# ── Classe principale ───────────────────────────────────────────────────────────

class PlayerIdentityMemory:
    """
    Mémoire persistante des identités joueurs pour un match donné.

    Un `player_record` a la structure :
    {
        "track_id":     "P27",
        "jersey":       9,
        "team":         0,            # 0 ou 1
        "team_color":   "rouge",
        "confidence":   0.92,
        "observations": 3,            # nombre de détections concordantes
        "sources":      ["gemini_visual", "gemini_batch"],
        "visual_desc":  { "boots": "foncées", "socks": "rouges", ... },
        "first_seen":   1234567890.0,
        "last_seen":    1234567990.0,
        "locked":       False,        # True = ne jamais écraser
    }
    """

    def __init__(self, match_id: str, video_path: Optional[str] = None):
        self.match_id   = match_id
        self.video_path = video_path
        self._records: dict[str, dict] = {}   # track_id → record
        self._dirty = False

        self._path = MEMORY_DIR / f"match_{match_id}.json"
        self._load()

    # ── Persistance ────────────────────────────────────────────────────────────

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._records = data.get("players", {})
                print(f"  [PIM] Mémoire chargée : {len(self._records)} joueurs "
                      f"({sum(1 for r in self._records.values() if r.get('jersey'))} avec numéro)")
            except Exception as e:
                print(f"  [PIM] Erreur chargement mémoire : {e}")
                self._records = {}

    def save(self):
        if not self._dirty:
            return
        try:
            data = {
                "match_id":   self.match_id,
                "updated_at": time.time(),
                "players":    self._records,
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  [PIM] Mémoire sauvegardée : {len(self._records)} joueurs → {self._path}")
        except Exception as e:
            print(f"  [PIM] Erreur sauvegarde : {e}")

    # ── Lecture ────────────────────────────────────────────────────────────────

    def get_jersey_map(self) -> dict:
        """
        Retourne un jersey_map compatible avec le pipeline existant.
        Format : { "P27": 9, "P14": 7, ... }
        Uniquement les joueurs avec confiance suffisante.
        """
        return {
            tid: r["jersey"]
            for tid, r in self._records.items()
            if r.get("jersey") is not None
            and r.get("confidence", 0) >= MIN_CONFIDENCE_THRESHOLD
        }

    def get_record(self, track_id: str) -> Optional[dict]:
        return self._records.get(str(track_id))

    def knows(self, track_id: str) -> bool:
        """True si on connaît déjà ce joueur avec une confiance suffisante."""
        r = self._records.get(str(track_id))
        return r is not None and r.get("confidence", 0) >= MIN_CONFIDENCE_THRESHOLD

    def unknown_tracks(self, track_ids: list) -> list:
        """Retourne les track_ids qu'on ne connaît pas encore — ceux à envoyer à Gemini."""
        return [tid for tid in track_ids if not self.knows(str(tid))]

    def summary(self):
        total    = len(self._records)
        known    = sum(1 for r in self._records.values()
                       if r.get("confidence", 0) >= MIN_CONFIDENCE_THRESHOLD)
        locked   = sum(1 for r in self._records.values() if r.get("locked"))
        multi    = sum(1 for r in self._records.values() if r.get("observations", 0) >= 2)

        print(f"  [PIM] ── Résumé mémoire ──────────────────────────")
        print(f"  [PIM]   Joueurs connus  : {known}/{total} (conf≥{MIN_CONFIDENCE_THRESHOLD})")
        print(f"  [PIM]   Multi-observés  : {multi} (≥2 détections concordantes)")
        print(f"  [PIM]   Locked          : {locked}")

        for tid, r in sorted(self._records.items()):
            if r.get("jersey"):
                src = "+".join(set(r.get("sources", [])))
                _tc = r.get('team_color') or '?'
                print(f"  [PIM]   {tid:>6} → #{r['jersey']:>2} "
                      f"team={_tc:<8} "
                      f"conf={r.get('confidence',0):.2f} "
                      f"obs={r.get('observations',0)} src={src}")
        print(f"  [PIM] ───────────────────────────────────────────")

    # ── Mise à jour ────────────────────────────────────────────────────────────

    def register(self, track_id: str, jersey: Optional[int],
                 team: Optional[int] = None,
                 team_color: Optional[str] = None,
                 source: str = "gemini_general",
                 visual_desc: Optional[dict] = None,
                 lock: bool = False):
        """
        Enregistre ou met à jour l'identité d'un joueur.

        Règles de fusion :
        - Si le joueur est locked, ignorer (sauf si lock=True)
        - Si le numéro est différent et confiance basse → mettre à jour
        - Si le numéro est différent et confiance haute → incrémenter le compteur
          de discordances sans écraser (mécanisme de sécurité)
        - Si le numéro est identique → incrémenter observations, augmenter confiance
        """
        tid = str(track_id)
        conf_new = SOURCE_CONFIDENCE.get(source, 0.50)
        now = time.time()

        if tid in self._records:
            r = self._records[tid]

            # Joueur locked → ignorer sauf si force lock
            if r.get("locked") and not lock:
                return

            existing_jersey = r.get("jersey")

            if existing_jersey is not None and jersey is not None:
                if existing_jersey == jersey:
                    # Observation concordante → augmenter confiance
                    obs = r.get("observations", 1) + 1
                    conf = min(1.0, conf_new + (obs / OBSERVATIONS_FOR_MAX_CONF) * 0.15)
                    conf = max(conf, r.get("confidence", 0))  # jamais descendre
                    r["observations"] = obs
                    r["confidence"]   = round(conf, 3)
                    r["last_seen"]    = now
                    if source not in r.get("sources", []):
                        r.setdefault("sources", []).append(source)
                else:
                    # Discordance — on n'écrase que si confiance actuelle basse
                    if r.get("confidence", 0) < 0.70:
                        r["jersey"]       = jersey
                        r["confidence"]   = round(conf_new * 0.8, 3)  # pénalité discordance
                        r["observations"] = 1
                        r["sources"]      = [source]
                        r["last_seen"]    = now
                        print(f"  [PIM] ⚠️  Discordance {tid}: #{existing_jersey}→#{jersey} "
                              f"(confiance basse, mise à jour)")
                    else:
                        print(f"  [PIM] ⚠️  Discordance ignorée {tid}: #{existing_jersey} "
                              f"conservé (conf={r['confidence']:.2f}) vs #{jersey}")
            elif jersey is not None:
                # Pas de numéro connu → enregistrer
                r["jersey"]       = jersey
                r["confidence"]   = round(conf_new, 3)
                r["observations"] = 1
                r["sources"]      = [source]
                r["last_seen"]    = now

            # Enrichir les métadonnées si manquantes
            if team is not None and r.get("team") is None:
                r["team"] = team
            if team_color and not r.get("team_color"):
                r["team_color"] = team_color
            if visual_desc and not r.get("visual_desc"):
                r["visual_desc"] = visual_desc
            if lock:
                r["locked"] = True

        else:
            # Nouveau joueur
            self._records[tid] = {
                "track_id":    tid,
                "jersey":      jersey,
                "team":        team,
                "team_color":  team_color,
                "confidence":  round(conf_new, 3),
                "observations": 1,
                "sources":     [source],
                "visual_desc": visual_desc or {},
                "first_seen":  now,
                "last_seen":   now,
                "locked":      lock,
            }

        self._dirty = True

    def update_from_pipeline(self, jersey_map: dict,
                              events: Optional[list] = None,
                              visual_pool: Optional[dict] = None,
                              teams: Optional[dict] = None):
        """
        Mise à jour en masse depuis les sorties du pipeline existant.

        jersey_map : { "P27": 9, ... }
        events     : liste d'events avec player, team, type
        visual_pool: { track_id: { "hair": ..., "boots": ..., ... } }
        teams      : { track_id: team_id } ou dans events
        """
        # Construire un index équipe depuis les events
        team_index: dict[str, int]   = {}
        color_index: dict[str, str]  = {}

        if events:
            for e in events:
                pid = str(e.get("player", ""))
                if not pid:
                    continue
                if "team" in e and pid not in team_index:
                    team_index[pid] = e["team"]
                if "team_color" in e and pid not in color_index:
                    color_index[pid] = e["team_color"]

        if teams:
            for pid, t in teams.items():
                if str(pid) not in team_index:
                    team_index[str(pid)] = t

        # Détecter la source selon d'où vient le jersey_map
        # (on ne peut pas le savoir ici, on utilise "gemini_batch" par défaut)
        for track_id, jersey in jersey_map.items():
            tid = str(track_id)
            visual = (visual_pool or {}).get(tid) or (visual_pool or {}).get(track_id)
            self.register(
                track_id   = tid,
                jersey     = jersey,
                team       = team_index.get(tid),
                team_color = color_index.get(tid),
                source     = "gemini_batch",
                visual_desc= visual,
            )

        # Enregistrer les joueurs sans numéro (pour avoir leur équipe)
        if events:
            seen = set()
            for e in events:
                pid = str(e.get("player", ""))
                if not pid or pid in seen or pid in jersey_map:
                    continue
                seen.add(pid)
                if pid not in self._records:
                    self.register(
                        track_id   = pid,
                        jersey     = None,
                        team       = team_index.get(pid),
                        team_color = color_index.get(pid),
                        source     = "inferred",
                    )

        print(f"  [PIM] Mise à jour depuis pipeline : {len(jersey_map)} numéros, "
              f"{len(self._records)} joueurs en mémoire")

    def update_from_goal_scorer(self, track_id: str, jersey: int,
                                 team_color: Optional[str] = None,
                                 visual_desc: Optional[dict] = None):
        """
        Méthode spécialisée pour les buteurs identifiés par Gemini sur frame de but.
        Source la plus fiable → haute confiance, peut écraser.
        """
        self.register(
            track_id    = track_id,
            jersey      = jersey,
            team_color  = team_color,
            source      = "gemini_visual",
            visual_desc = visual_desc,
        )
        print(f"  [PIM] 🎯 Buteur identifié : {track_id} → #{jersey} "
              f"({team_color or '?'}) conf={SOURCE_CONFIDENCE['gemini_visual']:.2f}")


# ── Helpers ──────────────────────────────────────────────────────────────────────

def match_id_from_video(video_path: str) -> str:
    """Génère un match_id stable depuis le chemin vidéo."""
    # Hash du nom de fichier (pas du contenu — trop lent)
    name = os.path.basename(video_path)
    return hashlib.md5(name.encode()).hexdigest()[:12]


def match_id_from_timestamp() -> str:
    """Fallback si pas de vidéo."""
    return str(int(time.time()))


# ── Intégration pipeline (fonctions d'aide) ──────────────────────────────────────

def load_memory_for_video(video_path: str) -> "PlayerIdentityMemory":
    """Point d'entrée unique : charger la mémoire pour une vidéo."""
    mid = match_id_from_video(video_path)
    return PlayerIdentityMemory(match_id=mid, video_path=video_path)


def filter_unknown_players(pim: "PlayerIdentityMemory",
                            players_with_frames: list) -> tuple[list, dict]:
    """
    Filtre la liste de joueurs à envoyer à Gemini.
    Retourne (à_envoyer_à_gemini, déjà_connus).

    Gain typique : si 15 joueurs identifiés sur le match précédent,
    un run suivant n'envoie que les nouveaux joueurs à Gemini.
    """
    to_gemini = []
    already_known = {}

    for p in players_with_frames:
        tid = str(p.get("id", ""))
        if pim.knows(tid):
            r = pim.get_record(tid)
            already_known[tid] = r["jersey"]
        else:
            to_gemini.append(p)

    if already_known:
        print(f"  [PIM] ⚡ {len(already_known)} joueurs déjà connus → skip Gemini")
        for tid, num in sorted(already_known.items()):
            print(f"         {tid} → #{num}")

    return to_gemini, already_known
