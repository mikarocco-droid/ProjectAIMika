"""
kickoff_group_candidate_dataset.py — Constructeur de dataset KICKOFF_GROUP_CANDIDATE
======================================================================================
Transforme les artefacts de pipeline en table canonique de recherche sur les
groupes de joueurs détectés comme candidats kickoff.

Philosophie : ce script COLLECTE, ne DÉCIDE pas.
Exactement le même modèle que goal_candidate_dataset.py.

Entrées :
    --log           pipeline.log (ou kickoff_cache.log)
    --match         identifiant du match (ex: andrimont_full)
    --kickoff_real  vrai kickoff en secondes de jeu (ex: 307)
    --output        répertoire de sortie (défaut: .)

Sorties :
    {match_id}_kickoff_groups.csv
    {match_id}_kickoff_groups.parquet

Colonnes
--------
  Identité     : match_id, candidate_id, video_name, camera_type
  Temps        : t_start, t_end, duration
  Joueurs      : sep_avg, sep_max, players_avg, players_max, group_size
  Mouvement    : p_motion_avg, p_motion_min, p_motion_max, p_motion_before, p_motion_after
  Ballon       : activity_after, ball_visible_ratio, ball_speed_after
  Décision     : selected_original, selected_rank, score_original
  Annotation   : distance_to_real_kickoff, is_within_30s, verified_label, annotation_comment
"""

from __future__ import annotations
import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Parseurs log ───────────────────────────────────────────────────────────────

def parse_camera_type(log: str) -> str:
    m = re.search(r'\[CAMERA_PROFILE\] type\s+=\s+(\S+)', log)
    return m.group(1) if m else "unknown"


def parse_video_name(log: str) -> str:
    m = re.search(r'video(?:_path)?\s*=\s*(.+?)(?:\s|$)', log)
    if m:
        return Path(m.group(1).strip()).stem
    return "unknown"


def parse_kickoff_final(log: str) -> dict:
    m = re.search(
        r'\[KICKOFF FINAL\] source=(\S+) offset=([\d.]+)s(?: conf=([\d.]+))?',
        log
    )
    if m:
        return {
            "source": m.group(1),
            "offset": float(m.group(2)),
            "conf":   float(m.group(3)) if m.group(3) else None,
        }
    return {"source": "not_found", "offset": None, "conf": None}


def parse_p1_motion(log: str) -> Optional[float]:
    m = re.search(r'\[KICKOFF P1\] p_motion moy 0-60s = ([\d.]+)', log)
    return float(m.group(1)) if m else None


def parse_motion_profile(log: str) -> dict[float, float]:
    """Retourne un dict {t_sec: p_motion} depuis le profil [MOTION]."""
    profile = {}
    in_motion = False
    motion_re = re.compile(r'\s+(\d+)s \[.+?\] ([\d.]+)')
    for line in log.split('\n'):
        if '[MOTION] Profil activité' in line:
            in_motion = True
            continue
        if in_motion:
            m = motion_re.match(line)
            if m:
                profile[float(m.group(1))] = float(m.group(2))
            elif profile:
                break
    return profile


def parse_kickoff_groups(log: str) -> list[dict]:
    """
    Extrait les groupes depuis [KICKOFF PLAYERS] N groupe(s) sep≥... :
    Format : grp[N] HH:MM→HH:MM dur=Xs  sep_avg=X.XX  n_avg=X.X  frames=X
    """
    groups = []
    header = re.search(r'\[KICKOFF PLAYERS\] (\d+) groupe\(s\) sep≥.*?dur≥.*?:', log)
    if not header:
        return []

    block = log[header.start():header.start() + 10000]
    grp_re = re.compile(
        r'grp\[(\d+)\] (\d+:\d+)→(\d+:\d+) dur=([\d.]+)s\s+'
        r'sep_avg=([\d.]+)\s+n_avg=([\d.]+)\s+frames=(\d+)'
    )
    for m in grp_re.finditer(block):
        t0_str = m.group(2)
        t1_str = m.group(3)
        t0 = int(t0_str.split(':')[0]) * 60 + float(t0_str.split(':')[1])
        t1 = int(t1_str.split(':')[0]) * 60 + float(t1_str.split(':')[1])
        groups.append({
            "idx":      int(m.group(1)),
            "t_start":  round(t0, 2),
            "t_end":    round(t1, 2),
            "t_start_fmt": t0_str,
            "duration": float(m.group(4)),
            "sep_avg":  float(m.group(5)),
            "n_avg":    float(m.group(6)),
            "frames":   int(m.group(7)),
        })
    return groups


def parse_kickoff_physical_candidates(log: str) -> list[dict]:
    """
    Extrait les candidats physiques depuis [KICKOFF] N candidat(s) :
    Format : t=MM:SS score=X.X sep=X.XX n=XX ball=✓/✗ near=X
    """
    candidates = []
    header = re.search(r'\[KICKOFF\] \d+ candidat\(s\) dans [\d.]+s :', log)
    if not header:
        return []
    block = log[header.start():header.start() + 8000]
    cand_re = re.compile(
        r't=(\d+:\d+) score=([\d.]+) sep=([\d.]+) n=(\d+) ball=([✓✗]) near=(\d+)'
    )
    for m in cand_re.finditer(block):
        t_str = m.group(1)
        t_sec = int(t_str.split(':')[0]) * 60 + float(t_str.split(':')[1])
        candidates.append({
            "t_str":  t_str,
            "t_sec":  round(t_sec, 2),
            "score":  float(m.group(2)),
            "sep":    float(m.group(3)),
            "n_team": int(m.group(4)),
            "ball":   m.group(5) == '✓',
            "near":   int(m.group(6)),
        })
    return candidates


def parse_selected_candidate(log: str) -> Optional[float]:
    """Retourne le t_start du groupe sélectionné par l'algorithme original."""
    # [KICKOFF PLAYERS] → N groupe(s) : précurseur=... → candidat kickoff t=MM:SS
    m = re.search(r'candidat kickoff t=(\d+:\d+)', log)
    if m:
        t_str = m.group(1)
        return int(t_str.split(':')[0]) * 60 + float(t_str.split(':')[1])
    return None


def parse_motion_around(motion_profile: dict, t_start: float,
                        window_before: float = 30.0,
                        window_after: float = 30.0) -> dict:
    """Calcule p_motion avant/après un timestamp depuis le profil."""
    before_vals = [v for t, v in motion_profile.items()
                   if t_start - window_before <= t < t_start]
    after_vals  = [v for t, v in motion_profile.items()
                   if t_start < t <= t_start + window_after]
    return {
        "p_motion_before": round(sum(before_vals) / max(len(before_vals), 1), 3) if before_vals else None,
        "p_motion_after":  round(sum(after_vals)  / max(len(after_vals),  1), 3) if after_vals else None,
    }


def _make_candidate_id(match_id: str, t_start: float) -> str:
    ts = f"{int(t_start):06d}"
    h  = hashlib.md5(f"{match_id}_ko_{ts}".encode()).hexdigest()[:6]
    return f"{match_id}_ko_{ts}_{h}"


# ── Parseurs enrichissement groupes ────────────────────────────────────────────

def enrich_groups_from_physical(groups: list[dict],
                                 candidates: list[dict]) -> list[dict]:
    """
    Ajoute sep_max, players_max depuis les candidats physiques
    en cherchant les candidats dans la fenêtre de chaque groupe.
    """
    for g in groups:
        t0, t1 = g["t_start"], g["t_end"]
        in_group = [c for c in candidates if t0 - 1 <= c["t_sec"] <= t1 + 1]
        if in_group:
            g["sep_max"]     = round(max(c["sep"]    for c in in_group), 3)
            g["players_max"] = max(c["n_team"] for c in in_group)
            g["score_max"]   = round(max(c["score"]  for c in in_group), 2)
        else:
            g["sep_max"]     = g["sep_avg"]
            g["players_max"] = int(g["n_avg"])
            g["score_max"]   = None
    return groups


# ── Builder principal ──────────────────────────────────────────────────────────

def parse_log(log: str) -> dict:
    """Parse toutes les informations disponibles dans le log."""
    return {
        "camera_type":         parse_camera_type(log),
        "video_name":          parse_video_name(log),
        "p1_motion":           parse_p1_motion(log),
        "motion_profile":      parse_motion_profile(log),
        "groups":              parse_kickoff_groups(log),
        "physical_candidates": parse_kickoff_physical_candidates(log),
        "selected_t":          parse_selected_candidate(log),
        "kickoff_final":       parse_kickoff_final(log),
    }


def extract_groups(parsed: dict) -> list[dict]:
    """Enrichit les groupes avec les données des candidats physiques."""
    groups = parsed["groups"]
    groups = enrich_groups_from_physical(groups, parsed["physical_candidates"])
    return groups


def build_dataframe(groups: list[dict], parsed: dict,
                    match_id: str, kickoff_real: Optional[float],
                    algorithm_version: Optional[str] = None,
                    detector_commit: Optional[str] = None) -> pd.DataFrame:
    """Construit le DataFrame canonique."""
    motion_profile = parsed["motion_profile"]
    selected_t     = parsed["selected_t"]

    rows = []
    for rank, g in enumerate(groups):
        t0 = g["t_start"]

        # Mouvement autour du groupe
        motion = parse_motion_around(motion_profile, t0)

        # p_motion dans le groupe (depuis profil 10s)
        in_grp_vals = [v for t, v in motion_profile.items()
                       if t0 <= t <= g["t_end"]]
        p_motion_avg = round(sum(in_grp_vals) / max(len(in_grp_vals), 1), 3) if in_grp_vals else None
        p_motion_min = round(min(in_grp_vals), 3) if in_grp_vals else None
        p_motion_max = round(max(in_grp_vals), 3) if in_grp_vals else None

        # Sélection par l'algorithme original
        is_selected = (selected_t is not None and abs(t0 - selected_t) < 3.0)

        # Distance au vrai kickoff — mesurée à l'INTERVALLE [t_start, t_end] du groupe,
        # pas seulement à t_start. Un groupe qui dure 227s (ex: raeren grp[6] 5:57→9:45)
        # peut contenir le vrai kickoff même si son t_start en est à 53s — la distance
        # doit être 0 dans ce cas, pas 53.
        t_end_g = g["t_end"]
        if kickoff_real is not None:
            if t0 <= kickoff_real <= t_end_g:
                dist = 0.0
            else:
                dist = round(min(abs(t0 - kickoff_real), abs(t_end_g - kickoff_real)), 1)
        else:
            dist = None
        within_30 = (dist <= 30.0) if dist is not None else None

        row = dict(
            # Identité
            match_id            = match_id,
            candidate_id        = _make_candidate_id(match_id, t0),
            video_name          = parsed["video_name"],
            camera_type         = parsed["camera_type"],

            # Temps
            t_start             = t0,
            t_start_fmt         = g["t_start_fmt"],
            t_end               = g["t_end"],
            duration            = g["duration"],

            # Joueurs
            sep_avg             = g["sep_avg"],
            sep_max             = g.get("sep_max"),
            players_avg         = round(g["n_avg"], 1),
            players_max         = g.get("players_max"),
            group_size          = g["frames"],

            # Mouvement
            p_motion_avg        = p_motion_avg,
            p_motion_min        = p_motion_min,
            p_motion_max        = p_motion_max,
            p_motion_before     = motion["p_motion_before"],
            p_motion_after      = motion["p_motion_after"],

            # Ballon (à enrichir depuis d'autres sources)
            activity_after      = None,   # enrichissable depuis _post_activity log
            ball_visible_ratio  = None,   # enrichissable
            ball_speed_after    = None,   # enrichissable

            # Décision algorithme
            selected_original   = is_selected,
            selected_rank       = rank,   # rang chronologique dans les groupes
            score_original      = g.get("score_max"),

            # Annotation
            distance_to_real_kickoff = dist,
            is_within_30s            = within_30,
            verified_label           = None,   # rempli manuellement
            annotation_comment       = None,   # rempli manuellement

            # Traçabilité — permet de mixer datasets de plusieurs versions
            algorithm_version        = algorithm_version,
            detector_commit          = detector_commit,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Point d'extension : ajouter des features dérivées sans toucher au reste.
    Aujourd'hui : normalisation sep, ratio durée.
    """
    if len(df) == 0:
        return df

    # sep normalisée sur le match (utile pour comparer inter-matchs)
    sep_max_match = df["sep_avg"].max() or 1.0
    df["sep_avg_norm"] = (df["sep_avg"] / sep_max_match).round(3)

    # Durée relative (fraction de la durée max du match)
    dur_max = df["duration"].max() or 1.0
    df["duration_norm"] = (df["duration"] / dur_max).round(3)

    # Score combiné provisoire (non utilisé pour décision — pour exploration)
    df["sep_x_dur"] = (df["sep_avg"] * df["duration"]).round(2)

    return df


def export_csv(df: pd.DataFrame, match_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path     = output_dir / f"{match_id}_kickoff_groups.csv"
    parquet_path = output_dir / f"{match_id}_kickoff_groups.parquet"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_parquet(parquet_path, index=False)
    print(f"\n✅ CSV     → {csv_path}")
    print(f"✅ Parquet → {parquet_path}")


def build_dataset(log_path: Path, match_id: str,
                  kickoff_real: Optional[float],
                  output_dir: Path,
                  algorithm_version: Optional[str] = None,
                  detector_commit: Optional[str] = None) -> pd.DataFrame:
    """Point d'entrée principal."""
    log = log_path.read_text(encoding="utf-8", errors="replace")

    print(f"match_id     : {match_id}")

    parsed = parse_log(log)
    print(f"camera_type  : {parsed['camera_type']}")
    print(f"p_motion 0-60s : {parsed['p1_motion']}")
    print(f"kickoff final  : {parsed['kickoff_final']}")
    print(f"groupes probe  : {len(parsed['groups'])}")
    print(f"candidats phys : {len(parsed['physical_candidates'])}")
    print(f"sélectionné    : {parsed['selected_t']}s")
    print(f"kickoff réel   : {kickoff_real}s")

    groups = extract_groups(parsed)
    df = build_dataframe(groups, parsed, match_id, kickoff_real,
                         algorithm_version=algorithm_version,
                         detector_commit=detector_commit)
    df = enrich(df)
    export_csv(df, match_id, output_dir)

    # Résumé
    if kickoff_real is not None and len(df) > 0:
        near = df[df["is_within_30s"] == True]
        print(f"\nGroupes dans ±30s du vrai kickoff : {len(near)}")
        if len(near) > 0:
            for _, r in near.iterrows():
                print(f"  t={r['t_start_fmt']}  sep={r['sep_avg']:.2f}"
                      f"  dur={r['duration']:.0f}s"
                      f"  selected={r['selected_original']}"
                      f"  rank={r['selected_rank']}")

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Construit le dataset KICKOFF_GROUP_CANDIDATE depuis les logs pipeline."
    )
    p.add_argument("--log",           required=True,
                   help="Chemin vers pipeline.log ou kickoff_cache.log")
    p.add_argument("--match",         required=True,
                   help="Identifiant du match (ex: andrimont_full)")
    p.add_argument("--kickoff_real",  type=float, default=None,
                   help="Vrai kickoff en secondes de jeu (ex: 307)")
    p.add_argument("--output",        default=".",
                   help="Répertoire de sortie")
    p.add_argument("--algorithm_version", default=None,
                   help="Version du détecteur (ex: v1.0)")
    p.add_argument("--detector_commit",   default=None,
                   help="SHA git du commit kickoff_detector.py (ex: a3f8c12)")
    args = p.parse_args()

    try:
        import pyarrow  # noqa — vérification silencieuse
    except ImportError:
        print("⚠️  pyarrow non installé — pip install pyarrow")
        sys.exit(1)

    build_dataset(
        log_path          = Path(args.log),
        match_id          = args.match,
        kickoff_real      = args.kickoff_real,
        output_dir        = Path(args.output),
        algorithm_version = args.algorithm_version,
        detector_commit   = args.detector_commit,
    )


if __name__ == "__main__":
    main()