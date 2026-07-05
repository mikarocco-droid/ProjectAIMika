"""
goal_candidate_dataset.py — Constructeur de dataset GOAL_CANDIDATE
====================================================================
Transforme les artefacts de pipeline en table canonique de recherche.

Entrées :
    --log       pipeline.log (ou test.txt)
    --analysis  analysis.json (outputs/test/analysis.json)
    --goals     goals_real en secondes de jeu, ex: 382,1830,2555,2938,3582,5046
    --match     identifiant du match (ex: andrimont_0)
    --output    répertoire de sortie (défaut: .)

Sorties :
    goal_candidates.parquet
    goal_candidates.csv

Colonnes
--------
  Identification  : match_id, candidate_id, candidate_t, linked_shot_t, camera_type
  Géométrie       : bx, by, bx_prev, delta_bx, goal_side
  Dynamique       : speed, peak_speed, stuck, rebound
  Pipeline        : candidate_stage, n_terminal, posthoc_score, reject_reason
  Gemini          : gemini_called, goal_score, goal_votes, gemini_conf
  BC4             : world_score, bx_at_goal, in_goal
  Vérité terrain  : auto_label, verified_label, cluster
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Helpers ────────────────────────────────────────────────────────────────────

def _t_to_sec(t: str) -> float:
    """'MM:SS' → float secondes."""
    parts = t.split(":")
    return int(parts[0]) * 60 + float(parts[1])


def _make_candidate_id(match_id: str, t_sec: float) -> str:
    """Identifiant stable : match_id + timestamp arrondi à la seconde."""
    ts = f"{int(t_sec):06d}"
    h  = hashlib.md5(f"{match_id}_{ts}".encode()).hexdigest()[:6]
    return f"{match_id}_{ts}_{h}"


def _label(t_sec: float, goals_real: list[float], tol: float = 90.0) -> Optional[str]:
    """Retourne 'VP' si dans ±tol d'un vrai but, 'FP' sinon, None si goals_real vide."""
    if not goals_real:
        return None
    for g in goals_real:
        if abs(t_sec - g) <= tol:
            return "VP"
    return "FP"


# ── Parseurs log ───────────────────────────────────────────────────────────────

def parse_cross_traces(log: str) -> dict[str, dict]:
    """
    Extrait les [CROSS TRACE] terminaux (terminal=True) du log.
    Retourne un dict {t_str: {bx_prev, bx, delta_bx, by, speed, stuck}}.
    On ne garde que les lignes suivies de [TERMINAL] goal dans un rayon de 2s.
    """
    # Terminaux confirmés
    terminal_re = re.compile(
        r'\[TERMINAL\] goal à (\d+:\d+) \| gauche bx=([\d.]+) by=([\d.]+) '
        r'stuck=(\d+)f peak=([\d.]+) delta_bx=([\d.]+)'
    )
    terminals: dict[str, dict] = {}
    for m in terminal_re.finditer(log):
        t, bx, by, stuck, peak, delta = m.groups()
        t_sec = _t_to_sec(t)
        # Dédupliquer (le log contient des doublons)
        if t not in terminals:
            terminals[t] = dict(
                bx       = float(bx),
                by       = float(by),
                stuck    = int(stuck),
                peak_speed = float(peak),
                delta_bx = float(delta),
                speed    = float(peak),   # approximation
                rebound  = False,         # rempli par CROSS TRACE si dispo
            )

    # Enrichir avec bx_prev depuis CROSS TRACE
    cross_re = re.compile(
        r'\[CROSS TRACE\] t=(\d+:\d+) bx_prev=([\d.]+) bx=([\d.]+) delta_bx=([\d.]+) '
        r'by=([\d.]+) speed=([\d.]+) stuck=(\d+) stuck_min=\d+ → →candidat'
    )
    for m in cross_re.finditer(log):
        t, bxp, bx, delta, by, speed, stuck = m.groups()
        if t in terminals:
            terminals[t]["bx_prev"] = float(bxp)

    return terminals


def parse_profiler_csv(csv_path: Path) -> list[dict]:
    """Lit le CSV du shot_profiler."""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def parse_analysis_json(json_path: Path) -> dict:
    """Lit analysis.json produit par le pipeline."""
    if not json_path.exists():
        return {}
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def parse_camera_type(log: str) -> str:
    m = re.search(r'\[CAMERA_PROFILE\] type\s+=\s+(\S+)', log)
    return m.group(1) if m else "unknown"


def parse_kickoff(log: str) -> float:
    m = re.search(r'kickoff_offset=(\d+\.\d+)s', log)
    return float(m.group(1)) if m else 0.0


def parse_goal_score(log: str, t_rel: str) -> dict:
    """Extrait le FINAL SCORE Gemini pour un candidat donné (par timestamp)."""
    # On cherche le bloc PRE-GEMINI le plus proche du timestamp
    pattern = re.compile(
        rf'\[CANDIDAT\] t={re.escape(t_rel)}.*?'
        r'\[FINAL SCORE\] goal=([\d.]+) neg=([\d.]+) total=([\d.-]+).*?'
        r'\[DEBUG FINAL\] goal_votes=(\d+) shot_votes=(\d+)',
        re.DOTALL
    )
    m = pattern.search(log)
    if m:
        return dict(
            goal_score  = float(m.group(1)),
            neg_score   = float(m.group(2)),
            final_score = float(m.group(3)),
            goal_votes  = int(m.group(4)),
            shot_votes  = int(m.group(5)),
        )
    return dict(goal_score=None, neg_score=None, final_score=None, goal_votes=None, shot_votes=None)


# ── Builder principal ──────────────────────────────────────────────────────────

def build_dataset(
    log_path:      Path,
    analysis_path: Path,
    goals_real:    list[float],
    match_id:      str,
    output_dir:    Path,
) -> pd.DataFrame:

    log = log_path.read_text(encoding="utf-8", errors="replace")
    camera_type = parse_camera_type(log)
    kickoff     = parse_kickoff(log)

    print(f"match_id     : {match_id}")
    print(f"camera_type  : {camera_type}")
    print(f"kickoff      : {kickoff:.1f}s")
    print(f"goals_real   : {goals_real}")

    # Terminaux créés
    terminals = parse_cross_traces(log)
    print(f"Terminaux uniques : {len(terminals)}")

    # Chercher le CSV shot_profiler dans analysis.json ou chemin standard
    analysis = parse_analysis_json(analysis_path)
    shot_csv_path = analysis_path.parent / "shot_profiles" / f"{match_id}_shots.csv"
    profiler_rows = parse_profiler_csv(shot_csv_path)
    shot_map = {r.get("shot_t_fmt", ""): r for r in profiler_rows}

    rows = []
    for t_str, geo in terminals.items():
        t_sec = _t_to_sec(t_str)

        # Identifiant stable
        cid = _make_candidate_id(match_id, t_sec)

        # Shot profiler : trouver le tir lié le plus proche
        linked_shot_t = None
        posthoc_score = None
        reject_reason = None
        n_terminal    = None
        rebound       = geo.get("rebound", False)

        # Gemini
        gemini_info = parse_goal_score(log, t_str)

        # BC4 — chercher dans le log autour du timestamp
        bc4_pat = re.compile(
            r'\[SHOT→GOAL BC4\] t=' + re.escape(t_str) +
            r'.*?bx=([\d.]+).*?world_score=([\d.]+)',
            re.DOTALL
        )
        bc4_m = bc4_pat.search(log)
        world_score = float(bc4_m.group(2)) if bc4_m else None
        bx_at_goal  = float(bc4_m.group(1)) if bc4_m else None
        in_goal_m   = re.search(r'in_goal=(True|False)', bc4_m.group(0)) if bc4_m else None
        in_goal     = (in_goal_m.group(1) == "True") if in_goal_m else None

        # Label
        auto_lbl = _label(t_sec, goals_real)

        row = dict(
            # Identification
            match_id       = match_id,
            candidate_id   = cid,
            candidate_t    = t_str,
            candidate_t_sec= round(t_sec, 2),
            linked_shot_t  = linked_shot_t,
            camera_type    = camera_type,

            # Géométrie
            bx             = geo.get("bx"),
            by             = geo.get("by"),
            bx_prev        = geo.get("bx_prev"),
            delta_bx       = geo.get("delta_bx"),
            goal_side      = "left",  # low_side toujours gauche ici

            # Dynamique
            speed          = geo.get("speed"),
            peak_speed     = geo.get("peak_speed"),
            stuck          = geo.get("stuck"),
            rebound        = rebound,

            # Pipeline
            candidate_stage= "terminal_goal",
            n_terminal     = n_terminal,
            posthoc_score  = posthoc_score,
            reject_reason  = reject_reason,

            # Gemini
            gemini_called  = gemini_info["goal_votes"] is not None,
            goal_score     = gemini_info["goal_score"],
            goal_votes     = gemini_info["goal_votes"],
            gemini_conf    = None,  # enrichissable

            # BC4
            world_score    = world_score,
            bx_at_goal     = bx_at_goal,
            in_goal        = in_goal,

            # Vérité terrain
            auto_label     = auto_lbl,
            verified_label = None,   # rempli manuellement
            cluster        = None,   # rempli par clustering
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Typage
    df["stuck"]         = df["stuck"].astype("Int64")
    df["goal_votes"]    = df["goal_votes"].astype("Int64")
    df["gemini_called"] = df["gemini_called"].astype(bool)
    df["rebound"]       = df["rebound"].astype(bool)

    # Tri
    df = df.sort_values("candidate_t_sec").reset_index(drop=True)

    print(f"\nDataset : {len(df)} lignes")
    print(df["auto_label"].value_counts().to_string())

    # Export
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out     = output_dir / f"{match_id}_goal_candidates.csv"
    parquet_out = output_dir / f"{match_id}_goal_candidates.parquet"

    df.to_csv(csv_out, index=False, encoding="utf-8")
    df.to_parquet(parquet_out, index=False)

    print(f"\n✅ CSV     → {csv_out}")
    print(f"✅ Parquet → {parquet_out}")

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Construit le dataset GOAL_CANDIDATE depuis les artefacts pipeline.")
    p.add_argument("--log",      required=True,  help="Chemin vers pipeline.log ou test.txt")
    p.add_argument("--analysis", default="",     help="Chemin vers analysis.json")
    p.add_argument("--goals",    default="",     help="Virgule-séparés: secondes de jeu des vrais buts, ex: 382,1830")
    p.add_argument("--match",    default="match", help="Identifiant du match (ex: andrimont_0)")
    p.add_argument("--output",   default=".",    help="Répertoire de sortie")
    args = p.parse_args()

    log_path      = Path(args.log)
    analysis_path = Path(args.analysis) if args.analysis else log_path.parent / "analysis.json"
    goals_real    = [float(x) for x in args.goals.split(",") if x.strip()] if args.goals else []
    output_dir    = Path(args.output)

    build_dataset(log_path, analysis_path, goals_real, args.match, output_dir)


if __name__ == "__main__":
    main()
