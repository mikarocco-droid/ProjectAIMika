"""
kickoff_profiler.py — Profiler du détecteur de kickoff
======================================================
Parse un pipeline.log et produit un rapport complet sur :
  - Les candidats kickoff considérés et leur score
  - La valeur de _p1_early_high et pourquoi elle est levée
  - Pourquoi le vrai kickoff (connu) est rejeté ou mal classé
  - Les signaux disponibles (motion, séparation, ballon)

Usage :
    python3 kickoff_profiler.py --log pipeline.log --true_kickoff 307
    python3 kickoff_profiler.py --log pipeline.log --true_kickoff 880
    python3 kickoff_profiler.py --log test.txt --true_kickoff 307

Sorties :
    kickoff_report.txt   rapport texte complet
    kickoff_candidates.csv  table des candidats
"""

from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path


# ── Parseurs ──────────────────────────────────────────────────────────────────

def parse_p1_motion(log: str) -> tuple[float | None, bool]:
    """Extrait la p_motion 0-60s et si _p1_early_high a été levé."""
    m = re.search(r'\[KICKOFF P1\] p_motion moy 0-60s = ([\d.]+)', log)
    if not m:
        return None, False
    val = float(m.group(1))
    early_high = val >= 30.0
    return val, early_high


def parse_kickoff_final(log: str) -> dict:
    """Extrait la ligne [KICKOFF FINAL]."""
    m = re.search(
        r'\[KICKOFF FINAL\] source=(\S+) offset=([\d.]+)s conf=([\d.]+)',
        log
    )
    if m:
        return {"source": m.group(1), "offset": float(m.group(2)), "conf": float(m.group(3))}
    m2 = re.search(r'\[KICKOFF FINAL\] source=(\S+) offset=([\d.]+)s', log)
    if m2:
        return {"source": m2.group(1), "offset": float(m2.group(2)), "conf": None}
    return {"source": "not_found", "offset": None, "conf": None}


def parse_kickoff_candidates(log: str) -> list[dict]:
    """
    Extrait les candidats depuis la section [KICKOFF] N candidat(s) :
        t=MM:SS score=X.X sep=X.XX n=XX ball=✓/✗ near=X
    """
    candidates = []
    header = re.search(r'\[KICKOFF\] (\d+) candidat\(s\) dans ([\d.]+)s :', log)
    if not header:
        return []

    block_start = header.start()
    # Lire les lignes suivantes jusqu'à la prochaine section
    block = log[block_start:block_start + 5000]
    cand_re = re.compile(
        r't=(\d+:\d+)\s+score=([\d.]+)\s+sep=([\d.]+)\s+n=(\d+)\s+ball=([✓✗])\s+near=(\d+)'
    )
    for m in cand_re.finditer(block):
        t_str = m.group(1)
        t_sec = int(t_str.split(':')[0]) * 60 + int(t_str.split(':')[1])
        candidates.append({
            "t_str":    t_str,
            "t_sec":    t_sec,
            "score":    float(m.group(2)),
            "sep":      float(m.group(3)),
            "n_team":   int(m.group(4)),
            "ball":     m.group(5) == '✓',
            "near_ball": int(m.group(6)),
        })
    return candidates


def parse_kickoff_groups(log: str) -> list[dict]:
    """Extrait les groupes sep>=0.45 depuis [KICKOFF GROUPS]."""
    groups = []
    block_m = re.search(r'\[KICKOFF GROUPS\] \d+ groupe\(s\) sep.*?:', log)
    if not block_m:
        return []
    block = log[block_m.start():block_m.start() + 3000]
    grp_re = re.compile(
        r'grp\[(\d+)\] t=(\d+:\d+) len=(\d+) sep=([\d.]+) p_motion avg=([\d.]+) min=([\d.]+) max=([\d.]+)'
    )
    for m in grp_re.finditer(block):
        t_str = m.group(2)
        t_sec = int(t_str.split(':')[0]) * 60 + int(t_str.split(':')[1])
        groups.append({
            "idx":        int(m.group(1)),
            "t_str":      t_str,
            "t_sec":      t_sec,
            "len_frames": int(m.group(3)),
            "sep":        float(m.group(4)),
            "pm_avg":     float(m.group(5)),
            "pm_min":     float(m.group(6)),
            "pm_max":     float(m.group(7)),
        })
    return groups


def parse_players_probe(log: str) -> list[dict]:
    """Extrait les groupes du [KICKOFF PLAYERS] probe."""
    groups = []
    header = re.search(r'\[KICKOFF PLAYERS\] \d+ groupe\(s\)', log)
    if not header:
        return []
    block = log[header.start():header.start() + 3000]
    grp_re = re.compile(
        r'grp\[(\d+)\] (\d+:\d+)→(\d+:\d+) dur=([\d.]+)s\s+sep_avg=([\d.]+)\s+n_avg=([\d.]+)\s+frames=(\d+)'
    )
    for m in grp_re.finditer(block):
        t0 = m.group(2)
        t0_sec = int(t0.split(':')[0]) * 60 + int(t0.split(':')[1])
        groups.append({
            "idx":     int(m.group(1)),
            "t_start": t0,
            "t_sec":   t0_sec,
            "t_end":   m.group(3),
            "dur":     float(m.group(4)),
            "sep_avg": float(m.group(5)),
            "n_avg":   float(m.group(6)),
            "frames":  int(m.group(7)),
        })
    return groups


def parse_motion_profile(log: str) -> list[tuple[int, float]]:
    """Extrait le profil d'activité [MOTION] (t, val)."""
    profile = []
    motion_re = re.compile(r'\s+(\d+)s \[([█░]+)\] ([\d.]+)')
    in_motion = False
    for line in log.split('\n'):
        if '[MOTION] Profil activité' in line:
            in_motion = True
            continue
        if in_motion:
            m = motion_re.match(line)
            if m:
                profile.append((int(m.group(1)), float(m.group(3))))
            elif '[' in line and profile:
                break  # fin du bloc
    return profile


def parse_kickoff_conflict(log: str) -> dict | None:
    """Extrait le résultat du [KICKOFF CONFLICT] si présent."""
    m = re.search(
        r'\[KICKOFF CONFLICT\] physical=(\d+:\d+) \((\d+)s\) players=(\d+:\d+) \((\d+)s\) '
        r'delta=(\d+)s → (.+)',
        log
    )
    if m:
        return {
            "physical_t": m.group(1), "physical_s": int(m.group(2)),
            "players_t":  m.group(3), "players_s":  int(m.group(4)),
            "delta":      int(m.group(5)),
            "decision":   m.group(6),
        }
    return None


def parse_kickoff_applied(log: str) -> float | None:
    """Extrait le kickoff_offset réellement appliqué par le pipeline."""
    m = re.search(r'kickoff_offset=([\d.]+)s', log)
    return float(m.group(1)) if m else None


# ── Rapport ────────────────────────────────────────────────────────────────────

def build_report(log: str, true_kickoff: float | None, output_dir: Path) -> None:
    lines = []
    w = lines.append

    w("=" * 70)
    w("KICKOFF PROFILER — Rapport d'analyse")
    w("=" * 70)

    # ── Résultat appliqué
    applied = parse_kickoff_applied(log)
    w(f"\n[1] KICKOFF APPLIQUÉ PAR LE PIPELINE")
    w(f"    kickoff_offset = {applied}s")
    if true_kickoff and applied:
        delta = abs(applied - true_kickoff)
        status = "✅ CORRECT" if delta < 15 else "❌ FAUX"
        w(f"    vrai kickoff   = {true_kickoff}s")
        w(f"    delta          = {delta:.1f}s  → {status}")

    # ── KICKOFF FINAL
    final = parse_kickoff_final(log)
    w(f"\n[2] KICKOFF FINAL (sélection)")
    w(f"    source  = {final['source']}")
    w(f"    offset  = {final['offset']}s")
    w(f"    conf    = {final['conf']}")

    # ── P1 early high
    p1_val, early_high = parse_p1_motion(log)
    w(f"\n[3] SIGNAL _p1_early_high")
    if p1_val is not None:
        w(f"    p_motion moy 0-60s = {p1_val:.1f}px")
        w(f"    seuil              = 30.0px")
        w(f"    _p1_early_high     = {early_high}")
        if early_high:
            w(f"    ⚠️  FLAG LEVÉ → probe joueurs DÉSACTIVÉ")
            w(f"       Conséquence : le seul signal qui trouvait 307s est coupé.")
            w(f"       L'échauffement génère p_motion > 30 comme le jeu réel.")
        else:
            w(f"    ✅ flag non levé → probe joueurs actif")
    else:
        w(f"    (signal absent dans le log)")

    # ── Profil d'activité
    profile = parse_motion_profile(log)
    w(f"\n[4] PROFIL D'ACTIVITÉ (résolution 10s)")
    if profile:
        max_val = max(v for _, v in profile) or 1.0
        for t, v in profile[:60]:  # max 10 minutes
            bar_len = int(v / max_val * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            marker = ""
            if true_kickoff and abs(t - true_kickoff) < 10:
                marker = " ← VRAI KICKOFF"
            if applied and abs(t - applied) < 10:
                marker += " ← DÉTECTÉ"
            w(f"    {t:4d}s [{bar}] {v:.1f}{marker}")
    else:
        w("    (profil absent dans le log)")

    # ── Candidats physiques
    candidates = parse_kickoff_candidates(log)
    w(f"\n[5] CANDIDATS PHYSIQUES ({len(candidates)} trouvés)")
    if candidates:
        w(f"    {'t':>6}  {'score':>6}  {'sep':>5}  {'n':>4}  {'ball':>5}  {'near':>5}  label")
        w(f"    {'-'*55}")
        for c in sorted(candidates, key=lambda x: x["t_sec"]):
            label = ""
            if true_kickoff and abs(c["t_sec"] - true_kickoff) < 30:
                label = "✅ VRAI KO"
            elif applied and abs(c["t_sec"] - applied) < 30:
                label = "← SÉLECTIONNÉ"
            w(f"    {c['t_str']:>6}  {c['score']:>6.1f}  {c['sep']:>5.2f}  "
              f"{c['n_team']:>4}  {'✓' if c['ball'] else '✗':>5}  "
              f"{c['near_ball']:>5}  {label}")

        # Chercher le vrai kickoff dans les candidats
        if true_kickoff:
            near = [c for c in candidates if abs(c["t_sec"] - true_kickoff) < 60]
            if near:
                w(f"\n    Candidats dans ±60s du vrai kickoff ({true_kickoff}s) :")
                for c in near:
                    w(f"      t={c['t_str']}  score={c['score']:.1f}  sep={c['sep']:.2f}")
            else:
                w(f"\n    ⚠️  AUCUN candidat physique dans ±60s du vrai kickoff ({true_kickoff}s)")
                w(f"       → Le vrai kickoff n'a pas été scoré correctement.")
    else:
        w("    (aucun candidat trouvé dans le log)")

    # ── Groupes sep>=0.45
    groups = parse_kickoff_groups(log)
    w(f"\n[6] GROUPES SEP≥0.45 ({len(groups)} groupes)")
    if groups:
        w(f"    {'idx':>4}  {'t':>6}  {'len':>5}  {'sep':>5}  {'pm_avg':>7}  {'pm_min':>7}  label")
        w(f"    {'-'*60}")
        for g in groups:
            label = ""
            if true_kickoff and abs(g["t_sec"] - true_kickoff) < 60:
                label = "✅ VRAI KO"
            w(f"    {g['idx']:>4}  {g['t_str']:>6}  {g['len_frames']:>5}  {g['sep']:>5.2f}  "
              f"{g['pm_avg']:>7.1f}  {g['pm_min']:>7.1f}  {label}")
        if true_kickoff:
            near_g = [g for g in groups if abs(g["t_sec"] - true_kickoff) < 60]
            if not near_g:
                w(f"\n    ⚠️  Vrai kickoff ({true_kickoff}s) absent des groupes sep≥0.45")
    else:
        w("    (aucun groupe trouvé ou probe désactivé)")

    # ── Probe joueurs
    probe = parse_players_probe(log)
    w(f"\n[7] PROBE JOUEURS ({len(probe)} groupes)")
    if probe:
        w(f"    {'idx':>4}  {'t_start':>8}  {'t_end':>6}  {'dur':>6}  {'sep':>5}  {'n_avg':>6}  label")
        w(f"    {'-'*60}")
        for g in probe:
            label = ""
            if true_kickoff and abs(g["t_sec"] - true_kickoff) < 60:
                label = "✅ VRAI KO"
            w(f"    {g['idx']:>4}  {g['t_start']:>8}  {g['t_end']:>6}  {g['dur']:>6.0f}  "
              f"{g['sep_avg']:>5.2f}  {g['n_avg']:>6.1f}  {label}")
    else:
        # Chercher si le probe a été désactivé
        if "_p1_early_high=True" in log and "probe désactivé" in log:
            w("    ⚠️  PROBE DÉSACTIVÉ car _p1_early_high=True")
            w("       C'est la cause principale du mauvais offset sur ce match.")
        else:
            w("    (probe absent ou aucun groupe trouvé)")

    # ── Conflict
    conflict = parse_kickoff_conflict(log)
    w(f"\n[8] CONFLIT physical/players")
    if conflict:
        w(f"    physical : {conflict['physical_t']} ({conflict['physical_s']}s)")
        w(f"    players  : {conflict['players_t']} ({conflict['players_s']}s)")
        w(f"    delta    : {conflict['delta']}s")
        w(f"    décision : {conflict['decision']}")
    else:
        w("    (aucun conflit détecté — les deux sources sont cohérentes ou l'une est absente)")

    # ── Diagnostic final
    w(f"\n[9] DIAGNOSTIC SYNTHÈSE")
    if early_high:
        w(f"    CAUSE RACINE : _p1_early_high = True")
        w(f"    Le seuil p_motion >= 30.0 sur 0-60s est déclenché par l'ÉCHAUFFEMENT.")
        w(f"    L'échauffement et le jeu réel ont la même amplitude de mouvement joueurs.")
        w(f"    → Ce flag masque l'information clé : la vidéo a bien un pré-match.")
        w(f"")
        w(f"    QUESTION CLÉ : comment distinguer échauffement de jeu réel ?")
        w(f"    Pistes :")
        w(f"      1. Séparation des équipes : à l'échauffement les joueurs se mélangent,")
        w(f"         au coup d'envoi ils sont strictement séparés en deux moitiés.")
        w(f"         → team_separation à 0-60s : si < 0.3, c'est de l'échauffement.")
        w(f"      2. Formation : au KO, ≥10 joueurs, 2 moitiés peuplées, 1-2 au centre.")
        w(f"         → score_frame() donne déjà ce signal, mais il est bypassé.")
        w(f"      3. Durée : l'échauffement dure 10-20 min. Le KO arrive après.")
        w(f"         → un seuil temporel min_t plus tardif aiderait.")
        w(f"")
        w(f"    CORRECTIF PROPOSÉ (à valider sur données) :")
        w(f"      Remplacer le test binaire p_motion >= 30.0")
        w(f"      par un test de SÉPARATION DES ÉQUIPES sur 0-60s.")
        w(f"      Si sep_moy(0-60s) < 0.25 → vidéo commence à l'échauffement")
        w(f"                                  → probe joueurs reste ACTIF")
        w(f"      Si sep_moy(0-60s) >= 0.25 → vidéo commence en jeu")
        w(f"                                  → probe joueurs désactivé (comportement actuel)")
    else:
        w(f"    _p1_early_high = False → probe joueurs actif.")
        if final["source"] != "players_conflict" and final["source"] != "players_fallback":
            w(f"    La sélection finale vient de '{final['source']}' → vérifier si correct.")

    # ── Export CSV
    if candidates:
        csv_path = output_dir / "kickoff_candidates.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t_str", "t_sec", "score", "sep", "n_team", "ball", "near_ball"])
            writer.writeheader()
            writer.writerows(candidates)
        w(f"\n[CSV] {csv_path}")

    # ── Export rapport
    report_path = output_dir / "kickoff_report.txt"
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\n✅ Rapport → {report_path}")
    if candidates:
        print(f"✅ CSV     → {output_dir / 'kickoff_candidates.csv'}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Profiler du détecteur de kickoff.")
    p.add_argument("--log",           required=True, help="Chemin vers pipeline.log")
    p.add_argument("--true_kickoff",  type=float, default=None,
                   help="Vrai kickoff en secondes de jeu (ex: 307 ou 880)")
    p.add_argument("--output",        default=".", help="Répertoire de sortie")
    args = p.parse_args()

    log = Path(args.log).read_text(encoding="utf-8", errors="replace")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_report(log, args.true_kickoff, output_dir)


if __name__ == "__main__":
    main()
