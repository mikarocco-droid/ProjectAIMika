"""
geometry/bc4_enrich.py — Sprint BC.4 : instrumentation monde + comparaison VP/FP
==================================================================================

Objectif BC.4 :
  Répondre définitivement à :
  "Les variables géométriques (in_goal, in_penalty, dist_goal_m)
   apportent-elles un signal que Gemini et les pixels n'ont pas ?"

  Si oui  → BC.5 homographie.
  Si non  → on ferme le chantier géométrie.

Ce module fait deux choses :

  1. enrich_events(events, anchor, frame_w)
     Écrit event["geo"] sur chaque event candidat (goal/shot).
     Ne modifie aucune autre logique.

  2. compare_vp_fp(events, goals_real, anchor)
     Produit le tableau statistique VP vs FP depuis goals_real.
     Retourne un dict loggable + sérialisable JSON.

Structure event["geo"] :
  {
    "in_goal":        bool | None,
    "in_penalty":     bool | None,
    "dist_goal_m":    float | None,
    "dist_penalty_m": float | None,
    "bx":             float | None,
    "hypothesis":     str,            # "H1_goal_line" | "H2_penalty_line" | ...
    "anchor_score":   float,          # score BC.3C de l'hypothèse gagnante
    "anchor_ready":   bool,
  }

Usage dans pipeline.py :
  from geometry.bc4_enrich import enrich_events, compare_vp_fp

  enrich_events(events, _anchor, _frame_w)

  if goals_real:
      bc4 = compare_vp_fp(events, goals_real, _anchor)
      print_bc4_report(bc4)
"""

from __future__ import annotations
from typing import Optional
import math


# ── Tolérance VP : ±T secondes pour matcher un but réel ──────────────────────
VP_MATCH_TOLERANCE_S = 30.0   # fenêtre large — on est post-Gemini


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENRICHISSEMENT EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def enrich_events(events: list[dict],
                  anchor,
                  frame_w: int,
                  event_types: tuple = ("goal", "score", "shot")) -> int:
    """
    Écrit event["geo"] sur chaque event de type goal/score/shot.

    Paramètres :
      events     : liste des events du pipeline (dicts mutables)
      anchor     : FieldAnchorModel (BC.3C)
      frame_w    : largeur frame en pixels (pour normaliser ball_x brut)
      event_types: types d'events à enrichir

    Retourne : nombre d'events enrichis avec bx valide.

    Notes :
      - Ne lève jamais d'exception (non bloquant).
      - Si anchor non ready, écrit quand même geo avec anchor_ready=False
        et les valeurs None — pour tracer la couverture dans les logs.
      - bx résolution : obs.bx > obs.ball_x/frame_w > None
    """
    n_enriched = 0
    h = anchor.get_hypothesis() if anchor else None

    for ev in events:
        if ev.get("type") not in event_types:
            continue

        bx = _resolve_bx(ev, frame_w)

        geo: dict = {
            "in_goal":        None,
            "in_penalty":     None,
            "dist_goal_m":    None,
            "dist_penalty_m": None,
            "bx":             bx,
            "hypothesis":     h.name  if h else "none",
            "anchor_score":   round(h.score, 3) if h else 0.0,
            "anchor_ready":   bool(anchor.is_ready()) if anchor else False,
        }

        # geo_debug — traçabilité complète pour diagnostic BC.4
        # Permet de savoir POURQUOI dist_goal = X, sans aller fouiller les logs
        geo_debug: dict = {
            "hypothesis":       h.name           if h else "none",
            "hypothesis_score": round(h.score, 3) if h else 0.0,
            "s_camera":         round(h.score_camera, 3) if h else None,
            "s_fifa":           round(h.score_fifa, 3)   if h else None,
            "s_stable":         round(h.score_stable, 3) if h else None,
            "anchor_x":         round(anchor.penalty_line_obs, 4)
                                if anchor and anchor.penalty_line_obs else None,
            "goal_line_x":      round(h.goal_line_x, 4)    if h and h.goal_line_x    else None,
            "penalty_line_x":   round(h.penalty_line_x, 4) if h and h.penalty_line_x else None,
            "px_per_m":         round(h.px_per_m, 2)       if h and h.px_per_m       else None,
            "camera_prior":     round(anchor._camera_goal_left, 4)
                                if anchor and anchor._camera_goal_left else None,
            "n_obs":            len(anchor._anchor_obs) if anchor else 0,
        }

        if bx is not None and anchor is not None:
            try:
                geo["in_goal"]        = bool(anchor.ball_in_goal(bx))   if anchor.ball_in_goal(bx)    is not None else None
                geo["in_penalty"]     = bool(anchor.ball_in_penalty(bx)) if anchor.ball_in_penalty(bx) is not None else None
                d_goal                = anchor.distance_to_goal_m(bx)
                d_pen                 = anchor.distance_to_penalty_m(bx)
                geo["dist_goal_m"]    = round(float(d_goal), 2) if d_goal is not None else None
                geo["dist_penalty_m"] = round(float(d_pen),  2) if d_pen  is not None else None
                if bx is not None:
                    n_enriched += 1
            except Exception:
                pass   # anchor partiel — geo reste à None

        ev["geo"]       = geo
        ev["geo_debug"] = geo_debug

    return n_enriched


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPARAISON VP / FP
# ─────────────────────────────────────────────────────────────────────────────

def compare_vp_fp(events: list[dict],
                  goals_real: list[float],
                  anchor) -> dict:
    """
    Produit le tableau statistique BC.4 sur les events goal/score.

    Paramètres :
      events     : events enrichis par enrich_events()
      goals_real : liste de timestamps réels (secondes) des vrais buts
                   ex: [140.0, 587.0] pour 02:20 et 09:47
      anchor     : FieldAnchorModel (pour méta-info)

    Retourne un dict :
      {
        "anchor_ready": bool,
        "anchor_hypothesis": str,
        "anchor_score": float,

        "n_vp": int,
        "n_fp": int,

        "vp": {
          "events": [...],
          "in_goal_pct":     float,   # % avec in_goal=True
          "in_penalty_pct":  float,
          "dist_goal_mean":  float | None,
          "dist_goal_median":float | None,
          "dist_goal_std":   float | None,
        },
        "fp": { ... même structure ... },

        "separation": {
          "in_goal_delta":    float,   # vp_pct - fp_pct
          "in_penalty_delta": float,
          "dist_goal_delta":  float | None,
          "has_signal":       bool,    # True si au moins une séparation > 20%
        }
      }
    """
    h = anchor.get_hypothesis() if anchor else None

    goal_candidates = [
        e for e in events
        if e.get("type") in ("goal", "score") and "geo" in e
    ]

    vp_events, fp_events = _split_vp_fp(goal_candidates, goals_real)

    vp_stats = _compute_geo_stats(vp_events)
    fp_stats = _compute_geo_stats(fp_events)

    # Séparation statistique — c'est la question centrale de BC.4
    ig_vp  = vp_stats["in_goal_pct"]
    ig_fp  = fp_stats["in_goal_pct"]
    ip_vp  = vp_stats["in_penalty_pct"]
    ip_fp  = fp_stats["in_penalty_pct"]
    dg_vp  = vp_stats["dist_goal_mean"]
    dg_fp  = fp_stats["dist_goal_mean"]

    dist_delta = None
    if dg_vp is not None and dg_fp is not None:
        dist_delta = round(dg_vp - dg_fp, 2)

    # Signal = au moins une variable sépare VP vs FP de > 20 points
    has_signal = (
        abs(ig_vp - ig_fp) > 0.20
        or abs(ip_vp - ip_fp) > 0.20
        or (dist_delta is not None and abs(dist_delta) > 5.0)
    )

    return {
        "anchor_ready":      bool(anchor.is_ready()) if anchor else False,
        "anchor_hypothesis": h.name  if h else "none",
        "anchor_score":      round(h.score, 3) if h else 0.0,

        "n_vp":  len(vp_events),
        "n_fp":  len(fp_events),

        "vp":    {**vp_stats, "events": _summarize_events(vp_events)},
        "fp":    {**fp_stats, "events": _summarize_events(fp_events)},

        "separation": {
            "in_goal_delta":    round(ig_vp - ig_fp, 3),
            "in_penalty_delta": round(ip_vp - ip_fp, 3),
            "dist_goal_delta":  dist_delta,
            "has_signal":       has_signal,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. RAPPORT LISIBLE
# ─────────────────────────────────────────────────────────────────────────────

def print_bc4_report(bc4: dict):
    """
    Affiche le rapport BC.4 dans les logs pipeline.
    Format lisible, pas de tableau ASCII complexe.
    """
    sep   = bc4.get("separation", {})
    vp    = bc4.get("vp", {})
    fp    = bc4.get("fp", {})
    n_vp  = bc4.get("n_vp", 0)
    n_fp  = bc4.get("n_fp", 0)

    print(f"\n  [BC4] ══ Rapport géométrie BC.4 ══════════════════════════")
    print(f"  [BC4]   Anchor : ready={bc4['anchor_ready']} "
          f"hyp={bc4['anchor_hypothesis']} "
          f"score={bc4['anchor_score']:.3f}")
    print(f"  [BC4]   Events : {n_vp} VP  {n_fp} FP")

    if n_vp == 0 and n_fp == 0:
        print(f"  [BC4]   ⚠️  Aucun event classifié — goals_real vide ou bx absent")
        print(f"  [BC4] ══════════════════════════════════════════════════════")
        return

    print(f"  [BC4]")
    print(f"  [BC4]   Variable         │  VP (n={n_vp})  │  FP (n={n_fp})  │  Δ")
    print(f"  [BC4]   ─────────────────┼────────────────┼────────────────┼──────")

    ig_vp = vp.get("in_goal_pct", 0.0)
    ig_fp = fp.get("in_goal_pct", 0.0)
    ip_vp = vp.get("in_penalty_pct", 0.0)
    ip_fp = fp.get("in_penalty_pct", 0.0)
    dg_vp = vp.get("dist_goal_mean")
    dg_fp = fp.get("dist_goal_mean")
    dm_vp = vp.get("dist_goal_median")
    dm_fp = fp.get("dist_goal_median")
    ds_vp = vp.get("dist_goal_std")
    ds_fp = fp.get("dist_goal_std")

    ig_delta = sep.get("in_goal_delta", 0.0)
    ip_delta = sep.get("in_penalty_delta", 0.0)
    dg_delta = sep.get("dist_goal_delta")

    _flag = lambda d: " ◄◄" if abs(d) > 0.20 else ""

    print(f"  [BC4]   in_goal          │  {ig_vp*100:>5.1f}%        │  {ig_fp*100:>5.1f}%        │  {ig_delta:+.2f}{_flag(ig_delta)}")
    print(f"  [BC4]   in_penalty       │  {ip_vp*100:>5.1f}%        │  {ip_fp*100:>5.1f}%        │  {ip_delta:+.2f}{_flag(ip_delta)}")

    if dg_vp is not None and dg_fp is not None:
        dg_d_str = f"{dg_delta:+.1f}m" if dg_delta is not None else "?"
        flag_d   = " ◄◄" if dg_delta is not None and abs(dg_delta) > 5.0 else ""
        print(f"  [BC4]   dist_goal mean   │  {dg_vp:>+7.1f}m      │  {dg_fp:>+7.1f}m      │  {dg_d_str}{flag_d}")
        print(f"  [BC4]   dist_goal median │  {dm_vp:>+7.1f}m      │  {dm_fp:>+7.1f}m      │")
        print(f"  [BC4]   dist_goal σ      │  {ds_vp:>7.2f}m      │  {ds_fp:>7.2f}m      │")
    else:
        print(f"  [BC4]   dist_goal        │  (px_per_m absent) │  (px_per_m absent) │")

    print(f"  [BC4]")
    if sep.get("has_signal"):
        print(f"  [BC4]   ✅ SIGNAL DÉTECTÉ — au moins une variable sépare VP/FP")
        print(f"  [BC4]      → BC.5 homographie justifié")
    else:
        print(f"  [BC4]   ❌ PAS DE SIGNAL — géométrie n'apporte pas de discrimination")
        print(f"  [BC4]      → BC.5 non justifié avec les données actuelles")

    print(f"  [BC4]")
    print(f"  [BC4]   Détail VP :")
    for ev_s in vp.get("events", []):
        dbg = ev_s.get("geo_debug", {})
        print(f"  [BC4]     t={ev_s['t']}  bx={ev_s['bx']}  "
              f"in_goal={ev_s['in_goal']}  in_pen={ev_s['in_penalty']}  "
              f"dist={ev_s['dist_goal_m']}m  src={ev_s['source']}")
        if dbg:
            print(f"  [BC4]       H={dbg.get('hypothesis')} score={dbg.get('hypothesis_score')} "
                  f"goal_x={dbg.get('goal_line_x')} pen_x={dbg.get('penalty_line_x')} "
                  f"px/m={dbg.get('px_per_m')} n_obs={dbg.get('n_obs')}")
    print(f"  [BC4]   Détail FP :")
    for ev_s in fp.get("events", []):
        dbg = ev_s.get("geo_debug", {})
        print(f"  [BC4]     t={ev_s['t']}  bx={ev_s['bx']}  "
              f"in_goal={ev_s['in_goal']}  in_pen={ev_s['in_penalty']}  "
              f"dist={ev_s['dist_goal_m']}m  src={ev_s['source']}")
        if dbg:
            print(f"  [BC4]       H={dbg.get('hypothesis')} score={dbg.get('hypothesis_score')} "
                  f"goal_x={dbg.get('goal_line_x')} pen_x={dbg.get('penalty_line_x')} "
                  f"px/m={dbg.get('px_per_m')} n_obs={dbg.get('n_obs')}")

    print(f"  [BC4] ══════════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES INTERNES
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_bx(ev: dict, frame_w: int) -> Optional[float]:
    """Résout bx normalisé depuis les champs disponibles dans l'event."""
    bx = ev.get("bx")
    if bx is not None and 0.0 <= bx <= 1.0:
        return bx
    ball_x = ev.get("ball_x")
    if ball_x is not None and ball_x > 0 and frame_w > 0:
        return ball_x / frame_w
    # Fallback : center du bbox ballon si disponible
    bbox = ev.get("ball_bbox") or ev.get("bbox_ball")
    if bbox and len(bbox) >= 3 and frame_w > 0:
        cx = (bbox[0] + bbox[2]) / 2.0
        if 0 < cx < frame_w:
            return cx / frame_w
    return None


def _split_vp_fp(candidates: list[dict],
                 goals_real: list[float]) -> tuple[list[dict], list[dict]]:
    """
    Sépare les candidats en VP (vrai positif) et FP (faux positif).

    Un candidat est VP si son timestamp est dans les ±VP_MATCH_TOLERANCE_S
    d'un but réel. FP sinon.

    Un but réel est consommé par le premier candidat qui le matche
    (fenêtre temporelle, pas unicité stricte — on est post-Gemini).
    """
    real_times = list(goals_real) if goals_real else []
    vp, fp = [], []

    for ev in candidates:
        t = ev.get("time", 0.0)
        matched = False
        for rt in real_times:
            if abs(t - rt) <= VP_MATCH_TOLERANCE_S:
                matched = True
                break
        if matched:
            vp.append(ev)
        else:
            fp.append(ev)

    return vp, fp


def _compute_geo_stats(events: list[dict]) -> dict:
    """Calcule les stats géométriques agrégées sur un groupe d'events."""
    n = len(events)
    if n == 0:
        return {
            "in_goal_pct":     0.0,
            "in_penalty_pct":  0.0,
            "dist_goal_mean":  None,
            "dist_goal_median":None,
            "dist_goal_std":   None,
        }

    geo_list = [e.get("geo", {}) for e in events]

    # in_goal / in_penalty — compter seulement quand la valeur est définie
    in_goal_vals   = [g["in_goal"]    for g in geo_list if g.get("in_goal")    is not None]
    in_pen_vals    = [g["in_penalty"] for g in geo_list if g.get("in_penalty") is not None]
    dist_vals      = [g["dist_goal_m"] for g in geo_list if g.get("dist_goal_m") is not None]

    ig_pct = sum(1 for v in in_goal_vals if v) / len(in_goal_vals) if in_goal_vals else 0.0
    ip_pct = sum(1 for v in in_pen_vals  if v) / len(in_pen_vals)  if in_pen_vals  else 0.0

    dg_mean   = _mean(dist_vals)
    dg_median = _median(dist_vals)
    dg_std    = _std(dist_vals)

    return {
        "in_goal_pct":      round(ig_pct, 3),
        "in_penalty_pct":   round(ip_pct, 3),
        "dist_goal_mean":   round(dg_mean,   2) if dg_mean   is not None else None,
        "dist_goal_median": round(dg_median, 2) if dg_median is not None else None,
        "dist_goal_std":    round(dg_std,    2) if dg_std    is not None else None,
    }


def _summarize_events(events: list[dict]) -> list[dict]:
    """Résumé compact de chaque event pour le rapport détail."""
    out = []
    for ev in events:
        t   = ev.get("time", 0.0)
        geo = ev.get("geo", {})
        out.append({
            "t":           f"{int(t//60):02d}:{int(t%60):02d}",
            "bx":          f"{geo.get('bx'):.3f}" if geo.get("bx") is not None else "?",
            "in_goal":     geo.get("in_goal"),
            "in_penalty":  geo.get("in_penalty"),
            "dist_goal_m": geo.get("dist_goal_m"),
            "source":      ev.get("source", ev.get("detected_from", "?")),
            "gemini":      ev.get("gemini_validated", False),
            "geo_debug":   ev.get("geo_debug", {}),
        })
    return out


# ── Stats de base (pas de dépendance numpy) ────────────────────────────────────

def _mean(vals: list[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None

def _median(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n//2 - 1] + s[n//2]) / 2

def _std(vals: list[float]) -> Optional[float]:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
