# bc4_calibration.py
# Persistance des observations BC4 VP/FP pour calibration du seuil world_score.
# Chaque run contribue des lignes au fichier bc4_calibration.json.
# Objectif : 10 VP + 10 FP pour décider du seuil définitif.

import json
import os
from datetime import datetime

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bc4_calibration.json"
)


def load(path=None):
    p = path or _DEFAULT_PATH
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"observations": [], "summary": {}}


def save(data, path=None):
    p = path or _DEFAULT_PATH
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [BC4 CAL] Sauvegarde échouée : {e}")


def record(
    video_name,
    t_sec,
    source,
    vp_fp,           # "VP" ou "FP" ou "unknown"
    in_goal,
    in_penalty,
    dist_goal,
    world_score,
    bx=None,
    desc="",
    path=None,
):
    """Ajoute une observation. vp_fp="unknown" si non encore validé terrain."""
    data = load(path)
    obs = {
        "ts":          datetime.now().isoformat()[:19],
        "video":       video_name,
        "t_sec":       round(float(t_sec), 1),
        "t_str":       f"{int(t_sec//60):02d}:{int(t_sec%60):02d}",
        "source":      source,
        "vp_fp":       vp_fp,
        "in_goal":     in_goal,
        "in_penalty":  in_penalty,
        "dist_goal":   round(float(dist_goal), 2) if dist_goal is not None else None,
        "world_score": round(float(world_score), 3) if world_score is not None else None,
        "bx":          round(float(bx), 3) if bx is not None else None,
        "desc":        desc[:80] if desc else "",
    }
    data["observations"].append(obs)
    _update_summary(data)
    save(data, path)
    return obs


def _update_summary(data):
    obs = data["observations"]
    vp  = [o for o in obs if o["vp_fp"] == "VP" and o["world_score"] is not None]
    fp  = [o for o in obs if o["vp_fp"] == "FP" and o["world_score"] is not None]

    def stats(lst):
        if not lst:
            return {}
        scores = [o["world_score"] for o in lst]
        in_g   = [o["in_goal"] for o in lst if o["in_goal"] is not None]
        return {
            "n":           len(lst),
            "ws_min":      round(min(scores), 3),
            "ws_max":      round(max(scores), 3),
            "ws_mean":     round(sum(scores) / len(scores), 3),
            "in_goal_true_pct": round(sum(1 for g in in_g if g) / max(len(in_g), 1) * 100, 1),
        }

    data["summary"] = {
        "total":    len(obs),
        "VP":       stats(vp),
        "FP":       stats(fp),
        "unknown":  len([o for o in obs if o["vp_fp"] == "unknown"]),
    }

    # Suggestion de seuil si assez de données
    if vp and fp:
        vp_min = min(o["world_score"] for o in vp)
        fp_max = max(o["world_score"] for o in fp)
        if vp_min > fp_max:
            data["summary"]["threshold_suggestion"] = round((vp_min + fp_max) / 2, 3)
            data["summary"]["threshold_clean_separation"] = True
        else:
            data["summary"]["threshold_suggestion"] = None
            data["summary"]["threshold_clean_separation"] = False


def print_summary(path=None):
    data = load(path)
    s = data.get("summary", {})
    obs = data.get("observations", [])
    print(f"  [BC4 CAL] {s.get('total', 0)} observations | "
          f"VP={s.get('VP', {}).get('n', 0)} FP={s.get('FP', {}).get('n', 0)} "
          f"unknown={s.get('unknown', 0)}")
    if s.get("VP"):
        v = s["VP"]
        print(f"  [BC4 CAL] VP  world_score : min={v['ws_min']} mean={v['ws_mean']} max={v['ws_max']} | in_goal={v['in_goal_true_pct']}%")
    if s.get("FP"):
        f = s["FP"]
        print(f"  [BC4 CAL] FP  world_score : min={f['ws_min']} mean={f['ws_mean']} max={f['ws_max']} | in_goal={f['in_goal_true_pct']}%")
    if s.get("threshold_suggestion"):
        print(f"  [BC4 CAL] ✅ Seuil suggéré : {s['threshold_suggestion']} (séparation nette VP/FP)")
    elif s.get("VP") and s.get("FP"):
        print(f"  [BC4 CAL] ⚠️  Pas de séparation nette VP/FP — ne pas activer le gate")
    # Tableau récent (10 dernières obs)
    if obs:
        print(f"  [BC4 CAL] Dernières observations :")
        for o in obs[-10:]:
            print(f"    {o['t_str']} {o['video'][:20]:20s} {o['vp_fp']:7s} "
                  f"in_goal={str(o['in_goal']):5s} ws={str(o['world_score']):5s} "
                  f"src={o['source'][:20]}")
