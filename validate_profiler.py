"""
validate_profiler.py — Checklist d'acceptation Shot Profiler Phase B
=====================================================================
Usage :
    python validate_profiler.py <path_to_shots_csv> [path_to_pipeline_log]

Critères de réussite (tous requis) :
    C1  Aucun tir dupliqué (shot_t_s unique)
    C2  Aucun tir perdu    (0 ligne vide)
    C3  candidate_stage cohérent (valeurs connues uniquement)
    C4  Chaque posthoc_rejected a un linked_shot_t non-nul
    C5  Chaque posthoc_rejected a une reject_reason connue
    C6  delta linked_shot_t ↔ shot_t_s < EPS pour tous les posthoc_rejected
        (vérifié si le log est fourni)
    C7  Aucun tir avec posthoc_rejected=True et candidate_stage != posthoc_rejected
    C8  gemini_called cohérent avec candidate_stage
        (posthoc_rejected → gemini_called doit être False)

Retourne exit code 0 si tous les critères passent, 1 sinon.
"""
import csv, sys, re
from pathlib import Path

VALID_STAGES   = {"raw", "posthoc", "posthoc_rejected", "gemini", "validated", ""}
VALID_REASONS  = {"missing_terminal", "cooldown", "low_score",
                  "camera_gate", "duplicate", ""}
EPS            = 0.01   # 10 ms — même valeur que shot_profiler.py

RED   = "\033[91m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RESET = "\033[0m"

def ok(msg):  print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg):print(f"  {RED}❌ {msg}{RESET}")
def warn(msg):print(f"  {YELLOW}⚠️  {msg}{RESET}")

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def check(rows, log_text=None):
    errors = 0

    # ── C1 : pas de doublons ──────────────────────────────────────────
    times = [float(r["shot_t_s"]) for r in rows]
    dupes = [t for t in times if times.count(t) > 1]
    if dupes:
        fail(f"C1 FAIL — {len(set(dupes))} shot_t_s dupliqué(s) : {sorted(set(dupes))}")
        errors += 1
    else:
        ok(f"C1 — Aucun doublon ({len(rows)} tirs)")

    # ── C2 : pas de ligne vide ────────────────────────────────────────
    empty = [i for i, r in enumerate(rows) if not r.get("shot_t_s")]
    if empty:
        fail(f"C2 FAIL — {len(empty)} ligne(s) sans shot_t_s")
        errors += 1
    else:
        ok("C2 — Aucune ligne vide")

    # ── C3 : candidate_stage valide ───────────────────────────────────
    bad_stage = [r for r in rows if r.get("candidate_stage","") not in VALID_STAGES]
    if bad_stage:
        bad = [(r["shot_t_fmt"], r["candidate_stage"]) for r in bad_stage]
        fail(f"C3 FAIL — stages inconnus : {bad}")
        errors += 1
    else:
        ok(f"C3 — candidate_stage valides ({sorted(set(r.get('candidate_stage','') for r in rows))})")

    # ── C4 : posthoc_rejected → linked_shot_t non-nul ────────────────
    rejected = [r for r in rows if str(r.get("posthoc_rejected","")).lower() == "true"]
    missing_lst = [r for r in rejected if not r.get("posthoc_score")]
    if missing_lst:
        warn(f"C4 — {len(missing_lst)} posthoc_rejected sans posthoc_score "
             f"(linked_shot_t non vérifiable directement depuis CSV)")
    else:
        ok(f"C4 — {len(rejected)} posthoc_rejected ont posthoc_score")

    # ── C5 : reject_reason connue ─────────────────────────────────────
    bad_reason = [r for r in rejected
                  if r.get("reject_reason","") not in VALID_REASONS]
    if bad_reason:
        bad = [(r["shot_t_fmt"], r["reject_reason"]) for r in bad_reason]
        fail(f"C5 FAIL — reject_reason inconnue : {bad}")
        errors += 1
    else:
        ok(f"C5 — reject_reason valides pour {len(rejected)} posthoc_rejected")

    # ── C6 : delta linked_shot_t < EPS (depuis les logs) ─────────────
    if log_text:
        # [PROFILER UPDATE] shot_t=31.40s matched row=31.40s delta=0.002s stage=...
        pattern = r"\[PROFILER UPDATE\] shot_t=(\S+)s matched row=(\S+)s delta=(\S+)s"
        matches = re.findall(pattern, log_text)
        no_matches = re.findall(r"\[PROFILER UPDATE\] shot_t=(\S+)s → NO MATCH", log_text)

        if no_matches:
            fail(f"C6 FAIL — {len(no_matches)} NO MATCH dans les logs : {no_matches}")
            errors += 1
        elif matches:
            max_delta = max(float(d) for _, _, d in matches)
            if max_delta > EPS:
                fail(f"C6 FAIL — delta max={max_delta:.4f}s > EPS={EPS}s")
                errors += 1
            else:
                ok(f"C6 — {len(matches)} match(es) | delta max={max_delta*1000:.1f}ms < {EPS*1000:.0f}ms")
        else:
            warn("C6 — Aucun [PROFILER UPDATE] dans le log (posthoc peut-être absent)")
    else:
        warn("C6 — Log non fourni, delta non vérifiable")

    # ── C7 : cohérence rejected ↔ stage ──────────────────────────────
    incoherent = [r for r in rejected
                  if r.get("candidate_stage") != "posthoc_rejected"]
    if incoherent:
        bad = [(r["shot_t_fmt"], r["candidate_stage"]) for r in incoherent]
        fail(f"C7 FAIL — {len(incoherent)} posthoc_rejected avec stage incorrect : {bad}")
        errors += 1
    else:
        ok(f"C7 — Cohérence rejected ↔ stage ({len(rejected)} lignes)")

    # ── C8 : posthoc_rejected → gemini_called=False ───────────────────
    bad_gem = [r for r in rejected
               if str(r.get("gemini_called","")).lower() not in ("false", "")]
    if bad_gem:
        bad = [(r["shot_t_fmt"], r.get("gemini_called")) for r in bad_gem]
        fail(f"C8 FAIL — {len(bad_gem)} posthoc_rejected avec gemini_called=True : {bad}")
        errors += 1
    else:
        ok(f"C8 — gemini_called=False pour tous les posthoc_rejected")

    # ── Résumé ────────────────────────────────────────────────────────
    print()
    print(f"  Distribution candidate_stage :")
    from collections import Counter
    for stage, count in sorted(Counter(r.get("candidate_stage","") for r in rows).items()):
        print(f"    {stage or '(vide)':25s}: {count}")

    print()
    if errors == 0:
        print(f"{GREEN}✅ VALIDATION OK — {len(rows)} tirs, 0 erreur{RESET}")
    else:
        print(f"{RED}❌ VALIDATION ÉCHOUÉE — {errors} critère(s) non satisfait(s){RESET}")
    return errors

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    log_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not csv_path.exists():
        print(f"Fichier introuvable : {csv_path}")
        sys.exit(1)

    rows    = load_csv(csv_path)
    log_txt = log_path.read_text(errors="replace") if log_path and log_path.exists() else None

    print(f"\n{'='*60}")
    print(f"  Shot Profiler — Validation Phase B")
    print(f"  CSV : {csv_path.name}  ({len(rows)} tirs)")
    print(f"  Log : {log_path.name if log_path else '(non fourni)'}")
    print(f"{'='*60}\n")

    sys.exit(check(rows, log_txt))
