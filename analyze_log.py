#!/usr/bin/env python3
# analyze_log.py
# ─────────────────────────────────────────────────────────────────
# Script d'analyse du pipeline.log ScoutIA
# Usage : python analyze_log.py [chemin/vers/pipeline.log]
# ─────────────────────────────────────────────────────────────────

import sys
import re
from collections import defaultdict

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/pipeline.log"

try:
    with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Log introuvable : {LOG_PATH}")
    sys.exit(1)

text = "".join(lines)

SEP = "=" * 55

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def find_all(pattern, flags=0):
    return re.findall(pattern, text, flags)

def find_first(pattern, default="—", flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1) if m else default

def fmt_time(sec):
    try:
        s = float(sec)
        return f"{int(s//60):02d}:{int(s%60):02d}"
    except Exception:
        return str(sec)


# ─────────────────────────────────────────
# 1. RÉSUMÉ GLOBAL
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("  RÉSUMÉ GLOBAL")
print(SEP)

duration    = find_first(r"Durée vidéo\s*:\s*(\S+)")
total_time  = find_first(r"Temps\s*:\s*([\d.]+)s")
goals       = find_first(r"Buts\s*:\s*(\d+)")
shots       = find_first(r"Tirs\s*:\s*(\d+)")
xg          = find_first(r"xG total\s*:\s*([\d.]+)")
passes      = find_first(r"Passes\s*:\s*(\d+)")
players     = find_first(r"Joueurs\s*:\s*(\d+)")
highlights  = find_first(r"Highlights\s*:\s*(\d+)")
mvp         = find_first(r"MVP\s*:\s*(.+)")

print(f"  Durée vidéo   : {duration}")
print(f"  Temps run     : {total_time}s ({float(total_time)/60:.1f} min)" if total_time != "—" else "  Temps run     : —")
print(f"  Buts          : {goals}")
print(f"  Tirs          : {shots}")
print(f"  xG total      : {xg}")
print(f"  Passes        : {passes}")
print(f"  Joueurs       : {players}")
print(f"  Highlights    : {highlights}")
print(f"  MVP           : {mvp}")


# ─────────────────────────────────────────
# 2. BUTS DÉTECTÉS
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("  BUTS DÉTECTÉS")
print(SEP)

# Pattern debug buts
goal_blocks = re.finditer(
    r"-- But a (\d{2}:\d{2}) --.*?"
    r"Buteur\s*:\s*(.+?)\n.*?"
    r"Source\s*:\s*(.+?)\n.*?"
    r"gemini\s*:\s*(\S+).*?"
    r"conf\s*:\s*([\d.]+).*?"
    r"xG\s*:\s*([\d.]+)",
    text, re.DOTALL
)

goal_list = []
for m in goal_blocks:
    goal_list.append({
        "time":    m.group(1),
        "buteur":  m.group(2).strip(),
        "source":  m.group(3).strip(),
        "gemini":  m.group(4).strip(),
        "conf":    float(m.group(5)),
        "xg":      float(m.group(6)),
    })

# Buts attendus : 02:14 et 09:44
EXPECTED = ["02:14", "09:44"]

if not goal_list:
    print("  ⚠️  Aucun but détecté dans le log")
else:
    for g in goal_list:
        match_expected = any(abs(
            int(g["time"].split(":")[0])*60 + int(g["time"].split(":")[1]) -
            int(e.split(":")[0])*60 - int(e.split(":")[1])
        ) <= 10 for e in EXPECTED)
        flag = "✅ VRAI BUT" if match_expected else "⚠️  INCONNU"
        print(f"  {flag}  t={g['time']}  {g['buteur']}")
        print(f"           source={g['source']}  gemini={g['gemini']}  conf={g['conf']:.2f}  xG={g['xg']:.3f}")

# Vérifier vrais buts manquants
print()
for exp in EXPECTED:
    exp_sec = int(exp.split(":")[0])*60 + int(exp.split(":")[1])
    found = any(
        abs(int(g["time"].split(":")[0])*60 + int(g["time"].split(":")[1]) - exp_sec) <= 10
        for g in goal_list
    )
    if found:
        print(f"  ✅  But attendu {exp} : DÉTECTÉ")
    else:
        print(f"  ❌  But attendu {exp} : MANQUANT")


# ─────────────────────────────────────────
# 3. GEMINI — CONCORDANCE HIGH_CONF
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("  GEMINI — VALIDATION")
print(SEP)

# Candidats
n_candidates = find_first(r"Validation Gemini\s*:\s*(\d+) candidats")
n_high       = find_first(r"\((\d+) high_conf >= 0.90\)")
validated    = find_first(r"Gemini\s*:\s*(\d+) valid")
corrected    = find_first(r"(\d+) corrig")
removed      = find_first(r"(\d+) supprim")

print(f"  Candidats     : {n_candidates} ({n_high} high_conf >= 0.90)")
print(f"  Validés       : {validated}  |  Corrigés : {corrected}  |  Supprimés : {removed}")

# Détail concordance high_conf
gemini_lines = re.finditer(
    r"Gemini but (\d{2}:\d{2}) → type=(\S+) conf=([\d.]+) high_conf=(\S+) concordance=(\S+)",
    text
)
high_conf_results = []
for m in gemini_lines:
    high_conf_results.append({
        "time":        m.group(1),
        "type":        m.group(2),
        "conf":        float(m.group(3)),
        "high_conf":   m.group(4) == "True",
        "concordance": m.group(5) == "True",
    })

if high_conf_results:
    print()
    n_hc     = sum(1 for r in high_conf_results if r["high_conf"])
    n_concord = sum(1 for r in high_conf_results if r["high_conf"] and r["concordance"])
    print(f"  High conf buts : {n_hc}")
    if n_hc > 0:
        pct = n_concord / n_hc * 100
        print(f"  Concordance    : {n_concord}/{n_hc} = {pct:.0f}%")
        if pct >= 99:
            print("  → Skip Gemini possible sur high_conf (à vérifier sur 100+ matchs)")
        elif pct >= 95:
            print("  → Encore trop risqué pour skip (< 99%)")
        else:
            print("  → Skip impossible — high_conf non fiable")

    print()
    for r in high_conf_results:
        hc_flag = "🔵 high" if r["high_conf"] else "⚪ norm"
        cc_flag = "✅" if r["concordance"] else "❌"
        print(f"  {cc_flag} {hc_flag}  t={r['time']}  type={r['type']}  conf={r['conf']:.2f}")


# ─────────────────────────────────────────
# 4. BALL TRACKER — MODE ÉCO
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("  BALL TRACKER & MODE ÉCO")
print(SEP)

eco_pct  = find_first(r"Mode éco\s*:\s*(\d+)%")
bt_speed = find_first(r"speed_base=([\d.]+)")
posthoc  = find_first(r"goal_posthoc_v9\.5.*?speed_base=([\d.]+)")

lost_frames = find_all(r"lost_frames.*?(\d+)")
max_lost_reached = [int(x) for x in lost_frames if int(x) >= 10]

print(f"  Mode éco      : {eco_pct}% frames skippées")
print(f"  Speed base    : {bt_speed} px/frame")

# Compter les buts posthoc
posthoc_goals = find_all(r"GOAL ([\d.]+)s \| score=([\d.]+) \| stuck=(\d+) \| rebound=(\w+)")
if posthoc_goals:
    print(f"\n  Goal posthoc détectés : {len(posthoc_goals)}")
    for g in posthoc_goals:
        t, score, stuck, rebound = g
        print(f"    t={fmt_time(t)}  score={score}  stuck={stuck}  rebound={rebound}")
else:
    print("  Goal posthoc : aucun détecté")

# Buts rejetés
rejected = find_all(r"goal REJETÉ à t=([\d.]+)s")
if rejected:
    non_zero = [t for t in rejected if float(t) > 0]
    print(f"\n  Buts rejetés (filtre xG) : {len(rejected)} ({len(non_zero)} hors t=0)")
    for t in non_zero[:5]:
        print(f"    t={fmt_time(t)}")


# ─────────────────────────────────────────
# 5. TIRS — TOP 8
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("  TIRS (TOP 8)")
print(SEP)

shot_lines = re.finditer(
    r"(\d{2}:\d{2})\s+(\S+)\s+xg=([\d.]+)\s+on_target=(\S+)(\s+\[synthetique\])?",
    text
)
shot_list = []
for m in shot_lines:
    shot_list.append({
        "time":       m.group(1),
        "player":     m.group(2),
        "xg":         float(m.group(3)),
        "on_target":  m.group(4),
        "synthetic":  bool(m.group(5)),
    })

if shot_list:
    for s in shot_list[:8]:
        syn = " [synth]" if s["synthetic"] else ""
        ot  = "✓" if s["on_target"] == "True" else "·"
        print(f"  {ot} t={s['time']}  {s['player']:8}  xG={s['xg']:.3f}{syn}")
else:
    print("  Aucun tir dans le log debug")


# ─────────────────────────────────────────
# 6. JOUEURS & JERSEY MAP
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("  JOUEURS DÉTECTÉS")
print(SEP)

jersey_lines = re.finditer(r"(\d+) -> #(\S+)", text)
jersey_map   = {}
for m in jersey_lines:
    jersey_map[m.group(1)] = m.group(2)

print(f"  Jersey map : {len(jersey_map)} joueurs identifiés")
if jersey_map:
    for pid, jersey in list(jersey_map.items())[:15]:
        print(f"    ID {pid:6} → #{jersey}")

# Stats joueurs
player_stats = re.finditer(
    r"ID\s+(\S+)\s+->\s+#(\S+)\s+\|\s+touches=\s*(\d+)\s+\|\s+tirs=\s*(\d+)\s+\|\s+buts=(\d+)",
    text
)
pstats = list(player_stats)
if pstats:
    print(f"\n  Stats joueurs (top 10 par touches) :")
    for m in pstats[:10]:
        buts_flag = " ⚽" if int(m.group(5)) > 0 else ""
        print(f"    ID {m.group(1):6} #{m.group(2):4} "
              f"touches={m.group(3):3} tirs={m.group(4):2} buts={m.group(5)}{buts_flag}")

    # Vérifier si #11 est présent
    jerseys_found = [m.group(2) for m in re.finditer(
        r"ID\s+\S+\s+->\s+#(\S+)\s+\|\s+touches", text)]
    if "11" in jerseys_found:
        print("\n  ✅ Joueur #11 détecté dans les stats")
    else:
        print("\n  ❌ Joueur #11 absent des stats")
        # Vérifier dans jersey_map
        if "11" in jersey_map.values():
            print("     → #11 dans jersey_map mais pas dans stats (touches < 15 ?)")
        else:
            print("     → #11 pas dans jersey_map non plus (non détecté par OCR/Gemini)")


# ─────────────────────────────────────────
# 7. ERREURS & WARNINGS
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("  ERREURS & WARNINGS")
print(SEP)

errors   = [l.strip() for l in lines if " ERROR " in l or "Traceback" in l or "Exception" in l]
warnings = [l.strip() for l in lines if "WARNING" in l or "⚠️" in l or "ignoré" in l.lower()]

if errors:
    print(f"  ❌ {len(errors)} erreur(s) :")
    for e in errors[:5]:
        print(f"    {e[:100]}")
else:
    print("  ✅ Aucune erreur")

if warnings:
    print(f"\n  ⚠️  {len(warnings)} warning(s) :")
    for w in warnings[:5]:
        print(f"    {w[:100]}")
else:
    print("  ✅ Aucun warning")


# ─────────────────────────────────────────
# 8. VERDICT FINAL
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("  VERDICT")
print(SEP)

n_goals_found = len(goal_list)
n_expected    = 2
n_found_exp   = sum(1 for exp in EXPECTED for g in goal_list
                    if abs(int(g["time"].split(":")[0])*60 + int(g["time"].split(":")[1]) -
                           int(exp.split(":")[0])*60 - int(exp.split(":")[1])) <= 10)

if n_found_exp == n_expected and n_goals_found <= 5:
    print("  ✅ PARFAIT — 2 vrais buts détectés, peu de faux positifs")
elif n_found_exp == n_expected:
    print(f"  ✅ Vrais buts OK mais {n_goals_found} buts total — vérifier les faux positifs")
elif n_found_exp == 1:
    print(f"  ⚠️  1/2 vrais buts détectés — amélioration partielle")
elif n_found_exp == 0 and n_goals_found > 0:
    print(f"  ❌ 0/2 vrais buts — {n_goals_found} faux positifs")
elif n_goals_found == 0:
    print("  ❌ 0 but détecté — pipeline bloqué en amont")
else:
    print(f"  ⚠️  {n_found_exp}/2 vrais buts | {n_goals_found} total")

print(f"\n  Fichier analysé : {LOG_PATH}")
print(f"  Lignes log      : {len(lines)}")
print()
