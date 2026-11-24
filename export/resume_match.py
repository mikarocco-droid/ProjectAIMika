# export/resume_match.py
# VIDÉO « RÉSUMÉ DU MATCH » et séquences pour « Composer mon résumé ».
# 21/11/2026 — Résumé de base : TOUS les buts + les occasions repérées par l'analyse (classification + journal Gemini).
# 22/11/2026 — Chaque séquence est gardée en petit fichier (sequences/), avec sequences.json. Elles comprennent aussi
#   des « AUTRES MOMENTS À VOIR », non retenus dans le résumé de base (décochés) : passages où Gemini décrit un tir, un
#   ballon contré ou une attaque dans la surface (descriptions déjà payées, 0 $ ; Warsage : 38 moments, dont 3 vraies
#   occasions absentes du résumé). L'utilisateur coche ce qu'il veut ; recomposer_resume() assemble DANS L'ORDRE DU
#   MATCH, sans réencoder (quelques secondes, 0 $). Les séquences ne se chevauchent jamais (préparées disjointes).
#
#   monter_resume_match(video, buts, occasions, sortie, autres=None, bornes=None) -> chemin du résumé de base (ou None)
#   recomposer_resume(dossier_sequences, ids_coches, sortie_mp4) -> chemin (ou None)
#   autres_moments(textes, deja, buts) -> [{"time_abs", "label"}]   (textes : [(heure_du_passage, phrase), ...])
import json
import os
import re
import shutil
import subprocess
import tempfile

AVANT_BUT_CONFIRME = 25.0     # s montrées avant un but confirmé (heure fiable à quelques secondes)
AVANT_BUT_POSSIBLE = 35.0     # s avant un but NON confirmé (heure parfois en retard)
APRES_BUT_CONFIRME = 12.0     # après un but confirmé (le ballon au fond, la célébration)
APRES_BUT_POSSIBLE = 20.0     # 21/11 : après un but NON confirmé — son heure peut être EN AVANCE (Warsage : 111:01 daté 110:52)
AVANT_OCCASION, APRES_OCCASION = 12.0, 8.0   # 21/11 : 6 → 8 s (heure d'une occasion parfois approximative)
ECART_FUSION = 3.0            # deux séquences à moins de 3 s l'une de l'autre n'en font qu'une
MAX_AUTRES = 40               # au plus 40 « autres moments » proposés
# 23/11 : durée MAXIMALE du résumé de base (Baelen : 44 séquences, 21 min). Les buts entrent toujours ; puis les occasions
# par ordre de priorité de leur type ; celles qui ne tiennent pas passent dans les « autres moments » (décochées) :
# rien n'est perdu, l'utilisateur peut les rajouter. SCOUTIA_RESUME_MAX_MIN=0 → pas de limite.
RESUME_MAX_MIN = float(os.environ.get("SCOUTIA_RESUME_MAX_MIN", "12"))
_PRIORITE = {"poteau": 0, "sauvé sur la ligne": 0, "arrêt": 1, "tir à côté": 2, "occasion": 3, "tir contré": 4}

# Mots des descriptions de Gemini qui signalent un moment à proposer (en plus du résumé de base)
_AUTRES_MOTS = [("ballon contré", r"\bblock(?:ed|s)\b|\bdeflect"),
                ("tir", r"\bshot\b|\bshoots\b|\bstrikes?\b|attempt on goal|header towards goal"),
                ("attaque dans la surface", r"penalty (?:area|box)|into the box|in the box|six[- ]yard")]
_AUTRES_NEG = re.compile(r"\bno (shot|save|chance|attempt)|without (a |any )?shot|no dangerous|harmless", re.I)
_LIBELLES = {"SAVE": "arrêt", "WIDE": "tir à côté", "WOODWORK": "poteau", "POST": "poteau", "BLOCKED": "tir contré",
             "LINE_CLEARANCE": "sauvé sur la ligne", "Occasion": "occasion"}


def _ffmpeg():
    """Chemin de ffmpeg : celui du système, sinon celui fourni par le paquet Python imageio-ffmpeg (pratique sous Windows,
    sans droits d'administrateur : python -m pip install imageio-ffmpeg). None si aucun."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def autres_moments(textes, deja=(), buts=()):
    """Moments « à voir » tirés des phrases de Gemini (déjà payées) : ni dans le résumé de base, ni près d'un but."""
    vus = []
    for t_ref, phrase in textes or []:
        for ph in re.split(r"(?<=[.;])\s+", str(phrase or "")):
            if _AUTRES_NEG.search(ph):
                continue
            hh = [int(a) * 60 + int(b) for a, b in re.findall(r"(\d{1,3}):(\d{2})", ph)]
            hh = [h for h in hh if float(t_ref) - 35 <= h <= float(t_ref) + 32]
            if not hh:
                continue
            for lab, mo in _AUTRES_MOTS:
                if re.search(mo, ph, re.I):
                    vus.append((float(max(hh)), lab))
                    break
    out = []
    for t, lab in sorted(vus):
        if any(abs(t - float(d)) <= 15 for d in deja) or any(abs(t - float(b)) <= 15 for b in buts):
            continue
        if out and t - out[-1]["time_abs"] <= 15:
            continue
        out.append({"time_abs": t, "label": lab})
    return out[:MAX_AUTRES]


def _fusion(f):
    f = sorted(f, key=lambda x: x[0])
    out = []
    for a, b, e in f:
        if out and a <= out[-1][1] + ECART_FUSION:
            out[-1][1] = max(out[-1][1], b)
            if e.startswith("but") and not out[-1][2].startswith("but"):
                out[-1][2] = e
        else:
            out.append([a, b, e])
    return out


def fenetres_resume(buts, occasions, duree_video=None):
    """Séquences du résumé de base (début, fin, étiquette), triées et fusionnées."""
    f = []
    for b in buts or []:
        t = float(b["time_abs"])
        conf = str(b.get("statut") or b.get("status") or "").startswith("confirm")
        f.append([t - (AVANT_BUT_CONFIRME if conf else AVANT_BUT_POSSIBLE),
                  t + (APRES_BUT_CONFIRME if conf else APRES_BUT_POSSIBLE), "but" if conf else "but possible"])
    for o in occasions or []:
        t = float(o["time_abs"])
        f.append([t - AVANT_OCCASION, t + APRES_OCCASION,
                  _LIBELLES.get(str(o.get("label")), str(o.get("label") or "occasion"))])
    f = [[max(0.0, a), min(b, duree_video) if duree_video else b, e] for a, b, e in f]
    return [tuple(x) for x in _fusion(f) if x[1] - x[0] >= 2.0]


def _decouper(video, a, b, m, copie):
    if copie:      # qualité identique à l'origine (aucune recompression) ; départ sur l'image-clé précédente
        cmd = [_ffmpeg() or "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-i", video, "-t", f"{b - a:.2f}",
               "-c", "copy", "-avoid_negative_ts", "make_zero", m]
    else:
        cmd = [_ffmpeg() or "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-i", video, "-t", f"{b - a:.2f}",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p", m]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(m), (r.stderr or "")[:80]


def _assembler(fichiers, sortie_mp4):
    tmp = tempfile.mkdtemp(prefix="resume_")
    liste = os.path.join(tmp, "liste.txt")
    with open(liste, "w") as fl:
        fl.writelines(f"file '{os.path.abspath(m)}'\n" for m in fichiers)
    r = subprocess.run([_ffmpeg() or "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", liste, "-c", "copy",
                        "-movflags", "+faststart", sortie_mp4], capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return sortie_mp4 if r.returncode == 0 and os.path.exists(sortie_mp4) else None


def monter_resume_match(video, buts, occasions, sortie, nom="resume_match.mp4", duree_video=None, copie=None,
                        autres=None, bornes=None):
    """Résumé de base + toutes les séquences (dont « autres moments », décochés) dans sortie/sequences/."""
    if copie is None:
        copie = os.environ.get("SCOUTIA_RESUME_COPIE", "1").strip() != "0"
    if not _ffmpeg():
        print("  [RÉSUMÉ MATCH] ffmpeg absent : résumé non produit")
        return None
    base = fenetres_resume(buts, occasions, duree_video)
    trop = []
    if RESUME_MAX_MIN > 0 and sum(b - a for a, b, _ in base) > RESUME_MAX_MIN * 60:
        garde_ = [x for x in base if x[2].startswith("but")]
        reste_ = sorted((x for x in base if not x[2].startswith("but")), key=lambda x: (_PRIORITE.get(x[2], 3), x[0]))
        tot = sum(b - a for a, b, _ in garde_)
        for x in reste_:
            if tot + (x[1] - x[0]) <= RESUME_MAX_MIN * 60:
                garde_.append(x); tot += x[1] - x[0]
            else:
                trop.append(x)
        base = sorted(garde_)
        print(f"  [RÉSUMÉ MATCH] limité à {RESUME_MAX_MIN:.0f} min : {len(trop)} occasion(s) déplacée(s) dans les « autres moments »")
    # autres moments : jamais à cheval sur une séquence du résumé de base (séquences disjointes → recomposition simple)
    aut = []
    for o in autres or []:
        t = float(o["time_abs"])
        a, b = max(0.0, t - AVANT_OCCASION), t + APRES_OCCASION
        if duree_video:
            b = min(b, duree_video)
        if any(a <= fb + ECART_FUSION and b >= fa - ECART_FUSION for fa, fb, _ in base):
            continue
        aut.append([a, b, str(o.get("label") or "moment")])
    aut = _fusion(aut + [list(x) for x in trop])
    if not base and not aut:
        print("  [RÉSUMÉ MATCH] aucune séquence (ni but ni occasion)")
        return None
    dos = os.path.join(sortie, "sequences")
    os.makedirs(dos, exist_ok=True)
    ko1 = (bornes or {}).get("ko1")
    try:
        from export.rapport_client import minute_de_match
    except Exception:
        minute_de_match = None
    seqs = []
    toutes = sorted([(a, b, e, True) for a, b, e in base] + [(a, b, e, False) for a, b, e in aut])
    for k, (a, b, e, dans) in enumerate(toutes):
        nom_s = f"s{k:03d}.mp4"
        ok, err = _decouper(video, a, b, os.path.join(dos, nom_s), copie)
        if not ok:
            print(f"  [RÉSUMÉ MATCH] séquence {k + 1} ignorée ({err})")
            continue
        t_action = b - (APRES_BUT_POSSIBLE if e == "but possible" else APRES_BUT_CONFIRME if e == "but" else APRES_OCCASION)
        minute = (minute_de_match(t_action, ko1, bornes) if (minute_de_match and ko1 is not None)
                  else f"{int(t_action) // 60}:{int(t_action) % 60:02d}")
        seqs.append({"id": f"s{k:03d}", "fichier": nom_s, "debut": round(a, 1), "fin": round(b, 1), "minute": minute,
                     "type": e, "dans_resume": dans})
    json.dump(seqs, open(os.path.join(dos, "sequences.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    out = recomposer_resume(dos, [s["id"] for s in seqs if s["dans_resume"]], os.path.join(sortie, nom))
    fmt = lambda s: f"{int(s) // 60}:{int(s) % 60:02d}"
    n_b = sum(1 for s in seqs if s["dans_resume"])
    print(f"  [RÉSUMÉ MATCH] {n_b} séquence(s), {sum(s['fin'] - s['debut'] for s in seqs if s['dans_resume']) / 60:.1f} min "
          f"→ {out} | + {len(seqs) - n_b} « autre(s) moment(s) à voir » (décochés) dans {dos}")
    for s in seqs:
        print(f"     {'✔' if s['dans_resume'] else '·'} {fmt(s['debut'])} → {fmt(s['fin'])}  {s['minute']:>6}  {s['type']}")
    return out


def recomposer_resume(dossier_sequences, ids_coches, sortie_mp4):
    """Assemble les séquences cochées DANS L'ORDRE DU MATCH (heure de début), sans réencoder."""
    try:
        seqs = json.load(open(os.path.join(dossier_sequences, "sequences.json"), encoding="utf-8"))
    except Exception:
        return None
    voulus = set(ids_coches or [])
    choix = sorted((s for s in seqs if s["id"] in voulus), key=lambda s: s["debut"])
    fichiers = [os.path.join(dossier_sequences, s["fichier"]) for s in choix
                if os.path.exists(os.path.join(dossier_sequences, s["fichier"]))]
    if not fichiers:
        return None
    return _assembler(fichiers, sortie_mp4)
