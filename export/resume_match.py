# export/resume_match.py
# 21/11/2026 — VIDÉO « RÉSUMÉ DU MATCH » livrée au client : TOUS les buts (confirmés et possibles) + toutes les
# occasions repérées par l'analyse (arrêts, tirs à côté, sur le poteau, contrés : les « temps forts » de la
# classification Gemini, déjà payés), dans l'ordre du match. Remplace l'ancien reel qui ne prenait que les buts
# (Warsage : 5 séquences, aucune occasion).
#   monter_resume_match(video, buts, occasions, sortie) -> chemin du MP4 (ou None)
#   buts      : [{"time_abs": s, "statut": "confirmé" | "à_vérifier"}, ...]
#   occasions : [{"time_abs": s, "label": "..."}, ...]  (heure de l'action, en secondes depuis le début de la vidéo)
import os
import shutil
import subprocess
import tempfile

AVANT_BUT_CONFIRME = 25.0     # s montrées avant un but confirmé (heure fiable à quelques secondes)
AVANT_BUT_POSSIBLE = 35.0     # s avant un but NON confirmé (heure parfois en retard)
APRES_BUT = 8.0               # la célébration
AVANT_OCCASION, APRES_OCCASION = 12.0, 6.0
ECART_FUSION = 3.0            # deux séquences à moins de 3 s l'une de l'autre n'en font qu'une


def fenetres_resume(buts, occasions, duree_video=None):
    """Séquences (début, fin, étiquette) triées et fusionnées."""
    f = []
    for b in buts or []:
        t = float(b["time_abs"])
        conf = str(b.get("statut") or b.get("status") or "").startswith("confirm")
        f.append([t - (AVANT_BUT_CONFIRME if conf else AVANT_BUT_POSSIBLE), t + APRES_BUT,
                  "but" if conf else "but possible"])
    for o in occasions or []:
        t = float(o["time_abs"])
        f.append([t - AVANT_OCCASION, t + APRES_OCCASION, o.get("label") or "occasion"])
    f = sorted(([max(0.0, a), min(b, duree_video) if duree_video else b, e] for a, b, e in f), key=lambda x: x[0])
    out = []
    for a, b, e in f:
        if out and a <= out[-1][1] + ECART_FUSION:
            out[-1][1] = max(out[-1][1], b)
            if e.startswith("but") and not out[-1][2].startswith("but"):
                out[-1][2] = e
        else:
            out.append([a, b, e])
    return [tuple(x) for x in out if x[1] - x[0] >= 2.0]


def monter_resume_match(video, buts, occasions, sortie, nom="resume_match.mp4", duree_video=None):
    if not shutil.which("ffmpeg"):
        print("  [RÉSUMÉ MATCH] ffmpeg absent : résumé non produit")
        return None
    seq = fenetres_resume(buts, occasions, duree_video)
    if not seq:
        print("  [RÉSUMÉ MATCH] aucune séquence (ni but ni occasion)")
        return None
    os.makedirs(sortie, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="resume_")
    morceaux = []
    for i, (a, b, e) in enumerate(seq):
        m = os.path.join(tmp, f"s{i:03d}.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-i", video, "-t", f"{b - a:.2f}",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k",
                            "-pix_fmt", "yuv420p", m], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(m):
            morceaux.append(m)
        else:
            print(f"  [RÉSUMÉ MATCH] séquence {i + 1} ignorée ({(r.stderr or '')[:80]})")
    if not morceaux:
        return None
    liste = os.path.join(tmp, "liste.txt")
    with open(liste, "w") as fl:
        fl.writelines(f"file '{m}'\n" for m in morceaux)
    out = os.path.join(sortie, nom)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", liste, "-c", "copy", out],
                       capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"  [RÉSUMÉ MATCH] assemblage impossible ({(r.stderr or '')[:80]})")
        return None
    fmt = lambda s: f"{int(s) // 60}:{int(s) % 60:02d}"
    print(f"  [RÉSUMÉ MATCH] {len(morceaux)} séquence(s), {sum(b - a for a, b, _ in seq) / 60:.1f} min → {out}")
    for a, b, e in seq:
        print(f"     {fmt(a)} → {fmt(b)}  {e}")
    return out
