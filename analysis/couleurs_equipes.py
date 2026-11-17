# analysis/couleurs_equipes.py
# 16/11/2026 — COULEURS DES ÉQUIPES APPRISES UNE FOIS ET FIXÉES POUR TOUT LE MATCH (SCOUTIA_COULEURS_FIXES=1,
# désactivé par défaut).
#
# Constat (diag_equipes_caches.py, 16/11) : 16 mi-temps sur 26 ont une répartition des équipes ratée (plus petite
# équipe < 25 % des joueurs classés, ou > 15 % de « gardiens »), avec YOLO comme avec RF-DETR. Cause : chaque
# processus de suivi apprend les couleurs sur ses 50 premières images (≈ 8 s, PlayerReID._calibrate_teams) ; pour la
# 2e mi-temps, ces images commencent 30 s avant la reprise. Or le détecteur de coups d'envoi repose sur l'équilibre
# entre les équipes (Franchimont, suivi RF-DETR : buts de 65:28 et 99:29 manqués faute de bon candidat).
#
# Principe (proposé par l'utilisateur) : les équipes changent de côté à la mi-temps, PAS de couleur. On apprend les
# couleurs UNE fois, avant le suivi, sur plusieurs moments de la 1re mi-temps ; on VÉRIFIE ce qu'elles donnent sur
# d'AUTRES moments (avec la règle de classement du suivi) ; si c'est bon, elles servent aux deux mi-temps. Sinon on
# essaie d'autres moments (début de 2e mi-temps estimée), sinon rien n'est fixé (comportement d'avant).
#
# Repère de qualité (fixé avant, 16/11) : plus petite équipe ≥ 25 % des joueurs classés ET « gardien » ≤ 15 %.
# Les couleurs passent aux processus de suivi par un fichier JSON (SCOUTIA_COULEURS_EQUIPES_FICHIER), lu par
# PlayerReID à sa création. Aucun appel Gemini.
import json
import os

import numpy as np

SEUIL_PETITE_EQUIPE = 0.25
SEUIL_GARDIEN = 0.15


def _classer(colors, centroids):
    """Même règle que PlayerReID._infer_team (gardien si loin des deux couleurs, ambigu si écart < 0,3)."""
    c0, c1 = np.asarray(centroids[0], np.float32), np.asarray(centroids[1], np.float32)
    seuil_gk = float(np.linalg.norm(c0 - c1)) * 1.5
    n = {"0": 0, "1": 0, "gk": 0, "none": 0}
    for c in colors:
        d0, d1 = float(np.linalg.norm(c - c0)), float(np.linalg.norm(c - c1))
        if d0 > seuil_gk and d1 > seuil_gk:
            n["gk"] += 1
        elif abs(d0 - d1) < 0.3:
            n["none"] += 1
        else:
            n["0" if d0 < d1 else "1"] += 1
    tot = max(sum(n.values()), 1)
    eq = n["0"] + n["1"]
    petite = min(n["0"], n["1"]) / eq if eq else 0.0
    return {"equipe0": n["0"] / tot, "equipe1": n["1"] / tot, "gardien": n["gk"] / tot, "aucune": n["none"] / tot,
            "plus_petite_equipe": petite, "n": tot,
            "ok": bool(petite >= SEUIL_PETITE_EQUIPE and n["gk"] / tot <= SEUIL_GARDIEN)}


def _apprendre(colors):
    import cv2
    samples = np.asarray(colors, np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    _, _, centroids = cv2.kmeans(samples, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centroids


def _couleurs_aux_moments(video_path, instants_s, detecteur, extraire):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = []
    for t in instants_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        try:
            players, _ = detecteur.detect(frame)
        except Exception:
            continue
        for p in players or []:
            c = extraire(frame, p.get("bbox", [0, 0, 0, 0]))
            if c is not None and float(np.asarray(c)[:16].sum()) > 0.9:     # même filtre que la calibration du suivi
                out.append(np.asarray(c, np.float32))
    cap.release()
    return out


def _instants(debut_s, fin_s, n_moments, images_par_moment=5, pas_s=1.0):
    if fin_s <= debut_s:
        return []
    centres = np.linspace(debut_s, fin_s, n_moments)
    return [float(c + k * pas_s) for c in centres for k in range(images_par_moment)]


def _aligner(ref, c):
    """Range les deux couleurs de c dans le même ordre que ref (équipe 0 / équipe 1 identiques d'une tranche à
    l'autre : la lumière change progressivement, deux tranches voisines ont des couleurs proches)."""
    ref, c = np.asarray(ref, np.float32), np.asarray(c, np.float32)
    direct = np.linalg.norm(ref[0] - c[0]) + np.linalg.norm(ref[1] - c[1])
    croise = np.linalg.norm(ref[0] - c[1]) + np.linalg.norm(ref[1] - c[0])
    return c if direct <= croise else c[::-1].copy()


def _txt(q):
    return (f"équipe0 {q['equipe0']:.0%} | équipe1 {q['equipe1']:.0%} | gardien {q['gardien']:.0%} | "
            f"aucune {q['aucune']:.0%} | plus petite équipe {q['plus_petite_equipe']:.0%} → {'ok' if q['ok'] else 'MAUVAISE'}")


def _h(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def tranches_du_match(ko1_s, half_duration_min):
    """Tranches d'environ 15 min, 3 par mi-temps : (nom, mi-temps, début, fin, (a_appr, b_appr), (a_verif, b_verif)).
    Début/fin = période où la norme s'applique (temps absolu de la vidéo) ; les moments d'apprentissage et de
    vérification sont DIFFÉRENTS (décalés de ≈ 1 min) et pris à l'intérieur de la tranche. 2e mi-temps estimée :
    reprise ≈ KO1 + mi-temps + 17 min ; ses moments commencent à KO1 + mi-temps + 22 min (hors pause même longue)."""
    H = half_duration_min * 60
    k = ko1_s
    coupure = k + H + 5 * 60
    r2 = k + H + 17 * 60
    return [
        ("1re MT 0-15",  1, 0.0,          k + 15 * 60, (k + 1 * 60,  k + 13 * 60),  (k + 2 * 60,  k + 14 * 60)),
        ("1re MT 15-30", 1, k + 15 * 60,  k + 30 * 60, (k + 16 * 60, k + 28 * 60),  (k + 17 * 60, k + 29 * 60)),
        ("1re MT 30-fin", 1, k + 30 * 60, coupure,     (k + 31 * 60, k + H),        (k + 32 * 60, k + H + 60)),
        ("2e MT 0-15",   2, coupure,      r2 + 15 * 60, (k + H + 22 * 60, r2 + 14 * 60), (k + H + 23 * 60, r2 + 15 * 60)),
        ("2e MT 15-30",  2, r2 + 15 * 60, r2 + 30 * 60, (r2 + 16 * 60, r2 + 28 * 60), (r2 + 17 * 60, r2 + 29 * 60)),
        ("2e MT 30-fin", 2, r2 + 30 * 60, 1e9,          (r2 + 31 * 60, r2 + H),       (r2 + 32 * 60, r2 + H + 60)),
    ]


def calibrer_couleurs_match(video_path, ko1_s, half_duration_min, out_json, detecteur, duree_s=None):
    """16/11 (3) : couleurs des équipes PAR TRANCHE D'ENVIRON 15 MIN (3 par mi-temps), proposé par l'utilisateur :
    la lumière change pendant le match (Franchimont : « gardien » 1 % → 24 % entre la 30e et la 45e minute, jour →
    éclairage). Tout est fait AVANT le suivi : pour chaque tranche, couleurs apprises sur 4 moments et VÉRIFIÉES sur 4
    autres (repère : plus petite équipe ≥ 25 %, « gardien » ≤ 15 %), rangées dans le même ordre que la tranche
    précédente. Pendant le suivi, chaque image prend la norme de sa tranche (PlayerReID.appliquer_norme_couleurs) ;
    une tranche sans norme fiable prend celle de la tranche fiable la plus proche DE LA MÊME mi-temps ; une mi-temps
    sans aucune norme fiable garde la calibration d'avant. Écrit out_json et renvoie son chemin, ou None.
    detecteur : objet avec .detect(frame) → (players, ball), comme vision.detector.Detector."""
    from analysis.player_reid import PlayerReID
    extraire = PlayerReID(fps=6, sans_normes=True)._extract_color
    fin = (duree_s - 5) if duree_s else 1e9
    tranches, ref = [], None
    for nom, mt, deb, fn, (a1, b1), (a2, b2) in tranches_du_match(ko1_s, half_duration_min):
        if a1 >= fin:
            print(f"  [COULEURS] {nom} : au-delà de la fin de la vidéo")
            continue
        A = _couleurs_aux_moments(video_path, _instants(a1, min(b1, fin), 4), detecteur, extraire)
        V = _couleurs_aux_moments(video_path, _instants(a2, min(b2, fin), 4), detecteur, extraire)
        t = {"nom": nom, "mi_temps": mt, "debut_s": float(deb), "fin_s": float(fn)}
        if len(A) < 40 or len(V) < 40:
            print(f"  [COULEURS] {nom} ({_h(deb)}→{_h(min(fn, fin))}) : trop peu de joueurs lus ({len(A)} / {len(V)})")
        else:
            c = _apprendre(A)
            if ref is not None:
                c = _aligner(ref, c)
            q = _classer(V, c)
            t["qualite"] = q
            print(f"  [COULEURS] {nom} ({_h(deb)}→{_h(min(fn, fin))}) : {_txt(q)}")
            if q["ok"]:
                t["centroids"] = np.asarray(c).tolist()
                ref = c
        tranches.append(t)
    n_ok = sum(1 for t in tranches if "centroids" in t)
    print(f"  [COULEURS] → {n_ok} tranche(s) fiable(s) sur {len(tranches)}")
    if not n_ok:
        print("  [COULEURS] aucune couleur fiable : chaque mi-temps apprend ses couleurs comme avant")
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    json.dump({"tranches": tranches}, open(out_json, "w"), indent=1)
    return out_json


def charger_normes():
    """Pour PlayerReID : liste [(début, fin, mi-temps, centroïdes ou None)] des tranches, ou None."""
    p = os.environ.get("SCOUTIA_COULEURS_EQUIPES_FICHIER", "").strip()
    if not p or not os.path.exists(p):
        return None
    try:
        out = []
        for t in json.load(open(p)).get("tranches", []):
            c = np.asarray(t["centroids"], np.float32) if t.get("centroids") else None
            out.append((float(t["debut_s"]), float(t["fin_s"]), int(t["mi_temps"]),
                        c if (c is not None and c.shape[0] == 2) else None))
        return out if any(x[3] is not None for x in out) else None
    except Exception:
        return None


def norme_pour(normes, t_s):
    """Norme de l'instant t_s : celle de sa tranche, sinon la tranche fiable la plus proche de la même mi-temps,
    sinon None (calibration d'avant)."""
    if not normes:
        return None
    ici = next((x for x in normes if x[0] <= t_s < x[1]), None)
    if ici is None:
        return None
    if ici[3] is not None:
        return ici[3]
    meme = [x for x in normes if x[2] == ici[2] and x[3] is not None]
    if not meme:
        return None
    return min(meme, key=lambda x: min(abs(t_s - x[0]), abs(t_s - x[1])))[3]
