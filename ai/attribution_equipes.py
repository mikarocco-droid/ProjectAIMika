# -*- coding: utf-8 -*-
"""ScoutIA — ATTRIBUTION DES BUTS AUX ÉQUIPES (intégrée au pipeline le 09/11/2026).

Méthode validée sur les 43 buts de référence (questions identiques au notebook de test) :
  • tenues des deux équipes : 16 images de la 1re mi-temps ;
  • par but : Q1 « qui a marqué » (8 images, 2 s d'écart) + Q2 « qui engage » (1 image/s jusqu'au coup d'envoi)
    + Q3 arbitrage (shorts) si Q1 et Q2 ne concordent pas ;
  • CONFIRMÉ = Q1 et Q2 d'accord (écart ≥ 4) ; PROBABLE = écart ≥ 2 et au moins deux réponses concordantes ;
    sinon NON ATTRIBUÉ.
11/11 : DÉFAUT REMIS EN RÉSOLUTION NORMALE. En production (4 matchs), « moyenne + escalade » a donné 2 attributions
FAUSSES (Franchimont, Raeren) : critère « 0 faux » non tenu. Historique du réglage abandonné :
Réglage de coût (test du 09/11 sur les 43 buts) : questions d'abord en résolution MOYENNE (0 erreur d'équipe, mais
quelques abstentions), puis ESCALADE en résolution NORMALE des seuls buts « non attribués ». Calcul sur ces données :
41 buts attribués justes / 43, 0 faux (normale partout : 40 / 0), ≈ −37 % de coût. Règle à confirmer en production.
Réglages : SCOUTIA_ATTRIBUTION=0 (désactive), SCOUTIA_ATTRIBUTION_RESOLUTION=high (défaut)|medium|low,
SCOUTIA_ATTRIBUTION_PAS_Q2 (secondes entre deux images de Q2, défaut 1 — en test le 11/11),
SCOUTIA_ATTRIBUTION_PLAFOND (dollars, défaut 1.5).
"""
import os, json, collections

AUTRE = {"A": "B", "B": "A"}
fmt = lambda s: f"{int(s)//60:02d}:{int(s)%60:02d}"

QTENUES = """These 16 images come from an amateur football match video (a camera following the play), taken at different moments.
Identify the OUTFIELD players' kit of each of the two teams (ignore goalkeepers and the referee, describe them separately).
For each team give the main colour(s) of the shirt, the shorts and the socks, in simple English colour words
(e.g. "yellow", "red and white", "dark blue", "light blue", "black"). Also say whether the two teams' shirts share a colour.
Answer ONLY with JSON:
{"team_1": {"shirt": "...", "shorts": "...", "socks": "..."}, "team_2": {"shirt": "...", "shorts": "...", "socks": "..."},
 "goalkeepers": "...", "referee": "...", "shirts_share_a_colour": true|false, "certainty": "high|medium|low"}"""

AENT = "These images come from an amateur football match video. Team A wears: shirt {a0}, shorts {a1}. Team B wears: shirt {b0}, shorts {b1}.{r}\nEach image has its number and time in the top-left corner. If you cannot see it clearly, answer unknown. Do not guess.\n"
AQ1 = AENT + """These images (2 seconds apart) should show the moment a GOAL is scored. Which team SCORED? Look at which team was attacking
and which team's players celebrate. If you do not see a goal at all, answer unknown.
Answer ONLY with JSON: {{"scoring_team": "A|B|unknown", "certainty": "high|medium|low", "evidence": "short, with image numbers"}}"""
AQ2 = AENT + """These images (one per second) cover the period after a goal, until the KICK-OFF that restarts the match. Find the ball placed
on the CENTRE SPOT. Identify the ONE player who has his foot on the ball or is bending over it, and who PLAYS the first ball. Other players,
including opponents, may stand right next to the ball: ignore them, only the player who actually touches and plays the ball counts.
Give the colour of THAT player's shirt AND shorts. If you do not clearly see the ball being played from the centre spot, answer unknown.
Answer ONLY with JSON: {{"kickoff_team": "A|B|unknown", "kicker_shirt": "...", "kicker_shorts": "...", "image": int|null,
"certainty": "high|medium|low", "evidence": "short"}}"""
AQ3 = AENT + """Images 1 to 8 (2 seconds apart) show a GOAL; the following images (2 seconds apart) show the restart after it.
1. Which team SCORED? Look at which team attacked and whose players celebrate — check their SHORTS.
2. Which team plays the KICK-OFF? Only the player who actually touches and plays the ball on the centre spot counts — check his SHORTS.
Answer ONLY with JSON: {{"scoring_team": "A|B|unknown", "kickoff_team": "A|B|unknown", "certainty": "high|medium|low",
"evidence": "short, with image numbers and shorts colours seen"}}"""


def _rappel(a0, a1, b0, b1, toujours=False):
    mots = lambda x: set(x.lower().replace(",", " ").split()) - {"and", "dark", "light", "with"}
    com = mots(a0) & mots(b0)
    if com or toujours:
        t = f" Both teams have {' and '.join(sorted(com))} on their shirts." if com else " Shirts alone can be misleading (shadows, light)."
        return t + f" To tell the teams apart, look at the SHORTS: team A has {a1} shorts, team B has {b1} shorts."
    return ""


def _eq(d, champ):
    if not d or str(d.get("certainty", "")).lower() == "low":
        return None
    v = str(d.get(champ, "unknown")).upper()
    return v if v in AUTRE else None


class _Demandeur:
    def __init__(self, video_path, plafond):
        import cv2, numpy as np
        from google import genai
        from google.genai import types
        self.cv2, self.np, self.types = cv2, np, types
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        self.cap = cv2.VideoCapture(video_path)
        self.plafond = plafond
        self.jet = {"q": 0, "e": 0, "s": 0}
        self.arret = None

    def cout(self):
        return self.jet["e"] / 1e6 * 0.75 + self.jet["s"] / 1e6 * 3.75

    def __call__(self, instants, consigne, resolution, etiquette=True):
        if self.arret:
            return None
        if self.cout() > self.plafond:
            self.arret = f"plafond {self.plafond} $ atteint"
            return None
        cv2, np = self.cv2, self.np
        ims = []
        for i, t in enumerate(instants, 1):
            self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = self.cap.read()
            img = img.copy() if ok else np.zeros((1080, 1920, 3), np.uint8)
            if etiquette:
                cv2.putText(img, f"{i}  {fmt(t)}", (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
            ims.append(self.types.Part.from_bytes(data=cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(),
                                                  mime_type="image/jpeg"))
        cfg = {"temperature": 0.0, "thinking_config": {"thinking_level": "low"}}
        if resolution in ("low", "medium"):
            cfg["media_resolution"] = f"MEDIA_RESOLUTION_{resolution.upper()}"
        try:
            r = self.client.models.generate_content(model="gemini-3.8-flash", contents=[consigne] + ims, config=cfg)
            um = getattr(r, "usage_metadata", None)
            self.jet["q"] += 1
            self.jet["e"] += int(getattr(um, "prompt_token_count", 0) or 0)
            self.jet["s"] += int(getattr(um, "candidates_token_count", 0) or 0) + int(getattr(um, "thoughts_token_count", 0) or 0)
            t_ = r.text or ""
            d, _ = json.JSONDecoder().raw_decode(t_[t_.index("{"):])
            return d
        except Exception as e:
            print(f"  [ATTRIBUTION] ⚠️ question en échec : {e}")
            return None


def _attribuer_un_but(dem, g, k, lim, ff, tenues, resolution):
    (a0, a1), (b0, b1) = tenues
    d1 = dem([g + d for d in range(-8, 7, 2)], AQ1.format(**ff), resolution)
    pas = max(1, int(os.environ.get("SCOUTIA_ATTRIBUTION_PAS_Q2", "1")))      # secondes entre deux images de Q2
    inst = list(range(int(k), int(min(k + 45, lim)) + 1, pas)) if (k is not None and g + 10 <= k < lim) else \
        list(range(int(g) + 15, int(min(g + 120, lim)) + 1, pas))
    d2 = dem(inst, AQ2.format(**ff), resolution)
    s1 = _eq(d1, "scoring_team"); s2 = AUTRE.get(_eq(d2, "kickoff_team"))
    pts = collections.Counter()
    if s1: pts[s1] += 2
    if s2: pts[s2] += 2
    d3 = None
    if not (s1 and s2 and s1 == s2):
        d3 = dem([g + 1 + d for d in range(-8, 7, 2)] + list(range(int(g) + 10, int(min(g + 121, lim)), 2)),
                 AQ3.format(**dict(ff, r=_rappel(a0, a1, b0, b1, toujours=True))), resolution)
        a3, k3 = _eq(d3, "scoring_team"), _eq(d3, "kickoff_team")
        if a3: pts[a3] += 1
        if k3: pts[AUTRE[k3]] += 1
    tete = "A" if pts["A"] > pts["B"] else "B" if pts["B"] > pts["A"] else None
    sout = 0
    if tete:
        sout = (s1 == tete) + (s2 == tete) + (bool(d3) and (_eq(d3, "scoring_team") == tete or AUTRE.get(_eq(d3, "kickoff_team")) == tete))
    ecart = abs(pts["A"] - pts["B"])
    if tete and s1 == tete and s2 == tete and ecart >= 4:
        return "confirmé", tete
    if tete and ecart >= 2 and sout >= 2:
        return "probable", tete
    return "non attribué", None


def attribuer_buts(video_path, buts, debut_1mt, fin_1mt):
    """buts : liste de dicts avec time_abs (s) et, si connu, candidate_time_abs (coup d'envoi validé, s).
    Ajoute à chaque but : equipe ('A'/'B'/None), niveau_equipe, tenue_equipe. Renvoie un dict de synthèse."""
    if os.environ.get("SCOUTIA_ATTRIBUTION", "1").strip() == "0" or not buts:
        return None
    res = os.environ.get("SCOUTIA_ATTRIBUTION_RESOLUTION", "high").strip().lower()   # 11/11 : normale (medium = 2 faux sur 4 matchs)
    plafond = float(os.environ.get("SCOUTIA_ATTRIBUTION_PLAFOND", "1.5"))
    dem = _Demandeur(video_path, plafond)
    deb = float(debut_1mt) + 60
    fin = float(fin_1mt) - 60 if fin_1mt else deb + 40 * 60
    if fin <= deb + 60:
        fin = deb + 30 * 60
    dt = dem([deb + i * (fin - deb) / 15 for i in range(16)], QTENUES, "high", etiquette=False)   # tenues : résolution normale
    if not dt:
        print("  [ATTRIBUTION] tenues non détectées : attribution impossible")
        return None
    T1, T2 = dt.get("team_1", {}), dt.get("team_2", {})
    tenues = ((str(T1.get("shirt")), str(T1.get("shorts"))), (str(T2.get("shirt")), str(T2.get("shorts"))))
    (a0, a1), (b0, b1) = tenues
    nom = {"A": f"maillot {a0} / short {a1}", "B": f"maillot {b0} / short {b1}"}
    print(f"  [ATTRIBUTION] tenues : A = {nom['A']} | B = {nom['B']} (certitude {dt.get('certainty')})")
    ff = dict(a0=a0, a1=a1, b0=b0, b1=b1, r=_rappel(a0, a1, b0, b1))
    buts_tries = sorted(buts, key=lambda b: float(b["time_abs"]))
    n_esc = 0
    for i, b in enumerate(buts_tries):
        g = float(b["time_abs"])
        k = b.get("candidate_time_abs")
        suiv = float(buts_tries[i + 1]["time_abs"]) if i + 1 < len(buts_tries) else None
        lim = (suiv - 5) if suiv else 10 ** 9
        niv, tete = _attribuer_un_but(dem, g, k, lim, ff, tenues, res)
        if niv == "non attribué" and res in ("low", "medium") and not dem.arret:
            n_esc += 1                                    # escalade en résolution normale
            niv, tete = _attribuer_un_but(dem, g, k, lim, ff, tenues, "high")
        b["equipe"], b["niveau_equipe"], b["tenue_equipe"] = tete, niv, (nom[tete] if tete else None)
        print(f"  [ATTRIBUTION] but {fmt(g)} ({b.get('status')}) → {nom[tete] if tete else 'équipe non identifiée'} ({niv})")
    conf = collections.Counter(b["equipe"] for b in buts_tries if b.get("niveau_equipe") == "confirmé")
    est = collections.Counter(b["equipe"] for b in buts_tries if b.get("equipe"))
    synthese = {"tenues": {"A": nom["A"], "B": nom["B"]},
                "score_confirme": {"A": conf["A"], "B": conf["B"]},
                "score_estime": {"A": est["A"], "B": est["B"]},
                "buts_sans_equipe": sum(1 for b in buts_tries if not b.get("equipe")),
                "questions": dem.jet["q"], "cout_dollars": round(dem.cout(), 3), "escalades": n_esc,
                "incomplet": dem.arret}
    print(f"  [ATTRIBUTION] score confirmé {nom['A']} {conf['A']} - {conf['B']} {nom['B']} | estimé {est['A']} - {est['B']} | "
          f"{dem.jet['q']} question(s) ≈ {dem.cout():.3f} $ | escalades {n_esc}" + (f" | ⚠️ {dem.arret}" if dem.arret else ""))
    try:
        dem.cap.release()
    except Exception:
        pass
    return synthese
