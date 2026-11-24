# export/rapport_client.py
# 18/11/2026 — RÈGLES D'AFFICHAGE DU RAPPORT CLIENT (option B : rapport automatique PRUDENT), utilisées par export/pdf.py.
#   • « Confirmé » n'a qu'UN sens : un but dont la détection est confirmée. Les autres buts détectés sont des
#     « buts à vérifier » (24/11 : « à revoir » prêtait à confusion), jamais comptés comme des buts.
#   • Score affiché SEULEMENT s'il est sûr : aucun but « à vérifier » et tous les buts confirmés ont une équipe.
#     Sinon : nombre de buts confirmés par équipe, sans le présenter comme un score.
#   • Équipes : noms donnés au paramétrage du match (SCOUTIA_EQUIPES), reliés aux tenues détectées par leur couleur ;
#     sinon, couleurs en français (« Rouge et blanc », « Bleu »).
#   • Minutes de match : 1re MT depuis le coup d'envoi, 2e MT = 45' + temps depuis la reprise (pause exclue),
#     temps additionnel noté 45+2', 90+3'.
import json
import os

COULEURS_FR = {
    "red": "rouge", "white": "blanc", "blue": "bleu", "light blue": "bleu clair", "sky blue": "bleu ciel",
    "dark blue": "bleu foncé", "navy": "bleu marine", "navy blue": "bleu marine", "royal blue": "bleu roi",
    "black": "noir", "green": "vert", "dark green": "vert foncé", "light green": "vert clair", "yellow": "jaune",
    "orange": "orange", "purple": "violet", "violet": "violet", "grey": "gris", "gray": "gris", "pink": "rose",
    "maroon": "bordeaux", "burgundy": "bordeaux", "claret": "bordeaux", "brown": "brun", "gold": "doré",
    "silver": "argenté", "turquoise": "turquoise", "cyan": "cyan", "lime": "vert citron",
}
_FR_VERS_EN = {}
for _en, _fr in COULEURS_FR.items():
    for _mot in _fr.split():
        _FR_VERS_EN.setdefault(_mot, set()).update(_en.split())
_FR_VERS_EN.update({"blanche": {"white"}, "blancs": {"white"}, "rouges": {"red"}, "bleue": {"blue"}, "bleus": {"blue"},
                    "noire": {"black"}, "noirs": {"black"}, "verte": {"green"}, "verts": {"green"}, "jaunes": {"yellow"},
                    "grise": {"grey", "gray"}, "mauve": {"purple", "violet"}, "ciel": {"sky", "light"},
                    "marine": {"navy"}, "clair": {"light"}, "foncé": {"dark"}, "fonce": {"dark"}})


def _couleur_fr(c):
    c = str(c or "").strip().lower()
    if not c or c in ("none", "unknown", "?"):
        return ""
    return COULEURS_FR.get(c, " ".join(COULEURS_FR.get(m, m) for m in c.split()))


def _maillot_short(tenue):
    """« maillot red / short white » → ("red", "white")."""
    t = str(tenue or "")
    if "/" in t:
        a, b = t.split("/", 1)
        return a.replace("maillot", "").strip(), b.replace("short", "").strip()
    return t.replace("maillot", "").strip(), ""


def libelle_couleurs(tenue):
    """« maillot red / short white » → « Rouge et blanc » ; « maillot blue / short blue » → « Bleu »."""
    m, s = (_couleur_fr(x) for x in _maillot_short(tenue))
    if not m:
        return "Équipe"
    txt = m if (not s or s == m) else f"{m} et {s}"
    return txt[:1].upper() + txt[1:]


def _mots_en(tenue):
    m, s = _maillot_short(tenue)
    return set(m.lower().split()), set(s.lower().split())


def lire_equipes_parametrees():
    """SCOUTIA_EQUIPES = JSON : [{"nom": "SRU Verviers", "couleur": "rouge et blanc"}, {"nom": "...", "couleur": "bleu"}]."""
    try:
        v = json.loads(os.environ.get("SCOUTIA_EQUIPES", "") or "[]")
        return [e for e in v if isinstance(e, dict) and str(e.get("nom") or "").strip()][:2]
    except ValueError:
        return []


def noms_equipes(tenues, equipes=None):
    """{"A": libellé, "B": libellé}. Les noms donnés au paramétrage sont reliés aux tenues détectées par leur couleur ;
    si ce lien est impossible ou ambigu, les deux équipes gardent leurs couleurs (jamais de nom au hasard)."""
    tenues = tenues or {}
    lib = {k: libelle_couleurs(tenues.get(k)) for k in ("A", "B")}
    if lib["A"] == lib["B"]:
        lib = {"A": lib["A"] + " (A)", "B": lib["B"] + " (B)"}
    equipes = lire_equipes_parametrees() if equipes is None else equipes
    if len(equipes) != 2 or not tenues.get("A") or not tenues.get("B"):
        return lib

    def _score(eq, k):
        mots = set()
        for m in str(eq.get("couleur") or "").lower().replace(",", " ").replace("/", " ").split():
            mots |= _FR_VERS_EN.get(m, set())
        maillot, short = _mots_en(tenues.get(k))
        return 2 * len(mots & maillot) + len(mots & short)

    direct = _score(equipes[0], "A") + _score(equipes[1], "B")
    croise = _score(equipes[0], "B") + _score(equipes[1], "A")
    if direct == croise:
        return lib
    e_a, e_b = (equipes[0], equipes[1]) if direct > croise else (equipes[1], equipes[0])
    return {"A": str(e_a["nom"]).strip(), "B": str(e_b["nom"]).strip()}


def minute_de_match(t_abs, ko1, bornes=None):
    """Minute affichée (« 53' », « 45+2' », « 90+3' ») à partir de l'heure dans la vidéo. Sans reprise connue :
    temps depuis le coup d'envoi du match."""
    bornes = bornes or {}
    ko2, fin1 = bornes.get("ko2"), bornes.get("fin1mt")
    # 2e mi-temps : dès la fin de la 1re (+30 s de marge) ; un événement pendant la pause est rattaché à la reprise
    seuil = (float(fin1) + 30) if fin1 is not None else (float(ko2) - 5 if ko2 is not None else None)
    if ko2 is not None and t_abs >= min(seuil, float(ko2) - 5):
        s = max(0.0, t_abs - float(ko2))
        return f"{45 + int(s // 60) + 1}'" if s < 45 * 60 else f"90+{int((s - 2700) // 60) + 1}'"
    s = max(0.0, t_abs - float(ko1 or 0))
    if ko2 is None:
        return f"{int(s // 60) + 1}'"
    return f"{int(s // 60) + 1}'" if s < 45 * 60 else f"45+{int((s - 2700) // 60) + 1}'"


def duree_de_jeu(bornes):
    """« 91 min de jeu » si les quatre bornes sont connues, sinon None."""
    b = bornes or {}
    if all(b.get(k) is not None for k in ("ko1", "fin1mt", "ko2", "finmatch")):
        s = (float(b["fin1mt"]) - float(b["ko1"])) + (float(b["finmatch"]) - float(b["ko2"]))
        return f"{int(round(s / 60))} min de jeu"
    return None


def resume_buts(buts, noms):
    """Option B. Renvoie (ligne_score_ou_None, ligne_detail). Le score n'est donné que s'il est sûr."""
    conf = [b for b in buts if b.get("status") == "confirmé"]
    a_revoir = [b for b in buts if b.get("status") != "confirmé"]
    par_eq = {"A": 0, "B": 0}
    sans_eq = 0
    for b in conf:
        if b.get("equipe") in ("A", "B"):
            par_eq[b["equipe"]] += 1
        else:
            sans_eq += 1
    if not a_revoir and not sans_eq:
        n_u = sum(1 for b in conf if b.get("verdict_utilisateur") == "but")      # 24/11 : vérifiés par un utilisateur
        return (f"Score : {noms['A']}  {par_eq['A']} - {par_eq['B']}  {noms['B']}",
                "Tous les buts sont confirmés" + (f" (dont {n_u} vérifié(s) sur la vidéo par un utilisateur)." if n_u else "."))
    det = f"Buts confirmés : {noms['A']} {par_eq['A']}, {noms['B']} {par_eq['B']}"
    if sans_eq:
        det += f", équipe non identifiée {sans_eq}"
    if a_revoir:
        det += f"  |  {len(a_revoir)} but(s) à vérifier"
    return None, det + "  |  score non affiché tant que tout n'est pas confirmé"
