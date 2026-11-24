# export/fiche_joueur.py
# 18/11/2026 — FICHE « BILAN DU JOUEUR » (une page PDF), ajoutée au rapport du match.
# Règle : n'afficher QUE ce qui a été vu avec certitude.
#   • présence reconnue = passages où SON NUMÉRO est lu de façon concordante (minimum : le vrai temps de jeu est plus long) ;
#   • moments avec le ballon = nombre total (le détail par type — passe, dribble… — n'est PAS affiché : non vérifié) ;
#   • buts : les buts CONFIRMÉS auxquels il a participé ; juste en dessous, les buts POSSIBLES (non confirmés) où il
#     est impliqué, présentés comme tels ;
#   • joueur introuvable : message clair, sans chiffres.
from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

AVERTISSEMENT = ("Ce bilan ne compte que ce qui a été vu avec certitude : lorsqu'il n'a pas pu être reconnu à l'image "
                 "(numéro ou maillot pas visible), une action ou un but auquel il a participé peut manquer.")


def _txt(s):
    return str(s).encode("latin-1", "replace").decode("latin-1")


def generer_fiche_joueur(chemin_pdf, numero, couleur, trouve, presence_s=0.0, n_passages=0, n_moments_ballon=0,
                         buts_confirmes=None, buts_possibles=None, match="", lignes_supp=None,
                         titre_buts="Buts du match (confirmés)",
                         titre_possibles="Buts possibles du match, NON confirmés par l'analyse",
                         source_presence="passages où son numéro est lu", chronologie=None, bornes=None):
    """buts_* : listes de (minute « 54' », raison « 3 touche(s) juste avant »)."""
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_fill_color(10, 13, 20)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_text_color(0, 200, 230)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _txt(f"Bilan du joueur n°{numero}" + (f" ({couleur})" if couleur else "")), align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(150, 160, 175)
    pdf.cell(0, 5, _txt(f"{match}  |  {datetime.now().strftime('%d/%m/%Y')}"), align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    def _ligne(titre, valeur):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(80, 8, _txt(titre))
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 8, _txt(valeur), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if not trouve:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(255, 80, 80)
        pdf.multi_cell(0, 7, _txt(f"Nous n'avons trouvé aucune action du joueur n°{numero}"
                                  + (f" ({couleur})" if couleur else "") + " dans cette vidéo. Vérifiez le numéro et la "
                                  "couleur du maillot indiqués. Le résumé du match vous est livré normalement."),
                       align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        _ligne("Présence reconnue", f"environ {presence_s / 60:.1f} min ({n_passages} {source_presence})")
        _ligne("Moments avec le ballon", str(n_moments_ballon))
        for t_, v_ in (lignes_supp or []):          # 19/11 : lignes propres au gardien (arrêts, tirs adverses…)
            _ligne(t_, v_)
        bc = buts_confirmes or []
        _ligne(titre_buts,
               ", ".join(f"{m} ({r})" for m, r in bc) if bc else "aucun vu")
        bp = buts_possibles or []
        if bp:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(170, 175, 185)
            pdf.multi_cell(0, 6, _txt(titre_possibles + " : "
                                      + ", ".join(f"{m} ({r})" for m, r in bp) + "."),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(170, 175, 185)
        pdf.multi_cell(0, 5, _txt(AVERTISSEMENT), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if trouve and chronologie:
        _page_chronologie(pdf, chronologie, bornes or {}, numero, couleur)
    pdf.output(chemin_pdf)
    return chemin_pdf


# 24/11/2026 : FRISE + TABLEAU du joueur, sur le modèle de la chronologie du match. Couleurs :
_C_REPERE, _C_BUT, _C_VERIF = (0, 200, 230), (255, 77, 77), (230, 190, 60)
_C_IMPLIQUE, _C_BUTEUR, _C_PASSEUR = (170, 110, 255), (60, 200, 120), (255, 150, 50)
_C_MOMENT, _C_ARRET = (170, 180, 195), (235, 235, 235)
_LIB = {"but": "But", "but_verif": "But possible (à vérifier)", "but_implique": "Impliqué dans le but",
        "buteur": "But marqué", "passeur": "Passe décisive", "moment": "Action avec le ballon", "arret": "Arrêt"}


def _couleur(it):
    if it.get("role") == "buteur":
        return _C_BUTEUR
    if it.get("role") == "passeur":
        return _C_PASSEUR
    ty = it["type"]
    if ty == "but_implique":
        return _C_IMPLIQUE
    if ty == "but":
        return _C_BUT if it.get("conf", True) else _C_VERIF
    return {"moment": _C_MOMENT, "arret": _C_ARRET}.get(ty, _C_MOMENT)


def _page_chronologie(pdf, chronologie, bornes, numero, couleur):
    """chronologie : [{"t_abs": s, "type": but|but_implique|moment|arret, "conf": bool, "role": buteur|passeur|None}]"""
    from export.rapport_client import minute_de_match
    k1, f1, k2, fm = bornes.get("ko1"), bornes.get("fin1mt"), bornes.get("ko2"), bornes.get("finmatch")
    complet = None not in (k1, f1, k2, fm)
    if k1 is None:
        return
    joue = (lambda t: max(0.0, min(t, f1) - k1) + (max(0.0, min(t, fm) - k2) if t >= k2 else 0.0)) if complet \
        else (lambda t: max(0.0, t - k1))
    total = ((f1 - k1) + (fm - k2)) if complet else max(1.0, max(float(x["t_abs"]) for x in chronologie) - k1 + 60)
    items = sorted(chronologie, key=lambda x: float(x["t_abs"]))
    pdf.add_page()
    pdf.set_fill_color(10, 13, 20)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_text_color(0, 200, 230)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, _txt(f"Le match du n°{numero}" + (f" ({couleur})" if couleur else "")), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # frise
    x0, y0, w = 12, pdf.get_y() + 2, 186
    pdf.set_fill_color(22, 28, 40)
    pdf.rect(x0, y0, w, 24, "F")
    pdf.set_fill_color(60, 68, 82)
    pdf.rect(x0 + 5, y0 + 13, w - 10, 1.2, "F")
    pos = lambda t: x0 + 5 + (w - 10) * min(max(joue(float(t)) / total, 0.0), 0.995)
    rep_ = [(k1, "Début")] + ([(f1, "MT"), (fm, "Fin")] if complet else [])
    pdf.set_font("Helvetica", "", 5)
    for t, lb in rep_:
        pdf.set_fill_color(*_C_REPERE)
        pdf.rect(pos(t) - 0.4, y0 + 8.5, 0.8, 9, "F")
        pdf.set_text_color(*_C_REPERE)
        pdf.set_xy(pos(t) - 6, y0 + 18)
        pdf.cell(12, 3, _txt(lb), align="C")
    for it in items:
        pdf.set_fill_color(*_couleur(it))
        h = 7 if it["type"] != "moment" else 4.5
        pdf.rect(pos(it["t_abs"]) - 0.8, y0 + 13.6 - h / 2, 1.6, h, "F")
    # légende
    leg = [("BUT", _C_BUT), ("À VÉRIFIER", _C_VERIF), ("IMPLIQUÉ", _C_IMPLIQUE), ("SES ACTIONS", _C_MOMENT)]
    if any(i.get("role") == "buteur" for i in items):
        leg.append(("A MARQUÉ", _C_BUTEUR))
    if any(i.get("role") == "passeur" for i in items):
        leg.append(("PASSE DÉCISIVE", _C_PASSEUR))
    if any(i["type"] == "arret" for i in items):
        leg.append(("ARRÊT", _C_ARRET))
    lx = x0 + 5
    pdf.set_font("Helvetica", "", 6)
    for lb, c in leg:
        pdf.set_text_color(*c)
        pdf.set_xy(lx, y0 + 2)
        pdf.cell(pdf.get_string_width(_txt(lb)) + 5, 4, _txt(lb))
        lx += pdf.get_string_width(_txt(lb)) + 6
    pdf.set_y(y0 + 28)
    # tableau
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(25, 32, 45)
    pdf.set_text_color(170, 175, 185)
    pdf.set_x(12)
    for wc, t in ((26, "Minute"), (160, "Action")):
        pdf.cell(wc, 7, _txt(t), fill=True, align="C" if t == "Minute" else "L")
    pdf.ln(7)
    lignes = [(float(k1), "Coup d'envoi", _C_REPERE)] + ([(float(f1), "Mi-temps", _C_REPERE), (float(k2), "Reprise", _C_REPERE),
                                                         (float(fm), "Fin du match", _C_REPERE)] if complet else [])
    for it in items:
        lib = _LIB.get(it.get("role") or "", None) or (_LIB["but_verif"] if it["type"] == "but" and not it.get("conf", True)
                                                      else _LIB.get(it["type"], "Action"))
        if it["type"] == "but_implique" and not it.get("conf", True) and not it.get("role"):
            lib += " (but à vérifier)"
        if it.get("detail"):
            lib += f" — {it['detail']}"
        lignes.append((float(it["t_abs"]), lib, _couleur(it)))
    lignes.sort(key=lambda x: x[0])
    for k, (t, lib, c) in enumerate(lignes):
        if pdf.get_y() > 275:
            pdf.add_page()
            pdf.set_fill_color(10, 13, 20)
            pdf.rect(0, 0, 210, 297, "F")
            pdf.set_y(15)
        pdf.set_x(12)
        pdf.set_fill_color(*((20, 26, 38) if k % 2 else (15, 20, 30)))
        pdf.set_text_color(*c)
        pdf.set_font("Helvetica", "B" if c != _C_MOMENT else "", 8.5)
        mn = "" if lib in ("Mi-temps", "Fin du match") else ("0'" if lib == "Coup d'envoi" else minute_de_match(t, k1, bornes))
        pdf.cell(26, 6, _txt(mn), fill=True, align="C")
        pdf.cell(160, 6, _txt(lib), fill=True)
        pdf.ln(6)
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(150, 160, 175)
    pdf.set_x(12)
    pdf.multi_cell(186, 4, _txt("« Impliqué dans le but » : il a touché le ballon ou était reconnu juste avant le but. "
                                "Qui a marqué ou donné la passe décisive n'est indiqué que lorsque c'est connu."))


def joindre_au_rapport(rapport_pdf, fiche_pdf, sortie_pdf):
    """Ajoute la fiche à la fin du rapport du match (pypdf). Renvoie le chemin produit, ou None si impossible."""
    try:
        from pypdf import PdfWriter
        w = PdfWriter()
        for p in (rapport_pdf, fiche_pdf):
            w.append(p)
        with open(sortie_pdf, "wb") as f:
            w.write(f)
        return sortie_pdf
    except Exception as e:
        print(f"  ⚠️ fiche non jointe au rapport ({e}) : elle reste disponible seule ({fiche_pdf})")
        return None
