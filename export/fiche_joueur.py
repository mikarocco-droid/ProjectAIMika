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
                         titre_buts="Buts confirmés auxquels il a participé",
                         titre_possibles="Buts possibles, NON confirmés par l'analyse, où il apparaît impliqué",
                         source_presence="passages où son numéro est lu"):
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
    pdf.output(chemin_pdf)
    return chemin_pdf


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
