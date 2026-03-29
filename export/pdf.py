# export/pdf.py
# -*- coding: utf-8 -*-

import os
from datetime import datetime

try:
    from fpdf import FPDF, XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    print("fpdf2 non installe — export PDF desactive")

import config


# ─────────────────────────────────────────
# NETTOYAGE TEXTE
# ─────────────────────────────────────────
def clean(text):
    """
    Nettoie le texte pour éviter les erreurs
    de caractères non supportés par fpdf2.
    """
    if not text:
        return ""
    return (
        str(text)
        .replace("—", "-")
        .replace("–", "-")
        .replace("'", "'")
        .replace(""", '"')
        .replace(""", '"')
        .replace("«", '"')
        .replace("»", '"')
        .replace("…", "...")
        .replace("•", "-")
        .replace("⭐", "*")
        .replace("✅", "OK")
        .replace("❌", "X")
        .replace("⚽", "[foot]")
        .replace("🏀", "[basket]")
        .replace("🤾", "[hand]")
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


# ─────────────────────────────────────────
# COULEURS
# ─────────────────────────────────────────
PRIMARY = (26,  115, 232)
DARK    = (30,   30,  30)
GRAY    = (100, 100, 100)
LIGHT   = (245, 245, 245)
WHITE   = (255, 255, 255)
GREEN   = (52,  168,  83)
RED     = (234,  67,  53)
YELLOW  = (251, 188,   4)


# ─────────────────────────────────────────
# CLASSE PDF
# ─────────────────────────────────────────
class ScoutReport(FPDF):

    def __init__(self, sport="football", title="Rapport d'analyse"):
        super().__init__()
        self.sport     = clean(sport.capitalize())
        self.doc_title = clean(title)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)

    def header(self):
        self.set_fill_color(*PRIMARY)
        self.rect(0, 0, 210, 18, "F")

        logo = config.PDF_LOGO_PATH
        if os.path.exists(logo):
            self.image(logo, x=12, y=3, h=12)

        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*WHITE)
        self.set_y(5)
        self.cell(0, 8, f"SCOUT IA - {self.sport.upper()}", align="R")

        self.set_text_color(*DARK)
        self.ln(14)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 5,
            f"Scout IA  |  {datetime.now().strftime('%d/%m/%Y')}  |  Page {self.page_no()}",
            align="C"
        )

    def section_title(self, text):
        self.ln(4)
        self.set_fill_color(*PRIMARY)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, f"  {clean(text)}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)
        self.ln(2)

    def kpi_box(self, x, y, w, h, label, value, color=None):
        color = color or PRIMARY
        self.set_xy(x, y)
        self.set_fill_color(*LIGHT)
        self.rect(x, y, w, h, "F")
        self.set_fill_color(*color)
        self.rect(x, y, w, 2, "F")

        self.set_xy(x, y + 4)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*color)
        self.cell(w, 10, clean(str(value)), align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_xy(x, y + 14)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GRAY)
        self.cell(w, 5, clean(label.upper()), align="C")

        self.set_text_color(*DARK)

    def table_header(self, cols):
        self.set_fill_color(*PRIMARY)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8)
        for label, width in cols:
            self.cell(width, 7, clean(label), border=0, fill=True, align="C")
        self.ln()
        self.set_text_color(*DARK)

    def table_row(self, values, widths, alternate=False):
        if alternate:
            self.set_fill_color(*LIGHT)
        else:
            self.set_fill_color(*WHITE)
        self.set_font("Helvetica", "", 8)
        for val, width in zip(values, widths):
            self.cell(width, 6, clean(str(val)), border=0, fill=True, align="C")
        self.ln()


# ─────────────────────────────────────────
# GÉNÉRATION DU RAPPORT
# ─────────────────────────────────────────
def generate_pdf(result, output_path, sport="football"):

    if not FPDF_AVAILABLE:
        print("fpdf2 non disponible")
        return None

    summary    = result.get("summary",    {})
    stats      = result.get("stats",      {})
    highlights = result.get("highlights", [])
    jersey_map = result.get("jersey_map", {})
    ai_summary = result.get("ai_summary")
    heatmap    = result.get("heatmap")

    pdf = ScoutReport(sport=sport, title="Rapport d'analyse")
    pdf.add_page()

    # ─────────────────────────────────────────
    # TITRE
    # ─────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*PRIMARY)
    pdf.ln(5)
    pdf.cell(0, 12, "RAPPORT D'ANALYSE", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 7,
        clean(f"{sport.capitalize()}  |  Duree : {summary.get('duration', '--')}"),
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(6)

    # ─────────────────────────────────────────
    # KPI
    # ─────────────────────────────────────────
    pdf.section_title("Resume du match")

    kpis = [
        ("Buts",          summary.get("goals",         0), GREEN),
        ("Tirs",          summary.get("shots",         0), PRIMARY),
        ("xG Total",      summary.get("total_xg",    0.0), YELLOW),
        ("Passes",        summary.get("passes",        0), PRIMARY),
        ("Interceptions", summary.get("interceptions", 0), RED),
        ("Dribbles",      summary.get("dribbles",      0), PRIMARY),
    ]

    box_w   = 28
    box_h   = 22
    gap     = 4
    start_x = 15
    x       = start_x

    for label, value, color in kpis:
        pdf.kpi_box(x, pdf.get_y(), box_w, box_h, label, value, color)
        x += box_w + gap

    pdf.ln(box_h + 6)

    # ─────────────────────────────────────────
    # RESUME IA
    # ─────────────────────────────────────────
    if ai_summary:
        pdf.section_title("Analyse IA")
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GRAY)
        pdf.multi_cell(0, 5, clean(ai_summary))
        pdf.set_text_color(*DARK)
        pdf.ln(2)

    # ─────────────────────────────────────────
    # HEATMAP
    # ─────────────────────────────────────────
    if heatmap and os.path.exists(heatmap):
        pdf.section_title("Carte de chaleur")
        img_w = 120
        img_x = (210 - img_w) / 2
        pdf.image(heatmap, x=img_x, w=img_w)
        pdf.ln(3)

    # ─────────────────────────────────────────
    # STATS JOUEURS
    # ─────────────────────────────────────────
    if stats:
        pdf.section_title("Statistiques joueurs")

        cols = [
            ("Joueur",        30),
            ("Maillot",       18),
            ("Touches",       22),
            ("Passes",        22),
            ("Tirs",          18),
            ("Buts",          18),
            ("Interceptions", 30),
            ("Dribbles",      22),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols)

        # Trier par touches décroissantes
        sorted_stats = sorted(
            stats.items(),
            key=lambda x: x[1].get("touches", 0),
            reverse=True
        )[:20]  # max 20 joueurs

        for i, (player_id, s) in enumerate(sorted_stats):
            jersey = jersey_map.get(
                str(player_id),
                jersey_map.get(player_id, "-")
            )
            row = [
                f"Joueur {player_id}",
                f"#{jersey}" if jersey and jersey != "-" else "-",
                s.get("touches",       0),
                s.get("passes",        0),
                s.get("tirs",          0),
                s.get("buts",          0),
                s.get("interceptions", 0),
                s.get("dribbles",      0),
            ]
            pdf.table_row(row, widths, alternate=(i % 2 == 0))

        pdf.ln(4)

    # ─────────────────────────────────────────
    # HIGHLIGHTS
    # ─────────────────────────────────────────
    if highlights:
        pdf.section_title("Moments cles")

        type_labels = {
            "goal":         "But",
            "score":        "But",
            "shot":         "Tir",
            "interception": "Interception",
            "dribble":      "Dribble",
            "long_pass":    "Passe longue",
            "pass":         "Passe",
            "action":       "Action",
        }

        cols = [
            ("#",         10),
            ("Type",      45),
            ("Debut",     25),
            ("Fin",       25),
            ("Duree",     25),
            ("Score",     25),
            ("Nb events", 25),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols)

        for i, h in enumerate(highlights):
            duration = round(h.get("time_end", 0) - h.get("time_start", 0), 1)
            row = [
                i + 1,
                type_labels.get(h.get("main_type", ""), h.get("main_type", "-")),
                f"{h.get('time_start', 0):.1f}s",
                f"{h.get('time_end',   0):.1f}s",
                f"{duration}s",
                h.get("score", "-"),
                len(h.get("events", [])),
            ]
            pdf.table_row(row, widths, alternate=(i % 2 == 0))

        pdf.ln(4)

    # ─────────────────────────────────────────
    # SAUVEGARDE
    # ─────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        pdf.output(output_path)
        print(f"PDF genere -> {output_path}")
        return output_path
    except Exception as e:
        print(f"PDF erreur : {e}")
        return None