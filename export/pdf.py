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
# CLEAN TEXT (ULTRA SAFE)
# ─────────────────────────────────────────
def clean(text):
    if not text:
        return ""
    return (
        str(text)
        .replace("—", "-")
        .replace("–", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("«", '"')
        .replace("»", '"')
        .replace("…", "...")
        .replace("•", "-")
        .replace("⭐", "*")
        .replace("⚽", "[goal]")
        .replace("🏀", "[basket]")
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


# ─────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────
PRIMARY = (20, 110, 220)
DARK    = (25, 25, 25)
GRAY    = (120, 120, 120)
LIGHT   = (245, 245, 245)
WHITE   = (255, 255, 255)
GREEN   = (40, 180, 90)
RED     = (220, 60, 60)
YELLOW  = (250, 180, 20)


# ─────────────────────────────────────────
# PDF CLASS
# ─────────────────────────────────────────
class ScoutPDF(FPDF):

    def __init__(self, sport="football"):
        super().__init__()
        self.sport = clean(sport)
        self.set_auto_page_break(auto=True, margin=12)
        self.set_margins(15, 15, 15)

    def header(self):
        self.set_fill_color(*PRIMARY)
        self.rect(0, 0, 210, 16, "F")

        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 10)
        self.set_y(5)
        self.cell(0, 5, f"SCOUT AI - {self.sport.upper()}", align="R")

        self.set_text_color(*DARK)
        self.ln(12)

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GRAY)
        self.cell(
            0, 5,
            f"{datetime.now().strftime('%d/%m/%Y')}  |  Page {self.page_no()}",
            align="C"
        )

    def section(self, title):
        self.ln(4)
        self.set_fill_color(*PRIMARY)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 7, f"  {clean(title)}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)
        self.ln(2)

    def kpi(self, x, y, w, h, label, value, color):
        self.set_xy(x, y)
        self.set_fill_color(*LIGHT)
        self.rect(x, y, w, h, "F")

        self.set_fill_color(*color)
        self.rect(x, y, w, 2, "F")

        self.set_xy(x, y + 4)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*color)
        self.cell(w, 8, clean(value), align="C")

        self.set_xy(x, y + 12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GRAY)
        self.cell(w, 4, clean(label.upper()), align="C")

        self.set_text_color(*DARK)


# ─────────────────────────────────────────
# MAIN EXPORT
# ─────────────────────────────────────────
def generate_pdf(result, output_path, sport="football"):

    if not FPDF_AVAILABLE:
        print("fpdf2 non dispo")
        return None

    summary    = result.get("summary", {})
    stats      = result.get("stats", {})
    highlights = result.get("highlights", [])
    jersey_map = result.get("jersey_map", {})
    heatmaps   = result.get("heatmaps", {})
    ratings    = result.get("player_ratings", {})
    story      = result.get("match_story")

    pdf = ScoutPDF(sport=sport)
    pdf.add_page()

    # ─────────────────────────
    # TITLE
    # ─────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 10, "MATCH ANALYSIS REPORT", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 6,
        clean(f"{sport} | duration: {summary.get('duration', '--')}"),
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)

    # ─────────────────────────
    # KPI
    # ─────────────────────────
    pdf.section("Match Summary")

    kpis = [
        ("Goals", summary.get("goals", 0), GREEN),
        ("Shots", summary.get("shots", 0), PRIMARY),
        ("xG",    round(summary.get("total_xg", 0), 2), YELLOW),
        ("Pass",  summary.get("passes", 0), PRIMARY),
    ]

    x = 15
    for label, val, color in kpis:
        pdf.kpi(x, pdf.get_y(), 35, 20, label, val, color)
        x += 40

    pdf.ln(25)

    # ─────────────────────────
    # STORY
    # ─────────────────────────
    if story:
        pdf.section("Match Story")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, clean(story))

    # ─────────────────────────
    # HEATMAPS
    # ─────────────────────────
    if heatmaps:
        pdf.section("Heatmaps")

        for name, path in heatmaps.items():
            if path and os.path.exists(path):
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(0, 5, clean(name), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                pdf.image(path, w=120)
                pdf.ln(2)

    # ─────────────────────────
    # PLAYER STATS + RATING
    # ─────────────────────────
    if stats:
        pdf.section("Players Performance")

        pdf.set_font("Helvetica", "B", 8)
        headers = ["Player", "Touches", "Pass", "Shots", "Goals", "Rating"]
        widths  = [30, 25, 20, 20, 20, 25]

        for h, w in zip(headers, widths):
            pdf.cell(w, 6, clean(h), border=0, align="C")
        pdf.ln()

        sorted_players = sorted(
            stats.items(),
            key=lambda x: x[1].get("touches", 0),
            reverse=True
        )[:15]

        pdf.set_font("Helvetica", "", 8)

        for pid, s in sorted_players:
            rating = ratings.get(pid, {}).get("rating", "-")

            row = [
                f"P{pid}",
                s.get("touches", 0),
                s.get("passes", 0),
                s.get("tirs", 0),
                s.get("buts", 0),
                rating
            ]

            for val, w in zip(row, widths):
                pdf.cell(w, 6, clean(val), align="C")
            pdf.ln()

    # ─────────────────────────
    # HIGHLIGHTS
    # ─────────────────────────
    if highlights:
        pdf.section("Top Highlights")

        pdf.set_font("Helvetica", "", 8)

        for i, h in enumerate(highlights[:10]):
            txt = f"{i+1}. {h.get('main_type')} | {h.get('time_start')}s → {h.get('time_end')}s | score {h.get('score')}"
            pdf.cell(0, 5, clean(txt),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ─────────────────────────
    # SAVE
    # ─────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        pdf.output(output_path)
        print(f"PDF OK -> {output_path}")
        return output_path
    except Exception as e:
        print(f"PDF error : {e}")
        return None