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


def clean(text):
    if not text:
        return ""
    return (
        str(text)
        .replace("—", "-").replace("–", "-")
        .replace("'", "'").replace("\u201c", '"').replace("\u201d", '"')
        .replace("«", '"').replace("»", '"')
        .replace("…", "...").replace("•", "-")
        .replace("⭐", "*").replace("⚽", "[goal]").replace("🏀", "[basket]")
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def player_label(pid, jersey_map=None):
    if jersey_map:
        jersey = jersey_map.get(str(pid)) or jersey_map.get(pid)
        if jersey:
            return f"#{jersey}"
    return f"ID-{pid}"


def deduplicate_stats(stats, jersey_map):
    merged = {}
    for pid, s in stats.items():
        label = s.get("label") or player_label(pid, jersey_map)
        if label not in merged:
            merged[label] = {
                "touches":       s.get("touches",       0),
                "passes":        s.get("passes",        0),
                "tirs":          s.get("tirs",          0),
                "buts":          s.get("buts",          0),
                "arrets":        s.get("arrets",        0),
                "xg_total":      s.get("xg_total",      0),
                "is_goalkeeper": s.get("is_goalkeeper", False),
                "_rating":       s.get("_rating",       0),
                "_pid":          pid,
            }
        else:
            merged[label]["touches"]  += s.get("touches",  0)
            merged[label]["passes"]   += s.get("passes",   0)
            merged[label]["tirs"]     += s.get("tirs",     0)
            merged[label]["buts"]     += s.get("buts",     0)
            merged[label]["arrets"]   += s.get("arrets",   0)
            merged[label]["xg_total"] += s.get("xg_total", 0)
            merged[label]["_rating"]   = max(
                merged[label]["_rating"], s.get("_rating", 0)
            )
            # Un joueur est GK si au moins une entrée l'est
            if s.get("is_goalkeeper"):
                merged[label]["is_goalkeeper"] = True

    return merged


def fix_highlight_times(highlights):
    fixed = []
    for h in highlights:
        t_start = float(h.get("time_start") or 0)
        t_end   = float(h.get("time_end")   or 0)
        if t_end <= t_start:
            t_end = t_start + 3.0
        h = dict(h)
        h["time_start"] = t_start
        h["time_end"]   = t_end
        fixed.append(h)
    return fixed


PRIMARY = (20, 110, 220)
DARK    = (25, 25, 25)
GRAY    = (120, 120, 120)
LIGHT   = (245, 245, 245)
WHITE   = (255, 255, 255)
GREEN   = (40, 180, 90)
RED     = (220, 60, 60)
YELLOW  = (250, 180, 20)
CYAN    = (0, 180, 210)


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


def generate_pdf(result, output_path, sport="football"):

    if not FPDF_AVAILABLE:
        print("fpdf2 non dispo")
        return None

    summary    = result.get("summary",        {})
    stats      = result.get("stats",          {})
    highlights = result.get("highlights",     [])
    jersey_map = result.get("jersey_map",     {})
    heatmaps   = result.get("heatmaps",       {})
    ratings    = result.get("player_ratings", {})
    story      = result.get("match_story")
    mvp_id     = result.get("mvp")

    highlights = fix_highlight_times(highlights)

    pdf = ScoutPDF(sport=sport)
    pdf.add_page()

    # ── TITLE ──────────────────────────
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

    # ── KPI ────────────────────────────
    pdf.section("Match Summary")
    kpis = [
        ("Goals", summary.get("goals",     0), GREEN),
        ("Shots", summary.get("shots",     0), PRIMARY),
        ("xG",   round(summary.get("total_xg", 0), 2), YELLOW),
        ("Pass",  summary.get("passes",    0), PRIMARY),
    ]
    x = 15
    for label, val, color in kpis:
        pdf.kpi(x, pdf.get_y(), 35, 20, label, val, color)
        x += 40
    pdf.ln(25)

    # ── MVP ────────────────────────────
    if mvp_id:
        mvp_label = player_label(mvp_id, jersey_map)
        pdf.section("MVP du match")
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*PRIMARY)
        pdf.cell(0, 8, clean(f"Meilleur joueur : {mvp_label}"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*DARK)
        pdf.ln(2)

    # ── STORY ──────────────────────────
    if story:
        pdf.section("Match Story")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, clean(story))

    # ── HEATMAPS ───────────────────────
    if heatmaps:
        pdf.section("Heatmaps")
        for name, path in heatmaps.items():
            if path and os.path.exists(path):
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(0, 5, clean(name),
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.image(path, w=120)
                pdf.ln(2)

    # ── PLAYER STATS ───────────────────
    if stats:
        pdf.section("Players Performance")

        # FIX — deux tableaux séparés : joueurs de champ / gardiens
        deduped = deduplicate_stats(stats, jersey_map)

        field_players = {
            lbl: s for lbl, s in deduped.items()
            if not s.get("is_goalkeeper")
        }
        goalkeepers = {
            lbl: s for lbl, s in deduped.items()
            if s.get("is_goalkeeper")
        }

        # ── Tableau joueurs de champ ──
        pdf.set_font("Helvetica", "B", 8)
        headers_field = ["Joueur", "Touches", "Passes", "Tirs", "Buts", "xG",  "Note"]
        widths_field  = [28,        22,        20,       18,     18,     18,     22]

        for h, w in zip(headers_field, widths_field):
            pdf.cell(w, 6, clean(h), border=0, align="C")
        pdf.ln()

        sorted_field = sorted(
            field_players.items(),
            key=lambda x: x[1].get("touches", 0),
            reverse=True
        )[:18]

        pdf.set_font("Helvetica", "", 8)
        for label, s in sorted_field:
            pid    = s.get("_pid")
            rating = ratings.get(str(pid), {}).get("rating") or \
                     ratings.get(pid,      {}).get("rating", "-") \
                     if pid else "-"
            xg_val = round(s.get("xg_total", 0), 2)

            row = [
                label,
                s.get("touches", 0),
                s.get("passes",  0),
                s.get("tirs",    0),
                s.get("buts",    0),
                xg_val if xg_val > 0 else "",
                rating
            ]
            for val, w in zip(row, widths_field):
                pdf.cell(w, 6, clean(val), align="C")
            pdf.ln()

        # ── Tableau gardiens ──
        if goalkeepers:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*CYAN)
            pdf.cell(0, 5, "Gardien(s)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*DARK)

            headers_gk = ["Joueur", "Touches", "Arrets", "Note"]
            widths_gk  = [28,        22,        22,       22]

            pdf.set_font("Helvetica", "B", 8)
            for h, w in zip(headers_gk, widths_gk):
                pdf.cell(w, 6, clean(h), border=0, align="C")
            pdf.ln()

            pdf.set_font("Helvetica", "", 8)
            for label, s in goalkeepers.items():
                pid    = s.get("_pid")
                rating = ratings.get(str(pid), {}).get("rating") or \
                         ratings.get(pid,      {}).get("rating", "-") \
                         if pid else "-"

                row = [
                    label,
                    s.get("touches", 0),
                    s.get("arrets",  0),
                    rating
                ]
                for val, w in zip(row, widths_gk):
                    pdf.cell(w, 6, clean(val), align="C")
                pdf.ln()

    # ── HIGHLIGHTS ─────────────────────
    if highlights:
        pdf.section("Top Highlights")
        pdf.set_font("Helvetica", "", 8)
        for i, h in enumerate(highlights[:15]):
            t_start = h.get("time_start", 0)
            t_end   = h.get("time_end",   0)
            mins_s  = int(t_start // 60)
            secs_s  = int(t_start % 60)
            mins_e  = int(t_end   // 60)
            secs_e  = int(t_end   % 60)
            htype   = h.get("main_type", "action")
            score   = h.get("score", 0)

            pid     = h.get("player")
            p_label = player_label(pid, jersey_map) if pid else "?"

            txt = (f"{i+1}. {htype.upper()} | "
                   f"{mins_s:02d}:{secs_s:02d} -> {mins_e:02d}:{secs_e:02d} | "
                   f"Joueur {p_label} | score {score:.1f}")
            pdf.cell(0, 5, clean(txt),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── SAVE ───────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        pdf.output(output_path)
        print(f"PDF OK -> {output_path}")
        return output_path
    except Exception as e:
        print(f"PDF error : {e}")
        return None