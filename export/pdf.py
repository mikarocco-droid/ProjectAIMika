# export/pdf.py
# -*- coding: utf-8 -*-
#
# ScoutIA — Rapport PDF V9.6
# Design : dark mode + accent neon + cards + timeline + stats avancees

import os
from datetime import datetime

try:
    from fpdf import FPDF, XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    print("fpdf2 non installe -- export PDF desactive")

import config


# ─────────────────────────────────────────
# PALETTE DARK + NEON
# ─────────────────────────────────────────
BG_DARK   = (13,  17,  23)
BG_CARD   = (22,  30,  40)
BG_MID    = (30,  40,  54)
ACCENT    = (0,   200, 230)
ACCENT2   = (255, 77,  77)
ACCENT3   = (255, 200, 0)
ACCENT4   = (80,  200, 120)
ACCENT5   = (180, 100, 255)
WHITE     = (255, 255, 255)
GRAY_L    = (170, 180, 195)
GRAY_D    = (70,  80,  95)
HEADER_BG = (8,   12,  20)


# ─────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────
def clean(text):
    if text is None:
        return ""
    return (
        str(text)
        .replace("\u2014", "-").replace("\u2013", "-")
        .replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u00ab", '"').replace("\u00bb", '"')
        .replace("\u2026", "...").replace("\u2022", "-")
        .replace("\u2b50", "*").replace("\u26bd", "[G]")
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def fmt_time(seconds):
    seconds = float(seconds or 0)
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def player_label(pid, jersey_map=None):
    if pid is None:
        return "?"
    pid_str = str(pid)
    # Eviter le double # si pid contient déjà un #
    if pid_str.startswith("#"):
        return pid_str
    if jersey_map:
        jersey = jersey_map.get(pid_str) or jersey_map.get(pid)
        if jersey:
            return f"#{jersey}"
    return f"#{pid_str}"


def deduplicate_stats(stats, jersey_map):
    merged = {}
    for pid, s in stats.items():
        label = s.get("label") or player_label(pid, jersey_map)
        if label not in merged:
            merged[label] = {
                "touches":          s.get("touches",          0),
                "passes":           s.get("passes", s.get("passes_total", s.get("n_passes", 0))),
                "key_passes":       s.get("key_passes",        0),
                "tirs":             s.get("tirs",              0),
                "buts":             s.get("buts",              0),
                "arrets":           s.get("arrets",            0),
                "interceptions":    s.get("interceptions",     0),
                "dribbles":         s.get("dribbles",          0),
                "progressive_runs": s.get("progressive_runs",  0),
                "xg_total":         s.get("xg_total",          0.0),
                "xa_total":         s.get("xa_total",          0.0),
                "is_goalkeeper":    s.get("is_goalkeeper",     False),
                "_rating":          s.get("_rating",           0),
                "_pid":             pid,
                "team":             s.get("team"),
            }
        else:
            for k in ["touches","passes","key_passes","tirs","buts","arrets",
                      "interceptions","dribbles","progressive_runs"]:
                merged[label][k] += s.get(k, 0)
            merged[label]["xg_total"] += s.get("xg_total", 0)
            merged[label]["xa_total"] += s.get("xa_total", 0)
            merged[label]["_rating"]   = max(merged[label]["_rating"], s.get("_rating", 0))
            if s.get("is_goalkeeper"):
                merged[label]["is_goalkeeper"] = True
    return merged


def fix_highlight_times(highlights, fps=25):
    fixed = []
    for h in highlights:
        h = dict(h)
        t_start = float(h.get("time_start") or 0)
        t_end   = float(h.get("time_end")   or 0)
        # Si time_start=0 mais frame>0, recalculer depuis frame
        if t_start == 0 and h.get("frame", 0):
            t_start = round(float(h["frame"]) / fps, 2)
        # Clip de but : remonter context_before secondes
        is_goal = (h.get("main_type") or "").lower() in ("goal", "score")
        if is_goal and t_start > 0:
            t_start = max(0, t_start - 12)  # context_before=12s
        if t_end <= t_start:
            t_end = t_start + (17 if is_goal else 7)
        h["time_start"] = t_start
        h["time_end"]   = t_end

        # Corriger main_type basé sur l'event source (pas le titre Gemini)
        # Un highlight ne devient "goal" que si l'event est réellement un but
        src_type = (h.get("main_type") or h.get("type") or "shot").lower()
        if src_type in ("goal", "score", "but"):
            h["main_type"] = "goal"
        elif src_type in ("shot", "tir", "shot_on_target",
                          "shot_missed", "big_chance"):
            h["main_type"] = "shot"
        else:
            h["main_type"] = "shot"  # défaut conservateur

        fixed.append(h)
    return fixed


def compute_team_stats(stats):
    teams = {}
    for pid, s in stats.items():
        team = s.get("team")
        if team is None:
            continue
        if team not in teams:
            teams[team] = {"passes": 0, "tirs": 0, "buts": 0,
                           "xg": 0.0, "interceptions": 0, "dribbles": 0}
        teams[team]["passes"]        += s.get("passes",        0)
        teams[team]["tirs"]          += s.get("tirs",          0)
        teams[team]["buts"]          += s.get("buts",          0)
        teams[team]["xg"]            += s.get("xg_total",      0)
        teams[team]["interceptions"] += s.get("interceptions", 0)
        teams[team]["dribbles"]      += s.get("dribbles",      0)
    return teams


# ─────────────────────────────────────────
# CLASSE PDF
# ─────────────────────────────────────────
class ScoutPDF(FPDF):

    def __init__(self, sport="football"):
        super().__init__()
        self.sport = clean(sport).upper()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(12, 18, 12)

    def _fill_bg(self):
        self.set_fill_color(*BG_DARK)
        self.rect(0, 0, 210, 297, "F")

    def header(self):
        self._fill_bg()
        self.set_fill_color(*HEADER_BG)
        self.rect(0, 0, 210, 14, "F")
        self.set_fill_color(*ACCENT)
        self.rect(0, 13, 210, 1, "F")
        self.set_text_color(*ACCENT)
        self.set_font("Helvetica", "B", 8)
        self.set_xy(12, 4)
        self.cell(60, 5, "SCOUT AI", align="L")
        self.set_text_color(*GRAY_L)
        self.set_font("Helvetica", "", 7)
        self.set_xy(0, 4)
        self.cell(198, 5, self.sport, align="R")
        self.set_text_color(*WHITE)
        self.ln(10)

    def footer(self):
        self.set_y(-10)
        self.set_fill_color(*GRAY_D)
        self.rect(12, self.get_y(), 186, 0.3, "F")
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GRAY_D)
        self.set_xy(12, self.get_y() + 1)
        self.cell(0, 4,
            f"Scout AI  |  {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  Page {self.page_no()}",
            align="C")

    def section_title(self, title):
        self.ln(4)
        y = self.get_y()
        self.set_fill_color(*ACCENT)
        self.rect(12, y, 3, 8, "F")
        self.set_text_color(*ACCENT)
        self.set_font("Helvetica", "B", 10)
        self.set_xy(18, y)
        self.cell(0, 8, clean(title), align="L")
        self.set_text_color(*WHITE)
        self.ln(10)

    def card_bg(self, x, y, w, h, color=None):
        self.set_fill_color(*(color or BG_CARD))
        self.rect(x, y, w, h, "F")

    def kpi_card(self, x, y, w, label, value, color):
        self.card_bg(x, y, w, 22)
        self.set_fill_color(*color)
        self.rect(x, y, w, 2, "F")
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*color)
        self.set_xy(x, y + 3)
        self.cell(w, 10, clean(str(value)), align="C")
        self.set_font("Helvetica", "", 6)
        self.set_text_color(*GRAY_L)
        self.set_xy(x, y + 14)
        self.cell(w, 5, clean(label.upper()), align="C")
        self.set_text_color(*WHITE)

    def progress_bar(self, x, y, w, pct, color, h=3):
        self.set_fill_color(*BG_MID)
        self.rect(x, y, w, h, "F")
        fw = max(1, int(w * min(float(pct or 0), 1.0)))
        self.set_fill_color(*color)
        self.rect(x, y, fw, h, "F")

    def table_row(self, cells, widths, is_header=False, alt=False, highlight=False):
        y       = self.get_y()
        x_start = 12
        total_w = sum(widths)

        if highlight:
            self.set_fill_color(*ACCENT2)
            self.rect(x_start, y, total_w, 7, "F")
        elif is_header:
            self.set_fill_color(*BG_MID)
            self.rect(x_start, y, total_w, 7, "F")
        elif alt:
            self.set_fill_color(*BG_CARD)
            self.rect(x_start, y, total_w, 7, "F")

        self.set_font("Helvetica", "B" if is_header else "", 6.5 if is_header else 7)
        if is_header:
            self.set_text_color(*ACCENT)
        elif highlight:
            self.set_font("Helvetica", "B", 7)
            self.set_text_color(*WHITE)
        else:
            self.set_text_color(*GRAY_L)

        x = x_start
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.set_xy(x, y)
            self.cell(w, 7, clean(str(cell)), align="L" if i == 0 else "C")
            x += w

        self.set_text_color(*WHITE)
        self.ln(7)


# ─────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────
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
    story      = result.get("match_story",    "")
    mvp_id     = result.get("mvp")
    formation  = result.get("formation",      "")
    tactical   = result.get("tactical",       {})
    possession = result.get("possession",     summary.get("possession", {}))
    commentary = result.get("commentary",     [])

    highlights = fix_highlight_times(highlights)
    deduped    = deduplicate_stats(stats, jersey_map) if stats else {}
    team_stats = compute_team_stats(stats) if stats else {}

    pdf = ScoutPDF(sport=sport)

    # ════════════════════════
    # PAGE 1 — COVER + KPI
    # ════════════════════════
    pdf.add_page()
    pdf.ln(8)

    # Titre
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, "SCOUT AI", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GRAY_L)
    pdf.cell(0, 5, "MATCH ANALYSIS REPORT", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_fill_color(*ACCENT)
    pdf.rect(70, pdf.get_y(), 70, 0.8, "F")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GRAY_D)
    dur = summary.get("duration", "--")
    pdf.cell(0, 5,
        clean(f"{sport.title()}  |  {dur}  |  {datetime.now().strftime('%d %b %Y')}"),
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # KPIs
    pdf.section_title("Match Summary")
    kpis = [
        ("Goals",   summary.get("goals",    0),               ACCENT2),
        ("Tirs cadrés", summary.get("shots", 0),             ACCENT),
        ("xG",      round(summary.get("total_xg", 0), 2),     ACCENT3),
        ("Passes",  summary.get("passes",   0),               ACCENT4),
        ("Players", summary.get("players",  0),               GRAY_L),
    ]
    y0 = pdf.get_y()
    x  = 12
    for label, val, color in kpis:
        pdf.kpi_card(x, y0, 34, label, val, color)
        x += 38
    pdf.set_y(y0 + 27)

    # Possession
    if possession:
        teams = sorted(possession.items())
        if len(teams) >= 2:
            t0, p0 = teams[0]
            t1, p1 = teams[1]
            cy = pdf.get_y() + 3
            pdf.card_bg(12, cy, 186, 20)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*ACCENT)
            pdf.set_xy(15, cy + 3)
            pdf.cell(180, 5, "POSSESSION", align="L")
            bx, by, bw = 15, cy + 10, 180
            pdf.set_fill_color(*ACCENT)
            pdf.rect(bx, by, int(bw * float(p0) / 100), 5, "F")
            fill1 = int(bw * float(p1) / 100)
            pdf.set_fill_color(*ACCENT2)
            pdf.rect(bx + bw - fill1, by, fill1, 5, "F")
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*ACCENT)
            pdf.set_xy(15, by)
            pdf.cell(50, 5, f"  {p0}%", align="L")
            pdf.set_text_color(*ACCENT2)
            pdf.set_xy(145, by)
            pdf.cell(50, 5, f"{p1}%  ", align="R")
            pdf.set_y(cy + 24)

    # MVP + Formation
    pdf.ln(3)
    yr = pdf.get_y()
    if mvp_id:
        mvp_lbl = player_label(mvp_id, jersey_map)
        pdf.card_bg(12, yr, 88, 26)
        pdf.set_fill_color(*ACCENT3)
        pdf.rect(12, yr, 88, 2, "F")
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*ACCENT3)
        pdf.set_xy(14, yr + 4)
        pdf.cell(84, 5, "MVP DU MATCH", align="L")
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(14, yr + 11)
        pdf.cell(84, 11, clean(mvp_lbl), align="L")

    if formation:
        pdf.card_bg(106, yr, 92, 26)
        pdf.set_fill_color(*ACCENT)
        pdf.rect(106, yr, 92, 2, "F")
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*ACCENT)
        pdf.set_xy(108, yr + 4)
        pdf.cell(88, 5, "FORMATION / STYLE", align="L")
        style = (tactical.get("style", "") if tactical else "")
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(108, yr + 11)
        pdf.cell(88, 11, clean(f"{formation}  {style}"), align="L")
    pdf.set_y(yr + 30)

    # Stats par equipe
    if len(team_stats) >= 2:
        pdf.section_title("Team Comparison")
        cols   = ["TEAM", "PASSES", "CADRÉS", "GOALS", "xG", "INTERC.", "DRIBBLES"]
        widths = [24, 27, 27, 27, 27, 27, 27]
        pdf.table_row(cols, widths, is_header=True)
        for idx, (team, ts) in enumerate(sorted(team_stats.items())):
            row = [
                f"Team {team}",
                ts.get("passes", 0),
                ts.get("tirs",   0),
                ts.get("buts",   0),
                round(ts.get("xg", 0), 2),
                ts.get("interceptions", 0),
                ts.get("dribbles",      0),
            ]
            pdf.table_row(row, widths, alt=(idx % 2 == 0))

    # ════════════════════════
    # PAGE 2 — HEATMAPS
    # ════════════════════════
    if heatmaps:
        pdf.add_page()
        pdf.section_title("Heatmaps")

        hmap_list = [(k, v) for k, v in heatmaps.items()
                     if v and os.path.exists(str(v))]
        i = 0
        while i < len(hmap_list):
            yh = pdf.get_y()
            for col in range(2):
                if i >= len(hmap_list):
                    break
                name, path = hmap_list[i]
                xh = 12 + col * 96
                pdf.card_bg(xh, yh, 92, 64)
                pdf.set_fill_color(*ACCENT)
                pdf.rect(xh, yh, 92, 2, "F")
                pdf.set_font("Helvetica", "B", 7)
                pdf.set_text_color(*ACCENT)
                pdf.set_xy(xh + 2, yh + 3)
                pdf.cell(88, 5, clean(name.upper()), align="C")
                try:
                    pdf.image(str(path), x=xh + 2, y=yh + 9, w=88)
                except Exception:
                    pdf.set_font("Helvetica", "", 7)
                    pdf.set_text_color(*GRAY_D)
                    pdf.set_xy(xh + 2, yh + 30)
                    pdf.cell(88, 5, "n/a", align="C")
                i += 1
            pdf.set_y(yh + 68)
            pdf.ln(2)

    # ════════════════════════
    # PAGE 3 — JOUEURS
    # ════════════════════════
    if deduped:
        pdf.add_page()
        pdf.section_title("Players Performance")

        field = {l: s for l, s in deduped.items() if not s.get("is_goalkeeper")}
        gks   = {l: s for l, s in deduped.items() if s.get("is_goalkeeper")}
        max_xg = max((s.get("xg_total", 0) for s in field.values()), default=1) or 1

        headers = ["Joueur", "Touch.", "Passes", "K.Pass", "Tirs", "Buts", "xG", "xA", "Note"]
        widths  = [28, 18, 18, 16, 14, 14, 16, 16, 16]
        pdf.table_row(headers, widths, is_header=True)

        sorted_field = sorted(
            field.items(),
            key=lambda x: x[1].get("touches", 0), reverse=True
        )[:20]

        xG_col_x = sum(widths[:6]) + 12  # position x de la colonne xG

        for idx, (label, s) in enumerate(sorted_field):
            pid    = s.get("_pid")
            rating = "-"
            if pid:
                r = ratings.get(str(pid), {}) or ratings.get(pid, {})
                if r and r.get("rating"):
                    rating = round(float(r["rating"]), 1)

            xg_val = round(s.get("xg_total", 0), 2)
            xa_val = round(s.get("xa_total", 0), 2)
            buts   = s.get("buts", 0)
            kp     = s.get("key_passes", 0)

            row = [
                label,
                s.get("touches", 0),
                s.get("passes",  0),
                kp if kp > 0 else "",
                s.get("tirs",    0),
                buts   if buts   > 0 else "",
                xg_val if xg_val > 0 else "",
                xa_val if xa_val > 0 else "",
                rating,
            ]
            pdf.table_row(row, widths, alt=(idx % 2 == 0), highlight=(buts > 0))

            # Barre xG
            if xg_val > 0:
                ybar = pdf.get_y() - 1
                pdf.progress_bar(xG_col_x, ybar, 16,
                                 xg_val / float(max_xg), ACCENT3, h=2)

        # Gardiens
        if gks:
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*ACCENT)
            pdf.cell(0, 5, "GOALKEEPERS",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            headers_gk = ["Joueur", "Touches", "Arrets", "Note"]
            widths_gk  = [32, 30, 30, 30]
            pdf.table_row(headers_gk, widths_gk, is_header=True)
            for idx, (label, s) in enumerate(gks.items()):
                pid    = s.get("_pid")
                rating = "-"
                if pid:
                    r = ratings.get(str(pid), {}) or ratings.get(pid, {})
                    if r and r.get("rating"):
                        rating = round(float(r["rating"]), 1)
                pdf.table_row(
                    [label, s.get("touches", 0), s.get("arrets", 0), rating],
                    widths_gk, alt=(idx % 2 == 0)
                )

        # Top Performers badges
        pdf.ln(5)
        pdf.section_title("Top Performers")
        yb = pdf.get_y()

        def top_by(key):
            candidates = [(l, s) for l, s in field.items() if s.get(key, 0) > 0]
            if not candidates:
                return "-"
            return max(candidates, key=lambda x: x[1].get(key, 0))[0]

        badges = [
            ("Top Buteur",   top_by("buts"),          ACCENT2),
            ("Top xG",       top_by("xg_total"),       ACCENT3),
            ("Top Passeur",  top_by("passes"),         ACCENT4),
            ("Top Interc.",  top_by("interceptions"),  ACCENT),
        ]
        bx = 12
        for blabel, bval, bcolor in badges:
            pdf.card_bg(bx, yb, 43, 18)
            pdf.set_fill_color(*bcolor)
            pdf.rect(bx, yb, 43, 2, "F")
            pdf.set_font("Helvetica", "", 6)
            pdf.set_text_color(*GRAY_L)
            pdf.set_xy(bx, yb + 3)
            pdf.cell(43, 4, clean(blabel), align="C")
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*bcolor)
            pdf.set_xy(bx, yb + 8)
            pdf.cell(43, 7, clean(str(bval)), align="C")
            bx += 46
        pdf.set_y(yb + 22)

    # ════════════════════════
    # PAGE 4 — HIGHLIGHTS
    # ════════════════════════
    if highlights:
        pdf.add_page()
        pdf.section_title("Top Highlights")

        try:
            dur_str = str(summary.get("duration", "15:00"))
            parts   = dur_str.split(":")
            dur_s   = int(parts[0]) * 60 + int(parts[1])
        except Exception:
            dur_s = 900

        # Timeline
        tl_x, tl_y, tl_w = 12, pdf.get_y(), 186
        pdf.card_bg(tl_x, tl_y, tl_w, 16)
        pdf.set_fill_color(*GRAY_D)
        pdf.rect(tl_x + 5, tl_y + 10, tl_w - 10, 1.5, "F")

        for h in highlights[:15]:
            t     = float(h.get("time_start", 0))
            htype = h.get("main_type", "shot")
            color = ACCENT2 if htype in ("goal", "score") else ACCENT
            pct   = min(t / max(dur_s, 1), 0.98)
            mx    = tl_x + 5 + int((tl_w - 10) * pct)
            pdf.set_fill_color(*color)
            pdf.rect(mx - 2, tl_y + 7, 4, 6, "F")
            pdf.set_font("Helvetica", "", 5)
            pdf.set_text_color(*GRAY_L)
            pdf.set_xy(mx - 5, tl_y + 11)
            pdf.cell(10, 4, fmt_time(t), align="C")

        # Legende
        pdf.set_font("Helvetica", "", 6)
        pdf.set_text_color(*ACCENT2)
        pdf.set_xy(tl_x + 5, tl_y + 2)
        pdf.cell(20, 4, "BUT", align="L")
        pdf.set_text_color(*ACCENT)
        pdf.set_xy(tl_x + 28, tl_y + 2)
        pdf.cell(20, 4, "TIR", align="L")
        pdf.set_y(tl_y + 20)

        # Cartes highlights
        for i, h in enumerate(highlights[:15]):
            yh      = pdf.get_y()
            t_start = h.get("time_start", 0)
            htype   = h.get("main_type",  "action")
            xg_val  = h.get("xg", 0) or 0
            score   = h.get("score", 0)
            pid     = h.get("player")
            p_label = player_label(pid, jersey_map) if pid else "?"
            reason  = h.get("reason", "")
            is_goal = htype in ("goal", "score")
            color   = ACCENT2 if is_goal else ACCENT

            pdf.card_bg(12, yh, 186, 13, BG_CARD)
            pdf.set_fill_color(*color)
            pdf.rect(12, yh, 3, 13, "F")

            # Num
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*color)
            pdf.set_xy(16, yh + 2)
            pdf.cell(10, 9, f"{i+1:02d}")

            # Badge type
            pdf.set_fill_color(*color)
            pdf.rect(27, yh + 3, 16, 6, "F")
            pdf.set_font("Helvetica", "B", 5.5)
            pdf.set_text_color(*HEADER_BG if is_goal else BG_DARK)
            pdf.set_xy(27, yh + 4)
            pdf.cell(16, 5, htype[:4].upper(), align="C")

            # Timestamp
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*WHITE)
            pdf.set_xy(46, yh + 2)
            pdf.cell(22, 9, fmt_time(t_start))

            # Joueur
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*GRAY_L)
            pdf.set_xy(70, yh + 2)
            pdf.cell(28, 9, clean(p_label))

            # xG + barre
            if xg_val and float(xg_val) > 0:
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*ACCENT3)
                pdf.set_xy(101, yh + 2)
                pdf.cell(22, 9, f"xG {float(xg_val):.2f}")
                pdf.progress_bar(101, yh + 9, 22, min(float(xg_val), 1.0), ACCENT3, h=2)

            # Raison
            pdf.set_font("Helvetica", "", 6.5)
            pdf.set_text_color(*GRAY_D)
            pdf.set_xy(126, yh + 2)
            pdf.cell(70, 9, clean(reason or f"score {score:.1f}"), align="R")

            pdf.set_text_color(*WHITE)
            pdf.set_y(yh + 15)

    # ════════════════════════
    # PAGE 5 — STORY
    # ════════════════════════
    if story or commentary:
        pdf.add_page()

        if story:
            pdf.section_title("Match Story")
            pdf.card_bg(12, pdf.get_y(), 186, 4)
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*GRAY_L)
            pdf.set_x(14)
            pdf.multi_cell(182, 5, clean(story))
            pdf.ln(5)

        if commentary:
            pdf.section_title("Commentary")
            for i, line in enumerate(commentary[:12]):
                yc = pdf.get_y()
                if i % 2 == 0:
                    pdf.card_bg(12, yc, 186, 8, BG_CARD)
                pdf.set_fill_color(*ACCENT)
                pdf.rect(12, yc, 2, 8, "F")
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(*GRAY_L)
                pdf.set_xy(16, yc + 1)
                pdf.multi_cell(180, 5, clean(str(line)))
                if pdf.get_y() < yc + 9:
                    pdf.set_y(yc + 9)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        pdf.output(output_path)
        print(f"PDF OK -> {output_path}")
        return output_path
    except Exception as e:
        print(f"PDF error : {e}")
        return None