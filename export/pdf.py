# export/pdf.py
# -*- coding: utf-8 -*-
#
# ScoutIA — Rapport PDF V9.7
# FIXES :
#   - Bug 1  : KPIs shots/xG — fallback sur clés alternatives (shots_on_target, total_xg, xg, etc.)
#   - Bug 2  : Timestamp highlights — afficher event_time/time, pas time_start (début clip)
#   - Bug 3  : Stats joueurs vides — alignement clés + log debug + fallback multi-alias
#   - Bug 4  : Top Buteur vide — fallback clés "goals"/"score" en plus de "buts"
#   - Bug 5  : Formation hardcodée — masquer bloc si formation vide
#   - Bug 6  : Commentaires génériques — placeholder si commentary[] vide
#   - Bug 7  : fix_highlight_times écrasait main_type en "shot" — garder type original

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


def bgr_to_name(bgr):
    """Convertit une couleur BGR en nom lisible."""
    if not bgr:
        return None
    try:
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    except Exception:
        return None
    # Blanc / Noir
    if r > 200 and g > 200 and b > 200: return "Blanc"
    if r < 60  and g < 60  and b < 60:  return "Noir"
    # Rouge vif
    if r > 160 and g < 80  and b < 80:  return "Rouge"
    # Bordeaux (rouge fonce avec composante bleu)
    if r > 100 and r > b and r > g and b > 40 and g < 80 and r < 180: return "Bordeaux"
    # Bleu roi
    if b > 150 and r < 100 and g < 130: return "Bleu"
    # Bleu marine / bleu fonce (composante B dominante mais sombre)
    if b > r and b > g and b > 60:      return "Bleu marine"
    # Vert
    if g > 150 and r < 100 and b < 100: return "Vert"
    # Jaune
    if r > 200 and g > 200 and b < 80:  return "Jaune"
    # Violet
    if r > 120 and g < 80  and b > 120: return "Violet"
    # Orange
    if r > 200 and g > 100 and b < 80:  return "Orange"
    # Fallback : couleur dominante
    mx = max(r, g, b)
    if mx == r: return "Rouge fonce"
    if mx == b: return "Bleu fonce"
    if mx == g: return "Vert fonce"
    return None


def team_display(team_id, teams_data=None):
    """
    Retourne le nom d'equipe dans cet ordre de priorite :
    1. Nom utilisateur (ex: "RSC Stavelot B")
    2. Couleur detectee (ex: "Equipe Bordeaux")
    3. Fallback generique (ex: "Equipe A")
    """
    fallback = "Equipe A" if str(team_id) in ("0", "team_0") else "Equipe B"
    if not teams_data:
        return fallback
    tdata = teams_data.get(team_id) or teams_data.get(str(team_id)) or {}

    # Priorite 1 — nom utilisateur fourni lors de la configuration
    user_name = tdata.get("name")
    if user_name and str(user_name).strip():
        return str(user_name).strip()

    # Priorite 2 — couleur detectee par KMeans
    bgr  = tdata.get("color_bgr")
    name = bgr_to_name(bgr)
    if name:
        color_label = tdata.get("color_name") or name
        return f"Equipe {color_label}"

    return fallback


def player_label(pid, jersey_map=None):
    if pid is None:
        return "?"
    pid_str = str(pid)
    if pid_str.startswith("#"):
        return pid_str
    if jersey_map:
        jersey = jersey_map.get(pid_str) or jersey_map.get(pid)
        if jersey:
            return f"#{jersey}"
    # PIDs DeepSort non résolus (grands entiers genre 2910, 3053...) → "?"
    # Un vrai numéro de maillot est ≤ 99, un PID DeepSort est souvent > 200
    try:
        if int(pid_str) > 200:
            return "?"
    except (ValueError, TypeError):
        pass
    return f"#{pid_str}"


# FIX 1 — résoudre une valeur depuis un dict avec plusieurs alias possibles
def _get_any(d, *keys, default=0):
    """Retourne la première valeur non-None trouvée parmi les clés."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def deduplicate_stats(stats, jersey_map):
    merged = {}
    for pid, s in stats.items():
        label = s.get("label") or player_label(pid, jersey_map)
        if label not in merged:
            merged[label] = {
                # FIX 3 — multi-alias pour chaque stat
                "touches":          _get_any(s, "touches", "touch", default=0),
                "passes":           _get_any(s, "passes", "passes_total", "n_passes", "pass_count", default=0),
                "key_passes":       _get_any(s, "key_passes", "keypasses", "key_pass", default=0),
                "tirs":             _get_any(s, "tirs", "shots", "shots_total", "shot_count", default=0),
                "buts":             _get_any(s, "buts", "goals", "score", "goal_count", default=0),
                "arrets":           _get_any(s, "arrets", "saves", "save_count", default=0),
                "interceptions":    _get_any(s, "interceptions", "intercepts", default=0),
                "dribbles":         _get_any(s, "dribbles", "dribble_count", default=0),
                "progressive_runs": _get_any(s, "progressive_runs", "prog_runs", default=0),
                "xg_total":         float(_get_any(s, "xg_total", "xg", "expected_goals", default=0.0)),
                "xa_total":         float(_get_any(s, "xa_total", "xa", "expected_assists", default=0.0)),
                "is_goalkeeper":    s.get("is_goalkeeper", False),
                "_rating":          _get_any(s, "_rating", "rating", default=0),
                "_pid":             pid,
                "team":             s.get("team"),
            }
        else:
            for k in ["touches", "passes", "key_passes", "tirs", "buts", "arrets",
                      "interceptions", "dribbles", "progressive_runs"]:
                merged[label][k] += s.get(k, 0)
            merged[label]["xg_total"] += float(_get_any(s, "xg_total", "xg", default=0.0))
            merged[label]["xa_total"] += float(_get_any(s, "xa_total", "xa", default=0.0))
            merged[label]["_rating"]   = max(merged[label]["_rating"], _get_any(s, "_rating", "rating", default=0))
            if s.get("is_goalkeeper"):
                merged[label]["is_goalkeeper"] = True
    return merged


# FIX 2 — afficher le temps de l'EVENT, pas le début du clip
def get_event_time(h, fps=25):
    """
    Retourne le timestamp réel de l'action (but, tir…),
    pas le début du clip qui inclut le context_before.
    """
    # Priorité : champ explicite event_time > time > frame
    t = h.get("event_time") or h.get("time")
    if t:
        return float(t)
    frame = h.get("frame")
    if frame:
        return float(frame) / fps
    # Dernier recours : time_start + context_before (25s buts, 12s tirs)
    t_start = float(h.get("time_start", 0))
    is_goal = (h.get("main_type") or "").lower() in ("goal", "score")
    context = 25.0 if is_goal else 12.0
    return t_start + context


def fix_highlight_times(highlights, fps=25):
    fixed = []
    for h in highlights:
        h = dict(h)
        t_start = float(h.get("time_start") or 0)
        t_end   = float(h.get("time_end")   or 0)

        if t_start == 0 and h.get("frame", 0):
            t_start = round(float(h["frame"]) / fps, 2)

        # FIX 7 — détecter le type source AVANT d'ajuster les temps
        src_type = (h.get("main_type") or h.get("type") or "").lower()

        # Fix badge SHOT/GOAL — un highlight "shot" peut être un but confirmé
        # si sa source contient "goal" ou si son event source est de type goal
        _src_str = str(h.get("source", "") or h.get("detected_from", "")).lower()
        _is_confirmed_goal = (
            "goal" in _src_str          # shot_to_goal_gemini, shot_to_goal_hybrid…
            or h.get("gemini_validated") is True
            or h.get("gemini_type") == "goal"
            or (h.get("events") and any(
                e.get("type") in ("goal", "score") for e in h.get("events", [])
            ))
        )
        if _is_confirmed_goal and src_type in ("shot", "tir", "shot_on_target",
                                                "shot_missed", "big_chance"):
            src_type = "goal"

        is_goal = src_type in ("goal", "score", "but")
        is_shot = src_type in ("shot", "tir", "shot_on_target", "shot_missed", "big_chance")

        if is_goal and t_start > 0:
            t_start = max(0, t_start - 25)
        elif is_shot and t_start > 0:
            t_start = max(0, t_start - 12)

        if t_end <= t_start:
            t_end = t_start + (17 if is_goal else 7)

        h["time_start"] = t_start
        h["time_end"]   = t_end

        # FIX 7 — ne pas écraser main_type en "shot" par défaut
        # Garder le type original si reconnu, sinon "action" (pas "shot")
        if is_goal:
            h["main_type"] = "goal"
        elif is_shot:
            h["main_type"] = "shot"
        elif src_type:
            h["main_type"] = src_type  # garder tel quel (fast_break, dribble…)
        else:
            h["main_type"] = "action"  # défaut neutre, pas "shot"

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
        teams[team]["passes"]        += _get_any(s, "passes", "passes_total", "n_passes", default=0)
        teams[team]["tirs"]          += _get_any(s, "tirs", "shots", "shots_on_target", default=0)
        teams[team]["buts"]          += _get_any(s, "buts", "goals", default=0)
        teams[team]["xg"]            += float(_get_any(s, "xg_total", "xg", default=0.0))
        teams[team]["interceptions"] += _get_any(s, "interceptions", default=0)
        teams[team]["dribbles"]      += _get_any(s, "dribbles", default=0)
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

    # FIX 3 — log debug pour identifier les clés réelles du pipeline
    _stats_raw = result.get("stats", {})
    if _stats_raw:
        _sample_pid = next(iter(_stats_raw))
        print(f"[PDF DEBUG] stats keys sample (pid={_sample_pid}): "
              f"{list(_stats_raw[_sample_pid].keys())}")
    _summary_raw = result.get("summary", {})
    print(f"[PDF DEBUG] summary keys: {list(_summary_raw.keys())}")

    summary    = _summary_raw
    stats      = _stats_raw
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
    teams_data = result.get("teams",          {})

    highlights = fix_highlight_times(highlights)
    deduped    = deduplicate_stats(stats, jersey_map) if stats else {}
    team_stats = compute_team_stats(stats) if stats else {}

    # FIX 1 — KPIs avec fallback multi-alias
    # PATCH : kpi_goals depuis les highlights validés (buts affichés)
    # évite de compter les FP que Gemini a validés mais qui sont filtrés ensuite
    _highlights_goals = [
        h for h in highlights
        if (h.get("main_type") or "").lower() in ("goal", "score")
    ]
    kpi_goals  = len(_highlights_goals) if _highlights_goals else _get_any(summary, "goals", "goal_count", "buts", default=0)
    kpi_shots  = _get_any(summary, "shots",  "shots_on_target", "tirs_cadres", "shots_total", default=0)
    kpi_xg     = round(float(_get_any(summary, "total_xg", "xg", "xg_total", default=0.0)), 2)
    kpi_passes = _get_any(summary, "passes", "pass_count",  "n_passes", default=0)
    kpi_players= _get_any(summary, "players","player_count","nb_joueurs", default=0)

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

    # KPIs — FIX 1 : utilise kpi_* calculés avec fallbacks
    pdf.section_title("Match Summary")
    kpis = [
        ("Goals",        kpi_goals,   ACCENT2),
        ("Tirs cadres",  kpi_shots,   ACCENT),
        ("xG",           kpi_xg,      ACCENT3),
        ("Passes",       kpi_passes,  ACCENT4),
        ("Joueurs",      kpi_players, GRAY_L),
    ]
    y0 = pdf.get_y()
    x  = 12
    for label, val, color in kpis:
        pdf.kpi_card(x, y0, 34, label, val, color)
        x += 38
    pdf.set_y(y0 + 27)

    # Possession
    if possession:
        _unreliable = possession.pop("_unreliable", False)
        teams = sorted((k, v) for k, v in possession.items() if isinstance(k, int))
        cy = pdf.get_y() + 3
        pdf.card_bg(12, cy, 186, 20)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*ACCENT)
        pdf.set_xy(15, cy + 3)
        pdf.cell(180, 5, "POSSESSION", align="L")
        if not _unreliable and len(teams) >= 2:
            t0, p0 = teams[0]
            t1, p1 = teams[1]
            bx, by, bw = 15, cy + 10, 180
            pdf.set_fill_color(*ACCENT)
            pdf.rect(bx, by, int(bw * float(p0) / 100), 5, "F")
            fill1 = int(bw * float(p1) / 100)
            pdf.set_fill_color(*ACCENT2)
            pdf.rect(bx + bw - fill1, by, fill1, 5, "F")
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*ACCENT)
            pdf.set_xy(15, cy + 10)
            pdf.cell(50, 5, f"  {p0}%", align="L")
            pdf.set_text_color(*ACCENT2)
            pdf.set_xy(145, cy + 10)
            pdf.cell(50, 5, f"{p1}%  ", align="R")
        else:
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*GRAY_D)
            pdf.set_xy(15, cy + 10)
            pdf.cell(180, 5, "Données insuffisantes", align="C")
        pdf.set_y(cy + 24)

    # MVP + Formation — FIX 5 : masquer formation si vide
    pdf.ln(3)
    yr = pdf.get_y()
    if mvp_id:
        mvp_lbl = player_label(mvp_id, jersey_map)
        _mvp_team = None
        for _pid, _s in stats.items():
            if str(_pid) == str(mvp_id) or player_label(_pid, jersey_map) == mvp_lbl:
                _mvp_team = _s.get("team")
                break
        _mvp_team_name = team_display(_mvp_team, teams_data) if _mvp_team is not None else ""
        pdf.card_bg(12, yr, 88, 26)
        pdf.set_fill_color(*ACCENT3)
        pdf.rect(12, yr, 88, 2, "F")
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*ACCENT3)
        pdf.set_xy(14, yr + 4)
        _mvp_subtitle = ("MVP DU MATCH — " + clean(_mvp_team_name)) if _mvp_team_name else "MVP DU MATCH"
        pdf.cell(84, 5, _mvp_subtitle, align="L")
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(14, yr + 11)
        pdf.cell(84, 11, clean(mvp_lbl), align="L")

    # Formation par equipe si disponible, sinon formation globale
    _style = (tactical.get("style", "") if tactical else "")
    _formations_by_team = {}
    if tactical and isinstance(tactical, dict):
        if tactical.get("formations"):
            _formations_by_team = tactical["formations"]
    if _formations_by_team and len(_formations_by_team) >= 2:
        _lines = []
        for _tid, _f in sorted(_formations_by_team.items()):
            _tname = team_display(_tid, teams_data)
            _lines.append(f"{clean(_tname)}: {_f}")
        _formation_display = " | ".join(_lines)
    elif formation:
        # PATCH : une seule formation → afficher les deux équipes avec couleur
        # "Equipe Bleu 4-3-3 | Equipe Rouge 4-3-3"
        _team0 = team_display(0, teams_data)
        _team1 = team_display(1, teams_data)
        _style_str = f"  {_style}".rstrip() if _style else ""
        if _team0 != "Equipe A" or _team1 != "Equipe B":
            _formation_display = f"{clean(_team0)}: {formation} | {clean(_team1)}: {formation}{_style_str}"
        else:
            _formation_display = f"{formation}{_style_str}".strip()
    else:
        _formation_display = ""
    if _formation_display:
        pdf.card_bg(106, yr, 92, 26)
        pdf.set_fill_color(*ACCENT)
        pdf.rect(106, yr, 92, 2, "F")
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*ACCENT)
        pdf.set_xy(108, yr + 4)
        pdf.cell(88, 5, "FORMATION / STYLE", align="L")
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(108, yr + 11)
        pdf.cell(88, 11, clean(_formation_display), align="L")

    pdf.set_y(yr + 30)

    # Stats par equipe
    if len(team_stats) >= 2:
        pdf.section_title("Team Comparison")
        cols   = ["TEAM", "PASSES", "CADRES", "GOALS", "xG", "INTERC.", "DRIBBLES"]
        widths = [24, 27, 27, 27, 27, 27, 27]
        pdf.table_row(cols, widths, is_header=True)
        for idx, (team, ts) in enumerate(sorted(team_stats.items())):
            row = [
                team_display(team, teams_data),
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

        _hmap_labels = {
            "global":   "Activite globale",
            "shot":     "Zones de tir",
            "goal":     "Positions buts",
            "team_a":   team_display(0, teams_data),
            "team_b":   team_display(1, teams_data),
            "team_0":   team_display(0, teams_data),
            "team_1":   team_display(1, teams_data),
            "passes":   "Passes",
            "pressing": "Pressing",
        }
        hmap_list = [
            (_hmap_labels.get(k, k.replace("_", " ").title()), v)
            for k, v in heatmaps.items()
            if v and os.path.exists(str(v))
        ]
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
    # Enrichir deduped avec les vrais numeros depuis jersey_map + highlights
    # ════════════════════════
    # Enrichissement : remplacer les labels pid par vrais numeros jersey
    _enriched = {}
    for pid, s in deduped.items():
        _jersey = jersey_map.get(str(s.get("_pid", pid))) or jersey_map.get(pid)
        if _jersey:
            _lbl = f"#{_jersey}"
        else:
            _lbl = pid  # fallback sur le label existant
        # Equipe du joueur
        _team_id = s.get("team")
        _team_name = team_display(_team_id, teams_data) if _team_id is not None else ""
        _display_lbl = f"{_lbl} ({clean(_team_name)})" if _team_name else _lbl
        _enriched[_display_lbl] = s

    # Reconstruire depuis highlights si deduped insuffisant (< 3 joueurs avec tirs/buts)
    _hl_players = {}
    for h in highlights:
        _pid = str(h.get("player") or "")
        if not _pid:
            continue
        _jersey = jersey_map.get(_pid)
        if not _jersey:
            continue
        _lbl = f"#{_jersey}"
        _team_id = h.get("team")
        _team_name = team_display(_team_id, teams_data) if _team_id is not None else ""
        _display_lbl = f"{_lbl} ({clean(_team_name)})" if _team_name else _lbl
        if _display_lbl not in _hl_players:
            _hl_players[_display_lbl] = {"tirs": 0, "buts": 0, "xg_total": 0.0,
                                          "touches": 0, "key_passes": 0,
                                          "_pid": _pid, "team": _team_id}
        if h.get("main_type") == "shot":
            _hl_players[_display_lbl]["tirs"] += 1
            _hl_players[_display_lbl]["xg_total"] += float(h.get("xg", 0) or 0)
        elif h.get("main_type") in ("goal", "score"):
            _hl_players[_display_lbl]["buts"] += 1
            _hl_players[_display_lbl]["xg_total"] += float(h.get("xg", 0) or 0)
    # Fusionner avec enriched
    for _lbl, _hs in _hl_players.items():
        if _lbl in _enriched:
            # Mettre à jour tirs/buts/xG depuis highlights (plus fiable)
            _enriched[_lbl]["tirs"]     = _hs["tirs"]
            _enriched[_lbl]["buts"]     = max(_enriched[_lbl].get("buts", 0), _hs["buts"])
            _enriched[_lbl]["xg_total"] = round(_hs["xg_total"], 3)
        else:
            _enriched[_lbl] = _hs

    if _enriched or deduped:
        pdf.add_page()
        pdf.section_title("Performances Joueurs")

        _use = _enriched if _enriched else deduped
        field = {l: s for l, s in _use.items() if not s.get("is_goalkeeper")}
        gks   = {l: s for l, s in _use.items() if s.get("is_goalkeeper")}
        max_xg = max((s.get("xg_total", 0) for s in field.values()), default=1) or 1

        headers = ["Joueur", "Touch.", "K.Pass", "Tirs", "Buts", "xG", "xA", "Note"]
        widths  = [28, 18, 16, 14, 14, 16, 16, 16]
        pdf.table_row(headers, widths, is_header=True)

        sorted_field = sorted(
            field.items(),
            key=lambda x: x[1].get("touches", 0), reverse=True
        )[:20]

        xG_col_x = sum(widths[:6]) + 12

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
                kp if kp > 0 else "",
                s.get("tirs",    0) if s.get("tirs", 0) > 0 else "",
                buts   if buts   > 0 else "",
                xg_val if xg_val > 0 else "",
                xa_val if xa_val > 0 else "",
                rating,
            ]
            pdf.table_row(row, widths, alt=(idx % 2 == 0), highlight=(buts > 0))

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

        # Top Performers
        pdf.ln(5)
        pdf.section_title("Top Performers")
        yb = pdf.get_y()

        # FIX 4 — fallback sur plusieurs clés pour top buteur
        def top_by(*keys):
            best_label, best_val = "-", 0
            for l, s in field.items():
                val = 0
                for k in keys:
                    v = s.get(k, 0)
                    if v:
                        val = v
                        break
                if val > best_val:
                    best_val  = val
                    best_label = l
            return best_label if best_val > 0 else "-"

        _top_xg = top_by("xg_total", "xg")
        badges = [
            ("Top Buteur",   top_by("buts", "goals", "score"),  ACCENT2),
            ("Top xG",       _top_xg if kpi_xg > 0 else "-",    ACCENT3),
            ("Top Assists",  top_by("xa_total", "xa", "key_passes"), ACCENT4),
            ("Top Interc.",  top_by("interceptions"),             ACCENT),
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
            # FIX 2 — utiliser get_event_time pour le bon timestamp
            _ev_time = get_event_time(h)
            htype = h.get("main_type", "shot")
            color = ACCENT2 if htype in ("goal", "score") else ACCENT
            pct   = min(_ev_time / max(dur_s, 1), 0.98)
            mx    = tl_x + 5 + int((tl_w - 10) * pct)
            pdf.set_fill_color(*color)
            pdf.rect(mx - 2, tl_y + 7, 4, 6, "F")
            pdf.set_font("Helvetica", "", 5)
            pdf.set_text_color(*GRAY_L)
            pdf.set_xy(mx - 5, tl_y + 11)
            pdf.cell(10, 4, fmt_time(_ev_time), align="C")

        # Légende
        pdf.set_font("Helvetica", "", 6)
        pdf.set_text_color(*ACCENT2)
        pdf.set_xy(tl_x + 5, tl_y + 2)
        pdf.cell(20, 4, "BUT", align="L")
        pdf.set_text_color(*ACCENT)
        pdf.set_xy(tl_x + 28, tl_y + 2)
        pdf.cell(20, 4, "TIR", align="L")
        pdf.set_y(tl_y + 20)

        # Cartes highlights — FIX 2 : afficher event_time pas time_start
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

            # FIX 2 — timestamp d'affichage = temps de l'event réel
            _display_time = get_event_time(h)

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

            # Timestamp — FIX 2 : _display_time au lieu de t_start
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*WHITE)
            pdf.set_xy(46, yh + 2)
            pdf.cell(22, 9, fmt_time(_display_time))

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

            # Raison / Confiance
            pdf.set_font("Helvetica", "", 6.5)
            pdf.set_text_color(*GRAY_D)
            pdf.set_xy(126, yh + 2)
            _conf_str = ""
            if is_goal:
                _gc = h.get("goal_confidence")
                if _gc is not None:
                    _conf_str = f"conf {float(_gc):.0%}  "
            pdf.cell(70, 9, clean(_conf_str + (reason or f"score {score:.1f}")), align="R")

            pdf.set_text_color(*WHITE)
            pdf.set_y(yh + 15)

    # ════════════════════════
    # PAGE 5 — STORY
    # ════════════════════════
    if story or commentary:
        pdf.add_page()

        # FIX 3 — fusionner commentary dans match story
        # Construire un résumé narratif enrichi avec les moments clés horodatés
        _story_text = story or ""

        # Ajouter les moments clés des highlights comme paragraphe narratif
        _goal_lines = []
        for h in highlights:
            htype = (h.get("main_type") or "").lower()
            if htype in ("goal", "score"):
                _t     = get_event_time(h)
                _pid   = h.get("player")
                _plbl  = player_label(_pid, jersey_map) if _pid else "?"
                _titre = h.get("titre", "")
                _desc  = h.get("description", "")
                _mm    = f"{int(_t // 60)}':{int(_t % 60):02d}"
                if _titre:
                    _goal_lines.append(f"{_mm} — {_plbl} : {_titre}.")
                elif _desc:
                    _goal_lines.append(f"{_mm} — {_plbl} : {_desc}.")
                else:
                    _goal_lines.append(f"{_mm} — But de {_plbl}.")

        # Ajouter les tirs importants
        _shot_lines = []
        for h in highlights:
            htype = (h.get("main_type") or "").lower()
            if htype == "shot":
                _t    = get_event_time(h)
                _pid  = h.get("player")
                _plbl = player_label(_pid, jersey_map) if _pid else "?"
                _xg   = float(h.get("xg", 0) or 0)
                _mm   = f"{int(_t // 60)}':{int(_t % 60):02d}"
                if _xg >= 0.20:
                    _shot_lines.append(f"{_mm} — Tir cadre de {_plbl} (xG {_xg:.2f}).")

        # Assembler le résumé complet
        _narrative_parts = []
        if _story_text:
            _narrative_parts.append(_story_text)
        if _goal_lines:
            _narrative_parts.append("Buts : " + "  ".join(_goal_lines))
        if _shot_lines:
            _narrative_parts.append("Occasions notables : " + "  ".join(_shot_lines))

        # Lignes de commentary pertinentes (filtrer les génériques, garder celles avec joueur ou action spécifique)
        _generic = {
            "action a suivre.", "phase de jeu interessante.",
            "le match continue a bon rythme.", "belle intensite dans les duels.",
            "action a suivre", "phase de jeu interessante",
        }
        _good_lines = [
            c for c in (commentary or [])
            if c and len(str(c).strip()) > 20
            and clean(str(c)).lower().rstrip(".").strip() not in _generic
            and any(ch.isdigit() or ch == "#" for ch in str(c))  # contient un numéro de joueur
        ]
        if _good_lines:
            _narrative_parts.append("Actions : " + "  ".join(_good_lines[:5]))

        _full_story = "\n\n".join(_narrative_parts)

        if _full_story:
            pdf.section_title("Match Story")
            pdf.card_bg(12, pdf.get_y(), 186, 4)
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*GRAY_L)
            pdf.set_x(14)
            pdf.multi_cell(182, 5, clean(_full_story))
            pdf.ln(5)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        pdf.output(output_path)
        print(f"PDF OK -> {output_path}")
        return output_path
    except Exception as e:
        print(f"PDF error : {e}")
        return None