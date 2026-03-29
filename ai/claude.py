# ai/claude.py
# -*- coding: utf-8 -*-

import anthropic
import os

import config

client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)


def summarize(highlights, summary=None, stats=None, sport="football"):
    """
    Génère un résumé IA du match.

    Paramètres :
        highlights : liste des moments clés
        summary    : dict résumé du match (goals, shots, etc.)
        stats      : dict stats joueurs
        sport      : sport analysé
    """

    # ── Résumé chiffré ───────────────────
    summary_text = ""
    if summary:
        summary_text = f"""
Statistiques du match :
- Buts       : {summary.get('goals', 0)}
- Tirs       : {summary.get('shots', 0)}
- xG total   : {summary.get('total_xg', 0)}
- Passes     : {summary.get('passes', 0)}
- Interceptions : {summary.get('interceptions', 0)}
- Dribbles   : {summary.get('dribbles', 0)}
- Joueurs    : {summary.get('players', 0)}
- Durée      : {summary.get('duration', '--')}
"""

    # ── Moments clés ─────────────────────
    highlights_text = ""
    if highlights:
        lines = []
        for i, h in enumerate(highlights[:10], 1):
            t     = h.get("time_start", 0)
            mins  = int(t // 60)
            secs  = int(t % 60)
            htype = h.get("main_type", "action")
            score = h.get("score", 0)
            lines.append(f"  {i}. {mins:02d}:{secs:02d} — {htype} (score: {score:.1f})")
        highlights_text = "Moments clés :\n" + "\n".join(lines)

    # ── Stats top joueurs ─────────────────
    stats_text = ""
    if stats:
        top = sorted(
            stats.items(),
            key=lambda x: x[1].get("touches", 0),
            reverse=True
        )[:5]
        lines = []
        for pid, s in top:
            lines.append(
                f"  Joueur {pid} : "
                f"{s.get('touches', 0)} touches, "
                f"{s.get('passes', 0)} passes, "
                f"{s.get('tirs', 0)} tirs, "
                f"{s.get('buts', 0)} buts"
            )
        stats_text = "Top joueurs :\n" + "\n".join(lines)

    # ── Prompt ───────────────────────────
    sport_labels = {
        "football":         "football",
        "mini-foot":        "futsal / mini-foot",
        "basketball":       "basketball",
        "handball":         "handball",
        "rugby":            "rugby",
        "hockey sur glace": "hockey sur glace",
        "hockey sur gazon": "hockey sur gazon",
        "tennis":           "tennis",
        "tennis de table":  "tennis de table",
        "padel":            "padel",
    }
    sport_label = sport_labels.get(sport, sport)

    prompt = f"""Tu es un analyste sportif expert en {sport_label}.

Voici les données d'une analyse vidéo d'un match de {sport_label} :

{summary_text}

{highlights_text}

{stats_text}

Rédige un résumé analytique du match en français, en 3-4 paragraphes.
Utilise uniquement les données fournies. Ne mentionne aucun autre sport que le {sport_label}.
Commente les moments clés, les tendances tactiques et les performances individuelles notables.
Sois précis, factuel et concis.
"""

    res = client.messages.create(
        model      = config.CLAUDE_MODEL,
        max_tokens = config.CLAUDE_MAX_TOKENS,
        messages   = [{"role": "user", "content": prompt}]
    )

    return res.content[0].text