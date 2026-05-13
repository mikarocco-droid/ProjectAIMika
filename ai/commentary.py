# ai/commentary.py
# -*- coding: utf-8 -*-
"""
Commentaires contextuels qui s'améliorent match après match.

Apprentissage :
- Mémorise quels commentaires ont été utilisés pour quels types d'actions
- Adapte le style selon les stats du match (possession, pressing, score)
- Enrichit les commentaires avec les numéros de maillots si disponibles
- Mémorise les patterns de jeu détectés pour varier le vocabulaire
"""

import random
import os
import json


# ─────────────────────────────────────────
# BASE DE COMMENTAIRES PAR TYPE
# ─────────────────────────────────────────
_COMMENTARY = {
    "goal": [
        "BUUUUT ! Quelle finition imparable !",
        "INCROYABLE ! Le ballon est au fond des filets !",
        "IL NE POUVAIT PAS MIEUX PLACER ! 1-0 !",
        "MAGNIFIQUE ACTION COLLECTIVE qui se conclut par un but !",
        "LE GARDIEN N'A PU QUE REGARDER PASSER CE TIR !",
        "C'EST LE BUT ! Explosion de joie dans les rangs !",
        "QUELLE FRAPPE ! Rien à faire pour le portier !",
        "LE STADE EST EN DÉLIRE ! But magnifique !",
    ],
    "shot": [
        "Frappe dangereuse, repoussée de justesse !",
        "Tir tendu, le gardien sort le grand jeu !",
        "Juste à côté ! Quelle occasion manquée !",
        "Quelle occasion ! Le poteau s'interpose !",
        "Tir croisé, le gardien plonge et détourne !",
        "Frappe enroulée, ça passe au-dessus !",
        "Il tente sa chance de loin — pas loin du cadre !",
    ],
    "fast_break": [
        "Contre-attaque fulgurante ! Danger imminent !",
        "Transition éclair ! Ils partent en nombre !",
        "Interception et départ en contre — ça s'accélère !",
        "Récupération et sprint vers l'avant !",
        "Le pressing payant — contre-attaque en 3 contre 2 !",
    ],
    "interception": [
        "Bonne récupération défensive !",
        "Il coupe la trajectoire avec autorité !",
        "Excellente lecture du jeu — ballon récupéré !",
        "Pressing efficace — la possession change de camp !",
        "Interception décisive qui casse l'action adverse !",
    ],
    "dribble": [
        "Super dribble, il élimine son adversaire !",
        "Quelle technique ! Il en efface deux d'un coup !",
        "Coup de rein dévastateur — il passe comme une flèche !",
        "Crochet intérieur, il crée le décalage !",
        "Il résiste à la charge et garde le ballon !",
        "Touche de balle somptueuse pour se défaire du marquage !",
    ],
    "progressive_run": [
        "Percée intéressante vers la surface !",
        "Il progresse balle au pied, les défenseurs reculent !",
        "Montée de balle propre — l'équipe avance bien !",
        "Course vers l'avant, le bloc adverse se resserre !",
    ],
    "pass": [
        "Passe précise dans le couloir libre !",
        "Bon décalage pour le partenaire !",
        "La circulation du ballon est fluide.",
        "Jeu en triangle, la défense peine à suivre.",
    ],
    "long_pass": [
        "Longue ouverture vers l'aile !",
        "Grand pont — changement de jeu côté opposé !",
        "Diagonale précise qui trouve son homme !",
        "Passe lobée pour contourner le pressing.",
    ],
    "build_up": [
        "Belle construction collective du fond du terrain.",
        "Sortie de balle propre — l'équipe monte bien.",
        "Relance soignée, le jeu se développe posément.",
    ],
    "under_pressure": [
        "Il est sous pression — va-t-il s'en sortir ?",
        "Deux défenseurs sur lui — décision à prendre vite !",
        "Le pressing s'intensifie autour du porteur du ballon.",
    ],
    "score": [
        "C'est rentré ! Le score s'ouvre !",
        "BUT VALIDÉ ! L'arbitre accorde le goal !",
        "ÇA RENTRE ! La foule exulte !",
    ],
}

_DEFAULT = [
    "Phase de jeu intéressante.",
    "Action à suivre.",
    "Le match continue à bon rythme.",
    "Belle intensité dans les duels.",
]

# ─────────────────────────────────────────
# COMMENTAIRES CONTEXTUELS (score + style)
# ─────────────────────────────────────────
_CONTEXT_COMMENTS = {
    "winning_goal": [
        "C'est peut-être le but qui fait la différence ce soir !",
        "Quel moment crucial dans cette rencontre !",
        "L'avantage au tableau d'affichage — l'équipe prend les commandes !",
    ],
    "equalizer": [
        "Égalisation ! Le match repart de zéro !",
        "Retour à la case départ — tout est à refaire !",
        "Le but qui relance complètement la rencontre !",
    ],
    "late_goal": [
        "But tardif qui change tout dans cette fin de match !",
        "Dans les dernières minutes — dramatique !",
        "Le dénouement arrive au fil final !",
    ],
    "high_pressing": [
        "Le pressing intense porte ses fruits !",
        "La pression collective crée des espaces dangereux.",
    ],
    "possession_style": [
        "La maîtrise du ballon est remarquable.",
        "Patience et précision — cette équipe joue collectif.",
    ],
}


# ─────────────────────────────────────────
# LEARNER DE COMMENTAIRES
# Mémorise les patterns et améliore la pertinence
# ─────────────────────────────────────────
class CommentaryLearner:
    """
    Apprend quels commentaires sont les plus adaptés
    selon le contexte du match (sport, style, score).
    """

    SAVE_PATH = "outputs/learning/commentary_patterns.json"

    def __init__(self):
        self.patterns = self._load()

    def _load(self):
        if os.path.exists(self.SAVE_PATH):
            try:
                with open(self.SAVE_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "used_comments":   {},    # {type: {comment: count}}
            "sport_styles":    {},    # {sport: {style: count}}
            "n_matches":       0,
        }

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)
            with open(self.SAVE_PATH, "w") as f:
                json.dump(self.patterns, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def record(self, event_type, comment):
        """Enregistre qu'un commentaire a été utilisé pour un type d'event."""
        used = self.patterns.setdefault("used_comments", {})
        used.setdefault(event_type, {})
        used[event_type][comment] = used[event_type].get(comment, 0) + 1

    def get_least_used(self, event_type, pool):
        """
        Retourne le commentaire le moins utilisé parmi le pool.
        Évite la répétition des mêmes phrases match après match.
        """
        used = self.patterns.get("used_comments", {}).get(event_type, {})
        if not used:
            return random.choice(pool)
        # Trier par fréquence d'utilisation croissante
        scored = sorted(pool, key=lambda c: used.get(c, 0))
        # Parmi les 3 moins utilisés, choisir aléatoirement
        return random.choice(scored[:max(3, len(scored) // 3)])

    def record_match(self, sport, style):
        """Mémorise le style de jeu observé par sport."""
        styles = self.patterns.setdefault("sport_styles", {})
        styles.setdefault(sport, {})
        styles[sport][style] = styles[sport].get(style, 0) + 1
        self.patterns["n_matches"] = self.patterns.get("n_matches", 0) + 1
        self.save()

    def get_dominant_style(self, sport):
        """Retourne le style de jeu dominant appris pour ce sport."""
        styles = self.patterns.get("sport_styles", {}).get(sport, {})
        if not styles:
            return None
        return max(styles, key=styles.get)


# Instance globale — partagée entre les appels
_learner = CommentaryLearner()


# ─────────────────────────────────────────
# CONTEXTE MATCH
# ─────────────────────────────────────────
def _build_context(events, stats=None, sport="football", formation=None, style=None):
    """
    Analyse le contexte du match pour enrichir les commentaires.
    """
    goals   = [e for e in events if e.get("type") in ["goal", "score"]]
    shots   = sum(1 for e in events if e.get("type") == "shot")
    passes  = sum(1 for e in events if e.get("type") == "pass")
    presses = sum(1 for e in events if e.get("type") == "under_pressure")

    score = {0: 0, 1: 0}
    for g in goals:
        t = g.get("team")
        if t in score:
            score[t] += 1

    return {
        "score":     score,
        "n_goals":   len(goals),
        "n_shots":   shots,
        "n_passes":  passes,
        "n_press":   presses,
        "sport":     sport,
        "formation": formation or "?",
        "style":     style or (_learner.get_dominant_style(sport) or "équilibré"),
        "high_press": presses > 10,
        "possession_style": passes > shots * 4,
    }


def _get_context_suffix(event, context, event_index, total_events):
    """
    Ajoute un commentaire contextuel après le commentaire principal
    si la situation le justifie.
    """
    etype = event.get("type", "")
    score = context["score"]

    if etype in ["goal", "score"]:
        team  = event.get("team", 0)
        other = 1 - team if team in [0, 1] else 0
        diff  = score.get(team, 0) - score.get(other, 0)

        # But égalisateur
        if score.get(0, 0) == score.get(1, 0) and score.get(0, 0) > 0:
            return random.choice(_CONTEXT_COMMENTS["equalizer"])

        # But décisif (écart de 1)
        if abs(diff) == 1:
            # But tardif (dans les 20% finaux)
            if event_index > total_events * 0.8:
                return random.choice(_CONTEXT_COMMENTS["late_goal"])
            return random.choice(_CONTEXT_COMMENTS["winning_goal"])

    # Contexte pressing
    if etype == "fast_break" and context["high_press"]:
        return random.choice(_CONTEXT_COMMENTS["high_pressing"])

    # Contexte possession
    if etype in ["pass", "build_up"] and context["possession_style"]:
        return random.choice(_CONTEXT_COMMENTS["possession_style"])

    return None


def _format_player(event, jersey_map=None):
    """
    Retourne une désignation du joueur si disponible.
    N'affiche rien si le pid n'est pas résolu dans jersey_map
    (évite d'afficher P4 comme #4 quand c'est en réalité #11).
    """
    if not jersey_map:
        return ""
    pid = str(event.get("player", ""))
    if not pid:
        return ""
    jersey = jersey_map.get(pid) or jersey_map.get(event.get("player"))
    if jersey is None:
        return ""   # pid non résolu — ne pas afficher l'ID interne comme numéro
    jersey_str = str(jersey).strip()
    if not jersey_str.startswith("#"):
        jersey_str = f"#{jersey_str}"
    return f" — {jersey_str}"


# ─────────────────────────────────────────
# GÉNÉRATION PRINCIPALE
# ─────────────────────────────────────────
def generate_commentary(
    events,
    stats      = None,
    jersey_map = None,
    sport      = "football",
    formation  = None,
    style      = None,
):
    """
    Génère une ligne de commentaire contextuel pour chaque event.

    Paramètres :
        events     : liste d'events triés chronologiquement
        stats      : dict stats joueurs (optionnel)
        jersey_map : dict {player_id: numero_maillot} (optionnel)
        sport      : sport du match
        formation  : formation détectée (ex: "4-3-3")
        style      : style de jeu (ex: "possession")

    Retourne : liste de strings (un commentaire par event)
    """
    if not events:
        return []

    context     = _build_context(events, stats, sport, formation, style)
    total       = len(events)
    lines       = []
    used_recent = []   # Évite les répétitions consécutives

    # Trier : buts en premier (toujours commentés), puis ordre chronologique
    priority_order = {"goal": 0, "score": 0}
    events_sorted = sorted(events, key=lambda e: (
        priority_order.get(e.get("type",""), 1),
        e.get("time", 0)
    ))

    for i, e in enumerate(events_sorted):
        etype  = e.get("type", "default")
        pool   = _COMMENTARY.get(etype, _DEFAULT)
        player = _format_player(e, jersey_map)

        # Choisir le commentaire le moins utilisé globalement
        comment = _learner.get_least_used(etype, pool)

        # Éviter les répétitions consécutives (fenêtre 3)
        attempts = 0
        while comment in used_recent[-3:] and attempts < len(pool):
            comment  = random.choice(pool)
            attempts += 1

        # Ajouter le numéro de joueur si disponible
        if player and etype in ["goal", "shot", "dribble"]:
            comment = comment.rstrip("!.") + f"{player} !"

        # Ajouter un suffixe contextuel si pertinent
        suffix = _get_context_suffix(e, context, i, total)
        if suffix:
            comment = f"{comment} {suffix}"

        lines.append(comment)
        used_recent.append(comment)

        # Enregistrer l'utilisation pour l'apprentissage
        _learner.record(etype, _COMMENTARY.get(etype, [comment])[0]
                        if comment in _COMMENTARY.get(etype, []) else comment)

    # Enregistrer le style du match pour apprentissage futur
    detected_style = style or context["style"]
    _learner.record_match(sport, detected_style)

    return lines


# ─────────────────────────────────────────
# GÉNÉRATION STORY ENRICHIE
# ─────────────────────────────────────────
def generate_live_commentary(events, jersey_map=None, sport="football",
                              formation=None, style=None, max_lines=15):
    """
    Génère un commentaire live enrichi.
    Les buts sont toujours commentés en premier avec leur vrai contexte.
    """
    priority = {
        "goal": 10, "score": 10,
        "shot": 7,
        "fast_break": 6,
        "interception": 5,
        "dribble": 4,
        "progressive_run": 3,
        "long_pass": 2,
    }

    scored = sorted(
        [e for e in events if e.get("type") in priority],
        key=lambda x: (priority.get(x.get("type", ""), 0), x.get("xg", 0)),
        reverse=True
    )[:max_lines]
    scored.sort(key=lambda x: x.get("time", 0))

    # Générer les commentaires avec contexte but enrichi
    lines = []
    for e in scored:
        etype = e.get("type", "")
        t = e.get("time", 0)
        mins = int(t // 60)
        secs = int(t % 60)
        
        if etype in ("goal", "score"):
            # Commentaire de but avec numéro joueur et timing réel
            pid = str(e.get("player", "") or "")
            jersey = None
            if jersey_map and pid:
                jersey = jersey_map.get(pid)
                if jersey:
                    jersey_str = str(jersey).strip()
                    if not jersey_str.startswith("#"):
                        jersey_str = f"#{jersey_str}"
                else:
                    jersey_str = None
            
            base = random.choice(_COMMENTARY.get("goal", ["BUUUUT !"]))
            if jersey_str:
                line = f"⚽ {mins:02d}:{secs:02d} — {base} {jersey_str} marque !"
            else:
                line = f"⚽ {mins:02d}:{secs:02d} — {base}"
        else:
            pool = _COMMENTARY.get(etype, _DEFAULT)
            line = random.choice(pool)
        
        lines.append(line)

    return lines