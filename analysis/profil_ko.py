# analysis/profil_ko.py
# 17/11/2026 — UN SEUL INTERRUPTEUR pour la configuration de la recherche des buts validée à 0 $ sur les 10 matchs de
# référence (16-17/11) : SCOUTIA_PROFIL_KO=2026-11. DÉSACTIVÉ PAR DÉFAUT (variable absente = comportement d'avant).
#
# Ce que le profil règle (et ce que chaque réglage a apporté, mesuré sur la référence YOLO) :
#   SCOUTIA_KO_PISTES=reelles        les pistes prédites (« fantômes ») ne comptent plus dans les signaux des candidats
#                                    (seul réglage qui place un candidat juste après le penalty de Raeren 69:12) ;
#   SCOUTIA_KO_DEDUP=episodes        regroupement sans effet de chaîne (penalties de Spa 19:50 et 39:06) ;
#   SCOUTIA_KO_DEBUT_MATCH_S=30      un candidat < 30 s après le coup d'envoi du match est ce coup d'envoi (Spa 06:47) ;
#   SCOUTIA_KO_REDATER=1             un candidat voisin qui voit le même but 20 à 40 s plus tard donne son heure
#                                    (Raeren 50:17, Franchimont 49:51) ;
#   SCOUTIA_KO_REDATER_PREMIERE=1    si l'escalade n'a pas d'heure, celle de la 1re réponse (Spa 39:06).
# Résultat (replays du 17/11, cache strict) : 44/45 vrais buts à ±25 s (seul manque : Andrimont 84:06, hors du suivi),
# 43/45 à ±10 s, 3/3 penalties, 10 faux buts (1 « confirmé ») contre 41/45 et 6 faux buts en production le 11/11.
# Suivi RF-DETR, Franchimont (17/11) : 5/5 à ±10 s, 4 faux buts (0 « confirmé »).
# ⚠️ Mis au point sur ces 10 matchs : à valider sur des matchs jamais utilisés avant la production.
#
# Le profil PRIME sur un réglage contraire déjà posé (les cellules posent SCOUTIA_KO_DEDUP=fenetre) : chaque valeur
# remplacée est écrite dans le journal. Il est appliqué au début de run_pipeline, avant le lancement des processus de
# suivi (qui héritent de l'environnement).
import os

PROFILS = {
    "2026-11": {
        "SCOUTIA_KO_PISTES": "reelles",
        "SCOUTIA_KO_DEDUP": "episodes",
        "SCOUTIA_KO_DEBUT_MATCH_S": "30",
        "SCOUTIA_KO_REDATER": "1",
        "SCOUTIA_KO_REDATER_PREMIERE": "1",
    },
}


def appliquer_profil_ko():
    """Applique le profil nommé par SCOUTIA_PROFIL_KO. Renvoie le nom du profil appliqué, ou None."""
    nom = (os.environ.get("SCOUTIA_PROFIL_KO", "") or "").strip()
    if nom in ("", "0", "aucun", "none"):
        return None
    reglages = PROFILS.get(nom)
    if reglages is None:
        print(f"  [PROFIL KO] ⚠️ profil « {nom} » inconnu (connus : {', '.join(PROFILS)}) — aucun réglage modifié")
        return None
    changes = []
    for cle, valeur in reglages.items():
        avant = (os.environ.get(cle, "") or "").strip()
        if avant and avant != valeur:
            changes.append(f"{cle} {avant} → {valeur}")
        os.environ[cle] = valeur
    print(f"  [PROFIL KO] profil « {nom} » appliqué : "
          + ", ".join(f"{k}={v}" for k, v in reglages.items())
          + (f" | remplacés : {'; '.join(changes)}" if changes else ""))
    return nom
