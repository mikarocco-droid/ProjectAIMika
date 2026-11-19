# V5.2 — État final KO2 (Flash), limites connues et architecture

## Décision retenue

**Modèle : `gemini-3.5-flash`**, pas `Pro`. Coût nettement inférieur, résultats
équivalents ou meilleurs sur les cas problématiques rencontrés aujourd'hui
(fanion de corner, `KO1` : `9/9` identique à `Pro`). Aucune raison de revenir
à `Pro` pour tenter de corriger la limite ci-dessous — ce n'est pas un
problème de modèle, c'est un problème de nature du signal demandé.

## Résultat final KO2 (9 matchs de référence)

| Match | Erreur | Statut |
|---|---|---|
| Andrimont | +2s | ✅ |
| Raeren | +1s | ✅ |
| P1Minerois | +2s | ✅ |
| MineroisSter | +1s | ✅ |
| Juprelle | +2s | ✅ |
| Franchimont | +3s | ✅ |
| Goe | +1s | ✅ |
| Wanze | +2s | ✅ |
| Stembert | +16s | ⚠️ limite connue, expliquée ci-dessous |

`9/9` confirmés, médiane `2,0s`, `8/9` sous `3s`.

## Corrections cumulées ayant mené à ce résultat (KO2 uniquement, KO1 intact)

1. `deux_equipes_visibles` — rejette les scènes à une seule équipe (échauffement)
2. Clarté JSON — évite qu'un texte brut "NON" casse le parsing
3. Fanion de corner — limité au terrain de jeu (pas un terrain voisin)
4. `nombre_joueurs_suffisant` (≥5) — rejette les scènes à 2-3 personnes isolées
5. `pas_scan=20s` + `delai_verif_q2=22s` — alignés (au lieu des valeurs `60s`
   héritées de `KO1`, incohérentes avec un pas plus fin)
6. Vote économique sur `Q1` (3 appels, uniquement si le 1er dit déjà OUI)
7. Exactement un ballon visible (rejette `0` ballon ET `2+`, pas seulement `2+`)
8. Signal `Q2` alternatif `AVANT`/`TRANSITION`/`APRES`/`INCERTAIN` (remplace
   "le match a-t-il déjà commencé", qui dépendait trop de la visibilité
   ponctuelle du ballon)

## Limite structurelle documentée : Stembert (+16s)

**Ce n'est pas un bug de Flash, ni un défaut corrigible par plus de règles.**

Démontré empiriquement à deux niveaux distincts aujourd'hui :

- **Wanze** : le ballon était en touche (hors cadre) à l'instant testé, alors
  que le jeu avait réellement déjà repris. Corrigé par le nouveau signal
  `AVANT`/`APRES` (le ballon n'y est plus une condition éliminatoire).

- **Stembert** : une touche sifflée survient très peu après l'engagement
  (`t≈4130s`, `13s` après le vrai KO2 à `4117s`). Sur `5` images testées
  autour de ce point, `Flash` répond `AVANT` avec un raisonnement confiant et
  cohérent à chaque fois — **aucune hésitation, aucun `INCERTAIN`**. Ni plus
  de retries, ni un vote à `3` voix, ni un prompt plus détaillé n'auraient pu
  corriger ce cas : le modèle n'est pas dans le doute, il se trompe avec
  confiance parce que l'image seule ne contient pas l'information nécessaire
  (une touche sifflée juste après un engagement peut visuellement ressembler
  à une formation d'avant-match : joueurs relativement statiques, arbitre
  visible au centre).

**Le point commun entre les deux cas** : la question posée à `Q2` ("le match
a-t-il déjà commencé ?") est une déduction temporelle à partir d'une
observation instantanée — intrinsèquement fragile, quelle que soit la
qualité du modèle ou du prompt.

## Direction future (non engagée maintenant)

Remplacer la question ponctuelle par une comparaison de plusieurs images
successives, cherchant une transition explicite "formation d'engagement →
joueurs en mouvement / ballon joué" plutôt qu'un jugement sur une seule
image. Piste identifiée, pas implémentée — nécessiterait une nouvelle
architecture de recherche (comparaison de paires/séquences d'images), à
traiter comme un chantier séparé si la précision sur ce type de cas devient
critique.

## Fichiers de référence

Voir `BACKUP_systeme_avant_redesign_q2.zip` pour l'état intermédiaire
(avant le signal `AVANT`/`APRES`), et les fichiers de production actuels
(`kickoff_gemini_cascade.py`, `match_boundaries_v2.py`) pour l'état final
décrit ici.
