# V5.2 — Bilan final : Pipeline complet KO1/KO2/Fin1MT/FinMatch

## Résumé exécutif

Le pipeline complet (`KO1 → KO2 → Fin1MT → FinMatch`) a été testé bout-en-bout sur les `9` matchs de référence, avec vision `Gemini 3.5 Flash`, tous les correctifs de la session intégrés. **`9/9` matchs traités avec succès** — aucun échec définitif, y compris sur `Wanze` qui avait échoué (`DEGENERATE_WINDOW`) avant l'ajout du mécanisme de relance.

| Détecteur | Médiane | `≤3s` | `≤15s` | Max | Cas le plus faible |
|---|---|---|---|---|---|
| `KO1` | `2s` | `8/9` | `9/9` | `5s` | `Raeren +5s` |
| `KO2` | `2s` | `8/9` | `8/9` | `16s` | `Stembert +16s` |
| `Fin1MT` | `9s` | `2/9` | `7/9` | `19s` | `MineroisSter +19s` |
| `FinMatch` | `12s` | `1/9` | `8/9` | `35s` | `MineroisSter +35s` |

**`1621` appels `Gemini` au total sur les `9` matchs (`~180`/match).** Estimation de coût approximative (ordre de grandeur, pas une facture réelle) : `~$4,81` pour ce run précis — mais le coût réel cumulé d'une journée complète de développement (incluant tous les diagnostics) est significativement plus élevé ; seule la consultation directe du compte `Google AI Studio`/`Vertex` donne un chiffre fiable.

---

## Corrections apportées pendant cette session (en plus de celles déjà documentées pour Fin1MT/FinMatch)

### 1. Bug de collision de fichiers temporaires entre matchs et détecteurs (critique)

**Symptôme** : `KO2` donnait `-175s` sur `P1Minerois` et `+175s`/`+140s` sur `FinMatch` d'`Andrimont` dans le premier run du pipeline complet, alors que ces mêmes détecteurs avaient été validés individuellement sans problème.

**Cause réelle** : tous les détecteurs (`KO1`, `KO2`, `Fin1MT`, `FinMatch`) et tous les matchs partageaient le même répertoire temporaire `/tmp`, avec des fichiers nommés uniquement par timestamp arrondi. Un détecteur pouvait lire l'image laissée par un autre détecteur (ou un autre match) ayant utilisé un timestamp numériquement proche.

**Correctif** : chaque match, chaque détecteur, et chaque tentative de relance utilise désormais son propre sous-répertoire (`/tmp/{match}/{détecteur}/tentative{n}`). Un paramètre `tmp_dir` manquant a été ajouté à `find_ko2_gemini` (les trois autres fonctions l'avaient déjà).

### 2. Mécanisme de relance globale (3 tentatives par détection)

**Principe** : si `KO1`, `KO2`, `Fin1MT` ou `FinMatch` échoue (statut non confirmé), le pipeline relance automatiquement cette détection jusqu'à `3` fois avant d'abandonner définitivement — "ne pas savoir n'est pas acceptable".

**Validation** : `Wanze/KO2` a échoué deux fois de suite (`DEGENERATE_WINDOW`) puis réussi à la `3e` tentative, confirmant que le mécanisme fonctionne comme prévu. Coût : `133` appels rien que pour `KO2` sur ce match (contre `~50-60` habituellement).

### 3. Vote adaptatif ciblé sur le comptage de ballons (`Q1` de `KO2` uniquement)

**Diagnostic** : `10` appels identiques sur une même image (`Wanze t=4671s`) ont montré que `7` critères sur `9` du prompt `Q1` de `KO2` sont stables à `100%`, mais que `nombre_ballons_compte` varie (`60%`/`40%` entre `1` et `2` ballons) — la même image contenant objectivement `2` ballons (confirmé par inspection visuelle directe), difficiles à distinguer de loin.

**Correctif** : quand le compte rapporté est `1` ou `2` (zone identifiée comme instable), un vote isolé (`3` appels, `+5` en cas d'égalité) posant **uniquement** la question du comptage est déclenché, et `un_seul_ballon` est recalculé à partir du compte vérifié. Aucun coût supplémentaire pour les cas non-ambigus (`0` ou `3+` ballons).

**Bug corrigé en cours de route** : la première version gardait par erreur l'ancien booléen `un_seul_ballon` de l'appel initial en `ET` logique avec le nouveau compte vérifié — ce qui aurait annulé la correction dans le cas `2→1`. Corrigé et testé sur `5` scénarios avant livraison.

---

## Ce qui a été exploré sans aboutir à un correctif (documenté honnêtement)

### Sensibilité de `FinMatch` aux décalages d'échantillonnage d'une seconde

**Constat expérimental solide** : sur `Andrimont`, figer `KO2` à `3526s` donne `FinMatch=6166s` (`+18s`) ; figer `KO2` à `3527s` (`1s` de plus) donne `FinMatch=6288s` (`+140s`). Un décalage d'une seconde en amont change complètement la trajectoire du scan `FinMatch` en aval.

**Pistes explorées, sans succès définitif** :
- Cartographie point par point (`t=6162` à `6170`) : signal réel présent mais oscillant, pas une fenêtre propre.
- Vérification narrative à `3` images avec critère de "tendance cohérente" : question trop permissive, valide presque toujours vrai une fois dans la zone d'après-match.
- Vérification narrative à `3` images avec critère de "rituels concrets" (poignées de main, direction vestiaires) : discrimine bien avant/après **quand centrée à la main sur le bon instant**, mais **échoue à nouveau** quand centrée sur le vrai point de la grille de scan (`1s` de décalage suffit à la faire échouer aussi).

**Conclusion actuelle** : c'est une vraie limite structurelle non résolue, pas juste de la variance de modèle. Elle n'a affecté qu'`Andrimont` dans le run final (`+15s`, acceptable), mais le mécanisme qui en est la cause reste actif et pourrait produire des résultats bien pires sur d'autres matchs/timings. À garder en tête pour un futur travail ciblé si ce problème redevient bloquant.

---

## Limites structurelles déjà connues (confirmées, pas de régression)

- **`Fin1MT`/`FinMatch` biais systématique en retard** (`10-30s`, parfois plus) : la transition n'est détectable visuellement qu'après le début du mouvement de sortie. Non corrigeable par prompt.
- **`Stembert/KO2 +16s`** : jugement `Gemini` confiant mais erroné sur une image isolée (touche sifflée juste après l'engagement, ressemble à un avant-engagement).
- **Variance résiduelle de `Gemini` à `température=0`** : confirmée expérimentalement à plusieurs reprises (`10` appels identiques sur la même image donnant des réponses différentes) — caractéristique du modèle, pas de notre code.

---

## Fichiers de production finaux

| Fichier | Rôle |
|---|---|
| `kickoff_gemini_cascade.py` | `KO1` + infrastructure partagée + vote adaptatif ballons (`KO2` uniquement) |
| `match_boundaries_v2.py` | `KO2` (paramètre `tmp_dir` ajouté) |
| `fin1mt_gemini_cascade.py` | `Fin1MT` |
| `finmatch_gemini_cascade.py` | `FinMatch` |
| `notebook_pipeline_complet_9matchs.py` | Test bout-en-bout, relance `3x`, `tmp_dir` isolé, sortie complète |

## Items en attente

- **[GELÉ]** Sensibilité `FinMatch` au décalage d'échantillonnage (documentée ci-dessus, non résolue)
- **[GELÉ]** Fin de match prolongations/tirs au but
- **[GELÉ]** Filtre `deux_equipes_visibles` pour `KO1`
- **[EN COURS]** Test sur nouveau match hors échantillon (`Melen P1`) pour validation en conditions réelles
