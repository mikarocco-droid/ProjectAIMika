# V5.2 — Détection Fin1MT et FinMatch par vision Gemini

## Contexte et décision de départ

Après la validation de `KO2` (`8/9 ≤3s`), l'objectif du jour était d'étendre la même approche vision `Gemini` aux deux bornes restantes :
- **Fin1MT** (fin de la 1ère mi-temps)
- **FinMatch** (fin du match)

**Point de départ** : le détecteur audio existant pour `Fin1MT` donnait `6/9 ≤19s`, mais avec `2` échecs catastrophiques (`Franchimont -318s`, `Stembert -229s`) — un autre événement acoustique plus fort que le vrai sifflet dominait le signal. Décision : basculer les deux bornes sur la même architecture vision que `KO2`.

**Prolongations et tirs au but** : volontairement non gérés pour l'instant (item gelé du roadmap), les fenêtres de recherche sont calibrées pour un temps réglementaire standard.

---

## Architecture finale (identique pour Fin1MT et FinMatch)

### Structure Q1 / Q2 / recherche fine

- **Q1 (scan grossier, pas=60s)** : signal large, répond `OUI` si l'un des signaux suivants est présent :
  - **Signal A** — mouvement collectif vers la ligne de touche
  - **Signal B** — terrain nettement dégarni (moins de la moitié de l'effectif), sans action de jeu
  - **Signal C** (FinMatch uniquement) — poignées de main / rituel de salutation (n'existe pas à la mi-temps)

- **Vérification narrative à 2 images** (nouveau, voir plus bas) : dès qu'un candidat `Q1` est trouvé, avant de lancer la vérification `Q2` coûteuse à `+90s`, on envoie **2 images (t et t+5s) ensemble** dans un seul appel `Gemini`, avec un prompt demandant un jugement narratif de cohérence — pas deux jugements indépendants comparés après coup.

- **Q2 (vérification, +90s après confirmation)** : critère plus strict — terrain vide **des deux équipes du match** spécifiquement (tolère explicitement enfants, pom-pom girls, staff).

- **Recherche fine (dichotomie 15s/5s/1s)** : utilise `Q1` (pas `Q2`, qui est trop strict et pousse la détection après le vrai instant), en remontant jusqu'au dernier point où `Q1` disait encore `NON` — pas seulement jusqu'au point où `Q1` a détecté le signal.

### Paramètres finaux

| Paramètre | Fin1MT | FinMatch |
|---|---|---|
| Fenêtre de recherche | `[KO2-16min, KO2-5min]` | `[KO2+40min, KO2+55min]` |
| `pas_scan` (scan grossier) | 60s | 60s |
| `delai_verif_q2` | 90s | 90s |
| Plafonnée à la durée réelle de la vidéo | Oui | Oui |

---

## Corrections cumulées, dans l'ordre où les problèmes ont été trouvés

### 1. Signal directionnel plutôt que "calme"
**Problème initial** : le premier prompt Q2 utilisait juste "les joueurs sont-ils calmes" — un but marqué (joueurs calmes près d'un but, image utilisateur) déclenchait un faux positif identique à la vraie fin de mi-temps (également près d'un but dans ce match précis, confirmé par comparaison d'images).
**Correction** : remplacé par le sens de la marche (vers la touche vs vers le centre), pas la présence d'un but.

### 2. Fenêtre de recherche trop courte après le vrai événement (Fin1MT)
**Problème** : `marge_apres_min=8` (hérité de l'audio) laissait parfois seulement `16-18s` de marge après le vrai `Fin1MT` (cas `Andrimont`, `Goe`) — pas assez pour voir la sortie se dérouler.
**Correction** : réduit à `marge_apres_min=5`, vérifié `≥196s` de marge sur les 9 matchs.

### 3. Dichotomie ne remontant pas assez loin en arrière
**Problème** : la recherche fine partait du point candidat `Q1` lui-même, jamais avant — alors que `Q2` peut être vrai plus tôt que le moment où le mouvement devient assez net pour déclencher `Q1`.
**Correction** : utiliser le dernier point où `Q1` disait `NON` comme borne inférieure de la dichotomie.

### 4. Q1 ne détectait que le mouvement, pas l'état déjà vidé
**Problème** : si le scan tombait *après* que la sortie soit terminée (terrain déjà clairsemé), `Q1` ne le reconnaissait pas car il ne cherchait que l'action de partir.
**Correction** : ajout du Signal B (terrain déjà dégarni), indépendant du mouvement observé.

### 5. Q2 trop strict, biais systématique en retard
**Problème** : `Q2` n'acceptait que le terrain **complètement** vide, rejetant "en train de sortir" comme `PAS_VIDE` — poussant la détection après le vrai instant (`+33` à `+46s` systématiques).
**Correction** : la recherche fine utilise `Q1` (signal large), `Q2` sert uniquement à la confirmation initiale à distance (`+90s`).

### 6. Bug de robustesse : extraction hors durée de la vidéo
**Problème** : `ffmpeg` tentant d'extraire une frame au-delà de la fin réelle du fichier faisait planter tout le pipeline (`FileNotFoundError` non gérée).
**Correction** : `_extraire_frame` retourne `None` proprement en cas d'échec ; nouvelle fonction `obtenir_duree_video` (via `ffprobe`) pour plafonner les fenêtres de recherche à la durée réelle, avec une garde supplémentaire permettant de plafonner (plutôt que rejeter) la vérification `+90s` quand elle dépasserait la fin de la vidéo.

### 7. Repli intelligent si aucun candidat confirmé (FinMatch)
**Problème** : si aucun candidat n'était confirmé avant la fin de la fenêtre (vidéo coupée peu après le vrai `FinMatch`), le système renvoyait `None`.
**Corrections successives** :
- D'abord : renvoyer la fin de vidéo brute comme estimation (`+59s` sur `Goe`).
- Puis : balayage `Q2` en avant depuis le dernier point connu jusqu'à la fin de la vidéo, avec léger affinage — `+11s` sur le même cas.

### 8. Un seul ballon requis pour une vraie phase de jeu
**Problème** : deux ballons visibles simultanément (image utilisateur, joueurs avec ballon de rechange) n'étaient pas reconnus comme signe de non-match.
**Correction** : critère objectif — `2` ballons ou plus visibles → `OUI` décisif, indépendant du nombre de joueurs.

### 9. Une seule équipe visible mal détectée
**Problème** : un seul joueur adverse aperçu au loin suffisait à faire dire "les deux équipes sont là" à tort.
**Correction** : exige au moins `2-3` joueurs identifiables de chaque équipe, pas un isolé.

### 10. Gardien et arbitre absents — signaux faibles uniquement
Ajoutés comme signes **secondaires**, jamais suffisants seuls (un gardien peut légitimement quitter sa zone pour un renvoi, un arbitre peut être hors-cadre).

### 11. Formation vs attroupement social
**Problème** : des joueurs des deux équipes regroupés pour discuter (effectif normal, un seul ballon) étaient classés `NON` à tort par le critère brut "effectif + ballon".
**Correction** : distinction explicite entre joueurs **répartis en formation de jeu** (`NON`) et **regroupés ensemble pour discuter** (`OUI`, même avec effectif normal).

### 12. Blessure / discussion arbitrale mal interprétée comme fin de match
**Problème le plus impactant** (`Wanze -397s`) : un attroupement de joueurs autour d'un blessé, avec le reste de l'effectif simplement hors-cadre, était lu comme "terrain dégarni = fin de match".
**Correction** : exception explicite — un attroupement autour d'un incident précis (blessé, discussion arbitrale) ne déclenche jamais le Signal B.

### 13. Coup franc / coup de pied arrêté non couvert
**Problème** (`Juprelle -530s`) : un ballon seul et immobile, joueurs à distance en attente, était lu comme fin de match.
**Correction** : exception explicite — ballon unique et immobile avec joueurs à distance = configuration de coup de pied arrêté, oriente vers `NON`.

### 14. Incohérence "deux signaux" vs "trois signaux" dans le prompt FinMatch
**Bug trouvé en vérifiant** : le texte d'en-tête disait encore "l'un ou l'autre des **deux** signaux" alors que le Signal C (poignées de main) avait été ajouté depuis, portant le total à trois. Corrigé (Fin1MT n'était pas concerné, il n'a que 2 signaux).

### 15. Vérification narrative à 2 images (dernier ajout majeur)
**Problème résiduel** (`Juprelle`, diagnostiqué précisément par appels répétés à `t=6601s`) : `Q1` confiant et stable sur `3` appels identiques déclenchait à tort sur un instant transitoire — juste après une faute sifflée, avant que le ballon soit replacé pour le coup franc.
**Conception** : plutôt que de comparer deux jugements indépendants (`Q1` à `t` et `Q1` à `t+5s`), envoyer les **2 images ensemble** dans un seul appel, avec un prompt demandant si elles racontent une histoire cohérente de fin de match/pause.
**Premier essai insuffisant** : le critère initial acceptait "la même configuration calme se maintient, le jeu ne reprend pas" comme preuve suffisante — mais n'importe quel arrêt de jeu normal ressemble à ça sur `5s` aussi.
**Correction finale** : exige un signe **positif et actif** (mouvement de sortie qui progresse, poignées de main, nombre de joueurs qui diminue encore) — l'absence de reprise seule ne suffit jamais.
**Placement** : au niveau du scan grossier, immédiatement après qu'un candidat `Q1` soit trouvé, **avant** la vérification `Q2` à `+90s` — pour économiser les appels sur les faux signaux plutôt que de les gaspiller sur une vérification à distance vouée à l'échec.

### 16. Bug de nettoyage incomplet
Après avoir déplacé la vérification narrative du niveau "recherche fine" (`+5s` sur `Q1` seul) vers le niveau "scan grossier" (`2` images ensemble), un résidu de l'ancienne implémentation était resté dans `fin1mt_gemini_cascade.py` (mais avait bien été retiré de `finmatch_gemini_cascade.py`) — les deux vérifications coexistaient. Trouvé lors d'une vérification systématique demandée explicitement, corrigé.

---

## Résultats sur 5 matchs testés avec la version finale (FinMatch)

| Match | Erreur | Appels Gemini |
|---|---|---|
| Juprelle | +13s | 37 |
| Franchimont | +27s | 26 |
| Goe | +21s | 38 |
| Wanze | +2s | 30 |
| Stembert | +12s | 43 |

Plus aucun échec catastrophique (comparé à `-530s` sur `Juprelle` et `-397s` sur `Wanze` avant les corrections 12-13). Test limité à `5` matchs pour raison de coût — les `4` restants (`Andrimont`, `Raeren`, `P1Minerois`, `MineroisSter`) n'ont pas encore été retestés avec la version finale.

---

## Ce qui reste à valider

- **Test complet des 9 matchs** avec le pipeline entier (`KO1+KO2+Fin1MT+FinMatch`) et la version finale des prompts — notebook prêt (`notebook_pipeline_complet_9matchs.py`), pas encore exécuté.
- **Coût réel total du pipeline** — mesuré partiellement (`KO2` seul : ~46,6 appels/match ; `FinMatch` seul sur 5 matchs : ~34,8 appels/match), pas encore la moyenne consolidée des 4 détecteurs ensemble sur les 9 matchs.
- **Limite structurelle documentée mais non résolue** : sur `KO2`, `Stembert` reste à `+16s` — cas où `Flash` est confiant mais se trompe sur une image isolée sans contexte temporel suffisant (touche sifflée juste après l'engagement). Cette même classe de limite (jugement sur image isolée) est probablement encore présente, à un degré moindre, sur `Fin1MT`/`FinMatch` malgré les corrections apportées.

---

## Fichiers de production actuels

| Fichier | Rôle |
|---|---|
| `kickoff_gemini_cascade.py` | `KO1` + infrastructure partagée (`_EtatRecherche`, `_appeler_json_robuste`, `_extraire_frame`, `obtenir_duree_video`) |
| `match_boundaries_v2.py` | `KO2` |
| `fin1mt_gemini_cascade.py` | `Fin1MT` (Q1/Q2/recherche fine/vérification narrative) |
| `finmatch_gemini_cascade.py` | `FinMatch` (même architecture que Fin1MT, + Signal C) |
| `notebook_pipeline_complet_9matchs.py` | Test bout-en-bout des 4 détecteurs, sortie condensée, coût total |
