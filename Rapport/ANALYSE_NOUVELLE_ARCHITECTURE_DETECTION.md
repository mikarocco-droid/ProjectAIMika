# Analyse — Refonte de la détection d'événements (ScoutIA)

**Contexte** : document de travail rédigé après une session de tests réels
sur un match complet (`Franchimont`, `113` min, caméra `low_side_zoom`),
destiné à recueillir un avis externe avant de lancer un chantier de
développement. Objectif : remplacer une architecture de détection
d'événements qui s'est révélée à la fois **trop chère**, **trop lente**
et **imprécise** sur ce test, par quelque chose de **peu coûteux et
robuste**.

---

## 1. Constat mesuré — pourquoi on change

### 1.1 Le système actuel (à remplacer)

La détection des buts repose sur une heuristique de tracking de ballon
(trajectoire, vitesse, position par rapport à la ligne de but — fichier
`analysis/terminal_events.py`), qui génère des **candidats**, ensuite
validés un par un par un appel `Gemini` vision (`ai/gemini_validator.py`,
fonction `find_goal_after_shot`).

### 1.2 Chiffres réels mesurés sur un vrai match complet (ce soir)

| Métrique | Valeur mesurée |
|---|---|
| Durée du match | `113` min (caméra `low_side_zoom`) |
| Candidats de buts bruts détectés (heuristique ballon) | `134` |
| Buts réellement validés au final | `15` (nombre plausible pour ce match) |
| **Appels `Gemini` pour la seule validation buts/tirs/maillots** | **`1294` sur `1401` au total (`92%`)** |
| Temps total de l'analyse | `~590` minutes (`~10h`) |
| Temps annoncé sur le site pour ce type de match | `1h`–`2h` |

**Un `92%` du coût `Gemini` total sert uniquement à rejeter des faux
positifs de détection de buts.** Ce n'est pas un problème de réglage
fin (seuils mal calibrés) — le design lui-même (chercher le but en
observant chaque mouvement suspect du ballon) oblige structurellement à
examiner un grand nombre de non-événements.

### 1.3 Autres constats du même test

- **Système `A/B` dupliqué actif en production** : chaque candidat
  passe par **deux** fonctions de validation différentes
  (`find_goal_after_shot` et `find_goal_after_shot_v2`), doublant
  potentiellement le coût sur cette étape. Un garde-fou (`audit_mode`)
  est censé désactiver la version `V2` en mode audit, mais son
  comportement lors de ce run précis reste **à reconfirmer** (possible
  contamination par un cache de module `Python`, pas formellement isolé).
- **Incohérence de comptage interne** : le nombre de buts affiché à
  deux endroits différents du même rapport (`22` vs `15`) ne concorde
  pas — bug de cohérence distinct du problème de coût.
- **Lecture des numéros de maillot** : taux de succès mesuré de
  seulement `~13-15%` des `track_id` tentés (`179`/`1385`,
  `217`/`1544` sur les deux mi-temps).
- **Le tracker de ballon montre des réinitialisations très fréquentes**
  (`[BALL TRACKER RESET]`) sur ce type de caméra — signal supplémentaire
  que l'approche par trajectoire de ballon est fragile sur certains
  angles de prise de vue.

### 1.4 Décision

**Remplacer l'architecture de détection d'événements**, plutôt que de
continuer à calibrer/patcher le système actuel. Deux besoins produit
distincts sont identifiés, avec des architectures et des prérequis
différents — traités comme **deux chantiers séparés**.

---

## 2. Chantier A — Résumé de match / temps forts (mode "match")

> **Révisé après relecture externe** — le concept initial ("l'échantillonnage à `5`s détecte l'action") a été corrigé en un modèle à **deux étages**, jugé plus robuste. Voir `2.2` pour la version actuelle.

### 2.1 Objectif

Remplacer entièrement la détection actuelle (ballon `+` `V1`/`V2`) par
un mécanisme en deux temps, indépendant du tracking de ballon.

### 2.2 Principe retenu — macro-scan puis micro-scan

**Erreur à éviter** (identifiée lors de la relecture) : demander à
l'échantillonnage grossier de détecter l'action elle-même. Un tir, une
interception, un dribble ou une passe peuvent durer `1-2` secondes et
passer entre deux échantillons pris à `5`s d'intervalle — l'action
elle-même est un mauvais objectif pour ce scan.

**Le bon rôle du scan grossier** : repérer des **zones d'intérêt**, pas
des actions précises. Il doit seulement répondre à *"il se passe
quelque chose d'intéressant ici"*, pas *"que s'est-il passé
exactement"*.

```
MATCH
 │
 ├── Macro-scan (1 image toutes les ~5s, Gemini, état simple)
 │      NORMAL        → rien
 │      ATTACK         → ouverture d'une zone d'intérêt
 │      CELEBRATION    → zone très probablement liée à un but
 │      RESTART_KO     → chercher un but juste avant (voir 2.3)
 │      DEAD_BALL      → probablement ignorable
 │
 ▼
 Fenêtre vidéo ciblée autour de la zone repérée
 │
 ▼
 Micro-scan (analyse fine, uniquement sur cette fenêtre)
 │
 ▼
 Résultat : but / tir / action notable, avec timing précis
```

Format de sortie du macro-scan (`JSON` strict, peu de tokens de
sortie) :

```json
{"state": "NORMAL" | "ATTACK" | "CELEBRATION" | "RESTART_KO" | "DEAD_BALL"}
```

### 2.3 `RESTART_KO` comme ancre pour la détection des buts — le point fort de cette architecture

**On sait déjà détecter `KO1` et `KO2` de façon fiable** (validé ce
soir sur `11` matchs de référence). N'importe quelle **autre** reprise
au centre du terrain détectée en cours de match ne peut être ni `KO1`
ni `KO2` — c'est donc, presque par élimination, la conséquence d'un but.

```
RESTART_KO détecté à 52:37
       │
       ├── Est-ce KO1 ? non (déjà connu)
       ├── Est-ce KO2 ? non (déjà connu)
       │
       ▼
Fouille ciblée dans [52:07 → 52:37] (micro-scan)
       │
       ▼
Gemini / analyse fine → BUT confirmé, timing précis
```

**Ce mécanisme élimine entièrement l'approche actuelle** (candidat
détecté par trajectoire de ballon → relecture dense → validation
`Gemini` → faux positif → `V2` → ...) — on ne cherche un but que là où
la reprise au centre prouve après coup qu'un but a bien eu lieu.

### 2.4 Autres états du macro-scan

| État détecté | Fenêtre ouverte pour le micro-scan |
|---|---|
| `CELEBRATION` | `[t-20s, t]` — capturer l'action qui a précédé |
| `ATTACK` | `[t-5s, t+10s]` |
| États `ATTACK` consécutifs | Fusionner en un seul clip continu avant micro-scan |
| `RESTART_KO` (hors `KO1`/`KO2`) | Voir `2.3` — recherche de but dans la fenêtre précédente |

### 2.5 Coût — objectif de mesure, pas un chiffre à viser

**Les estimations chiffrées produites initialement (`<0,05€` puis
`~0,10-0,15€` recalculé) sont retirées de ce document** — après avoir
constaté ce soir à quel point une estimation de coût théorique peut
s'éloigner de la réalité mesurée (le système actuel devait coûter
quelques centimes par appel et a fini à `1401` appels sur un seul
match), il serait malhonnête de fixer un chiffre cible avant
implémentation.

**Objectif reformulé** : réduire fortement le nombre d'appels `Gemini`
par rapport au système actuel (`1401` appels mesurés, dont `92%` de
validation de faux positifs). **Le nombre réel d'appels et le coût par
match seront mesurés expérimentalement après implémentation**, pas
supposés à l'avance.

**Point technique à ne pas oublier** : `Gemini 2.5 Flash` (modèle
initialement envisagé) est prévu pour être retiré le `16/10/2026` —
implémenter directement avec `gemini-3.5-flash` ou la version la plus
récente disponible au moment du développement.

### 2.6 Métriques de succès (à mesurer, pas à supposer)

- **Taux de rappel des temps forts** : proportion d'actions notables
  réellement présentes dans le match qui sont retrouvées par le
  macro-scan `+` micro-scan (vs actions manquées).
- **Ratio réel d'appels `Gemini`** (macro-scan `+` micro-scan combinés)
  par rapport au système actuel, mesuré sur l'ensemble des `11` matchs
  de référence — pas seulement `Franchimont`.
- **Temps total de traitement**, à comparer directement aux `~590`
  minutes mesurées ce soir sur le système actuel.

### 2.7 Risque connu, à tester empiriquement

Le macro-scan pourrait manquer une zone d'intérêt si l'action et sa
conséquence visible (célébration, reprise) sont toutes deux très
brèves ou masquées par l'angle de caméra. Le modèle à deux étages
réduit ce risque par rapport à la version initiale (détecter l'action
elle-même), mais **ce n'est toujours pas démontré** — à confirmer sur
les `11` matchs de référence avant de considérer ce chantier validé.

### 2.8 Ce que ce chantier remplace concrètement

- `analysis/terminal_events.py` (heuristique ballon)
- `find_goal_after_shot` / `find_goal_after_shot_v2` dans
  `ai/gemini_validator.py`
- Probablement une partie de `analysis/goal_posthoc.py`

### 2.9 Résultats du premier `POC` (`12/09/2026`, `Andrimont`, `300-900s`)

**Module testé** : `analysis/macro_scan.py` — implémente uniquement le
macro-scan (classification `1` image/`5s`), **sans micro-scan** (pas
encore construit). `3` itérations de prompt ont été nécessaires avant
d'obtenir ce résultat — détail dans le suivi de session, pas repris ici.

**Sur `RESTART_KO` (le mécanisme le plus critique, lié aux buts)** :
- ✅ `KO1` (`t=307s`) correctement détecté (`300s`, `305s` classés `RESTART_KO`)
- ✅ **Le vrai but à `382s` (`6:22`) correctement détecté** via sa
  reprise au centre (`400s`, `405s` classés `RESTART_KO`, cohérent avec
  la reprise réelle observée à `~412s`/`6:52` sur une capture d'écran)
- ⚠️ **`1` faux positif résiduel** à `t=895s` (`14:55`) — confirmé par
  capture d'écran comme n'étant PAS un vrai coup d'envoi (répartition
  des joueurs pas assez nette). Persiste malgré plusieurs resserrements
  du prompt.
- **Bug de test corrigé en cours de route** : la marge d'exclusion
  `KO1`/`KO2` (paramètre `marge_exclusion_ko` de
  `detect_goal_candidates_via_ko`) était initialement à `120s`,
  excluant à tort un but survenu `93-98s` après `KO1` — corrigée à
  `60s`.

**Sur `SHOT`** (état ajouté en cours de `POC`, pas dans la conception
initiale) :
- Sur `3` tirs connus communiqués par l'utilisateur (`10:35`, `11:23`,
  `14:40`), **`1` détecté correctement** (`10:35`, classé `SHOT` pile
  sur l'échantillon). Rappel mesuré `1/3` — **échantillon bien trop
  petit pour en tirer une conclusion generalisable**, à confirmer sur
  un vrai jeu de test avec davantage de tirs recensés.
- Le tir à `11:23` (sorti "à côté") n'est pas classé `SHOT`, mais
  l'échantillon suivant (`+2s`) montre `DEAD_BALL` — une trace
  indirecte exploitable existe, mais n'est pas utilisée actuellement.
- **Le tir à `14:40` (repris/dégagé des poings par le gardien, sans
  aucun arrêt de jeu) reste indétectable même après ajout d'un signal
  `ATTACK` positionnel (ballon dans le grand rectangle)** — ce cas
  spécifique ne laisse structurellement aucune trace observable après
  coup (contrairement à un but ou un tir sorti), et est probablement le
  cas le plus dur, par nature, pour toute architecture basée sur un
  échantillonnage périodique.

**Verdict de cette itération** : suffisant pour valider l'hypothèse de
base du macro-scan sur les buts (`RESTART_KO` fonctionne, avec un taux
de faux positifs résiduel faible sur cet échantillon — `1`/`120`), pas
suffisant pour conclure quoi que ce soit de solide sur les tirs
(échantillon de `3` cas, non représentatif).

**Décision** : itération du prompt du macro-scan arrêtée à ce stade
(accord explicite du `12/09/2026`) — construire le micro-scan plutôt
que de continuer à resserrer le macro-scan seul.

### 2.10 Micro-scan — deux approches testées, une seule retenue

**Première approche testée (rejetée)** : chercher un signal fin dans la
fenêtre précédant le candidat `RESTART_KO` (ballon dans les filets,
puis célébration d'un joueur en complément). **Échec constaté** : sur
une caméra grand-angle filmant tout le terrain, un joueur qui célèbre
n'occupe qu'une poignée de pixels — `Gemini` a classé `"attente"` la
frame exactement à l'instant où une célébration bras-levés était
confirmée par capture d'écran (`t≈386s`), y compris après avoir forcé
une évaluation frame-par-frame explicite. Le signal existe dans
l'image, mais sa taille en pixels le rend structurellement peu fiable
à détecter sur ce type de cadrage. Cause écartée : ni la densité
d'échantillonnage (testée jusqu'à `2s`), ni le nombre de frames par
appel, ni le forçage d'une évaluation par frame n'ont résolu le
problème — la limite est la résolution effective du sujet dans l'image,
pas le prompt ou la méthode d'appel.

**Deuxième approche testée (retenue)** : au lieu de chercher un signal
fin invisible en plan large, **re-vérifier le signal macroscopique qui
fonctionne déjà** — la formation `RESTART_KO` elle-même — sur plusieurs
frames autour du candidat, avec un vote majoritaire. Même principe que
le système `KO1`/`KO2` existant (voter sur plusieurs passes plutôt que
sur une seule classification isolée), appliqué ici à un candidat de but
en milieu de match. Fonction : `micro_scan_confirm_restart_ko()`.

**Résultat (`12/09/2026`, `2` candidats testés)** :
- `t=402s` (vrai coup d'envoi après le but à `~386s`) → **confirmé**
  (`6/9` votes `"oui"`), avec une signature temporelle cohérente
  (`non` avant `398s`, `oui` de `398s` à `408s`, `non` après `410s`)
- `t=895s` (faux positif confirmé par capture d'écran) → **rejeté**
  (`2/9` votes `"oui"`, répartis de façon isolée plutôt que stable dans
  le temps)

**Score : `2/2`** — mais échantillon de seulement `2` candidats, à
confirmer sur davantage de cas avant de considérer le mécanisme
robuste. Coût : `1` appel `Gemini` par candidat (`~9` frames dans
l'appel), contre `0` confirmation dans le système actuel.

**Architecture retenue pour le chantier `A` (buts)** :
```
Macro-scan (1 appel/échantillon, 5s) → candidat RESTART_KO
    → Micro-scan par re-vote (1 appel, ~9 frames autour du candidat)
        → confirmé (BUT) ou rejeté (faux positif)
```
Le micro-scan par recherche de ballon/célébration n'est pas retenu pour
la confirmation des buts, mais reste potentiellement utile pour d'autres
usages nécessitant un détail fin (non explorés dans ce POC).

### 2.11 Premier test à l'aveugle (`12/09/2026`, `Andrimont`, `250-1920s`)

Contrairement aux tests précédents (sur des cas déjà inspectés
manuellement), ce test a tourné sur des données **non vues au
préalable** — le vrai test qui compte.

**Résultat global** : `337` appels `Gemini` (`334` macro-scan `+` `3`
micro-scan) sur `~28` min, à comparer aux `1401` appels de l'ancien
système sur `113` min — volume par minute comparable (`~12`/min dans
les deux cas), mais la nature du travail diffère : l'ancien système
passait `92%` de ce volume à rejeter des faux positifs, celui-ci
classe l'état du jeu en continu.

**Sur les buts** : `3` candidats `RESTART_KO` trouvés hors `KO1`/`KO2`.
- `t=405s` → confirmé (`8/9`), cohérent avec le but connu à `~386s`
- `t=760s` → rejeté (`0/9`), sans ambiguïté
- `t=1860s` → rejeté (`2/9`) — **mais l'utilisateur confirme un vrai
  but à `~1831s`** (reprise attendue `~29s` après, cohérente avec
  `1860s`) : **c'est un faux négatif**, pas un rejet correct. Rappel
  mesuré sur ce test : `1/2` buts réels correctement confirmés (`50%`),
  **nettement en dessous** du `2/2` obtenu sur l'échantillon calibré
  précédent. Cause non investiguée (seuil de vote à `50%` peut-être
  trop strict pour ce cas, ou fenêtre de `±8s` mal centrée) — à
  creuser avant de considérer le mécanisme de vote fiable.

**Sur les tirs** — vérité terrain complète fournie par l'utilisateur
pour `7` tirs connus sur cette fenêtre :

| Timestamp | Réalité | Détection | Résultat |
|---|---|---|---|
| `635s` | tir réel | `SHOT` | ✅ trouvé |
| `683s` | tir non cadré | rien | ❌ raté |
| `880s` | tir réel (gardien) | rien | ❌ raté |
| `1235s` | tir non cadré | `ATTACK` (pile) | ⚠️ capté, mauvaise étiquette |
| `1346s` | tir cadré | `ATTACK` à `+4s` | ⚠️ capté, mauvaise étiquette |
| `1420s` | tir non cadré | `ATTACK` (pile) | ⚠️ capté, mauvaise étiquette |
| `1680s` | tir cadré | rien | ❌ raté |
| `770s` | rien | `SHOT` | ❌ faux positif |
| `1465s` | occasion sans tir réel | `SHOT` | ❌ faux positif |

**Conclusion sur `SHOT`** : `0/7` tirs réels correctement étiquetés
`SHOT`, et les `2` seules détections `SHOT` de tout le test sont des
faux positifs. **`SHOT` en tant qu'état distinct ne fonctionne pas** —
`ATTACK` (signal positionnel : ballon dans le rectangle) capte
davantage de tirs (`3/7` approximativement) mais sans distinguer tir
réel d'occasion sans tir, et reste un signal bruyant (peut se
déclencher pour toute action dans la surface, pas seulement un tir).

**Décision** : conserver `ATTACK` comme signal brut à rappel imparfait
assumé (piste `2` déjà actée), abandonner `SHOT` comme état dédié dans
sa forme actuelle.

### 2.12 Découverte — données de position déjà disponibles (`Re-ID`/radar `2D`)

En vérifiant la faisabilité d'un prototype de "radar `2D`" (position de
chaque joueur et du ballon à chaque instant, proposé par l'utilisateur
comme piste pour les tirs et les touches de balle), deux fichiers
existants ont été inspectés :

- **`frames_data_complet.pkl`** : `24726` frames, `99%` avec des
  joueurs détectés. Chaque joueur a déjà `id`, `tracker_id`, `bbox`,
  **`center` (position `x,y` déjà calculée)**, `team`, `conf`. C'est la
  donnée brute nécessaire à un prototype `V0` de radar — **déjà
  disponible, pas à instrumenter**. Aucun champ `ball` présent dans ce
  fichier précis (pas de position de ballon continue trouvée à ce
  stade).
- **`events__1_.pkl`** : `5983` événements déjà produits par un système
  heuristique existant (`possession`, `pass`, `dribble`, `shot`,
  `shot_blocked`, `goal`...), avec position, `xg`, score de `danger`,
  `on_target`. **`38` événements `shot` pour seulement `3` `goal`** —
  le même ordre de grandeur de sur-génération que le problème des
  `134` candidats de buts diagnostiqué en tout début de soirée. Un
  exemple de `shot` porte le champ `"source": "events_standard_fallback"`,
  suggérant un mécanisme de repli heuristique plutôt que le système
  principal.

**Conclusion** : la piste radar `2D` n'est pas une nouvelle idée
partant de zéro — les données de position existent déjà, et un premier
système géométrique existe même déjà pour les tirs. Mais ce système
existant souffre probablement du **même défaut structurel** (trop de
candidats) que celui qu'on corrige côté buts. Le vrai travail à faire
n'est donc pas d'obtenir les données, mais de **revoir la logique de
calcul** (vitesse, distance, trajectoire) par-dessus des données déjà
disponibles — et de vérifier au préalable la stabilité des `id` de
tracking lors des croisements de joueurs (risque de switch identifié
par l'utilisateur, cohérent avec le prérequis `Re-ID` déjà documenté en
section `3`).

**Question ouverte, non résolue** : quel sera le temps de traitement
réel d'une vidéo de `~90` min avec cette nouvelle architecture (tracking
`+` radar local `+` `Gemini` réduit) ? Aucune mesure isolée du temps de
tracking seul (indépendant des appels `Gemini`) n'existe à ce jour —
seul un futur run instrumenté séparément (chronomètre dédié pour la
phase de tracking vs la phase de validation `Gemini`) pourra y
répondre. Non résolu intentionnellement : tant que le volume réel
d'appels `Gemini` du nouveau système n'est pas stabilisé, une estimation
de temps total serait une supposition, pas une mesure.

---

## 3. Chantier B — Suivi et meilleures actions d'un joueur individuel (mode "joueur ciblé")

### 3.1 Objectif

Pour un joueur désigné (ex. "le `n°6` rouge"), identifier ses actions
notables (passe, tir, tacle, dribble) tout au long du match, à coût
minimal.

### 3.2 État actuel du système (vérifié ce soir)

- Le mode `"joueur ciblé"` existant (`analysis/player_identity.py`,
  `analysis/player_identity_memory.py`) **ne fait que filtrer** les
  `events`/`jersey_map` déjà calculés par le pipeline général — aucune
  logique dédiée de déclenchement ou de classification d'action par
  joueur n'existe actuellement.
- `analysis/player_reid.py` a déjà une tentative de `Re-ID` basique
  (histogramme de couleur `+` position, fonction `_extract_embedding`),
  mais **pas d'embedding d'apparence profond** —
  `reidentify_players()` utilise explicitement "pas d'embedding
  visuel". Un histogramme de couleur ne peut pas distinguer deux
  joueurs de la même équipe (même maillot).
  **Confirmé directement dans le code le `12/09/2026`** :
  `_extract_embedding()` (ligne `247`) appelle `cv2.calcHist` — c'est
  un histogramme de couleur, pas un réseau de neurones, malgré son nom.
  De plus, une vérification sur un vrai export
  (`frames_data_complet.pkl`, `321122` observations joueur) montre que
  `id`/`tracker_id`/`player_id` sont **strictement identiques dans
  100% des cas** — confirmant que `PlayerReID.process()` n'a pas été
  appliqué à ce run (ou n'a eu aucun effet de consolidation). Sur ce
  même export, **`389` `tracker_id` distincts ont été mesurés pour
  seulement `~22` personnes réelles sur le terrain** (`~18`
  identifiants par personne en moyenne) — confirmation chiffrée,
  directement sur données réelles, de l'ampleur du problème de
  fragmentation d'identité que ce chantier doit résoudre.
- Lecture des numéros de maillot par `Gemini` : `~13-15%` de réussite
  mesuré ce soir (cf. section `1.3`).

### 3.3 Pourquoi un préalable est nécessaire avant tout développement

Construire des déclencheurs sophistiqués par-dessus une identification
de joueur peu fiable reviendrait à construire sur une base instable.
**`3` problèmes techniques identifiés** :

1. **Taux de lecture de maillot trop faible** (`13-15%` mesuré) — on ne
   sait pas toujours qui est `#6`.
2. **Changements d'`ID` du tracker** lors de regroupements/occlusions
   — risque général bien documenté pour les trackers de type
   `SORT`/`DeepSort`, **pas mesuré spécifiquement** sur les données de
   ce soir (seul le tracker de ballon a montré des réinitialisations
   fréquentes dans les logs — le tracker de joueurs n'a pas été
   vérifié spécifiquement).
3. **Homogénéité visuelle entre coéquipiers** — un histogramme de
   couleur ne peut structurellement pas distinguer deux joueurs
   portant le même maillot.

### 3.4 Plan en 2 étapes proposé

**Étape `1` — Fiabiliser l'identification continue (prérequis, `0€`
d'`API` supplémentaire)**

- **Propagation temporelle du numéro de maillot** : une lecture fiable
  toutes les `~2` minutes suffit à maintenir la cartographie, si le
  `track_id` ne saute pas entre-temps — propager l'identité
  rétroactivement et vers l'avenir sur tout le segment de trajectoire
  continu d'un `track_id`.
- **Embeddings d'apparence profonds** (`OSNet` ou `TorchreID`, modèles
  locaux, aucun coût `API`) pour ré-identifier un joueur après un saut
  de `track_id` — résout spécifiquement le problème d'homogénéité
  visuelle entre coéquipiers, contrairement à l'histogramme de couleur
  actuel.
- **Mémoire spatiale** (position relative sur le terrain) pour éviter
  qu'un `ID` ne "téléporte" après un croisement de joueurs.

**Étape `2` — Déclencheurs `+` validation ciblée (seulement après
l'étape `1` validée)**

- Déclencheurs locaux légers sur la télémétrie du joueur suivi
  (accélération/sprint, entrée dans la surface adverse, proximité du
  ballon) — calcul local, `0€` d'`API`.
- Appel `Gemini` ciblé uniquement aux moments déclenchés, pour
  classifier l'action précise (passe/tir/tacle/dribble).

### 3.5 Coût — objectif de mesure, pas un chiffre à viser

**Comme pour le chantier `A`, les chiffres précis initialement avancés
(`~20-50` appels, `<0,05€`/match) sont retirés de ce document** — ils
risquaient de devenir un objectif implicite alors qu'ils n'ont jamais
été mesurés. Le système actuel devait lui aussi coûter quelques
centimes en théorie et a fini à `1401` appels sur un seul match : les
estimations de coût non testées se sont révélées peu fiables ce soir.

**Objectif reformulé** : réduire fortement le nombre d'appels `Gemini`
par rapport à une approche naïve (`Gemini` à chaque seconde, `~5400`
appels/match). **Le nombre réel d'appels et le coût par match seront
mesurés expérimentalement une fois l'étape `1` (`Re-ID` fiable)
implémentée et le chantier `B` testé** — pas supposés à l'avance.

### 3.6 Diagnostic approfondi de la fragmentation `Re-ID` (`13/09/2026`)

Suite à la découverte de la section `3.2` (`389` `tracker_id` pour
`~22` personnes réelles sur `frames_data_complet.pkl`), un diagnostic
complet a été mené — **toujours sans modifier `player_reid.py`**, par
instrumentation via sous-classe dans les notebooks de test, pour ne
jamais confondre mesure et correction.

**Correctif appliqué (celui-ci, dans le code)** : `TTL_ACTIVE`/`TTL_SLEEP`
étaient des constantes fixes en nombre d'appels (`125`/`500`), calibrées
implicitement pour `25fps` sans jamais utiliser le paramètre `fps` reçu
par `__init__` malgré le commentaire `"@25fps"`. Corrigé en `TTL`
calculés dynamiquement depuis le `fps` **effectif** (rythme réel des
appels à `process()`, pas le `fps` natif de la vidéo) — probablement
un bug latent en production aussi, puisque `frame_skip` y réduit déjà
le rythme réel des appels bien en dessous de `25fps`.

**Test `OSNet` (embedding profond)** : `_extract_embedding()` était en
réalité un histogramme de couleur (`cv2.calcHist`) malgré son nom —
confirmé directement dans le code. Remplacé par `OSNet`
(`torchreid`, poids `ImageNet` par défaut — **pas** de checkpoint
`Re-ID` spécialisé `Market1501`/`DukeMTMC`, à corriger si besoin plus
tard) avec repli automatique sur l'histogramme si indisponible.
**Effet mesuré après correction du bug `TTL`** : `+2%` d'amélioration
seulement (`105→103` `ID` distincts sur un segment de `3` min) — réel
mais négligeable face à l'ampleur du problème.

**Diagnostic des causes réelles de fragmentation** (instrumentation de
`_assign_id()`, `1079` appels, `Andrimont` `300-480s`) :

| Cause | Nombre | Part |
|---|---|---|
| `MAX_PLAYERS_STEAL` (vol d'identité, pas une création) | `361` | `77,5%` des blocages |
| `EQUIPE` (contrainte d'équipe incompatible) | `57` | `12,2%` |
| `SPATIAL` (portillon `SPATIAL_MAX_DIST=80px`) | `47` | `10,1%` |
| `AUCUN_CANDIDAT` | `1` | `0,2%` |
| `SCORE_POSITION`/`COULEUR`/`EMBEDDING` | `0` | jamais observé |

**`105` vraies créations d'`ID`** (`EQUIPE`+`SPATIAL`+`AUCUN_CANDIDAT`)
pour `~22` personnes réelles, durée de vie médiane `32,2s` (`53%`
durent plus de `30s` — pas principalement une fragmentation en rafale
de micro-`ID`).

**Mécanisme causal identifié** : boucle auto-entretenue. `YOLO` ne
détecte jamais plus de `17` personnes par image (médiane `10`) mais
`active_count` (mémoire interne du `Re-ID`) grimpe à `19` en médiane et
jusqu'à `35` — écart médian `+9` entre détections réelles et mémoire
active, signe direct de fragments non nettoyés qui s'accumulent. Plus
il y a de fragmentation initiale (`SPATIAL`/`EQUIPE`), plus
`active_count` gonfle, plus `MAX_PLAYERS=25` est atteint souvent
(`29%` des appels), plus il y a de vols d'identité — qui corrompent
d'autres trajectoires et créent encore plus de fragmentation.
**`MAX_PLAYERS` n'est probablement pas la cause racine** — augmenter
ce plafond traiterait le symptôme, pas la source.

**Détail `SPATIAL` (`47` cas)** : `62%` ont un candidat vu à l'appel
précédent immédiat (`frames_absent≤1`) — **pas principalement un
problème d'occlusion prolongée**. Cohérent avec la distribution
indépendante déjà mesurée (`8,6%` des déplacements réels dépassent
`80px`, `P90=68px`, `P95=151px`) : le seuil semble simplement trop
serré pour la variance normale de mouvement à ce `frame_skip`, pas
seulement pour les cas d'absence prolongée.

**Détail `EQUIPE` (`57` cas)** : seulement `33%` ont une distance
spatiale `<40px` (probable vraie même personne, couleur mal inférée) —
signal initial jugé **insuffisant pour conclure**, l'hypothèse
"couleur d'équipe instable" a donc été testée séparément et
rigoureusement (voir ci-dessous) plutôt que d'être actée sur ce seul
indice ambigu.

**Test de stabilité de l'étiquette d'équipe, indépendant du `Re-ID`**
(appariement plus proche voisin très strict `<25px` entre appels
consécutifs — quasi certain même personne physique, sans dépendre de
la logique d'identité elle-même) :
- Sur `Andrimont` seul (fenêtre fixe `300-480s`) : `0,9%` de
  changements d'étiquette sur `5164` comparaisons — bas.
- **Étendu aux `11` matchs de référence pour ne pas répéter l'erreur
  déjà documentée dans le code** (seuils `_infer_team` calibrés sur
  `Raeren` seul par le passé, jamais revalidés). Premier passage avec
  fenêtre fixe `300-480s` pour tous les matchs : `2` cas aberrants
  détectés (`MineroisSter` `8,3%`, `Spa` peu de samples) — mais causés
  par un **défaut du test lui-même** (fenêtre absolue ne tenant pas
  compte du `KO1` réel de chaque match, signalé par l'utilisateur).
  Corrigé avec fenêtre `[KO1_réel, KO1_réel+180s]` par match (`11`
  valeurs de vérité terrain déjà établies lors de sessions
  précédentes) : les `2` cas aberrants disparaissent
  (`MineroisSter` `10→595` samples de calibration, `Spa` `8→542`).
- **Résultat final sur `11` matchs, méthodologie correcte** : `1,11%`
  agrégé (`723`/`64991`), taux par match resserré entre `0,7%` et
  `2,0%` — homogène, aucun cas aberrant restant.

**Conclusion sur `EQUIPE`** : écartée comme cause principale de la
fragmentation, cette fois avec une validation multi-match rigoureuse
(contrairement à l'historique du code). Les `57` cas `EQUIPE` du
diagnostic sont donc plus probablement de vrais duels/croisements
entre joueurs d'équipes différentes (comportement voulu du portillon)
que des erreurs de couleur.

**`SPATIAL_MAX_DIST` testé empiriquement sur `11` matchs** (`80`/`100`/
`120`/`150`/`200px`, mêmes fenêtres alignées `KO1`), avec une métrique
de risque de fusion corrigée en cours de route (voir `3.7`) : `80` et
`100px` sont strictement identiques (aucun candidat ne bascule jamais
sur aucun match) ; le vrai effet démarre à `120px`. À `150px` :
fragmentation `-32%` (`103→70` `ID` médian), vols `MAX_PLAYERS` `-80%`
(`427→84` médian), pour une hausse de risque de fusion mesurée de
seulement `~7%` et une dérive de couleur des appariements acceptés
(`col_d_P99`) stable. **Valeur retenue et appliquée en production :
`150px`** (voir `3.7`).

### 3.7 Correctifs appliqués en production (`13/09/2026`)

Contrairement au reste de la section `3.6` (diagnostics et tests, sans
toucher au code réel), les correctifs suivants ont été **appliqués
directement** aux fichiers de production, après validation empirique :

**`player_reid.py`** :
- `TTL_ACTIVE`/`TTL_SLEEP` : constantes fixes en nombre d'appels
  (`125`/`500`, calibrées implicitement pour `25fps`) remplacées par
  un calcul dynamique depuis le `fps` **effectif** reçu
  (`TTL_ACTIVE_SECONDS=5.0`, `TTL_SLEEP_SECONDS=20.0`, converties en
  nombre d'appels via `fps`).
- `SPATIAL_MAX_DIST` : `200.0` (valeur par défaut de la classe,
  jamais utilisée telle quelle en production) → `150.0`, validée
  empiriquement sur `11` matchs (voir `3.6`).
- `_extract_embedding()` : histogramme de couleur (`cv2.calcHist`,
  malgré son nom) remplacé par `OSNet` (`torchreid`,
  `osnet_x0_25`, poids `ImageNet` — pas de checkpoint `Re-ID`
  spécialisé `Market1501`/`DukeMTMC`, amélioration possible future),
  avec repli automatique sur l'histogramme si indisponible.

**`vision/tracker.py`** :
- `set_spatial_max_dist(80)` → `set_spatial_max_dist(150)`, cohérent
  avec `player_reid.py`.
- `PlayerReID(fps=config.FPS)` → `PlayerReID(fps=fps_effectif)`, où
  `fps_effectif = config.FPS * (FRAME_SKIP_EVERY-1) / FRAME_SKIP_EVERY`
  — nécessaire pour que le correctif `TTL` ci-dessus fonctionne
  réellement en production (sinon `TTL` reste calculé sur la base du
  `fps` natif, pas du rythme réel d'appel).

**`main.py` — bug distinct, découvert en vérifiant le point d'appel
exact de `Tracker.update()`** :
- La boucle principale de lecture vidéo (ligne `~388`) utilisait
  `if frame_id % skip_every == (skip_every - 1): skip` — cette
  condition **saute `1` frame sur `skip_every`** et en **garde**
  `(skip_every-1)` sur `skip_every` (`~80%` des frames pour
  `skip_every=4`), **l'inverse exact** de l'effet recherché
  (`FRAME_SKIP_EVERY` visait à réduire `30fps→6fps`, soit garder `1`
  frame sur `5`). Corrigé en `if frame_id % skip_every != 0: skip` —
  ne garde maintenant que `1` frame sur `skip_every`, comme prévu à
  l'origine. **Probablement une cause majeure, distincte du coût
  `Gemini`, des `~590` minutes de traitement mesurées en tout début de
  soirée** — le levier principal censé réduire la charge de calcul ne
  réduisait le volume que d'environ `20%`, pas `80%` comme visé.
- Deux points annexes corrigés en cascade pour rester cohérents avec
  ce correctif : l'affichage du nombre de frames analysées
  (`analyzed_count`), et le débit de la vidéo annotée de sortie
  (`out_fps = fps/skip_every`, au lieu de `fps*2/skip_every` — le
  facteur `2` d'origine reste inexpliqué, aucune explication propre
  trouvée ni avec l'ancienne logique ni la nouvelle ; à surveiller si
  la vidéo annotée semble mal synchronisée après ce correctif).

**Cohérence méthodologique importante** : tous les tests `Re-ID` de la
section `3.6` (`TTL`, `SPATIAL_MAX_DIST`, stabilité couleur) ont été
réalisés avec un échantillonnage `1` frame gardée sur `5`
(`if (frame - début) % 5 == 0`) dans les notebooks de test — soit,
par coïncidence, **exactement le rythme visé une fois le bug `main.py`
corrigé**, mais **différent** du rythme réel de production avant ce
correctif (`~80%` des frames gardées). Les valeurs validées (`150px`,
`TTL` en secondes) restent donc valables pour la production
**corrigée**, sans besoin de revalidation.

### 3.8 Mesure réelle de vitesse et profilage complet (`13-14/09/2026`)

Après les correctifs de la section `3.7`, un test réel (pas un diagnostic
isolé) a été lancé via `process_video()` directement, sur un segment de
`5` min (`Andrimont`, à partir de `KO1`), avec un profilage temporel
détaillé ajouté à `3` niveaux (`main.py` → `process_batch`,
`vision/tracker.py` → `Tracker.update()`, `player_reid.py` →
`PlayerReID.process()`) pour mesurer plutôt que deviner où partait le
temps.

**Progression mesurée sur des runs réels successifs** (extrapolation à
un match complet `~100` min, à partir d'un segment de `5-15` min,
`Andrimont`) :

| Étape | Temps extrapolé | Cause |
|---|---|---|
| Mesure initiale (`Franchimont`, match complet, bug `frame_skip` non corrigé) | `~590` min (`~10h`) | Référence historique — **vidéo et méthodologie différentes**, pas directement comparable terme à terme aux lignes suivantes |
| Premier test réel post-correctifs `frame_skip`+`TTL`+`SPATIAL_MAX_DIST` (`Andrimont`, `OSNet` sur `CPU`) | `~408-599` min (`~7-10h`, `2` runs) | `device="cpu"` codé en dur dans `_charger_osnet()`, jamais revu au passage du test isolé au pipeline complet — **aucune mesure isolée de l'effet du seul correctif `frame_skip` n'a été faite**, il était déjà appliqué dès le premier test de cette série |
| `+` `OSNet` sur `GPU` (non groupé) | `~387-428` min (`~6,5-7h`) | Passage `device="cpu"`→`"cuda"` |
| `+` `OSNet` traité par lot | **`~192` min (`~3,2h`)** | Appel individuel par joueur (`extractor([crop])`) remplacé par un seul appel groupé par frame |

**Le vrai goulot d'étranglement, identifié précisément par le
profilage** : `PlayerReID.process()` → extraction d'embedding
(`OSNet`) = `95,6%` du temps `Re-ID` avant correctif (`838s` sur
`876,6s`, segment de `5` min), à `13,18ms` par appel — mais un appel
par **joueur détecté**, pas par frame. Avec `~28` joueurs actifs par
frame et `2248` frames analysées sur le segment, ça représente
`~63000` appels `GPU` individuels — l'overhead fixe par appel (lancement
noyau, transfert mémoire) dominait très largement le calcul réel pour
ce petit réseau (`203k` paramètres, `82M` flops).

**Correctif appliqué** : `PlayerReID.process()` restructuré en `3`
passes — extraction rapide (couleur/équipe/crop) par détection, **un
seul appel `OSNet` groupé pour tous les crops d'une frame**
(`_extraire_embeddings_batch()`), puis assignation d'identité. Le
repli histogramme de couleur (si `OSNet` indisponible) reste inchangé,
crop par crop — pas de gain de lot possible pour `cv2.calcHist`, déjà
peu coûteux par appel. **Effet mesuré : `838s→147,6s` pour
l'embedding (`-82%`)**, confirmant précisément l'hypothèse plutôt que
de la supposer.

**Répartition finale du temps, après tous les correctifs** (segment de
`5` min, `576s` réels) :

| Étape | Temps | Part |
|---|---|---|
| `tracker_update` (total) | `387,8s` | `80,8%` |
| — dont `deepsort` | `192,8s` | `40,2%` |
| — dont `reid` (`Re-ID` complet) | `195,0s` | `40,6%` |
| —— dont `embedding` (`OSNet`, par lot) | `147,6s` | `30,8%` |
| —— dont `assign_id` | `28,0s` | `5,8%` |
| —— dont `color` | `16,3s` | `3,4%` |
| `yolo_batch` | `48,5s` | `10,1%` |
| `ocr` (`pytesseract`) | `35,4s` | `7,4%` |

**`DeepSort` et `Re-ID` sont maintenant à quasi-égalité** (`192,8s` vs
`195,0s`) — plus aucune étape isolée ne domine à `>80%` comme avant le
correctif. `DeepSort` n'a pas été investigué pour un défaut de
traitement par lot similaire à celui trouvé sur `OSNet` — hypothèse
non vérifiée, à creuser si une optimisation supplémentaire est
recherchée.

**Résultat final : `~3,2h` extrapolées pour un match complet, sous la
cible initiale de `4h`** fixée en tout début de soirée (avant même de
savoir que le bug `frame_skip` ne faisait presque rien). **Non
revalidé sur un match complet réel** — cette extrapolation part d'un
segment de `5` min, à confirmer par un run bout-en-bout avant de
considérer ce chiffre comme définitif.

---

## 4. Séquencement — Chantier A d'abord, Chantier B ensuite

**Décision explicite** : ne pas développer les deux chantiers en
parallèle.

**Phase `1` — Chantier `A` seul.** Il répond directement au problème
démontré ce soir (`134` candidats → `1294` validations → `~10h` pour
un seul match). Question expérimentale à trancher : *est-ce qu'un
macro-scan `+` micro-scan peut reconstruire les temps forts d'un match
sans exploser le coût et sans rater trop d'actions ?* — testé sur
l'ensemble des `11` matchs de référence, pas seulement `Franchimont`.

**Phase `2` — Chantier `B`, uniquement une fois le chantier `A`
stabilisé et mesuré.** D'abord fiabiliser l'identité continue d'un
joueur (étape `1` de la section `3.4`), puis seulement détecter ses
actions (étape `2`).

**Justification** : le chantier `A` seul suffit déjà à répondre au
problème le plus urgent et le plus coûteux constaté ce soir. Ajouter le
chantier `B` en parallèle risquerait de diluer l'effort de validation
sur deux fronts à la fois, alors qu'aucun des deux n'est encore
démontré.

---

## 5. Questions ouvertes pour avis externe

1. **Le risque de rater un événement bref** avec le macro-scan à `5`s
   (chantier `A`) — un premier `POC` (section `2.9`) a confirmé le but
   testé ET produit `1` faux positif résiduel sur `RESTART_KO`, avec un
   rappel de seulement `1/3` sur les tirs testés. Est-ce que ce niveau
   de bruit est acceptable si le micro-scan sert de confirmation, ou
   faut-il déjà un pas plus fin (`2-3`s) avant même de construire le
   micro-scan ?
2. **`OSNet`/`TorchreID`** sont-ils le bon choix pour un `Re-ID` de
   joueurs de football en vidéo broadcast, ou existe-t-il des
   alternatives plus adaptées à ce contexte spécifique (mouvement
   rapide, résolution variable, angles de caméra larges) ?
3. **Existe-t-il des synergies utiles entre les deux chantiers** malgré
   le séquencement strict (ex. le `Re-ID` fiabilisé pour le chantier
   `B` pourrait-il aussi améliorer la robustesse du chantier `A` sur le
   long terme) ?
4. **Le format `JSON` minimal proposé pour le macro-scan** est-il
   suffisant, ou faudrait-il conserver un champ de raisonnement court
   pour permettre un contrôle qualité a posteriori sans reperdre le
   bénéfice de coût ?
5. Y a-t-il une meilleure approche générale que ces deux propositions,
   compte tenu des contraintes (vidéos amateur/semi-pro, angles de
   caméra variables, budget par analyse qui doit rester bas pour un
   modèle économique grand public) ?
6. **Le `POC` fait un seul appel `Gemini` par échantillon, sans vote**
   — contrairement au système `KO1`/`KO2` existant, qui vote sur
   plusieurs passes avant de conclure. Faut-il faire voter le
   macro-scan lui-même (coût doublé/triplé par échantillon), ou est-ce
   le rôle du micro-scan de jouer ce rôle de confirmation, en laissant
   le macro-scan volontairement bruité ?

### 3.9 Debug complet d'un but manqué (penalty à `t=382s`, `14/09/2026`)

Suite aux correctifs de la section `3.8`, un test réel ciblé a révélé
que le penalty confirmé par l'utilisateur (`6:22`/`382s`) n'était
**jamais** détecté malgré tous les correctifs — investigation menée
entièrement par instrumentation progressive (logs `[SHOT]`, `[BALL]`,
`[GOALZONE]`), sans deviner, jusqu'à isoler la cause exacte.

**Chaîne causale complète, trouvée par élimination successive** :

1. **`current=None` bloquait toute détection** — la détection tirs/buts
   dépendait de `if current:` (joueur le plus proche du ballon,
   seuil `6%` largeur image). Pendant le vol du ballon d'un penalty,
   aucun joueur n'est assez proche (confirmé empiriquement : `current=
   None` en continu de `t=380,4s` à `387,9s`, exactement la fenêtre du
   tir+but). **Corrigé** : détection découplée de l'attribution
   (`if True:` au lieu de `if current:`, `6` références à
   `current["id"]` sécurisées individuellement).

2. **Un faux but antérieur bloquait tout via un cooldown partagé** —
   après le correctif `1`, le compteur `[GOALZONE]` progressait
   normalement mais ne déclenchait jamais de `REJETÉ`/`CONFIRMÉ`.
   Cause : un second chemin de confirmation, totalement silencieux
   (`_bt.tick_shot_candidate()`), avait confirmé un but à `t=324,3s`
   (probable tir contré/rebond en zone de but, ballon détecté de façon
   erratique juste avant — `n_joueurs` anormalement élevé à `29-30`,
   cohérent avec une mêlée devant le but). Ce faux but posait
   `state["goal_cd"] = goal_cd_max`, une clé **partagée** entre les `2`
   chemins de détection (vérifié directement dans le code, `2`
   occurrences identiques). Le penalty (`58s` plus tard) tombait
   entièrement dans ce cooldown.

3. **Le cooldown lui-même était `4,17×` trop long** — même bug que le
   `TTL` de la section `3.6` : `fps=25` codé en dur dans `init_state()`
   pour convertir `goal_cooldown=150s` (intention du développeur,
   raisonnable) en nombre d'appels, alors que le compteur décrémente
   une fois par appel **analysé** (`~6fps` effectif avec `frame_skip`),
   pas `25fps` natif. `150s` voulus devenaient `625s` réels — dépassant
   l'écart minimum réel entre deux vrais buts de ce match (`384s`,
   entre `2554s` et `2938s`) : même sans faux positif, ce bug pouvait
   bloquer un vrai but avec un autre vrai but trop rapproché. **Corrigé**
   à la source : `fps` effectif calculé via `FRAME_SKIP_EVERY` et
   transmis à `init_state()`, restaurant les `150s`/`8s` voulus
   (au lieu d'imposer une nouvelle valeur arbitraire).

4. **La vraie cause du faux but à `324,3s`** — le chemin `_bt` exigeait
   seulement `min_frames=3` (`~0,5s`) sans aucune vérification de
   vitesse, contre `goal_frames_min=8` **et** une vérification que le
   ballon ralentit (`avg_speed < frame_w*0,09`) pour le chemin standard.
   Beaucoup plus permissif, donc plus sensible aux détections erratiques
   de ballon (mêlée, rebond). **Corrigé** : `min_frames` aligné à `8`,
   vérification de vitesse ajoutée (`ShotCandidate` retient maintenant
   les vitesses observées, `tick_shot_candidate()` transmet `ball_speed`
   et `frame_w`).

**Point non résolu, documenté pour référence future plutôt que corrigé
sans preuve** : un `3ème` chemin, `_shot_blocked_cd` (rebond après tir
contré détecté), réduit aussi le seuil standard — de `8` à `4` frames,
pendant une fenêtre courte et bornée (`_shot_blocked_cd_max=40` appels,
`≈6,7s` à `~6fps` effectif). Contrairement au chemin `_bt` avant
correctif, celui-ci passe par la **même** vérification de vitesse que
le chemin standard, et semble être un choix délibéré (les buts sur
rebond rapide après un tir contré arrivent légitimement plus vite
qu'un but "normal"). **Décision : ne pas corriger sans preuve d'un
problème réel** — le relever à `8` par précaution pourrait faire rater
de vrais buts sur rebond rapide, un risque qu'on ne veut pas prendre
sans avoir mesuré quoi que ce soit sur ce chemin spécifique. À
surveiller lors des prochains tests : si un but sur rebond est raté ou
si un nouveau faux positif apparaît via ce chemin précis, revenir ici.

**Fichiers concernés par ces `4` correctifs** : `analysis/events.py`
(`if current:` → `if True:`, `init_state(fps=...)`, `6` sécurisations
`current["id"]`), `vision/ball_tracker.py` (`ShotCandidate.speeds`,
`is_confirmed_goal(min_frames=8, max_avg_speed=...)`,
`tick_shot_candidate(speed=..., frame_w=...)`).

**Non encore validé** : ces `4` correctifs n'ont pas encore été testés
ensemble sur un run réel confirmant que le penalty à `382s` est
maintenant détecté — à faire avant de considérer cette chaîne causale
comme définitivement résolue.

### 3.10 Cinquième et sixième correctifs, et limite structurelle atteinte

**`5ème` correctif, confirmé par test réel** : `is_valid_jump()`
plafonnait les sauts de position du ballon à `250-350px` fixes,
calibrés implicitement pour un suivi proche de `25fps` — avec le
`frame_skip` réel, le premier saut rapide d'un tir était rejeté à
tort, la vélocité n'étant alors jamais mise à jour (elle ne se met à
jour que si le saut est accepté), créant un verrouillage
auto-entretenu où le ballon restait figé sur sa dernière position
pendant toute la durée du vol (`~7s` observées, `t=376-383s`).
**Corrigé** : plafond mis à l'échelle via `fps_scale` (`25/fps_effectif`),
`BallTracker.fps` fixé au vrai rythme effectif dans `main.py` (créé
trop tôt pour le connaître directement). **Confirmé par test réel** :
le ballon bouge enfin, `lost=0` en continu pendant la fenêtre du
penalty — plus de gel.

**Dernier verrou identifié, non corrigé — limite structurelle plutôt
qu'un bug de seuil** : `stability` (cohérence de direction sur les `3`
derniers points) reste quasiment toujours à `0,00` pendant le
penalty, bloquant `is_shot_candidate()`. Investigation par les
données : `y` oscille entre `~0,31` et `~0,45` de façon répétée à
`x` quasi fixe — pas une trajectoire cohérente. Hypothèse d'une
confusion entre `2` candidats ballon **testée et réfutée** (nouveau
log `[BALL_SELECT]`, `0` occurrence sur tout le run — un seul
candidat détecté à chaque frame, jamais d'ambiguïté simultanée).

**Confirmé par l'utilisateur (`6:20`-`6:24`, images vidéo) : un
penalty tout à fait normal**, sans rebond ni interruption inhabituelle
— élimine l'hypothèse d'un événement physique complexe (rebond sur
barre, déviation gardien). **Cause probable, sans correctif de code
sûr identifié** : un tir de penalty parcourt `~11m` en `~0,3-0,5s`
réels — à notre rythme effectif (`~6fps`), ça représente seulement
`2-3` échantillons captés pendant le vol propre, avant que le ballon
n'atteigne déjà les filets. Le calcul de `stability` à `t=380-384s`
capture donc très probablement le ballon **déjà dans les filets**
(mailles qui bougent, boîte de détection qui flotte entre deux
positions pendant que le ballon s'immobilise), pas son vol — une
limite intrinsèque de détection précise sur une fenêtre aussi courte
à ce rythme d'échantillonnage, pas un seuil mal calibré comme les `5`
correctifs précédents. Aucun correctif appliqué, faute de piste sûre
ne risquant pas de dégrader d'autres cas.

### 3.11 Piste retenue pour compenser cette limite : détection de coup d'envoi géométrique

**Proposition de l'utilisateur, jugée solide** : plutôt que de
chercher à rendre la détection du franchissement de ligne plus fiable
sur une fenêtre de `0,3-0,5s` (limite probablement irréductible au
rythme d'échantillonnage actuel), utiliser un signal **complémentaire,
géométrique et beaucoup plus lent** : la reprise du jeu au centre du
terrain après un but. C'est le même raisonnement qu'un arbitre humain
utilise implicitement — si le jeu reprend au centre peu après un tir
proche du but, c'est une confirmation forte, indépendante de la
précision instantanée du franchissement de ligne.

**Pourquoi c'est particulièrement adapté maintenant** : les positions
de tous les joueurs sont déjà calculées en continu (`Re-ID` fiabilisé
ce soir, tracking à chaque frame analysée) — un détecteur géométrique
de formation "coup d'envoi" pourrait tourner **gratuitement** sur ces
données déjà disponibles, sans appel `Gemini` supplémentaire. C'est
directement lié à la piste évoquée plus tôt ce soir sur le mélange
chantier `A`/`Re-ID`.

**Limite honnête à documenter** : une version fiable nécessiterait une
vraie calibration géométrique du terrain (ligne médiane, cercle
central en coordonnées pixel réelles) — travail de calibration caméra
non fait ce soir (rappel : `"[GOAL_BOX] Aucune structure détectée →
fallback"` déjà observé plus tôt pour la zone de but elle-même,
laissant penser que cette calibration reste globalement fragile sur
ce type de vidéo amateur). **Implémentation ce soir volontairement
limitée à un prototype expérimental, diagnostique uniquement** —
n'intervient pas dans la décision `CONFIRMÉ`/`REJETÉ` existante, log
seulement un signal pour validation ultérieure avant de l'intégrer
réellement à la logique de décision.

---

*Document produit le 11-12/09/2026, synthèse d'une session de travail
et de test réel sur le pipeline ScoutIA, révisé après une première
relecture externe (passage à un modèle macro-scan/micro-scan, retrait
des chiffres de coût non mesurés, ajout du séquencement strict `A` puis
`B`), puis après un premier `POC` réel du macro-scan sur `Andrimont`
(section `2.9`). Les chiffres de la section `1` et de la section `2.9`
sont mesurés sur de vrais runs ; le reste du document décrit des
propositions et hypothèses explicitement non validées.*

### 3.12 Revue approfondie de `events.py`/`ball_tracker.py`/`main.py` (`17/09/2026`)

Suite au `4ème` signal de stabilisation (section `3.11`), une revue
systématique complète (pas seulement les zones touchées ce soir) des
`3` fichiers a été menée pour anticiper les bugs plutôt que les
découvrir un par un lors des tests suivants.

**Découverte majeure — double traitement avec `_bt` périmé** :
`process_video()` (`main.py`) traite chaque frame **deux fois**. La
passe "live" (pendant `process_batch`, visible dans tous les logs
`[SHOT]`/`[GOALZONE]`/`CONFIRMÉ`) utilise le vrai `ball_tracker`,
correctement mis à jour frame par frame. Mais `process_video()`
rappelait ensuite `process_match(frames_data, ...)`, une **seconde**
passe qui rappelle `detect_events()` pour chaque frame stockée — **sans
jamais rappeler `ball_tracker.update()`**. Or `frames_data` stocke le
**même objet** `ball_tracker` (`ball["_tracker_ref"] = ball_tracker`,
une référence Python, pas une copie) pour toutes les frames. Résultat :
pendant la seconde passe, `_bt.ball_buffer` reste figé à son état de
**fin de traitement**, pour toutes les frames retraitées, peu importe
laquelle. Tout ce qui dépend de `_bt` (`is_shot_candidate()`,
`tick_shot_candidate()`, la vitesse brute du fallback
`goalzone_speed_fallback`) utilisait donc des données périmées
pendant cette seconde passe — **alors que c'est sa sortie qui était
retournée au final**, pas celle de la passe live.

Ça explique deux observations de la section `3.11` restées
mystérieuses : le but du run `A` qui disparaissait entre la
confirmation et le résultat final (la seconde passe, avec `_bt` figé,
ne satisfaisait plus les conditions du fallback), et le double print
`"CONFIRMÉ"` au même instant pour les runs `B`/`C` (une fois pendant
la passe live, une fois pendant la passe périmée).

**Corrigé** : `process_match()` n'est plus appelée. `main.py`
réutilise directement les événements déjà corrects de la passe live
(stockés par frame dans `frames_data[i]["events"]`), et applique
`team_map` en post-traitement léger (juste la mise à jour du champ
`"team"`), sans plus jamais rappeler `detect_events()`. `process_match()`
reste définie dans `events.py` (au cas où un autre appelant en
dépendrait) mais n'est plus utilisée par `main.py`.

**Autres correctifs appliqués lors de cette revue** :

- **`ball_tracker.py`** : `timestamp` n'était jamais transmis par
  `main.py` à `ball_tracker.update()` — l'horloge interne du buffer de
  ballon utilisait alors `self.frame_id/self.fps`, un temps **relatif**
  au début du tracking, pas le temps absolu de la vidéo (`current_time`
  dans `events.py`). Sans conséquence active tant que seules des
  différences de temps sont utilisées en interne (vitesse, stabilité —
  immunes à un décalage constant), mais fragilité latente. Corrigé :
  `main.py` transmet désormais `timestamp=frame_id/fps` explicitement.
- **`ball_tracker.py`** : `max_lost`, `_MAX_FILTERED_STREAK`, et les
  seuils `30`/`5000` dans `update()` étaient des nombres d'appels fixes,
  jamais mis à l'échelle du `fps` effectif — même motif systémique que
  `is_valid_jump()` (déjà corrigé section `3.10`). Convertis en
  propriétés calculées dynamiquement via le même `fps_scale`.
- **`ball_tracker.py`** : doublon exact de code dans
  `SimpleKalman.update()` (le même plafonnement de vélocité répété deux
  fois) — retiré, sans impact fonctionnel (idempotent) mais source de
  confusion à la lecture.
- **`ball_tracker.py`** : le print `"[BALL FILTERED]"` n'était pas
  conditionné à `DEBUG`, contrairement à tous les autres logs de
  diagnostic du module — corrigé.
- **`events.py`** : `min(state["shot_cd"], 15)` était un autre nombre
  d'appels fixe non mis à l'échelle — corrigé via une règle de
  proportion sur `shot_cd_max` (déjà correctement calibré), préservant
  la durée réelle voulue (`~0,6s`) peu importe le `fps` effectif.
- **`events.py`** : les valeurs de repli `state.get("_shot_cd_max", 75)`
  et `state.get("_goal_cd_max", 3750)` référençaient les anciennes
  valeurs pré-correctif — mortes en pratique (`init_state()` définit
  toujours ces clés), mais trompeuses à la lecture. Commentées
  clairement comme filet de sécurité, pas des défauts actifs.
- **`events.py`** : `process_events()` (fonction distincte de
  `process_match()`, contient un filtre dormant sur `xg=0.0`) confirmée
  non appelée nulle part dans le pipeline actuel — marquée comme code
  mort avec avertissement, pas supprimée.

**Non encore fait** : audit approfondi de `pipeline.py` (`~3600`
lignes, jamais audité en profondeur ce soir, seulement consulté
ponctuellement) — notamment pour vérifier si le chemin de production
réel (`run_pipeline()`) a un problème similaire ou différent de double
traitement, et si les correctifs de cette section s'y appliquent
correctement.

### 3.13 Audit complet de `pipeline.py` (`17/09/2026`)

Lecture intégrale des `3620` lignes de `pipeline.py`, suite à la
demande explicite de vérifier si le fichier de production réel
(`run_pipeline()`) est affecté par des problèmes similaires à ceux
trouvés en section `3.12`.

**Bonne nouvelle confirmée** : `detect_events` est importée mais
**jamais appelée directement** dans `pipeline.py` — le fichier passe
entièrement par `process_video()` (directement, ou via
`analyze_segments()`, qui l'appelle aussi en interne pour les chemins
`candidate_segments`/`use_match_boundary_segments`). Le correctif du
bug `process_match` (section `3.12`) s'applique donc automatiquement à
tous les chemins de `pipeline.py`, sans modification supplémentaire
nécessaire pour ce problème précis.

**`4` bugs trouvés et corrigés** :

1. **`fps=25` codé en dur** pour `detect_goal_box()` — alors que
   `_src_fps` (le vrai `fps` de la vidéo) est calculé juste avant et
   disponible. Contribuait peut-être à l'échec de détection déjà
   observé (`"[GOAL_BOX] Aucune structure détectée → fallback"`).
   Corrigé avec un repli sûr (`locals().get("_src_fps", 25.0)`) au cas
   où le bloc de calcul précédent aurait échoué.

2. **`teams` utilisé avant sa définition** — `teams` n'est réellement
   assigné qu'au `Step 4` (bien plus loin dans le fichier), mais
   référencé bien avant pour construire `_team_colors`, transmis à
   `find_match_end()`. Le `NameError` résultant était silencieusement
   avalé par un `except Exception: pass`, laissant `_team_colors`
   **toujours `None`** à cet endroit — `find_match_end()` en était donc
   privé sans que rien ne le signale. Corrigé en récupérant directement
   la source déjà disponible à ce stade (`vision.tracker._LAST_TEAM_COLORS`,
   peuplée pendant le `Step 1` — la même source utilisée un peu plus
   loin dans le fichier pour `_captured_team_colors`, juste récupérée
   plus tôt ici).

3. **Boucle morte sur liste vide** (`for _hl in []:`) — construction
   d'un dict `{goal_time: clip_path}` censé venir de `highlights`, mais
   `highlights` n'est peuplé qu'au `Step 8`, bien après ce point
   (`Step 1b`). Vérifié : **pas de conséquence fonctionnelle** (`{} or
   None` donne `None` de toute façon plus loin, avec ou sans la
   boucle) — mais code trompeur à la lecture, ressemblant à un oubli.
   Simplifié pour la clarté, comportement inchangé.

4. **Précédence d'opérateurs `and`/`or` ambiguë** sans parenthèses —
   pouvait inclure tous les tirs dans le pool de référence visuelle
   sans validation `Gemini`, contrairement à l'intention apparente du
   commentaire. Intention d'origine incertaine (pourrait être
   volontaire — les tirs comme référence brute, les autres types
   nécessitant validation) : parenthèses explicites ajoutées pour
   clarifier, **comportement exact préservé**, pas de changement de
   logique risquant une régression sans confirmation de l'intention
   réelle.

**Trouvé mais non corrigé (déjà connu, pas découvert ce soir)** : un
diagnostic déjà présent dans le code documente un écart de `0%` à
`93,9%` de doublons `tracker_id` entre la sortie du tracker et
`frames_data.pkl` final (mesuré sur `Raeren`) — activement en cours de
diagnostic par ailleurs (`[DIAG DOUBLONS]`), hors du périmètre de
cette session.

**Corrigé également** : `pipeline.py` ne reconnaissait pas les
nouvelles sources d'événements `events_bt` et `goalzone_speed_fallback`
(ajoutées ce soir) dans son instrumentation de log dédiée
(`[STANDARD BC4]`/`[BC4 AUTOPSY STD]`) — n'était pas cassant (aucun
filtre d'exclusion basé sur la source), juste moins de détail de log
pour ces sources spécifiquement. Ajoutées à la liste reconnue, une
seule occurrence dans le fichier — bénéficient désormais du même log
d'observation géométrique `BC4` (mesure `world_score` `VP`/`FP`) que
`events_standard`/`terminal_goal`.

### 3.14 Mesure et correctif : double système de classification d'équipe (`17/09/2026`)

Suite à l'échec du signal `GAP_EQUIPES` (aucune donnée `team`
disponible), investigation du flux `PlayerReID → TeamColorDetector →
detect_events()`, avec la règle explicite de **mesurer avant de
corriger** (proposée par l'utilisateur).

**Instrumentation ajoutée** (`main.py`, `assign_teams_by_color()`) :
compteurs purs, sans effet sur la décision, mesurant `6` métriques du
flux d'équipe sur un run continu de `20` min (`t=300-540s`, `2`
points de contrôle intermédiaires).

**Résultat mesuré, sans ambiguïté** :

| Métrique | Valeur |
|---|---|
| `PlayerReID` avait une équipe (avant effacement) | `90,9%` |
| `TeamColorDetector` fournit une équipe (final, cumulé) | `18,5%` → `21,5%` |
| `TeamColorDetector` fournit une équipe (**marginal**, post-calibration only) | `23,1%` |
| `PlayerReID` avait une équipe MAIS effacée, jamais récupérée | `71-74%` |
| `TeamColorDetector` calibré à la frame | `601` (`≈t=380s`) |

Le calcul marginal (uniquement la période entre les deux points de
contrôle, entièrement postérieure à la calibration de
`TeamColorDetector`) donne `23,1%` — quasi identique aux taux
cumulés. **Confirme que le taux bas n'est pas un artefact de la phase
de chauffe** : c'est le taux réel en régime stable.

**Cause exacte identifiée** : `main.py` efface explicitement
`p["team"] = None` avant d'appeler `color_detector.update()`. Or
`TeamColorDetector.update()` (Phase 2, `rendering/overlay.py`) contient
déjà `if p.get("team") is not None: continue` — **il ne force jamais
l'écrasement**, il ne fait que compléter les équipes manquantes.
L'effacement était donc strictement inutile pour son fonctionnement
propre, et responsable à lui seul de la perte de `~70` points de
couverture.

**Raison historique probable (non confirmée avec certitude)** :
`TeamColorDetector` porte la docstring *"pour l'overlay vidéo"*, avec
des garde-fous ajoutés *"suite au bug diagnostiqué sur Andrimont"* —
suggère qu'il a été conçu à l'origine pour le rendu visuel, puis
adopté comme source d'équipe pour les événements après un problème
constaté sur `PlayerReID` à cette époque, sans qu'on ait trouvé de
commentaire tranchant explicitement la question.

**Corrigé** : retrait de l'effacement inconditionnel. `PlayerReID`
garde désormais sa classification pour les `~91%` de joueurs qu'il
couvre déjà ; `TeamColorDetector` comble le reste (`~9%`) via son
mécanisme propre, inchangé. Aucune modification de `TeamColorDetector`
lui-même, ni de `PlayerReID` — uniquement le point de jonction dans
`main.py`.

**Non encore fait** : re-mesurer avec ce correctif pour confirmer que
la couverture finale grimpe bien vers `~90%+` (le notebook de
diagnostic a été régénéré avec le correctif inclus, prêt à relancer).
Reprendre ensuite `GAP_EQUIPES` (section `3.13`) une fois la donnée
`team` fiable confirmée disponible.

### 3.15 Audit approfondi : `player_reid.py`, `vision/tracker.py`, `rendering/overlay.py`, `config.py`, `vision/detector.py`, `vision/detect_goal_box.py` (`17/09/2026`)

Suite à la demande explicite d'auditer tous les fichiers encore non
couverts, pendant qu'un run de mesure tournait en arrière-plan.

**`3` bugs trouvés et corrigés** :

1. **`player_reid.py`, ligne `~500`** : `mem["team"] = det.get("team")
   or mem.get("team")` — motif `"falsy zero"` déjà rencontré ailleurs
   dans ce projet. Si `det.get("team") == 0` (équipe `0`, une
   classification valide), Python traite `0` comme faux : l'expression
   retombe sur l'ANCIENNE valeur en mémoire au lieu de garder le `0`
   frais. Perdait silencieusement des classifications `team=0` à
   chaque mise à jour de mémoire. Corrigé avec une vérification
   explicite `is not None`.

2. **`player_reid.py`, ligne `~544`** : `if inferred == "gk" and not
   existing_team:` — même motif. Si `existing_team == 0`, `not
   existing_team` vaut `True`, traitant à tort un joueur déjà classé
   équipe `0` comme s'il n'avait aucune équipe — risquant de le
   marquer `is_goalkeeper=True` à tort sur un simple échantillon de
   couleur ambigu. Corrigé avec `existing_team is None`.

3. **`vision/tracker.py`, ligne `~75`** : la formule de `fps_effectif`
   transmise à `PlayerReID` utilisait encore l'ANCIENNE logique buguée
   du `frame_skip` (`fps_natif * (skip_every-1)/skip_every` — celle
   qui "saute `1` frame sur `skip_every`") — la même qu'on a corrigée
   dans `main.py` plus tôt ce soir (elle gardait en réalité `1` frame
   sur `skip_every`, pas `skip_every-1`). Le commentaire n'avait jamais
   été mis à jour après ce correctif. Conséquence mesurée : `fps_effectif`
   surestimé de `3×` (`22,5` au lieu de `7,5` pour `skip_every=4`,
   `fps` natif `=30`) — confirmé incohérent avec `BallTracker` (déjà
   correct, `7,5fps` dans nos logs). `TTL_ACTIVE`/`TTL_SLEEP`
   (`5s`/`20s` intentionnés) duraient donc en réalité `~3×` plus
   longtemps (`~15s`/`~60s`), les fenêtres de mémoire `Re-ID` étant
   étirées bien au-delà de ce qui était prévu. Corrigé, formule alignée
   sur celle de `BallTracker` (`fps_natif / skip_every`).

**Correction importante à mon propre travail de ce soir** : le
correctif `fps=25` codé en dur pour `detect_goal_box()` (section
`3.13`, point `1`) s'est avéré **inoffensif mais sans aucun effet
réel** — vérifié que le paramètre `fps` n'est **jamais utilisé** dans
le corps de la fonction (une seule occurrence dans tout le fichier,
la définition du paramètre lui-même). L'échantillonnage se fait par
pourcentage du nombre total de frames, pas par calcul temporel. Le
correctif reste en place (harmless), mais ce n'était pas la vraie
cause de l'échec de détection du but observé sur cette vidéo — cause
encore non identifiée.

**Rien d'autre trouvé** dans `rendering/overlay.py` (au-delà de ce
qui a déjà motivé la section `3.14`), `config.py`, `vision/detector.py`.

### 3.16 Audit de `camera_profile.py`, `ocr.py`, `calibration.py`, `ball.py`, `model_manager.py` (`17/09/2026`, fichiers fournis par l'utilisateur)

**`1`. Correctif le plus significatif** : `vision/calibration.py`,
`calibrate()` échantillonnait les `10` TOUTES PREMIÈRES frames de la
vidéo (`t=0`), sans jamais avancer dans le temps — potentiellement
pré-match, terrain vide, écran noir/logo d'intro. Ce `shot_zones`
calibré ici est ensuite utilisé **en direct** pour tout le match
(`detect_events()`, via `pipeline.py`) — contrairement au système plus
robuste (`vision/camera_profile.py`, basé sur `50+` positions ballon
réelles accumulées), qui lui n'intervient qu'en **post-traitement**
(`BC4`, `goal_posthoc`), trop tard pour influencer cette calibration
initiale. **Corrigé** : avance à `t=90s` (heuristique raisonnable,
sans connaître l'heure exacte de `KO1` à ce stade précoce du
pipeline — `frames_data` pas encore construit), et étale les `10`
échantillons sur une fenêtre de `10s` plutôt que des frames
consécutives (`~0,33s` à `30fps`), plus robuste à un instant
malchanceux isolé.

**`2`-`3`. Même motif systémique, confirmé répandu** : `ocr.py`
(`ocr_every_n_frames=30`, censé être "1x/seconde" mais jamais mis à
l'échelle du `frame_skip` réel) et `ball.py` (`MAX_LOST=8`, "~0,3s à
25fps", même défaut). Corrigés selon le même schéma déjà établi ce
soir (propriété/valeur calculée dynamiquement depuis le vrai `fps`
effectif, fixée dans `main.py` une fois celui-ci connu — même
contrainte d'ordonnancement que `ball_tracker.fps`, ces objets étant
créés avant que le vrai `fps` ne soit disponible).

**Rien trouvé** dans `model_manager.py` (trivial), `__init__.py`
(vide). `detector.py`/`detect_goal_box.py` fournis étaient identiques
aux versions déjà auditées (section `3.15`).

### 3.17 Audit du dossier `analysis/` complet (`45` fichiers, `17/09/2026`)

Lecture ciblée (prioritaire sur les fichiers les plus liés à la
détection de buts/événements, puis recherche systématique des motifs
de bugs déjà identifiés ce soir sur les fichiers restants) suite à
l'envoi par l'utilisateur de l'archive complète du dossier `analysis/`.

**Confirmé identiques** à nos versions déjà corrigées : `events.py`,
`player_reid.py` (diff vide).

**`3` correctifs appliqués** :

1. **`goal_posthoc.py`** : `ABSENT_MIN_FRAMES = max(1, int(0.3 * 25 /
   _frames_spacing))` — `25` codé en dur pour le `fps` natif, alors que
   le vrai `fps` est déjà un paramètre de la fonction (et déjà transmis
   correctement par `pipeline.py`). Point positif à noter : `
   _frames_spacing` (l'équivalent de `skip_every`) est déjà mesuré
   **dynamiquement** ici (plus robuste que nos correctifs manuels
   ailleurs) — seul le `fps` restait codé en dur. Corrigé.

2. **`post_processing.py`** : f-string imbriquée avec le même type de
   guillemets (`f"...f'{g["time"]}'..."`) — syntaxe valide uniquement
   depuis `Python 3.12` (`PEP 701`). Si l'environnement de production/
   Kaggle utilise une version antérieure, cette ligne provoque une
   `SyntaxError` qui fait planter l'import de tout le module. Réécrite
   sans guillemets imbriqués, compatible toutes versions. Recherche
   systématique du même motif sur les `44` autres fichiers : aucune
   autre occurrence trouvée.

3. **`terminal_events.py`** : `_gk_center()` cherchait un champ
   `"role"`/`"label"` jamais défini nulle part dans le pipeline réel
   (vérifié par recherche exhaustive) — code mort, la fonction
   retombait toujours sur l'heuristique de position (joueur le plus
   proche d'une touche). Le vrai champ déjà calculé par notre pipeline
   (`is_goalkeeper`, `player_reid.py`) n'était jamais consulté. Corrigé.

**Découverte architecturale importante (pas un bug)** :
`terminal_events.py` implémente un **troisième système de détection
de but**, actif en production (`detect_terminal_events()`, appelé
depuis `pipeline.py`), totalement indépendant du lien-tir sur lequel
on a passé une bonne partie de la soirée à travailler dans `events.py`
— sa logique (`stuck`/`speed`/`rebound`) ne requiert aucun tir
associé. Ce système pourrait donc déjà capter, en production réelle
via le pipeline complet, des buts comme notre penalty que le chemin
`events.py` seul peinait à confirmer.

**Trouvé mais volontairement non corrigé** : `kickoff_detector.py`,
`find_match_end()` — `ball_speed` calculé comme distance brute en
pixels entre `2` frames analysées consécutives (jamais divisé par le
temps réel écoulé), avec des seuils (`_BALL_SPEED_MIN=3.0`,
`_SLIDING_MEDIAN_MIN=20.0`, etc.) calibrés empiriquement sur un match
de référence (`andrimont_2`) à un `frame_skip` donné. Si le
`frame_skip` change, ces mêmes valeurs représenteraient une vitesse
réelle différente — même motif que plusieurs correctifs de ce soir,
mais **non corrigé ici** : contrairement aux autres cas, une
correction changerait l'échelle de seuils déjà validés empiriquement
sur un vrai match, nécessitant une revalidation impossible à faire
depuis cette session. Documenté pour investigation future.

**Note pour une future session** : `kickoff_detector.py` implémente
déjà (dans `_score_frame()`, la fonction de scoring du coup d'envoi)
une séparation géométrique gauche/droite des joueurs très proche de
l'idée `GAP_EQUIPES` explorée en section `3.13` — à examiner comme
base de réutilisation possible plutôt que de repartir de zéro sur
cette piste.

**Fichiers vérifiés sans bug trouvé** (recherche ciblée, pas
nécessairement une lecture ligne par ligne complète comme pour
`events.py`) : `event_validator.py`, `team_cluster.py`,
`ball_physics.py`, `pass_detector.py`, `xg_model.py`, `xa_model.py`,
`utils_goals.py`, `match_boundaries_v2.py`, `player_identity_memory.py`
(début), `player_entity.py` (recherche de motifs). Recherche
systématique des motifs de bugs déjà identifiés ce soir (falsy-zero,
`fps=25` codé en dur pour du temps, boucles désactivées silencieuses)
sur l'ensemble des `45` fichiers : aucune autre occurrence trouvée au-
delà de ce qui précède.

**Non couvert en profondeur** (recherche par motif seulement, pas de
lecture complète) : les scripts de recherche/mining `kickoff`
(`kickoff_gemini_cascade.py`, `kickoff_group_candidate_dataset.py`,
`kickoff_profiler.py`, `candidates_mining.py`, `goal_candidate_dataset.py`,
`ko_features.py`, `notebook_test_motif_c_formation.py`,
`fin1mt_gemini_cascade.py`, `finmatch_gemini_cascade.py`,
`detect_c_capitaines.py`, `zone_analyzer.py`, `macro_scan.py`,
`recalibrate_colors.py`, `player_color_profiler.py`,
`aggregate_color_reports.py`, `detect_teams_preview.py`), et les
modules tactiques/insights moins directement liés à la détection de
buts (`tactical.py`, `tactical_v2.py`, `intelligence.py`,
`smart_game_ai.py`, `context_engine.py`, `match_story.py`,
`highlights.py`, `highlight_ranker.py`, `highlight_engine.py`,
`player_insights.py`, `player_identity.py`, `player_rating.py`,
`team_color_matching.py`).

### 3.18 Suite de l'audit : `player_identity_memory.py`, `player_entity.py` — interaction subtile avec le correctif 3.14 (`17/09/2026`)

Poursuite de la lecture complète (ligne par ligne) des fichiers
restants, à la demande explicite de l'utilisateur.

**`player_identity_memory.py`, `player_entity.py`** : lus
intégralement, rien trouvé au-delà du point suivant.

**Interaction subtile trouvée avec le correctif de la section `3.14`**
(retrait de l'effacement inconditionnel dans `assign_teams_by_color`) :
`player_reid.py` peut assigner la chaîne spéciale `"gk"` (pas une
vraie équipe `0`/`1`) à `det["team"]` quand la couleur d'un joueur est
trop éloignée des deux centroïdes d'équipe (probable gardien). Or
`TeamColorDetector.update()` (Phase 2) contient `if p.get("team") is
not None: continue` — avec le correctif `3.14` qui ne wipe plus les
classifications existantes, un joueur marqué `"gk"` ne recevrait
**plus jamais** de vraie équipe `0`/`1` de `TeamColorDetector` (bloqué
à tort, alors que ce marqueur n'est pas une vraie équipe).

**Compromis identifié et tranché avec l'utilisateur** : effacer
spécifiquement le marqueur `"gk"` (pas les vraies équipes `0`/`1`)
avant `TeamColorDetector`, pour lui laisser une chance de classifier
correctement ces joueurs. Coût accepté : `player_entity.py` (Phase
`1b`, détection de gardien via `event.get("team")=="gk"`) perd ce
signal spécifique pour les joueurs concernés — mais **pas totalement**,
puisque `player_reid.py` pose aussi un champ séparé `is_goalkeeper=True`
(indépendant de `"team"`) et que `player_entity.py` a une Phase `1c`
de repli (détection par dominance d'événements défensifs). Décision :
privilégier une vraie équipe `0`/`1` (utile pour `GAP_EQUIPES`, stats
par équipe, tactique) plutôt que le marqueur `"gk"`, en acceptant
l'affaiblissement (pas la suppression) de cette détection de gardien
spécifique.

### 3.19 Clôture de l'audit complet des `45` fichiers `analysis/` (`17/09/2026`)

Poursuite et achèvement de la lecture intégrale ligne par ligne des
fichiers restants, à la demande explicite de l'utilisateur de
continuer jusqu'au bout.

**`2` bugs supplémentaires trouvés et corrigés** :

1. **`zone_analyzer.py`** : `_ball_pos()` ne reconnaissait que le
   format `x`/`y`/`cx`/`cy` (celui produit localement par
   `_read_zone_frames`) — mais `compute_dynamic_window()` reçoit
   aussi les `frames_data` du **pipeline principal** (format
   `ball_tuple_to_dict()` : `{"center": [cx, cy]}`, aucune clé `x`/`cx`
   séparée). Conséquence : `_ball_pos()` retournait **toujours**
   `None` sur ces données, faisant croire que le ballon était
   perpétuellement absent — la fenêtre dynamique tombait
   systématiquement sur l'heuristique de repli, jamais sur la vraie
   logique de position. Corrigé en reconnaissant les deux formats.

2. **`highlight_ranker.py`** : tri sur `x.get("t", 0)` alors que
   **tous** les events du pipeline utilisent la clé `"time"` (confirmé
   partout ailleurs ce soir) — un no-op silencieux, l'ordre
   d'insertion original était conservé au lieu d'un vrai tri
   chronologique. Corrigé.

**Fichiers confirmés comme code mort ou recherche pure** (vérifié :
aucune référence, directe ou indirecte via un autre module actif,
dans `pipeline.py`) : `macro_scan.py` (POC explicitement documenté
comme non intégré), `kickoff_group_candidate_dataset.py`,
`candidates_mining.py`, `goal_candidate_dataset.py`, `ko_features.py`,
`kickoff_profiler.py`, `detect_c_capitaines.py`,
`aggregate_color_reports.py`, `recalibrate_colors.py`,
`player_color_profiler.py`, `highlights.py`, `highlight_engine.py`,
`player_insights.py`, `smart_game_ai.py`. Non lus en profondeur — le
temps a été concentré sur le code réellement actif.

**Fichiers actifs lus intégralement, rien trouvé au-delà de ce qui
précède** : `team_color_matching.py`, `zone_analyzer.py` (au-delà du
bug ci-dessus), `detect_teams_preview.py`, `intelligence.py`
(confirmé utilisé par `events.py` lui-même, via `compute_danger()`),
`context_engine.py`, `tactical.py`, `tactical_v2.py`,
`player_identity.py` (contient déjà, en interne, la même correction
`falsy-zero` que celle appliquée ce soir dans `player_reid.py` —
confirme que ce principe était déjà établi ailleurs dans le code),
`player_rating.py`, `match_story.py`, `kickoff_gemini_cascade.py`
(`938` lignes), `fin1mt_gemini_cascade.py`, `finmatch_gemini_cascade.py`
— ces `3` derniers d'une qualité exceptionnelle, chaque seuil et
garde-fou documenté avec sa justification empirique précise (match et
timestamp exact ayant motivé chaque correctif historique).

**Bilan final de l'audit du dossier `analysis/`** : sur `45` fichiers,
`~20` fichiers actifs en production ont été lus intégralement,
`~14` fichiers confirmés comme code mort/recherche ont été
déprioritisés à raison, et l'ensemble des correctifs appliqués ce
soir (`9` au total sur ce dossier, en comptant ceux des sections
`3.15`-`3.19`) ont été régénérés dans les `2` artefacts de test
complets (`Kaggle` et `Colab`).

### 3.20 Run complet de `20` min sur Andrimont : découverte critique et désactivation du fallback (`18/09/2026`)

Résultats du run continu de `20` min (`t=300-1500s`) lancé après le
correctif de la section `3.14`.

**Le correctif de classification d'équipe confirmé pleinement
fonctionnel** : couverture finale `93,2%` (contre `~18-23%` mesuré
avant), `"PlayerReID avait une équipe MAIS effacée : 0 (0,0%)"` —
plus aucune perte, exactement l'objectif visé.

**Découverte critique sur le fallback `GOALZONE_SPEED_FALLBACK`** : le
fallback confirme un "but" à des intervalles quasi exactement
réguliers de `~150s` (`302,3 / 452,2 / 602,1 / 752,0 / 901,8 / 1051,7
/ 1201,6 / 1351,5s`, écarts de `149,8` à `149,9s`) — **exactement la
durée du cooldown `goal_cd` corrigé en section `3.14`**. Ça démontre
qu'un signal **persistant** sur ce match satisfait en continu les `3`
conditions du fallback, se redéclenchant systématiquement dès que le
cooldown expire — pas des faux positifs isolés, un **générateur
périodique**. Aucun des `8` timestamps produits ne correspond à un
vrai but (buts réels : `382, 1831, 2554, 2938, 3581, 5046s`). Sur un
match complet (`~90min`), ce motif produirait environ `36` faux buts.

**Cause exacte non identifiée** — probablement une zone récurrente
près d'un bord de cadre avec mouvement régulier (remplaçants qui
s'échauffent, ligne de touche active, etc.), mais non investiguée en
détail.

**Décision, avec l'utilisateur** : désactiver le fallback plutôt que
de le retoucher une fois de plus à l'aveugle — pour éviter de
reproduire le risque déjà identifié plus tôt dans la soirée (section
`3.13`, "usine à heuristiques qui reconnaît les arrêts de jeu, pas les
buts"). Implémenté via un flag explicite
`FALLBACK_GOALZONE_SPEED_ACTIF = False` en tête de `events.py` : les
`3` conditions continuent d'être évaluées et logguées (nouveau message
`"⚠️ FALLBACK SUPPRIMÉ (désactivé)"`, distinct du rejet normal),
permettant de continuer à observer où/quand ce déclenchement aurait
lieu sans créer de faux événements ni poser de cooldown. Réactivable
en un seul changement (`True`) une fois la vraie cause du cycle de
`150s` comprise et corrigée.

**État du chemin standard** : la seule confirmation via le chemin
`events_standard` (tir lié, `t=1051,7s`) ne correspond pas non plus à
un but réel connu — probablement un faux positif également, non
encore investigué en détail (hors périmètre de cette section).

**Prochaine étape suggérée** : investiguer la cause du cycle de
`150s` (vérifier la position `x,y` exacte à chacun des `8`
déclenchements — si elle est similaire à chaque fois, ça confirmerait
une zone fixe du cadre plutôt qu'un événement de jeu variable).

### 3.21 Confirmation de la désactivation, et vraie fréquence révélée (`18/09/2026`)

Résultats du run relancé avec `FALLBACK_GOALZONE_SPEED_ACTIF = False`
actif.

**Désactivation confirmée fonctionnelle** : aucune occurrence de
`"✅ goal CONFIRMÉ (GOALZONE_SPEED_FALLBACK)"` dans tout le log —
uniquement le nouveau message `"⚠️ FALLBACK SUPPRIMÉ (désactivé)"`,
apparu `397` fois sur `~20` min.

**Découverte majeure, avec la bonne méthode (proposée par
l'utilisateur)** : le cycle de `150s` observé en section `3.20` était
un **artefact du cooldown**, pas la vraie fréquence. Sans cooldown
pour la masquer, les `397` déclenchements bruts se regroupent, par
proximité temporelle (`>5s` d'écart = nouvel épisode), en **`38`
épisodes distincts** — un calcul naïf (`20min / 397`) aurait donné
`~3s` entre déclenchements, largement trompeur : le vrai motif est
`38` épisodes de durées variables (`0` à `63,4s`), pas un cycle
régulier.

**Détail intéressant confirmant le mécanisme du cooldown partagé** :
un trou de `163s` entre l'épisode `29` (fin `1037,7s`) et l'épisode
`30` (début `1200,9s`) — correspond précisément à la confirmation du
chemin standard à `t=1051,0s` (`_bt/tick_shot_candidate`,
`cooldown=1123 appels`). Confirme que `state["goal_cd"]` étant
partagé entre les `2` chemins, une confirmation sur l'un bloque
l'évaluation même de `GOALZONE` (pas seulement la confirmation du
fallback) pendant tout le cooldown de l'autre.

**Log enrichi** (`events.py`) suite à la demande de l'utilisateur :
le message `"FALLBACK SUPPRIMÉ"` n'incluait ni position ni valeur de
vitesse — ajout de `x`/`y` (position brute et normalisée), vitesse
maximale sur `3s` et instantanée, et détail explicite de chacune des
`3` conditions (`vitesse_ok`, `trajectoire_ok` avec `ball_is_real`/
`lost_frames_ok`, `pas_contradictoire` avec `gk_blocking`/
`shot_blocked_cd`). Objectif : permettre de déterminer si les `38`
épisodes se produisent dans une **zone fixe du cadre** (position
quasi identique), une **ligne/bord récurrent** (position qui bouge
mais dans une même zone), ou si les positions sont **complètement
dispersées** (auquel cas le signal lui-même, pas une zone
particulière, serait en cause).

**Non encore fait** : le run précédent ne contient pas ces nouvelles
données de position (log généré avant cet enrichissement) — à
relancer avec le notebook mis à jour pour obtenir `x`/`y` sur les
`38` épisodes et trancher entre les `3` hypothèses.

### 3.18 Suite de l'audit `analysis/` — fichiers restants (`17/09/2026`)

Poursuite de la lecture complète (ligne par ligne, même rigueur que
`events.py`) sur les fichiers listés comme "non couverts" en section
`3.17`, à la demande explicite de l'utilisateur, plutôt que de
considérer la recherche par motif comme suffisante.

**`2` bugs trouvés et corrigés** :

1. **`zone_analyzer.py` — le plus significatif de cette série** :
   `_ball_pos()` (unique point d'accès à la position du ballon dans
   tout le fichier) cherchait `ball.get("x")`/`ball.get("cx")` — mais
   la structure réelle du ballon dans notre pipeline (`main.py`
   `ball_tuple_to_dict()`, et partout ailleurs) n'a **jamais** de clés
   `"x"`/`"cx"` séparées, uniquement `"center"` (liste `[cx,cy]`).
   Cette fonction retournait donc **toujours `None`** sur de vraies
   données, rendant tout ce module **silencieusement non fonctionnel**
   depuis le début — aucune erreur, juste `0` candidat trouvé à chaque
   run (confirmé : `pipeline.py` capture les exceptions autour de cet
   appel, mais n'en aurait vu aucune ici, juste un résultat vide).
   Corrigé pour vérifier `"center"` en priorité, cohérent avec
   `terminal_events.py`/`goal_posthoc.py`. **Trouvé au passage** :
   l'intégration `BallTracker` dans ce même fichier (`_read_zone_frames`)
   appelait `update()` avec une signature complètement différente de
   la vraie (`update(ball, small, frame_id)` au lieu de
   `update(detected_balls, frame_w, frame_h, timestamp=None)`) —
   échouait systématiquement, avalé silencieusement par un
   `except: pass`. Retiré plutôt que réparé (une vraie correction
   nécessiterait de capturer la largeur/hauteur du ballon, absente du
   code actuel — risque de bug non testé trop élevé à ce stade de la
   session). La détection YOLO brute (déjà fonctionnelle une fois
   `_ball_pos()` corrigée) reste opérationnelle sans cet ajout. Ajouté
   aux `2` artefacts complets (`Kaggle`/`Colab`) — ce module pourrait
   désormais capter des buts (comme notre penalty) qu'il ratait
   silencieusement jusqu'ici.

2. **`highlight_ranker.py`** : triait les highlights via
   `x.get("t", 0)` — mais la clé réellement utilisée sur tous les
   events du codebase (confirmé par recherche systématique) est
   `"time"`, jamais `"t"`. Ce tri ne faisait donc rien (tout retombait
   sur le défaut `0`), les highlights restant dans leur ordre
   d'entrée plutôt que chronologique. Corrigé. Profité de l'occasion
   pour sécuriser aussi `e["type"]` (accès direct, risque de
   `KeyError`) en `e.get("type", "")`, cohérent avec le reste du
   codebase.

**Effet de bord découvert et corrigé, lié au correctif de la section
`3.14`** : `player_entity.py` vérifie `if team == "gk"` sur les events
pour détecter un gardien (`"gk"` étant un marqueur spécial de
`PlayerReID._infer_team()`, pas une vraie équipe `0`/`1`). Or notre
correctif de `main.py` (retrait de l'effacement inconditionnel avant
`TeamColorDetector`) avait un effet de bord non anticipé :
`TeamColorDetector.update()` fait `if p.get("team") is not None:
continue` — un joueur marqué `"gk"` (non `None`) ne recevrait donc
**plus jamais** de vraie équipe `0`/`1`. Corrigé (avec accord explicite
de l'utilisateur après lui avoir présenté le compromis) : `main.py`
efface désormais spécifiquement le marqueur `"gk"` (mais jamais les
vraies valeurs `0`/`1`) avant `TeamColorDetector`, lui laissant une
chance de classer ces joueurs correctement. Le champ séparé
`is_goalkeeper` (déjà posé par `player_reid.py`, indépendant de
`"team"`) et la Phase `1c` de repli de `player_entity.py` (détection
par dominance d'événements défensifs) continuent d'assurer une
détection de gardien, même si ce chemin spécifique s'affaiblit
légèrement.

**Fichiers vérifiés sans bug trouvé** (lecture complète ou recherche
ciblée selon la taille) : `player_identity_memory.py`, `player_entity.py`
(au-delà du point ci-dessus), `team_color_matching.py`,
`detect_teams_preview.py` (fonctionnalité d'aperçu séparée, hors
pipeline principal — signale un risque de calibration précoce
similaire à `calibration.py` mais moins prioritaire, non corrigé),
`context_engine.py`, `player_rating.py`, `intelligence.py`,
`tactical.py`, `player_identity.py`, `match_story.py`,
`ko_features.py` (division par zéro suspectée, vérifiée protégée par
une garde précédente), `kickoff_gemini_cascade.py`/
`finmatch_gemini_cascade.py` (recherche ciblée uniquement, taille et
validation empirique déjà documentée rendant une relecture complète
moins prioritaire à ce stade).

**Point signalé mais non corrigé, distinct de celui de la section
`3.17`** : `kickoff_detector.py`, `find_match_end()` — seuils de
vitesse ballon calibrés empiriquement, sensibles au `frame_skip` (déjà
documenté section `3.17`), confirmé non traité pour les mêmes raisons
(revalidation empirique nécessaire, non faisable depuis cette
session).

### 3.22 Clôture définitive : `goalzone_speed_fallback` et la piste "détection basée sur le ballon seul" (`18/09/2026`)

Suite à l'enrichissement du log (section `3.21`) avec positions `x`/`y`,
un nouveau run a produit `122` déclenchements avec position, regroupés
en épisodes distincts. **`6` captures d'écran** aux timestamps
représentatifs du cœur du cluster spatial (`x_norm<0,12`,
`y_norm≈0,35-0,40`) ont permis une vérification visuelle directe,
plutôt qu'une hypothèse statistique seule.

**Confirmé visuellement** : cette zone correspond au **but gauche du
terrain**, visible comme une structure fixe du plan caméra (large,
fixe, filmant tout le terrain sans suivre le jeu) sur `3` des `6`
captures — dont une montrant un gardien au sol en pleine action juste
à côté du but.

**Conclusion technique, définitive** :

Le fallback mesurait en réalité *« activité rapide dans la zone du
but »*, un événement **beaucoup plus fréquent** que *« le ballon a
effectivement franchi la ligne de but »* — arrêt du gardien,
dégagement, corner, coup de pied de but, ou toute action offensive
normale produisent le même signal. C'est le **même défaut conceptuel**
que la tentative précédente avec la stabilisation de position (section
`3.11`) : *ballon près du but + comportement particulier du ballon ≠
but*. Les deux signaux essayés ce soir (stabilisation, vitesse)
décrivent une zone d'activité normale du jeu, pas un événement
spécifique de franchissement de ligne.

**Bilan des pistes explorées ce soir sur ce sujet** :

| Piste | Conclusion |
|---|---|
| `goalzone_speed_fallback` | ❌ Abandonnée définitivement |
| Seuil de vitesse | ❌ Ne résout pas le problème conceptuel |
| Zone fixe du but | ❌ Zone d'activité normale, pas un signal de but |
| Stabilisation du ballon | ❌ Même défaut conceptuel |
| Détection basée uniquement sur le ballon | ❌ Insuffisante pour confirmer un but |
| Détection du passage de ligne | ✅ Signal réellement spécifique — à rechercher |
| Reprise du jeu après but (`GAP_EQUIPES`, section `3.13`) | ✅ Alternative indirecte à conserver |

**Décision explicite** : ne plus modifier `goalzone_speed_fallback`
(`FALLBACK_GOALZONE_SPEED_ACTIF = False` reste en l'état, code de
diagnostic conservé mais plus touché). Ce fallback a rempli son rôle
— pas comme détecteur de production, mais comme **outil de diagnostic**
ayant permis de démontrer, avec des preuves mesurées puis vérifiées
visuellement, pourquoi une détection basée uniquement sur la
trajectoire/vitesse du ballon est structurellement insuffisante pour
confirmer un but. Les pistes prometteuses pour la suite
(`GAP_EQUIPES`, détection de franchissement de ligne) restent
documentées en section `3.13` et ci-dessus, à reprendre lors d'une
prochaine session.

### 3.23 Clôture de `GAP_EQUIPES` comme signal discriminant, après campagne élargie (`18/09/2026`)

Suite au signal prometteur observé sur `C` seul (section `3.21`),
campagne élargie proposée par l'utilisateur : **aucune modification du
code `GAP_EQUIPES`**, uniquement élargissement de l'échantillon testé
via `notebook_test_penalty.py` — les `5` autres vrais buts connus de
ce match (`D`-`H`) et `3` faux positifs supplémentaires (`I`-`K`,
épisodes déjà identifiés section `3.20`), en plus des `3` cas déjà
connus (`A`, `B`, `C`) et de `KO1`.

**Résultat, sur `~2400` points `GAP_EQUIPES` extraits au total** :

| Groupe | Motif présent | Taux |
|---|---|---|
| `6` vrais buts (`C`,`D`,`E`,`F`,`G`,`H`) | `C`, `H` uniquement | `2/6` (`33%`) |
| `5` faux positifs (`A`,`B`,`I`,`J`,`K`) | `A` uniquement | `1/5` (`20%`) |
| `KO1` (coup d'envoi réel) | Absent | — |

**L'écart entre les deux groupes est trop faible pour constituer un
signal discriminant fiable.** Particulièrement révélateur : `KO1`
lui-même (un vrai coup d'envoi, resserrement collectif quasi certain
en théorie) ne produit pas le motif attendu — fragilise fortement
l'hypothèse que le calcul actuel (`bord le plus avancé de chaque
équipe l'une vers l'autre`) mesure correctement un regroupement
collectif, indépendamment de la question du but.

**Distinction importante à conserver** (formulation de l'utilisateur,
plus précise que ce qu'une clôture rapide aurait dit) : ce test ne
démontre **pas** que *« les joueurs ne se regroupent jamais après un
but »* — il démontre que *« le `GAP_EQUIPES` actuellement calculé ne
fournit pas une signature suffisamment fiable et reproductible pour
distinguer un but d'une situation normale »*. La nuance compte : elle
laisse la porte ouverte à un calcul différent (vraie calibration
géométrique du terrain, méthode de mesure du regroupement différente)
sans rouvrir la question du principe lui-même.

**Décision explicite** : ne pas chercher un nouveau seuil, une autre
durée de compaction, ou une autre fenêtre temporelle sur ce même
calcul — risque de reproduire exactement le cycle déjà vécu avec le
fallback vitesse (`3.11`→`3.22`). `GAP_EQUIPES` reste dans le code
comme diagnostic pur (aucun effet sur la décision), non retouché.

**Point positif à retenir** : le correctif de classification d'équipe
(section `3.14`) n'était pas inutile malgré ce résultat — il a permis
de tester `GAP_EQUIPES` dans des conditions enfin valides (`~93%` de
couverture, contre `~20%` avant) et de conclure que le problème vient
de la valeur discriminante du signal lui-même, pas d'un manque de
données d'équipe.

**Bilan des pistes de confirmation de but, à ce stade** :

| Piste | État |
|---|---|
| Ballon proche du but + vitesse (`goalzone_speed_fallback`) | ❌ Abandonnée |
| Ballon proche du but + stabilisation | ❌ Abandonnée |
| `GAP_EQUIPES` | ❌ Abandonnée comme signal discriminant |
| Classification d'équipe | ✅ Corrigée, `~93%` de couverture |
| Détection directe du franchissement de ligne | 🔎 À explorer si nécessaire |
| Reprise du jeu après but (autre approche que `GAP_EQUIPES`) | 🔎 Piste indirecte, non fermée dans son principe |

**Méthode à reproduire pour la prochaine piste** : petit échantillon
ciblé → comparaison systématique vrais/faux événements → décision
seulement après, pas avant. C'est cette méthode, appliquée
rigoureusement par l'utilisateur sur `GAP_EQUIPES` et le fallback
vitesse, qui a permis d'éviter de valider à tort deux signaux non
fiables ce soir.

### 3.24 Clôture de `KICKOFF_SCORE_REEL` — même défaut conceptuel, troisième fois (`18/09/2026`)

Suite à la clôture de `GAP_EQUIPES` (section `3.23`), nouvelle
tentative : réutiliser directement `_score_frame()` de
`kickoff_detector.py` (déjà validé empiriquement en production pour
`KO1`/`KO2`/`Fin1MT`/`FinMatch`) plutôt que réinventer une métrique.
Nouveau diagnostic `[KICKOFF_SCORE_REEL]` ajouté à `events.py`,
purement diagnostique, réutilisant ce score composite à `7` facteurs
sans modification.

**Bug trouvé et corrigé en cours de route** : la fenêtre de
surveillance (`_kickoff_watch_until`) ne s'ouvrait que dans la
branche `"REJETÉ"`, pas dans la nouvelle branche `"SUPPRIMÉ
(désactivé)"` (introduite section `3.20` lors de la désactivation du
fallback vitesse) — conséquence concrète : à `t=382,1s` (le vrai
penalty), aucune donnée `KICKOFF_SCORE_REEL` n'était produite, alors
que c'était le cas le plus important à tester. Corrigé : la fenêtre
s'ouvre maintenant dans les deux branches.

**Première analyse (piégeuse)** : en ne regardant que la valeur des
sous-composantes au point de score total maximal, `players_in_circle`
semblait discriminer parfaitement (`1` pour les faux positifs/`KO1`,
`≥3` pour les vrais buts sur un petit échantillon initial). **Corrigé
immédiatement** : c'était un artefact d'échantillonnage — un seul
point choisi arbitrairement (le score max global), pas le
comportement réel de cette composante sur toute la fenêtre.

**Analyse rigoureuse, sur la séquence complète (`30s`) et l'échantillon
élargi (`5` vrais buts disponibles à ce stade)** :

```
players_in_circle (max sur 30s) :
  Vrais buts (5)         : [7, 5, 7, 7, 13]
  Faux positifs/KO1 (3)  : [7, 7, 7]
```

**Chevauchement total** — aucune séparation. Le score global
(`score_max`) montre le même chevauchement (`faux : 11,08-11,10` vs
`vrais : 9,38-12,38`). Confirmé sur `KO1` en particulier (`11,08`,
identique au faux positif `A` avec lequel il partage la fenêtre
temporelle).

**Conclusion** : même défaut conceptuel que `GAP_EQUIPES` et le
fallback vitesse. Le score capture bien *« il y a une reprise de jeu
avec regroupement au centre »*, mais rien ne distingue une reprise
après but d'une reprise après un autre arrêt de jeu nécessitant une
remise en jeu au centre. La sophistication du signal réutilisé
(`7` facteurs, déjà validé en production pour un usage différent) n'a
pas suffi à transférer un pouvoir discriminant absent du principe
lui-même pour ce cas d'usage précis.

**Décision explicite** : fermer cette piste aussi, sans chercher un
autre seuil ou une autre composante sur ce même signal — troisième
fois ce soir que ce même schéma se reproduit (`ballon+vitesse`,
`GAP_EQUIPES`, `KICKOFF_SCORE_REEL`), confirmant qu'un signal basé
uniquement sur *"y a-t-il une reprise/un regroupement après ce
candidat"*, quelle que soit sa sophistication technique, ne suffit
pas à lui seul à distinguer un but d'un arrêt de jeu ordinaire.
`[KICKOFF_SCORE_REEL]` reste dans le code comme diagnostic pur,
non retouché davantage.

**Bilan mis à jour des pistes de confirmation de but** :

| Piste | État |
|---|---|
| Ballon proche du but + vitesse | ❌ Abandonnée |
| Ballon proche du but + stabilisation | ❌ Abandonnée |
| `GAP_EQUIPES` (bord extrême de chaque équipe) | ❌ Abandonnée |
| `KICKOFF_SCORE_REEL` (score composite réutilisé) | ❌ Abandonnée |
| Classification d'équipe | ✅ Corrigée, `~93%` de couverture |
| Détection directe du franchissement de ligne | 🔎 Seule piste de confirmation directe non explorée |
| Reprise du jeu après but, comme signal isolé | ❌ Testé sous `3` formes différentes ce soir, jamais discriminant |

**Point de réflexion pour la suite** : les `3` tentatives de ce soir
partagent toutes la même limite — elles cherchent un signal
**après-coup** (reprise, regroupement) sans jamais vérifier
**directement** l'événement lui-même (le ballon a-t-il franchi la
ligne). La détection de franchissement de ligne reste la seule piste
non explorée qui attaquerait le problème à la racine plutôt que par
un signal indirect — mais nécessiterait une vraie calibration
géométrique du terrain, déjà identifiée comme fragile sur ce type de
vidéo (`"[GOAL_BOX] Aucune structure détectée"`).

### 3.25 Reclassification : de « confirmateurs de but » à « features de dangerosité » (`18/09/2026`)

Proposition de l'utilisateur, acceptée : les `3` signaux fermés comme
confirmateurs de but (sections `3.22`-`3.24`) échouent tous pour la
même raison — ils détectent *« une reprise/un regroupement après ce
candidat »*, sans distinguer un but d'un autre arrêt de jeu. Mais
cette absence de spécificité, qui les invalide comme confirmateurs,
ne les invalide **pas** pour un usage différent, avec une exigence
plus faible : détecter une **action dangereuse** (à faire remonter
pour les highlights), pas confirmer précisément un but.

**Distinction posée** :
- Confirmation de but → exige une forte spécificité (« un but a-t-il
  été marqué ? »)
- Détection de dangerosité → exige seulement une pertinence
  contextuelle (« se passe-t-il quelque chose d'intéressant dans
  cette zone ? »). Un arrêt spectaculaire du gardien, un corner
  dangereux, un `2ème` ballon disputé dans la surface sont tous des
  candidats highlights légitimes, pas des faux positifs à éliminer.

**Reclassification** (rien supprimé du code, tout reste comme
diagnostic disponible) :

| Signal | Comme confirmateur de but | Comme feature de dangerosité |
|---|---|---|
| Ballon proche du but `+` vitesse (`goalzone_speed_fallback`) | ❌ Abandonné | 🔎 Candidat |
| `GAP_EQUIPES` | ❌ Abandonné | 🔎 Candidat (congestion d'équipe) |
| `KICKOFF_SCORE_REEL` | ❌ Abandonné | 🔎 Candidat (densité/regroupement) |

**Architecture envisagée** (proposée par l'utilisateur) :

```
ballon → zone dangereuse + vitesse/trajectoire + densité joueurs
       + GAP_EQUIPES + activité offensive
       → SCORE DANGEROSITÉ → Gemini (confirmation/classification)
```

**Premier prototype implémenté dans la foulée** (voir plus bas) :
`_diag_score_dangerosite()`, purement diagnostique, combine les
signaux déjà calculés ce soir (zone de but, vitesse du ballon,
densité de joueurs à proximité, écart d'équipes) en un score composite
simple. Non validé — première ébauche à affiner/mesurer lors d'une
prochaine session, avec la même méthode rigoureuse que pour les
tentatives précédentes (échantillon ciblé → comparaison → décision).

### 3.26 Faux négatif structurel du déclencheur `is_shot_z`, identifié sur `D` (`18/09/2026`)

Premier test de `[SCORE_DANGEROSITE]` sur les `11` fenêtres : aucune
donnée produite pour `D` (`t=1831s`) et `H` (`t=5046s`) — le score
n'est calculé que si `is_shot_z` est vrai, et cette condition n'est
jamais satisfaite dans ces `2` fenêtres.

**Cause identifiée précisément sur `D`** : `is_shot_zone()` (repli
générique football, sans calibration `shot_zones`) exige `y ≥
frame_h*0,15` (`~162px` sur `1080`). Le ballon à `D` est à `y=124-143px`
— sous le seuil de `19` à `38px` seulement. Le `x` est bien dans la
zone ; c'est uniquement la contrainte `y` qui échoue. Le but apparaît
simplement plus haut dans le cadre à cet instant du match (variation
de cadrage/zoom caméra), et le seuil `y` fixe et générique ne s'y
adapte pas.

**Décision méthodologique explicite (proposée par l'utilisateur)** :
ne **pas** élargir ce seuil maintenant, pour la même raison qui a
guidé toute la soirée — ne pas modifier un seuil au milieu d'une
campagne simplement parce qu'un cas échappe au détecteur, sans mesure
rigoureuse préalable. `D` est conservé tel quel comme **cas
documenté** : un vrai mécanisme de faux négatif structurel du
déclencheur (`is_shot_z`), pas du score lui-même (`SCORE_DANGEROSITE`
n'est même pas atteint dans ce cas, donc pas en cause).

**Information utile retenue pour une future session** : la zone
offensive générique (`is_shot_zone()`, repli sans calibration) n'est
pas robuste aux variations verticales de cadrage. Si l'analyse
complète des `11` runs (ci-dessous) montre que le score de
dangerosité est par ailleurs pertinent mais que plusieurs occasions
sont perdues uniquement à cause de `y`, ce sera une justification
expérimentale (pas une supposition) pour travailler sur la
calibration de zone dans une prochaine session.

### 3.27 Pause stratégique et découverte de la cause racine : sélection du ballon parmi plusieurs candidats `YOLO` (`19/09/2026`)

**Remise en question du système, proposée par l'utilisateur**, après
`3` échecs consécutifs de signaux construits par-dessus la position du
ballon (fallback vitesse, `GAP_EQUIPES`, `KICKOFF_SCORE_REEL`) :
plutôt que de continuer à empiler des heuristiques, valider d'abord si
la couche de perception elle-même (position du ballon) est fiable,
avant de construire quoi que ce soit dessus.

**Test décisif** : sur `3` fenêtres de jeu neutre (`L`, `M`, `N`,
section `3.26`), comparaison de la position médiane des **joueurs**
contre celle du **ballon** :

| Fenêtre | Médiane joueurs | Médiane ballon |
|---|---|---|
| `L` | `0,302` | `0,097` |
| `M` | `0,551` | `0,104` |
| `N` | `0,697` | `0,077` |

**Les joueurs montrent des médianes variées et cohérentes avec du jeu
normal réparti sur le terrain. Le ballon reste figé près de
`x_norm≈0,08-0,10` dans les `3` cas, indépendamment de ce qui se
passe réellement.** Ça élimine l'hypothèse d'un effet de perspective
caméra (qui affecterait joueurs et ballon de la même façon) et pointe
vers un problème spécifique à la détection/sélection du ballon.

**Cause racine trouvée, avec certitude au niveau du code** (`main.py`,
boucle de traitement par lots, et `vision/detector.py`, méthode
`detect()`) :

```python
elif cls == detector.ball_cls:
    yolo_ball = {...}
```

Cette ligne **écrase silencieusement** `yolo_ball` à **chaque**
itération où `YOLO` détecte un objet classé "ballon" — sans jamais
comparer confiance ni position. **Si `YOLO` détecte `2` objets ou
plus classés "ballon" dans une même frame, seul le dernier de la
liste (ordre interne de `YOLO`, arbitraire) survit.** Aucune étape
plus loin (`_detect_ball()`, puis `select_best_ball()` dans
`ball_tracker.py`, pourtant conçue avec une vraie logique de
sélection par distance) ne voit jamais les autres candidats — un
seul, potentiellement faux, leur est transmis avant même qu'ils
puissent intervenir.

**Hypothèse cohérente avec toutes les observations de la soirée** :
un objet fixe près du bord gauche du cadre (peut-être un élément du
but, un panneau, un équipement au sol) est régulièrement classé
"ballon" par `YOLO` avec une confiance suffisante, et gagne l'écrasement
final dès qu'il apparaît après le vrai ballon dans la liste de
détections — expliquant potentiellement plusieurs échecs de ce soir
(le motif `is_shot_z` quasi permanent sur les fenêtres neutres,
section `3.26`, en particulier).

**Corrigé, dans les `2` endroits concernés** (`main.py` et
`vision/detector.py`) : tous les candidats "ballon" de la frame sont
maintenant collectés, puis le meilleur est sélectionné — le plus
proche de la dernière position connue si disponible (même principe
que `select_best_ball()`), sinon le plus confiant. Diagnostic
`[MULTI_BALL]` ajouté (conditionné à `DEBUG`) pour confirmer
empiriquement, au prochain test, la fréquence réelle de ce scénario
multi-candidats.

**Non encore fait** : validation empirique du correctif (est-ce que
`[MULTI_BALL]` se déclenche effectivement sur `L`/`M`/`N`, et est-ce
que la médiane du ballon redevient cohérente avec celle des joueurs
après correction). Confirmation visuelle (capture d'écran de l'objet
suspect) proposée mais pas encore réalisée — devenue moins prioritaire
si le correctif de sélection résout le problème par la mesure.

**Point de méthode à retenir** : cette découverte n'aurait pas été
faite en continuant à raffiner les scores de dangerosité/but — elle
n'est apparue qu'en redescendant délibérément d'un niveau
d'abstraction, pour vérifier la donnée fondamentale (position du
ballon) plutôt que les heuristiques construites dessus. Confirme la
justesse de la pause stratégique proposée par l'utilisateur.

### 3.28 Vérification visuelle sur `9` frames — confirmation directe, pas seulement statistique (`19/09/2026`)

Extraction des `9` frames exactes (numéros de frame précis, pas des
secondes approximatives) correspondant aux positions `x_norm≈0,08-0,10`
identifiées sur `L`/`M`/`N`. Examen visuel direct de chacune.

**Résultat : `6/9` frames montrent une position "ballon" manifestement
incorrecte** — le vrai ballon est parfois clairement visible ailleurs
dans la même image. Les fausses positions se concentrent sur `2`
catégories récurrentes :
1. **Fragments de joueurs coupés par le bord gauche du cadre** (frames `3`, `6`, `7`)
2. **Structure du but gauche** — poteaux, filet, personnel présent à
   proximité (arbitre assistant) (frames `4`, `5`, `8`)

**Chaîne causale complète, maintenant établie par preuve visuelle** :
```
fausse détection ballon (poteau/joueur hors-cadre)
        ↓
x_norm ≈ 0,08–0,10
        ↓
is_shot_z = True (couverture ~90-100% mesurée sur les fenêtres neutres)
        ↓
toutes les heuristiques en aval se déclenchent sur du bruit
```

**Correction importante de formulation** (proposée par l'utilisateur,
avec l'appui d'une relecture externe) : les `9` images démontrent que
`ball_pos` est faux dans plusieurs frames — elles ne permettent **pas**
encore de dire à quelle étape précise l'erreur se produit. La chaîne
complète est :
```
YOLO (candidats) → sélection du candidat → BallTracker → ball_pos → heuristiques
```
Trois causes distinctes restent possibles, avec une action corrective
différente pour chacune :
- **A — YOLO faux** : un seul candidat, déjà erroné à la source → travail sur le modèle
- **B — mauvaise sélection** : plusieurs candidats existaient, le bon
  n'a pas été choisi → correctif d'arbitrage
- **C — dérapage du tracker** : la sélection était correcte, mais la
  position finale de `BallTracker` diverge quand même → correctif de
  suivi temporel (`Kalman`/`TTL`/association)

**Mise en garde explicite retenue** : ne pas présupposer qu'un filtre
de plausibilité géométrique (ratio largeur/hauteur, masque spatial)
résoudra le problème avant de l'avoir mesuré — un poteau vu sous un
certain angle peut très bien produire une bbox petite, carrée, à
confiance élevée (auquel cas `aspect_ratio` ne suffit pas), et un
masque sur le bord gauche risquerait d'aveugler le système sur une
vraie action de jeu se déroulant légitimement le long de cette zone
(ligne de touche, corner).

**Décision de gel explicite** : `SCORE_DANGEROSITE`, `GAP_EQUIPES`,
`KICKOFF_SCORE_REEL`, `is_shot_z` et toute heuristique événementielle
ne sont **plus modifiés ni réévalués** tant que la fiabilité de
`ball_pos` n'est pas établie. Le vrai progrès de cette session est
d'être descendu d'un niveau : de *« quelle heuristique inventer ? »*
à *« `ball_pos` correspond-il réellement au ballon visible dans
l'image ? »*.

**Audit quantitatif lancé** (`[PERCEPTION_BALL]`, `main.py`,
inconditionnel — pas seulement si `2+` candidats) : logue pour
**chaque** candidat "ballon" brut détecté par `YOLO` — confiance, `x`,
`y`, `x_norm`, largeur/hauteur bbox, surface, ratio largeur/hauteur,
distance au bord gauche, et marque explicitement lequel a été retenu.
À croiser par timestamp avec le log `[BALL]` déjà existant (sortie
**après** le tracker) pour distinguer le cas `C` si la sélection était
bonne mais que la position finale diverge quand même.

**Échantillon contrôlé** : notebook dédié
(`notebook_audit_perception.py`), réduit aux `3` fenêtres `L`/`M`/`N`
(élargies à `60s` chacune) déjà prouvées problématiques par
vérification visuelle — pas besoin de refaire les `11` autres fenêtres
de la campagne `GAP_EQUIPES`, indépendante de cet audit de perception.

**Feuille de route actée pour la prochaine session** :
1. Gel total de la couche événementielle (déjà en vigueur)
2. Exécution de l'audit `[PERCEPTION_BALL]` sur l'échantillon contrôlé
3. Classification stricte de chaque erreur en `A`/`B`/`C`
4. Décision factuelle seulement après mesure : filtre logique,
   correctif d'arbitrage/tracking, ou réentraînement en dernier
   recours — jamais présupposée à l'avance

### 3.29 Audit quantitatif `[HSV_DIAG]` et `[PERCEPTION_BALL]` : deux résultats, l'un négatif, l'un positif (`20/09/2026`)

**Bug de timestamp trouvé et corrigé** dans `[PERCEPTION_BALL]` :
utilisait `analyzed/fps` (compteur post-`frame_skip`) au lieu de
`frame_id/fps` (numéro de frame absolu) — aucun timestamp en commun
avec `[BALL]`, croisement impossible. Corrigé.

**Résultat du croisement `[PERCEPTION_BALL]`↔`[BALL]` sur `L`/`M`/`N`** :
`YOLO` détecte `0` candidat dans `97%` des frames (`1309/1349`), jamais
`2+` (élimine définitivement l'hypothèse `B`, mauvaise sélection
parmi plusieurs candidats `YOLO`). Sur ces `1286` frames à `0`
candidat, `[BALL]` affiche pourtant `interp=False` (position "réelle")
dans `98,4%` des cas — la position vient donc du **repli `HSV`**
(niveau `2` de `_detect_ball()`), pas de `YOLO`.

**Contexte utilisateur déterminant** : le système a été initialement
réglé et validé sur des matchs filmés à la caméra **manuelle, tenue à
la barrière** (ballon proche, gros à l'écran) — la vidéo `VEO`
actuelle (plan large fixe, terrain entier visible) est un contexte
différent. Mesure : aire médiane des candidats `HSV` `~54px²` (`~8px`
de diamètre), contre des bornes `30-3000px²` implicitement calibrées
pour la caméra manuelle.

**Réutilisation du système `camera_type` existant** (`low_side`/
`low_side_zoom`/`high_side`, `build_camera_profile()`), pas de flag
séparé — `camera_type` propagé comme paramètre normal
(`process_video()`→`Detector`→`BallHSVDetector`), défaut `low_side`
préserve strictement le comportement existant. **Dépendance
circulaire identifiée et documentée** : `camera_type` dépend des
positions ballon (via `frames_data`), qui dépendent elles-mêmes de ce
détecteur — non résolue ce soir, `camera_type` doit être fourni
explicitement pour l'instant (pas encore alimenté automatiquement par
`build_camera_profile()`).

**Test `1` (RÉSULTAT NÉGATIF, utile)** : bornes de taille resserrées
pour `high_side` (`15-200px²` au lieu de `30-3000px²`, inchangé pour
`low_side`/`low_side_zoom`). Médiane `ball_x_norm` sur `L`/`M`/`N`
quasi inchangée (`0,087`/`0,099`/`0,078` contre `0,097`/`0,104`/`0,077`
avant) — **la taille seule ne discrimine pas les deux populations**.
Abandon de cette piste, sans chercher d'autre seuil intermédiaire
(règle établie plus tôt ce soir : ne pas re-optimiser une variable qui
ne discrimine manifestement pas).

**Instrumentation `[HSV_DIAG]`** ajoutée à `BallHSVDetector.detect()`
(`main.py`+`vision/detector.py`) : logue, pour **chaque** candidat
valide (pas seulement le gagnant), `score_forme` (intrinsèque :
circularité×taille) et `score_proximite` (bonus, séparément) en plus
du score final — sans modifier la logique elle-même.

**Test `2` (RÉSULTAT POSITIF, décisif)** : sur `1280` frames à `2+`
candidats (`L`/`M`/`N`), le bonus de proximité **change le gagnant**
par rapport au score intrinsèque seul dans `597` cas (`46,6%`). Les
scores intrinsèques des candidats en jeu sont presque toujours très
proches ou à égalité — **aucun candidat ne domine clairement sur le
seul critère `HSV`/forme/taille** ; c'est le bonus de proximité qui
tranche dans cette quasi-égalité.

**Chaîne causale retenue, plus solide que toute hypothèse
précédente** :
```
candidats HSV faibles et ambigus (aucun net gagnant intrinsèque)
        ↓
bonus de proximité (favorise la continuité avec last_pos)
        ↓
départage arbitraire, mais COHÉRENT frame après frame
        ↓
last_pos se stabilise sur un candidat (potentiellement faux)
        ↓
le nouveau ball_pos influence à son tour le last_pos futur
        ↓
boucle auto-entretenue vers x_norm≈0,08-0,10
```
Correspond precisement au motif observe depuis le debut de cette
investigation (position bloquee, persistante sur toute une fenetre).

**Décision explicite** : ne pas toucher à la plage `HSV` (déjà testée,
non concluante) ni au reste de la logique. Seul test suivant prévu :
faire varier **uniquement** le poids du bonus de proximité (`0,5`
actuel → `0,25` → `0`), sur les mêmes `L`/`M`/`N`, en mesurant
médiane `ball_x_norm`, taux de changement de gagnant, et persistance
du point suspect — rien d'autre modifié.

**Non encore fait** : lancement effectif du notebook d'audit — prêt,
pas encore exécuté à la fin de cette session.

### 3.30 Test A/B/C du poids de proximité, vérification visuelle, et audit statistique (`20/09/2026`)

**Test A/B/C exécuté** : `poids=0,5` (déjà mesuré section `3.29`),
`poids=0,25`, `poids=0`, sur les mêmes `L`/`M`/`N`. Médiane
`ball_x_norm` :

| Fenêtre | poids `0,5` | poids `0,25` | poids `0` |
|---|---|---|---|
| `L` | `0,087` | `0,111` | `0,218` |
| `M` | `0,099` | `0,124` | `0,166` |
| `N` | `0,078` | `0,087` | `0,113` |

**Tendance monotone** (pas de stabilisation nette) — la médiane
s'éloigne progressivement de `~0,08-0,10` à mesure que le poids
diminue, sans qu'on puisse dire si c'est vers la vérité ou vers un
autre point arbitraire, sans vérification visuelle.

**Vérification visuelle sur `6` frames** choisies aux écarts maximaux
entre poids `0,25` et poids `0` (numéros de frame exacts, pas de
secondes approximatives) :

| # | Vrai ballon | poids `0,25` | poids `0` | Gagnant |
|---|---|---|---|---|
| `1` (`L`) | `~0,54` | `0,004` ❌ | `0,383` ❌ | Aucun (les deux faux) |
| `2` (`M-1`) | `~0,48` | `0,943` ❌ | `0,442` ✅ | `poids=0` |
| `3` (`M-2`) | `~0,57` | `0,123` ❌ | `0,470` ✅ | `poids=0` |
| `4` (`M-3`) | `~0,37` | `0,072` ❌ | `0,401` ✅ | `poids=0` |
| `5` (`N-1`) | `~0,44` | `0,030` ❌ | `0,386` ✅ | `poids=0` |
| `6` (`N-2`) | non visible | `0,026` (isolé) | `0,379` (zone d'action) | `poids=0` (contextuel) |

`poids=0` gagne sur `5/6` cas. **Frame `1` est importante** : même
`poids=0` échoue — confirme qu'il existe au moins `2` problèmes
distincts (sélection intrinsèque parfois mauvaise, ET proximité
parfois nuisible), pas une cause unique.

**Preuve mécanique sur `M-2`/`M-3`** (candidats `HSV` bruts extraits
avant/après proximité) :

`M-2` : vrai ballon (`x_norm=0,470`) a le **meilleur score intrinsèque**
(`0,18`) — un faux candidat (`0,123`, intrinsèque `0,16`) reçoit
`+0,15` de proximité et gagne (`0,32` final). Sans proximité, le vrai
ballon gagne correctement.

`M-3` : vrai ballon (`x_norm=0,400`) a un score intrinsèque très
supérieur (`0,25` vs `0,14-0,16` pour les autres) — deux faux
candidats, boostés par la proximité (`+0,10`/`+0,12`), le dépassent
de justesse (`0,26` vs `0,25`).

**`M-1` : contre-exemple utile** — écart intrinsèque déjà large
(`0,19` vs second), le bonus (`+0,03`) ne renverse rien. Confirme que
la proximité n'est pas *toujours* nuisible — elle peut coexister sans
dégât quand l'écart intrinsèque est déjà tranché.

**Audit statistique complet** (`1281` frames à `2+` candidats,
poids `0,25`, `L`/`M`/`N`) :

- **`370` frames (`28,9%`) montrent un renversement** du meilleur
  candidat intrinsèque après application de la proximité.
- `intrinsic_gap` (écart entre `1er` et `2e` intrinsèque) **quand
  renversé** : médiane `0,030`, `p75=0,050`, max `0,160`.
- `intrinsic_gap` **quand PAS renversé** : médiane `0,090`,
  `p75=0,160`, max `0,420`.
- **Séparation nette entre les deux distributions** — les
  renversements se concentrent sur les petits écarts, cohérent avec
  le mécanisme mesuré sur `M-2`/`M-3`.
- Bonus de proximité maximum observé : `0,24`.

**Ce qui est démontré** : la proximité intervient principalement
quand les candidats intrinsèques sont déjà proches, et peut alors
inverser le classement — mécanisme confirmé statistiquement, pas
seulement sur `2` exemples isolés.

**Ce qui N'est PAS encore démontré** : la proportion des `370`
renversements qui sont réellement nuisibles (candidat faux gagne) vs
bénéfiques (stabilisation légitime d'une ambiguïté réelle) — seuls
`2` cas ont une vérification visuelle à ce stade (`M-2`, `M-3`, tous
deux nuisibles). Pas de généralisation possible sans échantillon plus
large.

**Décision explicite : ne pas figer de seuil maintenant**
(`0,05`/`0,09` restent des hypothèses de calibration, pas des valeurs
validées). Architecture cible retenue, à valider avant implémentation :

```
score intrinsèque → classement des candidats
        ↓
intrinsèque suffisamment meilleur (gap > seuil)
        ↓ oui                              ↓ non
conserver le meilleur intrinsèque    utiliser la proximité (départage)
```

C'est un changement de principe (proximité = tie-breaker conditionnel,
pas un terme additif systématique), pas juste un ajustement du poids
existant.

**Prochaine étape actée** : échantillon visuel ciblé par tranche de
`intrinsic_gap`, PAS uniquement sur les renversements — regarder
renversement ET non-renversement dans chaque tranche, pour distinguer
renversement bénéfique / nuisible / indéterminable :

| `intrinsic_gap` | Intérêt |
|---|---|
| `0-0,03` | renversements très ambigus |
| `0,03-0,05` | zone où `≥75%` des renversements se trouvent |
| `0,05-0,09` | zone candidate pour un futur seuil |
| `0,09-0,12` | vérifier si la proximité reste utile ici |
| `>0,12` | contrôle : renversements rares mais potentiellement importants |

**Point méthodologique important à retenir pour la suite** : chaque
gros fichier de résultats uploadé écrase le précédent au même nom de
fichier (`New_Text_Document.txt`) — sans copie immédiate vers un
espace persistant, les données brutes sont perdues dès l'upload
suivant. Toute donnée volumineuse à ré-analyser plus tard doit être
sauvegardée dès réception.

### 3.31 Vérification visuelle par tranche de `intrinsic_gap`, et implémentation de la proximité conditionnelle (`20/09/2026`)

**`10` frames vérifiées visuellement** (`2` par tranche de
`intrinsic_gap` : `0-0,03`, `0,03-0,05`, `0,05-0,09`, `0,09-0,12`,
`>0,12` — un cas renversé et un non-renversé par tranche) :

| Tranche | Cas | Verdict |
|---|---|---|
| `0-0,03` | Renversé | Les deux candidats faux |
| `0-0,03` | Non-renversé | Approximatif |
| `0,03-0,05` | Renversé | Ballon non visible, indéterminable |
| `0,03-0,05` | Non-renversé | Approximatif |
| **`0,05-0,09`** | **Renversé** | **Meilleur intrinsèque proche de la vérité, gagnant final s'en éloigne — renversement nuisible net** |
| `0,05-0,09` | Non-renversé | Ballon non visible, indéterminable |
| `0,09-0,12` | Renversé | Les deux candidats faux |
| `0,09-0,12` | Non-renversé | Ballon non visible, indéterminable |
| `>0,12` | Renversé | Meilleur intrinsèque plus proche, renversement probablement nuisible |
| `>0,12` | Non-renversé | **Faux même sans renversement** (le candidat unique ne correspond pas au vrai ballon) |

**`40%` de l'échantillon (`4/10`) non exploitable** — ballon non
identifiable visuellement dans ces plans larges à cette résolution
(limite réelle de la méthode de vérification, pas seulement du
tracking).

**Sur les `6` frames interprétables** : `1` renversement nuisible net
(`gap=0,05-0,09`, confirmant mécaniquement la zone de seuil
candidate), `1` renversement probablement nuisible (`gap>0,12`), et
**`3` cas où le problème est independant de la proximité** — la
génération/sélection intrinsèque des candidats échoue par
elle-même (frames `1`, `7`, `10`), y compris à `gap` élevé, réputé
"fiable" (frame `10`).

**Conclusion retenue** : les `2` problèmes coexistent bien
(sélection intrinsèque ET proximité), confirmant l'architecture à
`2` niveaux discutée plus tôt. La proximité conditionnelle reste
justifiée pour le sous-problème qu'elle cause spécifiquement, mais ne
résoudra pas le problème de génération de candidats identifié en
parallèle — à traiter séparément dans une session future.

**Implémenté** : proximité conditionnelle dans `BallHSVDetector` —
seuil `intrinsic_gap` initial fixé à `0,08` (milieu de la zone
candidate `0,05-0,09` où le renversement nuisible a été confirmé
visuellement), **explicitement marqué comme première estimation, pas
une valeur validée de façon exhaustive**. Si l'écart entre le meilleur
et le second candidat intrinsèque dépasse ce seuil, la proximité
n'est plus appliquée au meilleur candidat (il gagne quel que soit son
bonus de proximité) — sinon, comportement inchangé (proximité agit
comme avant, poids `0,5` par défaut).

### 3.32 Validation du mécanisme de protection, `seuil=0,08` (`20/09/2026`)

**Test exécuté sur `L`/`M`/`N`** (`proximite_poids=0,5` par défaut,
`seuil_gap_protection=0,08`) :

| Fenêtre | Avec protection | Sans protection |
|---|---|---|
| `L` | `0,106` | `0,087` |
| `M` | `0,126` | `0,099` |
| `N` | `0,084` | `0,078` |

**Mécanisme validé au niveau code** : croisement `[HSV_DIAG]`
(candidats bruts) ↔ `[BALL]` (sortie réelle du tracker) sur les
frames à `gap>0,08` — **`478/483` (`99%`)** montrent que le ballon
final correspond bien au meilleur candidat intrinsèque, confirmant
que la protection s'applique exactement comme conçue (les `5` écarts
résiduels probablement dus à l'arrondi de timestamp ou au lissage
`Kalman`, pas un bug de logique).

**Amélioration modeste, cohérente avec le mécanisme** : la protection
ne s'applique qu'à `~40%` des frames (`483/1195`, celles à
`gap>0,08`) — les `60%` restantes gardent le comportement inchangé.
Décalage de médiane limité, pas spectaculaire comme avec `poids=0`
complet, mais c'est le compromis chirurgical visé (ne bloquer la
proximité que quand elle serait vraiment nuisible).

**Prochain test prévu** : `seuil=0,05` (plus bas, protège plus de
frames) pour voir si l'amélioration se renforce sans perdre la
fonction de stabilisation légitime démontrée sur `M-1` (section
`3.30`).

### 3.33 Confirmation `seuil=0,05` et clôture de session — bilan de la chaîne complète (`20/09/2026`)

**Test `seuil=0,05` exécuté** : mécanisme validé (`647/655`, `99%` de
correspondance meilleur-intrinsèque ↔ sortie réelle), tendance
monotone confirmée avec `seuil=0,08` :

| | Sans protection | `seuil=0,08` | `seuil=0,05` |
|---|---|---|---|
| Frames protégées | `0` | `483` | `655` |
| Médiane `L`/`M`/`N` | `0,087`/`0,099`/`0,078` | `0,106`/`0,126`/`0,084` | `0,131`/`0,134`/`0,088` |

**Décision de clôture** : le mécanisme de protection est validé aux
deux seuils testés (`99%` de fiabilité mécanique, effet monotone et
proportionnel). Le choix du seuil final de production reste ouvert —
bénéficierait d'un échantillon de vérification visuelle plus large
avant calibration définitive. Session arrêtée à ce stade, prête pour
reprise ultérieure sur ce point précis.

**Bilan de la chaîne complète investiguée ce soir**, pour mémoire :

```
YOLO (0 candidat dans 97% des frames sur ce match)
        ↓
Repli HSV (plage de couleur tres large, quasi tout objet clair/rond)
        ↓
Sélection parmi candidats (jamais 2+ simultanés sur cet échantillon -
        hypothèse "mauvaise sélection multi-candidats" réfutée)
        ↓
Score intrinsèque (forme+taille) — souvent correct en tête, confirmé
        visuellement sur plusieurs cas (M-2, M-3)
        ↓
Bonus de proximité à last_pos — démontré capable de renverser ce
        classement intrinsèque correct dans les cas à faible écart
        (28,9% des frames à 2+ candidats, mécanisme confirmé
        statistiquement sur 1281 frames ET visuellement sur plusieurs
        exemples ciblés)
        ↓
ball_pos final — biais persistant mesuré (x_norm≈0,08-0,10, near
        structure de but gauche), confirmé visuellement sur 9 images
        (frames extraites à numéro exact, pas approximation)
        ↓
[Toutes les heuristiques event/but testées plus tôt dans la
 session — fallback vitesse, GAP_EQUIPES, KICKOFF_SCORE_REEL,
 SCORE_DANGEROSITE — héritaient de ce biais en amont]
```

**Point le plus important de cette session** : la cause racine des
échecs répétés des heuristiques événementielles n'était pas leur
logique propre, mais la donnée `ball_pos` qui les alimentait,
identifiée seulement en redescendant délibérément d'un niveau
d'abstraction (« quelle heuristique inventer ? » → « `ball_pos`
correspond-il au ballon réel ? »), avec une méthode rigoureuse
(mesure avant modification, vérification visuelle systématique,
distinction claire entre corrélation et causalité) appliquée de bout
en bout.

**Correctifs livrés, tous avec comportement par défaut STRICTEMENT
inchangé** (activables explicitement uniquement) :
- Sélection multi-candidats ballon (`main.py`, `vision/detector.py`)
- `camera_type` réutilisant le système existant (`low_side`/
  `low_side_zoom`/`high_side`), bornes `HSV` adaptées pour `high_side`
- `proximite_poids` configurable
- `seuil_gap_protection` (protection conditionnelle de la proximité)

**Ouvert pour une prochaine session** : calibration fine du seuil de
protection sur un échantillon visuel plus large, connexion de
`camera_type` au calcul automatique existant (`build_camera_profile()`,
dépendance circulaire non résolue), et retour aux heuristiques
événementielles (`GAP_EQUIPES` notamment) une fois `ball_pos`
suffisamment fiabilisé pour re-tester dans de bonnes conditions.

### 3.34 Cause racine de la génération de candidats trouvée : `search_radius` exclut structurellement la vérité (`20/09/2026`)

**Vérification visuelle sur `6` frames de la zone `P`** (jamais
explorée, `camera_type=high_side` + `seuil_gap_protection=0,08`
actifs) : `4/5` interprétables montrent une détection nettement
fausse, confirmant que le problème de génération de candidats
persiste malgré les correctifs de sélection déjà validés.

**Cause précise identifiée** : croisement des `4` cas faux avec les
données `[HSV_DIAG]` (`last_pos`, `search_radius`) — **`4/4`
montrent la vraie position du ballon HORS de la zone de recherche**
recadrée autour de `last_pos` :

| Instant | `last_pos_x` | Zone recherchée | Vrai ballon |
|---|---|---|---|
| `P-1` | `72-74` | `[-178, 324]` | `446` ❌ |
| `P-2` | `9-106` | `[-241, 356]` | `538` ❌ |
| `P-4` | `58-59` | `[-192, 309]` | `352` ❌ |
| `P-6` | `20-122` | `[-230, 372]` | `429` ❌ |

**Ce n'est pas un problème de couleur/forme/taille** — `HSV` ne
regarde structurellement jamais où se trouve le ballon une fois
`last_pos` dérivé. Boucle auto-entretenue : `last_pos` dérive → la
recherche suivante l'exclut → impossible de le retrouver → `last_pos`
reste bloqué, brisée seulement par les rares succès `YOLO` directs
(`~3%` des frames).

**Mécanisme de reprise implémenté**, proposé par l'utilisateur puis
raffiné : recherche globale (image entière, pas de `search_radius`)
déclenchée **périodiquement** (toutes les `N` frames,
`intervalle_recherche_globale`), pas seulement après des échecs
consécutifs — raffinement nécessaire car la recherche locale trouve
presque toujours *un* candidat (rarement rien du tout), juste souvent
le mauvais, rendant un déclencheur "N échecs consécutifs" quasi
inopérant en pratique.

**Détail important ajouté** : pendant une recherche globale forcée,
le bonus de proximité est aussi désactivé pour cet appel — sinon le
score continuerait de favoriser un candidat proche de l'ancien
`last_pos` (connu comme non fiable), reproduisant le même piège à
l'intérieur même du mécanisme censé le casser.

**Implémenté, défaut `None` = désactivé, comportement strictement
inchangé** : `intervalle_recherche_globale` propagé
`process_video()` → `Detector` → `BallHSVDetector`, même schéma que
tous les paramètres précédents. Log `[RECHERCHE_GLOBALE]` ajouté pour
confirmer empiriquement le déclenchement en test.

**Notebook de test prêt** : `3` runs ciblés uniquement sur `P`
(notre seul cas à vérité terrain confirmée sur zone neuve) pour
comparaison directe sur le même échantillon — sans mécanisme
(référence déjà mesurée), `intervalle_recherche_globale=10`, et `=5`
(plus fréquent). `camera_type=high_side` + `seuil_gap_protection=0,08`
inchangés dans les `3` cas.

### 3.35 Bug d'échelle trouvé — potentiellement la vraie cause racine de toute la session (`20/09/2026`)

**Test du mécanisme de reprise sur `P`** : amélioration partielle
seulement (`2/4` cas corrigés avec `intervalle=5`, `0/4` avec
`intervalle=10`). En creusant pourquoi `P-1`/`P-2` résistaient même
avec recherche globale forcée, découverte d'une incohérence majeure :
lors d'un déclenchement `[RECHERCHE_GLOBALE]` à `t=1200,80s`, le
**vrai gagnant** (meilleur `score_forme`, `0,550`, protection
`seuil=0,08` confirmant qu'il aurait dû être retenu) est à
`x_norm=0,761` — mais le `last_pos` de l'appel **suivant** affiche
`x_norm=0,180`, totalement incohérent.

**Cause exacte identifiée** : `main.py`, calcul de `last_pos_small` :
```python
lx, ly = detector._last_ball_pos
last_pos_small = (lx / scale_x, ly / scale_y)
```
Cette division suppose `detector._last_ball_pos` à l'échelle
**originale** (`1920`). Or `_detect_ball()` est appelé avec
`frame_small` (échelle `960`, `PROCESS_W`) et fixe
`_last_ball_pos` en interne **avant** que `rescale_detections()` ne
s'exécute — `_last_ball_pos` est donc **toujours** à l'échelle `960`,
que ce soit via `YOLO` (pré-rescale) ou `HSV`. Cette division
erronée divise une valeur déjà à l'échelle `960` par `scale_x`
(`~2`) **une seconde fois**, poussant `last_pos` vers zéro/le bord
gauche à **chaque mise à jour réussie**.

**Portée potentielle** : ce bug est probablement présent depuis
l'origine du projet (pas introduit ce soir) et pourrait expliquer,
à lui seul, une part importante de la dérive systématique vers
`x_norm≈0,08-0,10` observée sur toute la session — bien plus
fondamental que tout ce qu'on avait diagnostiqué jusqu'ici (bonus de
proximité, `search_radius`, plage `HSV`), qui ne faisaient
qu'amplifier ou exploiter ce biais mécanique sous-jacent.

**Corrigé** : suppression de la division, `last_pos_small` utilise
directement `detector._last_ball_pos` (déjà à la bonne échelle).
Vérifié qu'aucun autre point du code ne mélange les échelles de la
même façon (tous les usages de `_last_ball_pos` dans `_detect_ball()`
opèrent cohéremment sur `frame_small`).

**Test isolé préparé** : `P` avec **uniquement** ce correctif
(`intervalle_recherche_globale=None`, mécanisme de reprise
désactivé), pour mesurer l'effet de ce bug précis seul, sans le
combiner à d'autres mécanismes — méthode de mesure isolée déjà
appliquée toute la soirée.

### 3.36 Validation du correctif d'échelle, et découverte de sa complémentarité avec le mécanisme de reprise (`20/09/2026`)

**Test isolé exécuté sur `P`** (correctif d'échelle seul, sans
mécanisme de reprise) :

| | Médiane `x_norm` | Distribution |
|---|---|---|
| Avant (bug présent) | `0,098` | concentrée, bord gauche |
| Après (bug corrigé) | **`0,294`** | large (`p25=0,171`, `p75=0,625`, max`=0,996`) |

**Vérifié précisément sur les `4` instants à vérité terrain** :

| Instant | Vrai ballon | Avant correctif | Après correctif |
|---|---|---|---|
| `P-1` | `0,465` | `0,154` | `0,154` (inchangé) |
| `P-2` | `0,561` | `0,031` | **`0,662`** (quasi exact) |
| `P-4` | `0,366` | `0,123` | **`0,348`** (quasi exact) |
| `P-6` | `0,446` | `0,203` | `0,203` (inchangé) |

**`2/4` cas résolus quasi parfaitement** — confirmation forte que le
bug d'échelle est un vrai facteur causal, pas une coïncidence.

**Investigation de `P-1`/`P-6` (pourquoi inchangés)** : `HSV_DIAG` à
`t=1200,13s` (tout début de la fenêtre `P`) montre `last_pos=(118,222)`
déjà établi, avec tous les candidats disponibles proches de
`cx=27-175` — le vrai ballon (`~446` en échelle `960`) est **hors de
la zone de recherche** (`[118-250,118+250]=[-132,368]`, `446>368`).
Le correctif d'échelle fonctionne correctement ici (propagation
`last_pos` cohérente, `145/960=0,151` correspond exactement à la
sortie observée) — mais si le **tout premier ancrage** de la fenêtre
tombe déjà au mauvais endroit (pour une raison indépendante du bug
d'échelle), le `search_radius` limité maintient le système bloqué
localement, faute de mécanisme d'évasion.

**Conclusion : les deux correctifs de la session sont complémentaires,
pas redondants** :
- Le correctif d'échelle empêche la dérive **artificielle** vers zéro
  (bug de calcul pur)
- Le mécanisme de reprise (`intervalle_recherche_globale`) permet de
  s'**échapper** quand l'ancrage de départ est mauvais, quelle qu'en
  soit la cause

`P-1`/`P-6` sont précisément les cas que le mécanisme de reprise
visait à résoudre — non testé en combinaison avec le correctif
d'échelle jusqu'ici (le test précédent du mécanisme de reprise,
section 3.34-3.35, tournait encore avec le bug d'échelle actif).

**Prochain test préparé** : les deux correctifs combinés (échelle
corrigée + `intervalle_recherche_globale=5` ou `10`) sur `P`,
hypothèse que la combinaison résout aussi `P-1`/`P-6`.

### 3.37 Résultat mitigé du test combiné — la recherche globale peut détruire un suivi local déjà bon (`20/09/2026`)

**Test combiné exécuté** (échelle corrigée + `intervalle_recherche_globale=10` puis `=5`) :

| Instant | Vrai | Échelle seule | `+intervalle=10` | `+intervalle=5` |
|---|---|---|---|---|
| `P-1` | `0,465` | `0,154` | `0,154` | `0,154` |
| `P-2` | `0,561` | `0,662` ✅ | `0,031` ❌ | `0,662` ✅ |
| `P-4` | `0,366` | `0,348` ✅ | **`0,974`** ❌❌ | **`0,987`** ❌❌ |
| `P-6` | `0,446` | `0,203` | `0,203` | `0,298` |

**`P-4`, quasi parfait avec le correctif d'échelle seul, devient
franchement faux dans les deux variantes de recherche globale** — la
recherche globale **détruit un suivi local qui fonctionnait déjà**,
dans les deux fréquences testées (pas un simple problème
d'intervalle trop court/long).

**Diagnostic proposé par l'utilisateur, avec une mise en garde
importante** : comparer le score global au score local ne doit **pas**
se faire avec le score final (qui inclut la proximité côté local mais
pas côté global, rendant la comparaison biaisée) — uniquement sur le
score **intrinsèque seul** des deux côtés. Distinction à établir :
- **Cas A** : global réellement moins bon intrinsèquement → la garde
  proposée (ne remplacer que si le global est meilleur) suffit
- **Cas B** : global intrinsèquement meilleur mais visuellement faux
  → problème plus profond dans la recherche globale elle-même, une
  garde simple ne suffira pas

**Diagnostic `[GLOBAL_VS_LOCAL]` implémenté** dans
`BallHSVDetector.detect()` : lors d'une recherche globale forcée,
filtre les candidats déjà trouvés sur l'image entière à ceux tombant
dans la fenêtre locale (`last_pos ± search_radius`), compare le
meilleur score intrinsèque de chaque groupe — **sans relancer de
détection séparée** (tous les candidats globaux incluent déjà ce
qu'aurait trouvé une recherche locale sur la même frame). Pur
diagnostic, aucune décision affectée, aucun autre paramètre modifié
(`seuil_gap_protection=0,08` inchangé, mêmes fenêtres `P`) pour
préserver l'attribution causale du résultat.

**Notebook de test prêt**, identique au précédent sauf ce diagnostic
ajouté — objectif : déterminer si `P-4` correspond au cas `A` ou `B`
avant toute modification de code.

### 3.38 Cas `B` confirmé, puis correction d'hypothèse : gauche ≠ droite (`20/09/2026`)

**Diagnostic `[GLOBAL_VS_LOCAL]` exécuté** : cas `B` confirmé sans
ambiguïté — sur les recherches globales menant à `P-4`, le candidat
global (`x_norm≈0,94-0,99`) obtient un score intrinsèque
**systématiquement supérieur** au candidat local (ex. `0,307` vs
`0,171`, `0,391` vs `0,213`), et ce sur de nombreuses frames
consécutives. **La garde proposée (« ne remplacer que si global >
local ») aurait accepté le mauvais candidat à chaque fois** — le
problème n'est pas l'arbitrage global/local, il est en amont, dans le
score intrinsèque lui-même qui favorise une zone du bord droit.

**Vérification visuelle sur `5` frames consécutives** (`x_norm`
`0,945`-`0,995`, juste avant `P-4`) : **hypothèse initiale réfutée**.
Contrairement au but gauche (structure spatialement fixe), **les `5`
images montrent des joueurs différents à chaque fois** — aucun but,
poteau ou panneau fixe visible à droite dans cet échantillon.

**Correction d'hypothèse actée, avec le niveau de certitude approprié
pour chaque affirmation** :

| Zone | Observation | Certitude |
|---|---|---|
| Gauche | Structure fixe produit régulièrement de faux candidats | Démontré (section 3.28) |
| Droite | Candidats à `x_norm≈0,95` = joueurs différents à chaque fois | Démontré (`5` captures) |
| Droite | Pas de but/poteau/panneau fixe visible | Démontré sur cet échantillon |
| Droite | Fragments de joueurs / formes atypiques en cause | Hypothèse plausible, non démontrée |
| Droite | Ces formes favorisées par le score intrinsèque `HSV` | À vérifier quantitativement |

**Point méthodologique explicite** : ne pas symétriser artificiellement
le diagnostic. Gauche = source spatialement stable (un objet fixe).
Droite = source mobile en identité mais concentrée spatialement (près
du bord), mécanisme potentiellement différent, pas encore établi.

**Décision** : aucune modification de code à ce stade (ni poids de
proximité, ni intervalle de recherche globale, ni seuils). Prochain
test : extraire la signature géométrique précise (aire, largeur/
hauteur, ratio, circularité) des candidats faux à droite et la
comparer à celle de candidats correspondant réellement au ballon,
pour vérifier si une différence mesurable existe avant d'envisager un
filtre.

### 3.39 Résultat négatif sur la signature géométrique, pivot vers la cohérence temporelle (`20/09/2026`)

**Échantillon élargi** (`2189` gagnants `HSV_DIAG`, `3` fichiers de
test, `359` côté droit `x_norm>0,85` vs `1830` ailleurs) :

| Variable | Côté droit | Ailleurs | Discriminant ? |
|---|---|---|---|
| Ratio `w/h` | médiane `1,08` | médiane `1,08` | ❌ Non — hypothèse initiale (`5` vs `5`) réfutée à l'échelle |
| Circularité | médiane `0,63` | médiane `0,63` | ❌ Non |
| Aire | médiane `103` | médiane `140` | 🔎 Écart réel mais faible |
| `cy` | `62%` dans `[200,250)` | plus dispersé | 🔎 Concentration confirmée, mécanisme inconnu |

**Aucune caractéristique géométrique simple ne discrimine de façon
robuste** — confirme et généralise ce qu'on savait déjà pour la
taille seule (`L`/`M`/`N`, section 3.29).

**Pivot proposé par l'utilisateur** : plutôt que de continuer à
chercher un filtre géométrique par frame, exploiter la **cohérence
temporelle** — un fragment de joueur au bord peut ponctuellement
ressembler au ballon sur une image isolée, mais ne devrait pas
produire une trajectoire physiquement cohérente sur plusieurs frames
consécutives, contrairement au ballon.

**Test qualitatif initial concluant** : sur une séquence de candidats
côté droit, alternance nette entre segments fluides (`15-18px` de
déplacement sur `~0,13s`, progression `X` régulière) et sauts
erratiques (`80-120px` sur le même intervalle) — cohérent avec des
changements d'identité entre objets différents.

**Test quantitatif sur `2179` transitions** : distribution des
vitesses implicites montre un pic net à basse vitesse (`0-200px/s`,
`46%` des cas) mais **pas de séparation bimodale nette** — traîne
continue jusqu'à `7292px/s`. Un simple seuil de vitesse risquerait de
reproduire le même écueil que les filtres géométriques précédents
(coupure arbitraire sur une distribution continue), d'autant que le
vrai ballon peut légitimement se déplacer vite (tir, dégagement).

**Prochaine étape actée** : ne pas se limiter à la vitesse
instantanée, mais évaluer la **cohérence d'accélération** (le
candidat est-il compatible avec la position ET la vitesse déjà
établies sur les frames récentes, pas seulement proche du point
précédent) — plus proche de ce qu'un filtre de `Kalman` ferait, pas
encore exploité au niveau de la sélection de candidat `HSV` (seulement
en aval, dans `BallTracker`).

### 3.40 Clôture de session — bilan des 6 signaux testés, aucun ne discrimine isolément (`20/09/2026`)

**Test de cohérence d'accélération** (`2169` triplets, changement de
vitesse `|Δv|` entre segments consécutifs) : même motif que la
vitesse seule — distribution continue (`0-500px/s` : `35%`, puis
traîne longue jusqu'à `8636px/s`), **pas de séparation bimodale
nette**.

**Bilan complet des signaux testés ce soir pour discriminer le vrai
ballon des faux candidats** :

| Signal | Résultat |
|---|---|
| Taille (aire) | ❌ Ne discrimine pas (`L`/`M`/`N`, section 3.29) |
| Ratio `w/h` | ❌ Ne discrimine pas à l'échelle (section 3.39) |
| Circularité | ❌ Ne discrimine pas (section 3.39) |
| `cy` (position verticale) | 🔎 Concentration observée, mécanisme non expliqué |
| Vitesse instantanée | ❌ Distribution continue, pas de coupure nette |
| Changement de vitesse (accélération) | ❌ Distribution continue, pas de coupure nette |

**Aucune métrique simple, prise isolément sur `1-2` frames, ne
sépare proprement les deux populations** — motif systématique, pas
un échec isolé d'une seule piste.

**Conclusion retenue pour la suite** : le niveau d'analyse pertinent
n'est probablement pas la comparaison ponctuelle candidat-contre-
candidat, mais un **vrai suivi multi-hypothèses sur plusieurs frames
consécutives** (plusieurs pistes maintenues en parallèle, élaguées
selon leur cohérence **soutenue** dans le temps — type filtre de
`Kalman` complet), pas encore exploité au niveau de la sélection de
candidat `HSV` elle-même (seulement en aval, dans `BallTracker`).
Chantier substantiel, identifié comme sujet pour une prochaine
session dédiée plutôt qu'une suite immédiate.

---

## Bilan général de la session (`19-20/09/2026`)

**Point de départ** : plusieurs heuristiques événementielles
(fallback vitesse, `GAP_EQUIPES`, `KICKOFF_SCORE_REEL`,
`SCORE_DANGEROSITE`) échouaient de façon répétée à discriminer les
vrais buts des faux positifs.

**Découverte centrale** : en redescendant d'un niveau d'abstraction
(« quelle heuristique inventer ? » → « `ball_pos` correspond-il
réellement au ballon visible dans l'image ? »), identification que
la cause racine se situait en amont — dans la fiabilité de la
position du ballon elle-même, pas dans la logique événementielle
construite dessus.

**Chaîne de causes racines identifiées et corrigées** (comportement
par défaut strictement préservé à chaque étape) :
1. Sélection multi-candidats ballon (jamais déclenchée sur cet
   échantillon, mais corrigée par principe)
2. `camera_type` réutilisant le système existant, bornes `HSV`
   adaptées pour `high_side`
3. Bonus de proximité rendu configurable, puis conditionnel
   (`seuil_gap_protection`)
4. **Bug d'échelle critique** dans `last_pos_small` (division
   erronée par `scale_x` sur une valeur déjà à l'échelle
   `frame_small`) — probablement la découverte la plus importante de
   la session, validée par mesure directe (médiane `x_norm` passée
   de `0,098` à `0,294` sur le seul correctif)
5. Mécanisme de reprise par recherche globale périodique — utile
   mais avec un effet de bord découvert (peut détruire un suivi
   local déjà bon), non encore résolu

**Investigation finale** : recherche d'une signature discriminante
(géométrique ou temporelle) entre vrai ballon et faux candidats —
`6` signaux testés, aucun ne suffit isolément. Chantier ouvert pour
la prochaine session : suivi multi-hypothèses temporel complet.

**Méthode appliquée de bout en bout** : mesure avant modification,
vérification visuelle systématique (frames extraites à numéro exact,
jamais approximées), distinction rigoureuse entre corrélation et
causalité, documentation continue de chaque hypothèse (confirmée,
réfutée, ou encore ouverte) plutôt que des conclusions prématurées.

### 3.41 Pistes complémentaires évaluées et priorisées pour la prochaine session (`20/09/2026`)

**Proposition externe reçue** (liste de `9` pistes couvrant
masquage polygonal, ombre portée, consistance de taille par
perspective, pose estimation/regard collectif, vecteurs de vitesse
joueurs, crop ciblé pour `Gemini`, validation par arbitre, `Kalman`
inertiel, graphe de possession) — évaluée à la lumière des résultats
de ce soir plutôt qu'adoptée telle quelle.

**Convergence notable** : la piste *"Kalman avec prédiction
d'inertie physique"* correspond exactement à la conclusion déjà
atteinte indépendamment en section 3.40 (suivi multi-hypothèses,
cohérence de trajectoire) — confirmation croisée de cette direction.

**`3` pistes retenues comme prioritaires pour la prochaine session**,
par coût croissant :

1. **Vecteurs de vitesse des joueurs** — aucune nouvelle donnée
   nécessaire (`POSITION_JOUEURS`/`frames_data` déjà disponibles).
   Hypothèse : le vrai ballon se situe près de la zone de convergence
   des vecteurs de déplacement des joueurs de champ.

2. **Graphe de possession (ballon lié au joueur)** — réutilise le
   tracker de joueurs déjà existant. Dès que la distance joueur-ballon
   passe sous un seuil, le suivi du ballon hérite temporairement du
   `tracker_id` du joueur porteur (plus stable, moins sujet aux
   réinitialisations) jusqu'à détection d'une vitesse d'éjection
   (tir/passe). Pourrait réduire directement la dépendance aux `97%`
   d'échecs `YOLO` déjà mesurés, sans toucher au détecteur `HSV`.

3. **`Kalman` multi-hypothèses complet** — chantier déjà identifié en
   section 3.40, plusieurs pistes maintenues en parallèle, élaguées
   selon leur cohérence soutenue dans le temps.

**Pistes notées mais déclassées pour l'instant, avec justification** :
- Masquage polygonal du terrain : n'aurait pas éliminé le cas `P-4`
  (faux positif situé sur le terrain, pas en zone hors-jeu)
- Pose estimation / regard collectif : nouvelle dépendance lourde,
  réservée à une refonte plus large
- Ombre portée / taille par perspective : nécessitent une calibration
  géométrique déjà identifiée comme fragile sur cette vidéo
  (dépendance circulaire de `camera_type`/`build_camera_profile()`,
  section 3.34)
- Crop ciblé pour micro-scan `Gemini`, validation par arbitre :
  concernent une étape différente du pipeline (validation `LLM`, pas
  la détection `HSV` du ballon elle-même), à considérer séparément

**Rien implémenté ce soir** — ces `3` pistes sont documentées comme
point de départ pour la prochaine session, pas encore testées.

### 3.42 Implémentation démarrée : piste `1`, vecteurs de vitesse des joueurs (`20/09/2026`)

**Nouveau diagnostic `[VECTEURS_JOUEURS]`** ajouté à `events.py` :
logue, pour chaque joueur individuellement, sa position (`x,y`) et sa
vitesse (`vx,vy`, calculée par différence avec sa position précédente
de même `id`, stockée dans `state`). Volontairement simple — le
calcul de convergence lui-même sera fait en post-traitement sur les
données brutes, pas inline dans le diagnostic, pour rester vérifiable
indépendamment. Aucune nouvelle dépendance, réutilise les positions
déjà trackées.

**Test préparé** sur `P` : correctif d'échelle actif (déjà validé),
**sans** le mécanisme de reprise (`intervalle_recherche_globale=None`
— effet de bord connu, section 3.37) pour isoler proprement ce
nouveau diagnostic.

### 3.43 Piste `1` testée : résultat mitigé, et confirmation que le ratio `w/h` est noyé dans le bruit de résolution (`20/09/2026`)

**Convergence des vecteurs de vitesse des joueurs calculée** (méthode
des moindres carrés sur droites) aux `4` instants à vérité terrain
connus :

| Instant | Vrai ballon | Convergence joueurs | Écart | `HSV` seul | Écart `HSV` |
|---|---|---|---|---|---|
| `P-1` | `0,465` | — (`0` vecteur, trop tôt) | — | `0,154` | `0,311` |
| `P-2` | `0,561` | `0,451` | `0,110` | `0,662` | `0,101` |
| `P-4` | `0,366` | `0,498` | `0,132` | `0,348` | `0,018` |
| `P-6` | `0,446` | `0,594` | `0,148` | `0,203` | `0,243` |

**Résultat mitigé** : la convergence n'est jamais très précise
(écarts `0,11-0,15`), mais surpasse `HSV` précisément sur `P-6` (le
cas où `HSV` échouait le plus), et est moins bonne sur `P-4` (où
`HSV` était quasi parfait). **Piste retenue comme signal
complémentaire potentiel** (utile quand `HSV` échoue), pas comme
remplacement — échantillon de seulement `3` points exploitables,
prématuré pour conclure plus fermement.

**Découverte importante, en marge** : question posée sur si le
résultat négatif précédent sur le ratio `w/h` (section 3.39,
`2189` échantillons) ne serait pas simplement un problème de
résolution plutôt que l'absence réelle de différence de forme.
**Confirmé directement** : sur une séquence de `8` frames où le
**même ballon**, physiquement quasi immobile (`cx=334` fixe), est
suivi correctement, le ratio mesuré oscille entre `1,31` et `1,42`
**uniquement à cause d'une différence d'`1` pixel** en hauteur
(`12` vs `13px`, `~8%` de variation relative pour un objet de
`~17×12px`).

**Conclusion révisée** : le résultat "le ratio ne discrimine pas à
l'échelle" (section 3.39) reste correct dans sa conclusion pratique
(pas de seuil géométrique fiable exploitable), mais la cause n'est
probablement pas l'absence de vraie différence de forme — c'est que
**la mesure elle-même est trop bruitée à cette résolution** pour
révéler un éventuel signal sous-jacent, même s'il existait. Ferme
définitivement toute piste de seuil géométrique fin (largeur/hauteur/
ratio) à cette résolution — renforce la pertinence des pistes qui ne
dépendent pas de mesures pixel-précises (vecteurs joueurs, cohérence
temporelle, possession).

### 3.44 Implémentation de la piste `2` : secours par possession dans `BallTracker` (`20/09/2026`)

**Nouveau paramètre optionnel `players`** ajouté à
`BallTracker.update()` — défaut `None`, comportement strictement
inchangé. Quand fourni ET qu'aucun candidat ballon normal n'est
retenu (`best is None`), cherche le joueur le plus proche de la
dernière position fiable connue (réutilise la même logique que
`closest_player()` déjà existante) ; si un joueur est trouvé à moins
de `80px`, sa position sert de **candidat de secours**, qui passe
ensuite par **exactement la même validation** (`is_valid_jump()`)
que n'importe quel candidat normal — pas de traitement privilégié,
juste une source de candidat supplémentaire en cas d'échec.

**Propagation vérifiée** : `players=tracked` transmis depuis
`main.py`. Point de vigilance particulier après la découverte du bug
d'échelle (section 3.35) : confirmé que `rescale_detections()`
s'exécute **avant** `tracker.update(players, frame_orig)`, donc
`tracked` (joueurs) et `balls_list` (ballon) sont tous deux à
l'échelle **originale**, cohérents entre eux — pas de nouveau bug
d'échelle introduit.

**Notebook de test prêt** sur `P`, avec échelle corrigée (sans
mécanisme de reprise, effet de bord déjà connu) et ce nouveau secours
par possession activé.

**Correctif important, suite à une remarque de l'utilisateur** : le
simple fait de passer `players=tracked` à `update()` cassait le
principe tenu toute la session (comportement `low_side`/caméra
manuelle strictement inchangé) — le secours activait dès que
`players` était non-vide, indépendamment de `camera_type`. Corrigé
avec un flag explicite `activer_secours_possession` sur `BallTracker`
(défaut `False`), fixé à la construction selon
`camera_type=="high_side"` uniquement — `low_side`/`low_side_zoom`
retrouvent un comportement strictement identique à avant cette piste,
même si `players` est désormais toujours transmis (le flag interne
bloque toute utilisation tant qu'il n'est pas explicitement activé).

**Bug critique trouvé et corrigé : le secours n'était jamais implémenté
dans le fichier réellement testé.** Test lancé (`0` déclenchement de
`[POSSESSION_SECOURS]`) — en investiguant, découverte que
`activer_secours_possession` était absent des `2` fichiers
(`ball_tracker.py` et `main.py`), malgré la présentation précédente —
perdu lors d'une compaction de contexte entre l'implémentation et la
génération finale du notebook, comme déjà rencontré plus tôt ce soir
avec d'autres fichiers.

**Deuxième bug trouvé en reconstruisant** : le secours était placé
**avant** `select_best_ball()`, ne couvrant que le cas où
`detected_balls` est vide — or le repli `HSV`/`YOLO` renvoie presque
toujours *quelque chose* (rarement une liste vide), le vrai cas
d'échec étant le **rejet par `is_valid_jump()`** plus loin dans le
flux. Corrigé : le secours est maintenant placé dans le bloc
`if best is None:` (après le rejet éventuel par `is_valid_jump`),
couvrant les deux cas d'échec réels. Vérifié explicitement par
analyse de position dans le fichier avant de considérer le correctif
complet.

**Fichiers copiés immédiatement vers les sorties après chaque étape**
cette fois, pour éviter une nouvelle perte.

**Test exécuté avec succès** (plus d'erreur) : `1` seul déclenchement
sur `60s`, résultat quasi identique au correctif d'échelle seul
(médiane `0,294`, `P-2`/`P-4` inchangés). **Le mécanisme fonctionne
mais se déclenche trop rarement** — la condition initiale (candidat
rejeté par `is_valid_jump()` ET joueur à moins de `80px`) ne couvre
pas le cas le plus fréquent observé cette nuit : un candidat `HSV`
**accepté** par `is_valid_jump()` mais néanmoins faux (un saut
plausible en distance, mais vers le mauvais objet).

**Condition élargie** : ajout d'une substitution **proactive**, avant
même la validation du candidat normal — si un joueur est nettement
plus proche (`<60px` ET moins de la moitié de la distance du candidat
normal) de la dernière position fiable, sa position remplace le
candidat normal **avant** `is_valid_jump()`, qui s'applique ensuite
identiquement (pas de passe-droit). Reste derrière le même flag
`activer_secours_possession`, donc toujours limité à
`camera_type="high_side"`.

### 3.45 Régression trouvée et corrigée : bug de bornes dans la substitution proactive (`20/09/2026`)

**Test de la substitution proactive** : `282` déclenchements (contre
`1` seul pour la version stricte) — mais **régression nette** sur
nos `4` instants connus : `P-4` (auparavant quasi parfait, `écart
0,018`) chute à `écart 0,466` ; `P-6` produit **`x_norm=1,335`**,
physiquement impossible (`>1,0`, position hors cadre).

**Cause identifiée précisément** : séquence de positions `[BALL]`
marquées `interp=False` (donc "réelles", pas extrapolées) montrant
une progression parfaitement linéaire mais **entièrement hors cadre**
dès le départ (`x=2301` sur une image de `1920px`, progressant
ensuite de façon régulière). Un joueur suivi avec des coordonnées
déjà corrompues (`id` fantôme ou artefact de tracking) a été utilisé
sans aucune vérification de bornes comme candidat de substitution,
contaminant `last_valid_ball` et la vitesse pour les frames
suivantes.

**Cause racine du bug** : le code de substitution ne vérifiait jamais
que la position du joueur candidat était dans les bornes valides de
l'image avant de l'utiliser comme "vraie" position ballon.

**Corrigé** : vérification `0 ≤ pcx < frame_w` et `0 ≤ pcy < frame_h`
ajoutée aux `2` points de substitution (proactive et sur rejet) —
un joueur aux coordonnées hors cadre est désormais silencieusement
ignoré comme candidat, quelle que soit sa distance à la dernière
position fiable.

**Point de méthode confirmé une nouvelle fois** : un mécanisme
d'amélioration trop empressé (ici, une vérification de bornes
manquante, comme précédemment la recherche globale trop fréquente)
peut activement dégrader un suivi qui fonctionnait déjà — la
vigilance sur les effets de bord doit rester systématique, même pour
des ajouts qui semblent, à première vue, purement additifs.

### 3.46 Retrait de la substitution proactive — retour au secours conservateur, clôture de la piste `2` (`20/09/2026`)

**Test avec correctif de bornes** : anomalie hors cadre corrigée
(max `0,999`, plus de valeur `>1,0`), mais **régression de fond
persistante** — `P-4` reste dégradé (`écart 0,466` contre `0,018`
sans cette piste), `P-6` aussi (`écart 0,403`). Confirme que ce
n'était pas qu'un bug de bornes : "plus proche" n'implique pas "plus
correct", défaut de conception, pas d'implémentation.

**Décision** : substitution proactive entièrement retirée. Retour au
secours conservateur uniquement (déclenché sur rejet explicite par
`is_valid_jump()`, section 3.44) — rare (`1` déclenchement sur `60s`
mesuré) mais sans effet destructeur constaté.

**Bilan de la piste `2` (graphe de possession)** : implémentée,
testée en `2` variantes, la version conservatrice fonctionne sans
régression mais a un effet négligeable sur cet échantillon ; la
version élargie améliore la couverture mais dégrade des cas
auparavant corrects. **Conservée en l'état conservateur** dans le
code (`activer_secours_possession`, toujours limité à
`camera_type="high_side"`), sans bénéfice mesurable démontré à ce
stade — piste à reprendre avec une approche différente (peut-être
pondérer plutôt que substituer, ou exiger une cohérence sur plusieurs
frames avant de basculer) plutôt qu'à abandonner.

### 3.47 Correction méthodologique importante : `P-6` identifié comme cas non représentatif (`20/09/2026`)

**Vérification visuelle sur `7` frames consécutives autour de `P-6`**
(demandée suite à un motif de distance ballon-joueur chaotique, sans
approche-minimum-départ contrairement à `P-2`/`P-4`, section `3.46`) :
**`P-6` correspond à une séquence complète de dégagement du gardien**
— ballon ramassé, tenu en main, posé au sol, gardien qui recule pour
prendre son élan. **Ce n'est pas du jeu ouvert**, contrairement à
`P-1`/`P-2`/`P-4`.

**Cohérent rétrospectivement** avec un signal déjà noté sans être
approfondi lors de la toute première vérification visuelle de la
zone `P` (section `3.34`) : `P-6` y était déjà annoté *"non visible"*
— le ballon, immobile dans les mains/aux pieds du gardien, est
naturellement plus difficile à distinguer visuellement qu'un ballon
en mouvement dans le jeu.

**Conséquence sur les conclusions de la soirée — révision nécessaire,
mais partielle** :

| Conclusion utilisant `P-6` | Statut après correction |
|---|---|
| Bug d'échelle (`lx/scale_x`) | ✅ Reste solide — établi par trace de code, confirmé indépendamment sur `P-2`/`P-4` |
| Exclusion par `search_radius` | ✅ Reste solide — fait mécanique/calcul direct, indépendant du contenu de la scène |
| Régression de la substitution proactive | ✅ Reste solide — démontrée sur `P-4` seul (`écart 0,018→0,466`) |
| Vecteurs joueurs "battent" `HSV` sur `P-6` (section `3.43`) | ⚠️ **Affaibli** — pendant un dégagement de gardien, les joueurs se replacent pour la reprise, ils ne convergent pas vers un ballon en jeu ; la "victoire" mesurée pourrait être une coïncidence contextuelle plutôt qu'une validation du principe |
| Toute moyenne/comparaison traitant les `4` points comme équivalents | ⚠️ **Méthodologiquement fragile** — `P-1`/`P-2`/`P-4` sont du jeu ouvert normal, `P-6` est une balle arrêtée ; les mélanger sans distinction n'était pas justifié |

**Décision actée** : `P-2` et `P-4` (tous deux confirmés comme jeu
ouvert représentatif) devraient servir de points de référence
principaux pour toute mesure future de ce type. `P-6` est conservé
dans l'historique pour traçabilité, mais explicitement écarté des
comparaisons "jeu ouvert" futures — à traiter séparément si une
piste sur les phases de balle arrêtée devient pertinente un jour.

**Point de méthode générale à retenir** : un point de "vérité terrain"
choisi pour sa position temporelle (bien espacé des autres, zone
jamais explorée) ne garantit pas qu'il soit représentatif du
phénomène étudié — la vérification visuelle, déjà systématique ce
soir pour confirmer une position, doit aussi servir à confirmer que
le **contexte de jeu** lui-même est comparable aux autres points de
l'échantillon.

### 3.48 Requalification précise des tests affectés, et recadrage `P-2`/`P-4` seuls (`20/09/2026`)

**Tableau de requalification** de chaque test ayant utilisé les `4`
points, établi précisément plutôt que de suspecter l'ensemble de la
session :

| Test / conclusion | Impact de `P-6` | Statut |
|---|---|---|
| Bug `lx / scale_x` | Aucun | ✅ solide |
| `search_radius` exclut le vrai ballon | Aucun | ✅ solide |
| Régression de la recherche globale sur `P-4` | Aucun | ✅ solide |
| Proximité joueur-ballon ponctuelle (`P-2`/`P-4`/`P-6`) | Oui | 🟠 à relire uniquement avec `P-2`/`P-4` |
| Convergence vecteurs joueurs (`P-2`/`P-4`/`P-6`) | Oui, fortement | 🟠 validation affaiblie |
| Persistance des pistes (`P-2`/`P-4`/`P-6`) | Oui pour `P-6` | 🟠 `P-2`/`P-4` restent exploitables |
| Conclusions traitant les `4` points comme homogènes | Oui | ❌ à ne plus considérer comme tel |

**Distance ponctuelle, relue sans `P-6`** : `P-2=86px` et `P-4=86px`
— **exactement identiques**, malgré une piste correcte radicalement
différente (`1` frame vs `21` frames). **Confirme, sur l'échantillon
propre, que la distance instantanée seule n'explique toujours pas
pourquoi `P-2` échoue en persistance** — pas un artefact de `P-6`.

**Évolution temporelle, relue sans `P-6`** (déjà mesurée section
`3.46`, avant la découverte sur `P-6` — les données `P-2`/`P-4`
elles-mêmes restent valides, seule leur comparaison à `P-6` doit être
retirée) :

- **`P-4`** : motif en `V` — approche nette (`416→63px`) → minimum
  stable (`~40px` pendant `~1s`) → éloignement progressif
  (`45→142px`). Cohérent avec un ballon **réceptionné puis reparti**.
- **`P-2`** : éloignement monotone et régulier (`70→137px`), sans
  phase d'approche visible dans la fenêtre observée. Cohérent avec un
  ballon **juste après une passe/un dégagement**, déjà en train de
  s'éloigner.

**Reformulation importante actée** : les deux cas sont en réalité
cohérents avec une vraie dynamique de possession — simplement à des
**phases différentes du cycle** (réception pour `P-4`, départ pour
`P-2`), pas un cas "qui marche" contre un cas "qui échoue". La piste
courte de `P-2` pourrait s'expliquer par une **transition de vitesse
brutale au moment du tir/de la passe**, rendant l'appariement
frame-à-frame naturellement plus difficile à cet instant précis — pas
par une proximité insuffisante au joueur.

**Bilan de la feuille de route (piste `3b`, suivi multi-hypothèses)** :

Déjà démontré :
- Le suivi instantané peut sélectionner de mauvais objets (bug
  d'échelle, `search_radius`)
- La persistance seule ne suffit pas (`P-2` : piste courte mais
  correcte)
- La distance instantanée au joueur seule ne suffit pas (`P-2`/`P-4`
  identiques en distance, très différents en persistance)
- `P-6` ne doit plus servir de référence pour le jeu ouvert

Encore ouvert :
- Cohérence temporelle ballon ↔ joueur (approche/éloignement, pas
  distance ponctuelle)
- Cohérence temporelle du ballon lui-même (déjà partiellement testée,
  section `3.39`, résultat continu sans coupure nette)
- Combinaison de ces signaux

**Toujours aucune implémentation** — uniquement des mesures
rétrospectives sur données déjà enregistrées. Prochain test le plus
propre identifié : reprendre **uniquement** `P-2` et `P-4`, comparer
plus finement leurs motifs de distance temporelle (déjà en main) pour
formuler un critère testable de "cohérence de possession", avant tout
codage.

### 3.49 Découverte majeure : `P-1` a le même problème que `P-6` — reclassification complète de l'échantillon `P` (`20/09/2026`)

**Vérification visuelle directe par l'utilisateur** (captures au
timestamp `±1s`, depuis le lecteur vidéo original) sur les `6` points
`P-1` à `P-6` :

| Point | Description | Statut |
|---|---|---|
| `P-1` | Dégagement du gardien **au pied** | ❌ Balle arrêtée |
| `P-2` | Jeu continu, ballon aux pieds d'un jaune | ✅ Jeu ouvert |
| `P-3` | Jeu continu, ballon aux pieds d'un jaune | ✅ Jeu ouvert |
| `P-4` | Jeu continu, ballon un peu plus loin du joueur | ✅ Jeu ouvert |
| `P-5` | Ballon caché puis visible près d'un joueur (même seconde) | ✅ Jeu ouvert |
| `P-6` | Ballon en main du gardien après une frappe, prépare un `6m` | ❌ Balle arrêtée |

**`P-1` était utilisé dans plusieurs tests ce soir** (comparaison
échelle, mécanisme de reprise, test combiné) comme l'un des `4`
points de référence, et restait **systématiquement faux dans tous
les tests, sans jamais s'améliorer** (section `3.36`, `3.39`,
`3.44`). Ce n'était probablement pas un échec de nos correctifs sur
le jeu ouvert — `P-1` est une scène de dégagement du gardien, comme
`P-6`, donc structurellement hors du périmètre "jeu ouvert" que ces
correctifs visaient à améliorer.

**Conséquence directe** : toute conclusion utilisant `P-1` comme
preuve d'échec (ex. section `3.44`, `"P-1"` cité comme cas où le
mécanisme de reprise échoue) doit être relue avec la même prudence
que pour `P-6` — l'échec pourrait refléter le contexte de balle
arrêtée, pas un défaut du mécanisme testé.

**Point positif** : `P-3` et `P-5`, jamais exploités jusqu'ici, sont
confirmés comme du **jeu ouvert valide** — l'échantillon utilisable
passe de `2` points (`P-2`/`P-4`) à **`4` points**
(`P-2`/`P-3`/`P-4`/`P-5`), une base nettement plus solide pour toute
analyse de cohérence temporelle.

**Échantillon final retenu pour le jeu ouvert** : `P-2`, `P-3`, `P-4`,
`P-5`. **Écarté pour "balle arrêtée"** : `P-1`, `P-6` — conservés dans
l'historique pour traçabilité, non réutilisés comme référence de jeu
ouvert.

### 3.50 `P-3` ajouté à l'analyse — réfute l'hypothèse "vitesse seule explique la piste courte" (`20/09/2026`)

**Position de vérité terrain pour `P-3` établie avec moins de
précision que `P-2`/`P-4`** — aucun candidat `HSV` ne correspond
étroitement à l'estimation visuelle initiale (`x_norm≈0,45`) ; le
meilleur candidat disponible par score intrinsèque est à
`x_norm=0,690`, retenu comme meilleure estimation mais avec cette
réserve explicite.

**Longueur de piste, échantillon élargi à `3` points de jeu ouvert
propre** :

| Point | Vitesse médiane | Longueur de piste |
|---|---|---|
| `P-2` | `106px/s` | `1` frame |
| `P-3` | **`301px/s`** | **`24` frames** |
| `P-4` | `10px/s` | `21` frames |

**Réfute directement l'hypothèse de la section `3.48`** ("la piste
courte de `P-2` s'explique par une vitesse élevée au moment
tir/passe") : `P-3` est près de `3×` plus rapide que `P-2`, mais
conserve une piste `24×` plus longue. La vitesse absolue n'est donc
**pas** le facteur explicatif.

**Nouvelle hypothèse à tester** : peut-être la **régularité** du
mouvement (accélération, changement de direction), pas la vitesse
absolue — un ballon rapide mais à trajectoire stable (dribble, passe
en ligne droite) resterait suivable, tandis qu'un changement brutal
de vitesse/direction (l'instant précis d'un tir) casserait la
continuité même sur une seule frame, indépendamment de la vitesse
elle-même.

**Point de méthode confirmé une nouvelle fois** : une hypothèse qui
semblait solide sur `2` points (`P-2` vs `P-4`) ne tient plus avec un
`3e` point (`P-3`) — exactement le type de correction que
l'élargissement systématique de l'échantillon est censé produire.

### 3.51 Résolution du mystère `P-2` : un seuil `13px` trop strict, pas un problème de trajectoire (`20/09/2026`)

**Guidage précis de l'utilisateur** (dégagement en profondeur du
joueur jaune, visible entre `20:08` et `20:09`) a permis d'examiner
tous les candidats bruts (pas seulement les gagnants par frame) dans
la bande `x_norm≈0,50-0,58` sur `t=1208,0` à `1209,2s` : **une
trajectoire continue et plausible existe à chaque frame de cette
fenêtre**, mais avec des scores intrinsèques faibles (`0,07-0,34`),
jamais gagnants face à d'autres candidats ailleurs dans l'image.

**Reconstruction pas-à-pas** de la piste à partir du point `P-2`
connu (`t=1208,41s`) : le rattachement échoue à `t=1208,54s` — le
candidat le plus proche est à **`53px`**, seulement `13px` au-dessus
du seuil `MAX_SAUT=40px` utilisé dans l'analyse rétrospective.

**Avec un seuil à `60px`** (`+20px` seulement) : la piste s'étend
immédiatement à `7` frames, avec des pas ensuite très fluides
(`8-13px` à chaque transition suivante) jusqu'à une vraie coupure
nette à `t=1209,34s` (`197px`, décrochage réel — fin du dégagement).

**Conclusion, réfutant les hypothèses des sections `3.48`-`3.50`** :
`P-2` n'avait ni un problème de vitesse, ni d'accélération, ni de
régularité de trajectoire — uniquement un seuil de rattachement
légèrement trop strict à la toute première transition. Une fois ce
seul maillon franchi, la trajectoire est parfaitement cohérente.

**Prochaine vérification** : confirmer que ce seuil modestement élargi
(`60px`, contre `70-100px` testés section `3.51` précédente qui
montraient une sur-fusion de pistes) ne dégrade pas la fiabilité
globale de la construction de pistes sur l'échantillon complet.

### 3.52 Implémentation du suivi multi-hypothèses (piste `3b`) (`20/09/2026`)

**Nouveau paramètre `activer_multi_hypotheses`** ajouté à
`BallHSVDetector` — défaut `False`, comportement strictement
inchangé. Maintient plusieurs pistes candidates en parallèle (état
persistant `self._pistes_actives`), au lieu de la sélection
gagnant-par-frame habituelle :

1. Chaque piste active tente de se rattacher au candidat le plus
   proche (`<60px`, valeur validée section `3.51`) de sa dernière
   position connue
2. Les pistes non rattachées sont **tolérées une frame** avant
   expiration (`0,5s` sans rattachement) — évite qu'un maillon manqué
   isolé ne tue une piste par ailleurs cohérente
3. De nouvelles pistes sont semées sur les candidats non appariés
   (score intrinsèque `>0,05`, filtre anti-bruit minimal)
4. Le ballon retenu est le point courant de la **meilleure piste**
   (`longueur × score moyen cumulé`), pas le gagnant ponctuel d'une
   seule frame — sauf si aucune piste n'a encore `>1` point, auquel
   cas la logique normale (score + `seuil_gap_protection`) s'applique
   pour rester cohérente en début de séquence

**Propagé** `process_video()` → `Detector` → `BallHSVDetector`, même
schéma que tous les paramètres de la soirée.

**Notebook de test prêt** sur `P` entier (`60s`), avec
`camera_type=high_side` + `seuil_gap_protection=0,08` (déjà validés)
mais **sans** `intervalle_recherche_globale` (effet de bord connu,
section `3.37`) pour isoler proprement l'effet de ce nouveau
mécanisme.

### 3.53 Régression majeure trouvée et diagnostiquée précisément (`20/09/2026`)

**Test exécuté avec succès techniquement** (correctif de regex
nécessaire côté analyse — positions maintenant en format décimal,
`pos=(296.0,452.0)`), mais **régression sévère sur les `3` points de
référence** :

| Point | Vrai | Sans multi-hyp | Avec multi-hyp |
|---|---|---|---|
| `P-2` | `0,561` | `0,662` | `0,784` (pire) |
| `P-4` | `0,366` | **`0,348`** (quasi parfait) | **`0,980`** (catastrophique) |

**Diagnostic précis via `[MULTI_HYPOTHESES]`** : de `t=1223,62s` à
`t=1224,56s` (englobant l'instant `P-4` exact), la piste
"gagnante" reste bloquée à `x≈918-940` (bord droit — la même
structure identifiée section `3.38`), avec une longueur croissante
(`9→10→11→12`). **À `t=1224,29s` et `t=1224,56s`, la longueur reste
identique à `12`** — signe que cette piste n'a en réalité **pas été
réappariée** ces `2` frames, seulement maintenue en vie par la
tolérance "une frame manquée". Le vrai ballon (`x=334`, correct)
n'apparaît qu'à `t=1224,82s`, avec seulement `3` frames d'historique
— trop récent pour rivaliser avec les `12` frames (même stagnantes)
de la fausse piste.

**Cause racine identifiée** : le critère `longueur × score_moyen` ne
distingue pas une piste **activement reconfirmée cette frame** d'une
piste simplement **encore tolérée** après un maillon manqué — une
piste fausse mais ancienne peut ainsi dominer indéfiniment une piste
vraie mais récente, y compris pendant les frames où elle-même n'a
pas été retrouvée.

**Correctif ciblé implémenté** : la sélection multi-hypothèses ne
peut désormais l'emporter que si la piste gagnante a été
**réappariée à la frame courante exacte** (`dernier point de la
piste == debug_t` de cet appel), pas seulement "encore active" via
la tolérance d'expiration. Si aucune piste n'a été fraîchement
reconfirmée cette frame, retour silencieux à la logique de sélection
normale (score + `seuil_gap_protection`) — qui donnait déjà la bonne
réponse pour `P-4` avant l'introduction du suivi multi-hypothèses.

**Correctif validé par simulation** sur les données déjà enregistrées
(rejouant l'algorithme corrigé sur les candidats bruts déjà
sauvegardés, sans relancer de test coûteux) :

| Point | Sans correctif | Avec correctif |
|---|---|---|
| `P-4` (`t=1224,56s`) | `x_norm=0,979` (faux) | **`x_norm=0,348`** (exact) |
| `P-2` (`t=1208,41s`) | — | `x_norm=0,661` (proche de la référence `0,662` déjà connue, pas de dégradation) |

`P-4` est totalement corrigé, `P-2` reste stable à son niveau de
référence. Notebook de test régénéré avec le correctif, prêt pour
confirmation en conditions réelles.

### 3.54 Test réel du correctif — validation partielle, nouveau problème révélé, clôture de la piste `3b` (`20/09/2026`)

**Résultat du test réel, comparé à la simulation** :

| Point | Simulation | Test réel |
|---|---|---|
| `P-2` | `~0,66` | `0,662` ✅ conforme |
| `P-4` | `0,348` (correct) | **`0,229`** (pire que la référence `0,348` sans multi-hyp) |

**Trois conclusions distinctes, à ne pas mélanger** :

**1. Correctif de réappariement (frame fraîche uniquement) : utile,
mais insuffisant.** Le mécanisme cible bien un vrai problème (piste
maintenue en vie par la seule tolérance d'expiration, sans
réappariement réel) et fonctionne comme prévu en simulation. Mais la
simulation ne reproduit pas entièrement le comportement du test réel
— l'écart `0,348` (attendu) vs `0,229` (obtenu) reste inexpliqué à ce
stade.

**2. Nouvelle observation, distincte du problème corrigé : une piste
concurrente réellement persistante.** À `P-4`, une piste à
`x_norm≈0,21-0,23` est **réappariée à chaque frame** (`26→36` de
longueur), pas simplement tolérée — le correctif de la section `3.53`
ne peut rien contre ce cas, puisqu'il ne cible que les pistes
artificiellement maintenues en vie. Démontre que **la persistance
temporelle, même mesurée correctement, ne suffit pas à elle seule à
distinguer le ballon d'un autre objet suivable**. Nature de cet objet
concurrent non identifiée (joueur, structure, autre élément visuel) —
non investiguée ce soir.

**3. Bilan retenu pour la session** : correctif multi-hypothèses en
validation partielle — corrige le mécanisme identifié (`P-4` en
simulation, `P-2` non dégradé) mais un écart simulation/réel non
expliqué persiste, et une cause structurelle plus profonde (pistes
concurrentes légitimement persistantes) a été mise au jour sans être
résolue.

**Décision explicite : arrêt ici, sans ajouter de nouveau critère de
sélection ce soir.** Deux chantiers clairement identifiés pour la
suite, distincts et à traiter séparément :
1. Comprendre l'écart entre simulation et exécution réelle à `P-4`
2. Déterminer ce qui distingue une piste persistante correspondant au
   ballon d'une piste persistante correspondant à un autre objet — la
   combinaison de plusieurs signaux reste une piste possible, mais
   aucun signal discriminant n'a encore été démontré à ce stade

**Bilan honnête de la piste `3b` (suivi multi-hypothèses)** : correctif
ciblé utile et validé sur son périmètre précis, problème global de
sélection ballon-vs-faux-objet persistant non résolu, prochaine étape
clairement délimitée plutôt que floue.

### 3.55 Chantier `1` résolu : l'écart simulation/réel expliqué (`20/09/2026`)

**Cause identifiée précisément** : la piste concurrente à
`x_norm≈0,21-0,23` avait déjà **`longueur=12` dès `t=1221,35s`**,
soit `3s` avant l'instant `P-4` — largement avant le début de ma
fenêtre de simulation isolée (`t=1223,5s`). Elle continue de croître
énormément sur toute la suite (`36` à `t=1225,2s`, jusqu'à **`164`**
à `t=1251s`) — un objet extrêmement stable sur une grande partie de
la fenêtre `P`, pas un artefact ponctuel.

**Explication complète** : ma simulation, en démarrant "à froid" sur
une fenêtre étroite, ne pouvait pas reproduire l'historique déjà
accumulé par cette piste lors du traitement séquentiel continu de `P`
depuis `t=1200s`. Ce n'est pas un bug de code — c'est une limite
méthodologique de la vérification par simulation isolée : tester un
segment hors contexte ne reflète pas fidèlement un traitement continu
où des pistes concurrentes ont eu le temps de s'installer bien avant
le moment observé.

**Point de méthode à retenir pour toute future simulation
rétrospective** : simuler depuis le tout début de la séquence
réellement traitée (ou au moins une fenêtre largement antérieure au
point d'intérêt), jamais un segment isolé démarrant artificiellement
près de l'instant à vérifier.

### 3.56 Synthèse finale — écart expliqué, identité de la piste concurrente non confirmée, clôture de session (`20/09/2026`)

**1. Écart simulation/réel : expliqué.** À `t=1221,35s`, la piste
concurrente (`x_norm≈0,21-0,23`) possède déjà une longueur de `12`
frames — la simulation, démarrant à `t=1223,5s`, ne disposait pas de
cet historique. La piste atteint ensuite `164` frames à `t=1251s`.
**Conclusion : l'écart s'explique par une différence de contexte
temporel entre simulation isolée et traitement continu — il ne
démontre pas un bug du correctif.**

**2. Identification visuelle : résultat incomplet.** `6` captures
examinées, réparties sur toute la durée de la piste :

| Captures | Constat |
|---|---|
| `1` à `3` (`t=1221-1224s`) | Deux officiels immobiles près de la barrière, candidat plausible pour la piste initiale |
| `4` à `6` (`t=1236-1251s`) | Changement d'angle de caméra ; aucun objet distinctif identifié à la position relative correspondante |

**Hypothèses encore ouvertes, non tranchées** : soit la piste
initiale correspond aux officiels puis dérive vers un autre objet
après le changement de caméra, soit il s'agit de plusieurs épisodes
de suivi distincts se succédant à une position relative similaire par
coïncidence. **Impossible de confirmer que les `6` captures
correspondent au même objet physique.**

**3. Conclusion méthodologique retenue** : toute future simulation
rétrospective doit reproduire l'historique complet du traitement
continu, pas un segment isolé. L'identification visuelle de la piste
concurrente reste non concluante — la présence initiale d'officiels
immobiles est une explication plausible, non démontrée pour la
totalité de la durée.

**Statut final du chantier `3b` (suivi multi-hypothèses)** :
- Écart simulation/réel : ✅ expliqué (différence d'historique)
- Identité de la piste concurrente : ❌ non confirmée
- Correctif de réappariement : toujours insuffisant pour résoudre
  `P-4` en conditions réelles
- **Prochaine étape** : reprendre l'analyse avec un historique
  continu (pas de simulation isolée), et chercher un signal
  permettant de distinguer le ballon des objets concurrents
  légitimement persistants — aucune modification supplémentaire du
  détecteur ce soir.

---

## Clôture de session

Session arrêtée sur ce constat, sans modification supplémentaire du
détecteur. Bilan complet en section `3.40` (bilan général) et
sections `3.41`-`3.56` (pistes complémentaires explorées après la
clôture initiale) — correctifs validés livrés (bug d'échelle,
sélection multi-candidats, `camera_type`, `seuil_gap_protection`),
pistes `1`/`2`/`3b` explorées avec résultats honnêtement mitigés,
chantiers clairement délimités pour la reprise.

### 3.57 Piste complémentaire : suivi continu depuis le coup d'envoi (`20/09/2026`)

**Proposition reçue** : plutôt que de rechercher le ballon
indépendamment à chaque image, l'ancrer dès le coup d'envoi et
maintenir un suivi continu, avec une machine à états explicite
(ballon en jeu / masqué / sorti / remise en jeu / nouveau ballon /
perdu) pour gérer proprement les interruptions.

**Test scopé exécuté** (sans construire la machine à états complète,
hors de portée pour ce soir) : comparaison directe entre démarrage à
froid exactement à `t=1200s` et traitement continu depuis `t=1100s`
(`160s` d'historique réel avant `P-2`/`P-4`), mêmes correctifs validés
dans les deux cas (échelle + `high_side` + `seuil_gap_protection=0,08`),
aucun mécanisme à risque.

**Résultat identique dans les deux cas** :

| Point | `P_froid` | `P_continu` |
|---|---|---|
| `P-2` | `0,662` (écart `0,101`) | `0,662` (identique) |
| `P-4` | `0,348` (écart `0,018`) | `0,348` (identique) |

**Explication, pas un échec de l'hypothèse** : la sélection actuelle
(`seuil_gap_protection`) n'a de mémoire que sur `last_pos` (une seule
position précédente), jamais un historique accumulé — que le
traitement commence `100s` plus tôt ou non ne peut donc rien changer
à un mécanisme qui n'accumule structurellement rien au-delà d'une
frame en arrière.

**Conclusion retenue** : la continuité ne peut faire une différence
que combinée à un suivi qui exploite réellement l'historique (le
suivi multi-hypothèses, section `3.52`-`3.56`) — or ce test l'excluait
délibérément pour isoler la continuité seule. Les deux composantes de
la proposition initiale (partir du coup d'envoi **et** suivre en
continu avec mémoire) n'ont donc été testées qu'**à moitié**,
séparément : continuité seule sans mémoire (ce test, sans effet) et
mémoire seule sans démarrage précoce (suivi multi-hypothèses,
régression sur `P-4` en conditions réelles). **La combinaison des deux
n'a pas encore été testée.**

**Machine à états proposée pour une gestion propre des interruptions**
(non implémentée ce soir, chantier à part) :

| État | Comportement |
|---|---|
| Ballon en jeu | Suivre trajectoire et déplacements |
| Ballon masqué | Prédire brièvement, puis tenter de retrouver |
| Ballon sorti du terrain | Suspendre le suivi actif, rechercher la remise en jeu |
| Remise en jeu | Réinitialiser la trajectoire depuis le ballon remis en jeu |
| Nouveau ballon utilisé | Nouvelle trajectoire, sans conserver l'identité de l'ancien |
| Ballon perdu | Signaler la perte plutôt que d'inventer une position |

**Protocole de test proposé pour une prochaine session** : comparer
suivi classique (recherche indépendante à chaque image) contre suivi
depuis le coup d'envoi (trajectoire + historique, réinitialisé après
sortie/remise en jeu), en mesurant : pourcentage d'images correctement
localisées, nombre de changements accidentels d'objet suivi, nombre de
pertes de suivi, temps de récupération après interruption.

**Décision explicite : ne pas lancer le test combiné (continuité +
multi-hypothèses) ce soir.** Combiner deux mécanismes avant de
comprendre la régression déjà observée sur l'un d'eux (`P-4`, section
`3.53`-`3.56`) risquerait de reproduire le même problème sans savoir
lequel des deux en serait la cause. Le test combiné ne sera repris
qu'après avoir défini précisément **comment** l'historique doit
influencer la sélection du ballon (pas juste "ajouter de la mémoire"),
et **comment mesurer** son apport réel — l'objectif restant de
démontrer une amélioration mesurée, pas d'ajouter de la complexité.

**Bilan retenu pour cette piste** : continuité seule testée et
écartée (aucun effet avec la sélection actuelle, sans mémoire) ;
mémoire seule testée et non concluante (régression sur `P-4`) ;
combinaison des deux explicitement différée, condition claire posée
avant reprise.

### 3.58 Étape `1` du plan en `3` étapes : cause structurelle de la régression `P-4` identifiée (`20/09/2026`)

**Séquence complète `[MULTI_HYPOTHESES]` examinée** autour de `P-4`
(`t=1220` à `1226s`) : la piste gagnante reste à la position
**exactement identique** `(219,186)` pendant `8` frames consécutives
(`t=1224,29s` à `1225,22s`), réellement réappariée à chaque fois (pas
simplement tolérée — le correctif de fraîcheur, section `3.53`, ne
peut rien y changer puisqu'un point immobile se réapparie trivialement
à lui-même).

**Chiffres précis** :

| | Piste fausse | Piste vraie |
|---|---|---|
| Position | `(219,186)`, immobile `8` frames | `x≈336`, fraîchement repérée |
| Longueur à `P-4` | `≈32` | `2` à `3` |
| Score moyen | `0,17` à `0,20` | `≈0,27` |
| Valeur (`longueur×score`) | `≈5,4-6,4` | `≈0,5-0,8` |

**Écart d'un facteur `8` à `10×`** en faveur de la piste fausse,
uniquement du fait de la longueur accumulée pendant son immobilité —
pas d'une meilleure ressemblance avec le ballon.

**Cause structurelle proposée** : le critère `longueur × score_moyen`
n'a aucune limite ni décroissance — une piste immobile continue
d'accumuler de la valeur sans frein, alors qu'un ballon réellement en
jeu bouge presque continuellement.

**Mise en garde explicite retenue** : un biais du critère est
identifié, mais **pas encore démontré que toute piste immobile est
fausse** — un ballon peut légitimement être immobile lors d'un arrêt
de jeu (comme confirmé pour `P-6`, section `3.49`). Ne pas modifier le
code avant d'avoir comparé la durée d'immobilité et la variation de
position des pistes **vraies** et **fausses** sur `P-2` et `P-4`, pour
vérifier si une pénalité liée à l'immobilité serait justifiée sans
risquer d'éliminer des ballons réellement immobiles en jeu normal.

**Mesure effectuée, mêmes bornes temporelles exactes pour les deux
pistes (`t=1220,0` à `1226,0s`)** :

| Mesure | Piste fausse | Piste vraie |
|---|---|---|
| Répétitions exactes | `20%` | `20%` (identique) |
| **Séquence maximale de répétitions consécutives** | **`8` frames** | **`3` frames** |
| Positions distinctes | `34/42` | `24/31` |
| Déplacement médian | `8px` | `16px` |

**Ce que ça révèle** : le taux global de répétition exacte ne
distingue pas les deux pistes (confirmant la mise en garde retenue
plus haut — un vrai ballon répète aussi occasionnellement la même
position). **Mais la longueur maximale d'une séquence ininterrompue
de répétitions diffère nettement** (`8` contre `3`) — un vrai ballon,
même largement stable, ne reste jamais figé au pixel exact plus de
`2`-`3` frames d'affilée, contrairement à la piste fausse.

**Limite explicitement reconnue** : un seul cas comparatif (`P-4`) —
pas encore démontré que ce signal distingue généralement vrais
ballons et faux candidats. Aucune pénalité ni modification du score
appliquée à ce stade.

**Suite actée** : tester ce même critère sur `P-2` avant toute
intégration au score, pour vérifier si la limite de répétitions
consécutives généralise.

**Test sur `P-2` (piste gagnante, correspondant déjà à la réponse
connue `0,662`)** :

| Piste | Répétitions | Séquence max | Distinctes | Déplacement médian |
|---|---|---|---|---|
| Fausse (`P-4`) | `20%` | **`8`** | `81%` | `8,0px` |
| Vraie (`P-4`) | `20%` | `3` | `77%` | `16,0px` |
| Gagnante `P-2` | `3%` | **`1`** | `97%` | `9,4px` |

**Le motif généralise** : sur les `3` pistes mesurées (`1` fausse,
`2` vraies), la séquence maximale de répétitions exactes est nettement
plus basse pour les pistes vraies (`1` et `3`) que pour la fausse
piste identifiée (`8`) — écart net, pas une différence marginale.
Échantillon encore limité (`3` pistes, `2` points), mais signal
cohérent sur un deuxième cas indépendant.

**Critère retenu pour un test ciblé** : plafonner ou pénaliser la
contribution de longueur d'une piste au-delà d'une séquence
consécutive de répétitions exactes (`~4-5`), sans toucher au reste du
calcul.

**Implémenté** : nouveau paramètre `seuil_streak_gele` (défaut `None`
= désactivé, comportement du suivi multi-hypothèses strictement
inchangé). Quand fourni, plafonne la longueur effective d'une piste
dès qu'elle a montré une séquence de positions exactement identiques
consécutives d'au moins cette taille — n'affecte que le calcul de
valeur (`longueur × score_moyen`), rien d'autre. Propagé
`process_video()` → `Detector` → `BallHSVDetector`, même schéma que
tous les paramètres de la soirée.

**Notebook de test prêt** sur `P` entier, `seuil_streak_gele=5`
(valeur intermédiaire entre le `3` mesuré sur piste vraie et le `8`
mesuré sur piste fausse), combiné avec `activer_multi_hypotheses=True`.

### 3.59 Problème critique découvert : `ball_tracker.py` périmé dans plusieurs notebooks récents (`20/09/2026`)

**Découvert en préparant le test de résolution native** : le log
montrait `"BallTracker indisponible :
BallTracker.__init__() got an unexpected keyword argument
'activer_secours_possession' — fallback YOLO"` — la construction de
`BallTracker` échouait silencieusement, basculant sur un chemin de
repli différent pour **tout le run**.

**Cause identifiée** : le `ball_tracker.py` embarqué dans
`notebook_test_multi_hypotheses.py` (construit à partir d'un notebook
antérieur à l'ajout du secours par possession, section `3.44`) ne
contenait **pas** `activer_secours_possession` — mais `main.py`,
correctement mis à jour lui, essayait de passer ce paramètre à la
construction. Tous les notebooks construits **à partir de** cette
lignée (`multi_hypotheses` → `streak_gele` → `resolution_native`)
héritaient de ce même fichier périmé.

**Portée du problème** : les tests suivants ont potentiellement tourné
sur un chemin de repli différent de notre pipeline `HSV`/multi-
hypothèses prévu, **sans qu'aucune erreur visible ne l'indique** dans
les résultats eux-mêmes (juste ce message d'avertissement noyé dans
les logs) :
- Section `3.53`-`3.56` (test du suivi multi-hypothèses et son
  correctif de fraîcheur)
- Section `3.58` (généralisation du critère à P-2, et test du plafonnement par séquence gelée)

**Conséquence à prendre en compte** : les résultats de ces sections
restent documentés tels quels (le diagnostic sur `[HSV_DIAG]` et
`[MULTI_HYPOTHESES]` provient de `detector.py`, qui LUI était à jour —
seule la construction de `BallTracker` échouait) — mais toute
conclusion s'appuyant spécifiquement sur le comportement de
`BallTracker` (secours par possession, interaction avec
`ball_tracker.update()`) pendant ces runs est à considérer avec
prudence supplémentaire, le chemin de repli exact emprunté n'étant
pas entièrement caractérisé.

**Corrigé** : `ball_tracker.py` à jour ré-embarqué explicitement dans
`notebook_test_resolution_native_v2.py`, vérifié conforme aux `4`
fichiers de référence (`main.py`, `detector.py`, `ball_tracker.py`,
`events.py`) avant tout nouveau test.

**Point de méthode ajouté** : désormais, vérifier explicitement la
présence d'un marqueur caractéristique de CHAQUE fichier embarqué
(pas seulement `main.py`/`detector.py`, qui recevaient déjà cette
vérification systématique) avant de considérer un notebook prêt à
tester — cette omission spécifique sur `ball_tracker.py` explique
l'incident.

### 3.61 Test résolution native corrigé — pas d'amélioration, piste close (`20/09/2026`)

**Test avec `AIRE_BORNES` ajustées** (`60-800px²`, échelle `×4` en
aire pour compenser le doublement linéaire de résolution) :

| | `P-4` écart | Médiane globale |
|---|---|---|
| `960` (référence) | `0,018` | `0,348` |
| `1920` + bornes ajustées | `0,131` (pire) | `0,349` (quasi identique) |

**Conclusion** : la médiane globale reste quasi inchangée — le
comportement général du détecteur n'est pas fondamentalement
différent à cette résolution, même bornes correctement adaptées.
`P-4` se dégrade spécifiquement, plausiblement parce que la plage
élargie laisse passer davantage de candidats bruit sans que le
ballon devienne plus facilement discriminable parmi eux. **Confirme
que le problème n'est pas la détectabilité brute du ballon (déjà
trouvé la plupart du temps, à `960` comme à `1920`), mais la
discrimination entre le ballon et d'autres objets ronds/clairs de
taille comparable** — un problème qui ne se résout pas en changeant
l'échelle de traitement. Piste résolution native close pour cette
session.

### 3.62 Portée exacte du bug `ball_tracker.py` périmé — traçage des tests affectés (`20/09/2026`)

**Cause précise retracée** : `notebook_test_multi_hypotheses.py` a
été construit en repartant de `notebook_test_global_vs_local.py` —
un notebook **antérieur** à l'ajout du secours par possession dans
`ball_tracker.py` (section `3.44`). Toute la lignée construite
**à partir de** `multi_hypotheses` a hérité de ce `ball_tracker.py`
périmé : `streak_gele`, `resolution_native` (version `v1`, corrigée
depuis en `v2`).

**Détail déterminant** : `main.py` construit `BallTracker` avec
`activer_secours_possession=(camera_type=="high_side")` — **tous**
nos tests de cette lignée utilisaient `camera_type="high_side"`, donc
ce paramètre valait **toujours `True`**, déclenchant **systématiquement**
l'échec de construction et le repli sur `YOLO` seul, jamais de façon
intermittente.

**Tests concrètement affectés** (tous les runs de ces sections ont
tourné sur le chemin de repli `YOLO`, pas sur notre pipeline
`HSV`/multi-hypothèses prévu) :
- Section `3.53` à `3.56` : test initial du suivi multi-hypothèses,
  régression `P-4` observée, correctif de fraîcheur, écart
  simulation/réel, identification de la piste concurrente
- Section `3.58`-`3.59` : découverte de la séquence gelée, test du
  plafonnement (`seuil_streak_gele`)

**Tests NON affectés, vérifiés indépendamment** :
- Toute la lignée `notebook_test_possession` (`v1`-`v4`, sections
  `3.44`-`3.46`) — ball_tracker.py y était systématiquement à jour,
  vérifié à chaque étape
- `notebook_test_vecteurs_joueurs.py` (section `3.42`-`3.43`) — ne
  sollicitait pas `activer_secours_possession`, donc insensible à son
  absence
- `notebook_test_resolution_native_v2.py` et `resolution_ajustee.py`
  (sections `3.60`-`3.61`) — `ball_tracker.py` explicitement
  revérifié et corrigé avant ces runs

**Implication** : les découvertes mécanistiques elles-mêmes (piste
figée `8` frames à `(219,186)`, écart simulation/réel expliqué par
l'historique, séquence gelée comme signal candidat) proviennent des
logs `[HSV_DIAG]`/`[MULTI_HYPOTHESES]`, produits par `detector.py` —
**qui, lui, était à jour**. Ces observations décrivent donc bien le
comportement réel de notre logique multi-hypothèses. Mais la
comparaison finale avec `[BALL]` (sortie du tracker) et toute
interaction avec le secours par possession pendant ces runs
spécifiques reste à considérer avec prudence, puisque le chemin de
repli `YOLO` a pu influencer `last_pos`/`players` transmis en amont.

**Décision** : relancer le test le plus déterminant (`P-4`, suivi
multi-hypothèses avec correctif de fraîcheur, section `3.53`-`3.54`)
avec la chaîne de fichiers désormais entièrement vérifiée, pour
confirmer si la régression et son diagnostic tiennent une fois le
pipeline réellement fonctionnel.

