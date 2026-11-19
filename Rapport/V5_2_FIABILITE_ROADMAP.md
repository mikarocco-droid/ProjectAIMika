# V5.2 — De la détection à la fiabilité : refonte du contrat de confiance

**Statut : document de cadrage, à valider avant tout nouveau correctif de code.** Rédigé après une session d'investigation ayant mis au jour plusieurs défaillances indépendantes dans la chaîne équipe/identité/début de match — trop nombreuses et trop imbriquées pour continuer à les corriger une par une sans d'abord figer l'architecture cible.

---

## -1. CLARIFICATION ARCHITECTURALE — À LIRE EN PREMIER, IMPOSSIBLE À MANQUER

**Il existe deux architectures de détection du coup d'envoi dans l'historique de ce projet. Elles sont concurrentes, pas complémentaires. L'une remplace l'autre.**

### Architecture A — DINOv2 + Claude Vision (documents `BASELINE_*`, `EXPERIENCE_*`, `V3_*`, une partie de `V4_CLUSTERING_ET_AUTOPSIE_RF.md` §1-2)
Pipeline : scan DINOv2 (embeddings visuels) → clustering temporel → filtre de contexte lointain → Claude Vision (classification comparative) → règle de décision Python. **Résultat documenté : 7/9 succès automatiques, 1/9 ambiguïté honnête (Juprelle), 1/9 échec caractérisé (Goe), 0/9 faux résultat confiant.**

### Architecture B — Mining + RF sur `frames_data` (tout le travail d'aujourd'hui, §1-10 ci-dessous)
Pipeline : tracking DeepSort/BallTracker → `ko_features.py` (features géométriques + vitesse + audio, **aucune ne vient de DINO ni de Claude**) → `candidates_mining.py` → RandomForest → sélection finale.

### Décision actée (session du jour)
**Architecture B remplace Architecture A — elle ne l'alimente pas.** Ce pivot s'est produit après la découverte de `kickoff_detector.py` (le vrai système géométrique de production) en plein milieu de l'investigation du jour. Les clusters DINOv2, les scores de compatibilité, le prompt Claude Vision de l'Architecture A ne sont réutilisés nulle part dans le travail d'aujourd'hui.

**Si DINO/Claude revient un jour**, ce serait comme une 9ᵉ feature potentielle dans `ko_features.py` (ex: score de compatibilité Claude sur l'instant candidat) — jamais comme l'architecture porteuse.

### Point de vigilance sur la numérotation
`V4_CLUSTERING_ET_AUTOPSIE_RF.md` utilise déjà "V5.1" pour désigner l'audit de fiabilité d'identité joueur (§14 de ce document). **Ce document-ci (`V5_2_FIABILITE_ROADMAP.md`) est un renommage/prolongement indépendant**, pas la suite numérotée officielle de ce "V5.1" — à clarifier si les deux fils sont un jour fusionnés.

### Trois éléments de l'Architecture A/V4 directement réutilisables pour la suite d'Architecture B

1. **`class_weight='balanced'`** — déjà validé sur le RF historique (9 matchs, 5 améliorés, 0 régression, chaîne causale complète confirmée sur P1Minerois), **jamais intégré en production**. Techniquement indépendant de la contamination `v1_manuel`/`v2_mining` découverte aujourd'hui (§10) — à tester en complément, pas en remplacement.
2. **Le bug `tracker_id` (§6 ci-dessous) était déjà partiellement documenté sur Goe** (V4.10 : 65% des frames de la zone 1085-1095s, cause non trouvée, investigation volontairement arrêtée). **Le correctif d'aujourd'hui (§6) résout ce bug précisément identifié à l'époque** — confirmé par un nouveau test sur Goe (0% de doublons après correctif, contre 65% avant). Ne résout PAS le mystère `43 vs 17` (déjà su à l'époque — deux anomalies indépendantes).
3. **"Match Start Resolver" à 3 paliers** (§14.1 de `V4_CLUSTERING_ET_AUTOPSIE_RF.md`) — spécification fonctionnelle déjà rédigée (`AUTO_CONFIRMED`/`NEEDS_CONFIRMATION`/`MANUAL_REQUIRED`, ancrée sur les 9 matchs), seuils numériques non calibrés. Correspond exactement à ce qu'on envisageait de construire (§2.1 ci-dessous) — à reprendre plutôt que réinventer.

---

## 0. Le changement de règle fondamental

**Ancien contrat implicite du pipeline** : produire une réponse, toujours — un début de match, une identité de joueur, une statistique — même quand les preuves disponibles sont insuffisantes.

**Nouveau contrat visé** : un pipeline qui sait **refuser de répondre** quand il n'a pas assez de preuves, plutôt que de fabriquer une réponse plausible mais potentiellement fausse.

C'est le principe directeur de toute la V5.2. Chaque point ci-dessous en est une déclinaison concrète.

---

## 1. Pourquoi maintenant — ce qui a été découvert aujourd'hui

Cette feuille de route n'est pas une anticipation théorique : chaque point est motivé par une anomalie réellement mesurée dans cette session.

| Découverte | Preuve mesurée |
|---|---|
| Filtre de collecte de calibration cassé | `teams_calibrated=False` sur tout un match malgré des milliers de détections |
| Seuil d'ambiguïté ~50× trop grand | Percentiles réels `[0.68, 1.15, 1.25, 1.48, 1.59]` contre seuil `15.0` |
| `team=0` traité comme "faux" (bug Python) | 3 occurrences trouvées et corrigées (`pipeline.py`, `player_identity.py`, `smart_game_ai.py`) |
| Seuil "gardien" mal calibré | `44,2%` des détections classées gardien (statistiquement impossible) — ramené à `17,4%` après correctif, mais pas encore validé comme définitif |
| Trois systèmes d'équipe indépendants, incohérents | `PlayerReID` (19D, calibré) / `team_map` (agrégation main.py) / `team_cluster.py` (repli RGB, fonction pont inexistante) |
| OCR Tesseract peu fiable sur maillots à 2 chiffres | `#17`, `#44`, `#80` visuellement confirmés lisibles à l'écran, quasi absents des lectures OCR répétées |
| Gemini améliore sans résoudre | Fusions erronées `942 → 664 → 539` après intégration Gemini — réel progrès, toujours incomplet |
| Collision de `tracker_id` dans DeepSort | `65%` des frames d'une zone étudiée montrent des doublons stricts |
| `43` (Goe historique) non reproductible | Replay production-fidèle donne `17`, écart jamais expliqué malgré investigation approfondie (V4.10) |
| Possession de match absurde | `{0: 97,6%, 1: 2,4%}` sur un run réel — conséquence directe du repli RGB de `team_cluster.py` |

Aucune de ces anomalies n'est anecdotique : prises ensemble, elles montrent que **la chaîne complète tracking → équipe → identité → événements → statistiques** peut silencieusement propager une erreur sans qu'aucune étape individuelle ne plante.

---

## 2. Structure de la V5.2 — 6 chantiers, dans cet ordre

### 2.1. Sécurisation du début de match

**Objectif** : ne jamais produire de statistiques de match sur une vidéo dont le vrai début (coup d'envoi) n'a pas été identifié avec une confiance suffisante.

- Détecter une zone réellement exploitable, distincte de : échauffement, entrée des joueurs, briefing, attente.
- Si aucun signal ne dépasse le seuil de confiance retenu → statut `MANUAL_REQUIRED`, pas de faux début imposé arbitrairement.
- Reprend et complète la spécification "Match Start Resolver" déjà rédigée (3 paliers `AUTO_CONFIRMED`/`NEEDS_CONFIRMATION`/`MANUAL_REQUIRED`), enrichie des signaux concrets identifiés depuis :
  - apparition simultanée d'un grand nombre de joueurs ;
  - disparition du comportement d'échauffement ;
  - regroupement autour du rond central ;
  - changement brutal de mouvement global ;
  - première phase de jeu cohérente ;
  - sifflet, si détectable ;
  - cohérence avec les événements détectés dans les minutes suivantes.

**Sortie attendue, par exemple :**
```
Début estimé : 1092s
Confiance : 72%
KO automatique non confirmé.
```

### 2.2. Chaîne équipe de bout en bout — une seule source de vérité

**Objectif** : supprimer la coexistence actuelle de trois systèmes de classification d'équipe qui ne se parlent pas et produisent des résultats différents.

**Statut mis à jour (voir §5 pour le détail complet des mesures)** :

| Système | Rôle | Statut au 26/08 |
|---|---|---|
| `PlayerReID._calibrate_teams()` / `_infer_team()` | Classification par détection, K-means 19D | 🟢 Filtre de collecte corrigé, seuil d'ambiguïté recalibré (`15,0→0,3`), seuil gardien corrigé (`0,6→1,5`, `44,2%→17,4%`) |
| `team_map` (agrégation dans `main.py`) | Vote majoritaire par `track_id` | 🟢 Bug `team=0` falsy corrigé (3 occurrences distinctes trouvées et corrigées) |
| `events.py` (`_locked_team`) | Propagation aux événements | 🟢 **Nouveau** : les événements héritent désormais de `team_map` plutôt que d'une classification par frame isolée (principe "Veo" — identité verrouillée). Mesuré : `RAW team=None` 3,5%→2,1%, `CLEAN` 18,4%→12,9% |
| `team_cluster.py` | Classification par événement (repli) | 🟡 Toujours cassé (fonction pont inexistante, repli RGB incompatible) — **mais son impact réel s'est effondré** : sur le dernier run mesuré, `team_map` fournit 82-97% des équipes, `team_cluster` ne contribue plus rien de visible. Reporté, plus prioritaire pour l'instant. |
| `goal_posthoc.py` → `team` | Attribution d'équipe sur les buts détectés par trajectoire | 🟠 Confirmé comme cause principale du `~79%` de `team=None` sur les buts. A `dominant_team_at_goal` disponible mais **délibérément écarté** comme repli (proxy de momentum, pas une identification fiable — voir §5.4) |

**Travail requis** :
- Choisir UNE représentation d'équipe faisant autorité (probablement les centroïdes calibrés de `PlayerReID`, déjà les plus robustes après les correctifs d'aujourd'hui).
- Faire circuler cette information jusqu'à `team_cluster.py`, avec un format réellement compatible (couleur BGR moyenne par équipe, pas le vecteur 19D brut).
- Ne poursuivre vers les étapes suivantes qu'après avoir vérifié que les deux équipes d'un match sont représentées de façon plausible (proche de 50/50, pas de rapport 10× ou 97%/3%).

### 2.3. Identité joueur

**Objectif** : fusionner les `track_id` fragmentés en identités de joueurs réelles, sans jamais fusionner à tort deux personnes différentes.

- Clé de fusion conceptuelle : **`(team, jersey_number)`**, jamais `jersey_number` seul (implémenté dans `player_identity.py`).
- 🟢 **Deux bugs `team=0` falsy corrigés dans cette même logique** (`team_map.get(str(pid)) or team_map.get(pid)` et `if team else`) — sans ces corrections, l'équipe majoritaire (`team=0`, 41% des cas sur Raeren) était traitée comme "équipe inconnue" pour la fusion.
- 🟢 **`reconcile_unknown_team_fragments()` implémentée et testée** : fusionne un fragment "équipe inconnue" dans son fragment "équipe connue" **uniquement** si le numéro de maillot est confirmé sur une seule équipe (vérifié : 8 numéros/17 sur Raeren sont ambigus — confirmés sur les deux équipes — et restent délibérément séparés) + absence de chevauchement temporel (double sécurité).
- Auditable via `audit_identite_joueurs.py` (construit et utilisé toute la journée) : test de présence simultanée comme preuve mathématique de fusion erronée.
- Principe : **conserver l'incertitude plutôt que fabriquer une identité.** Un `track_id` sans équipe ni numéro fiable reste un fragment non résolu, pas une fausse identité.

### 2.4. Fiabilité joueur / statistique — score de confiance

**Objectif** : ne jamais présenter une statistique douteuse comme une certitude.

Un score de fiabilité **global** par joueur ne suffit pas — la découverte clé d'aujourd'hui est qu'un joueur peut avoir une identité fusionnée à tort (donc une distance parcourue faussée) tout en ayant un tir individuel parfaitement valide, détecté indépendamment de toute question d'identité.

**Décomposition proposée, par statistique :**

| Statistique | Dépend de |
|---|---|
| Identité | Chaîne équipe + fusion `(team, jersey)` |
| Tirs / buts | Détection d'événement — largement indépendant de l'identité |
| Passes | Identité de l'émetteur ET du récepteur |
| Distance parcourue | Continuité du tracking sur toute la durée |
| Vitesse | Continuité du tracking, sensible aux discontinuités |
| Duels | Identité des deux joueurs impliqués |

**Exemple de restitution :**

| Joueur | Identité | Tracking | Équipe | Fiabilité globale |
|---|---|---|---|---|
| #10 | très bonne | bonne | bonne | 🟢 94% |
| #17 | bonne | moyenne | bonne | 🟠 71% |
| #4 | ambiguë | bonne | inconnue | 🔴 38% |

Dans le rapport final, une statistique associée à un joueur `🔴` ne doit jamais être affichée avec la même autorité visuelle qu'une statistique `🟢`.

### 2.5. Gestion du cas Goe — benchmark permanent, pas problème à résoudre

**Changement de posture explicite** : après investigation approfondie (V4-V5.1, voir `V4_CLUSTERING_ET_AUTOPSIE_RF.md` §13-14), l'écart `43 vs 17` reste inexpliqué malgré une réplication fidèle de la configuration de production. Plutôt que de continuer à chercher la cause exacte, Goe devient un **cas de régression permanent** :

- Le pipeline doit être capable de produire, sur Goe : *"Début de match non suffisamment fiable / données joueur insuffisamment fiables"* — plutôt que de continuer l'analyse comme si le début avait été correctement identifié.
- **Règle de non-régression** : une modification future du pipeline n'est considérée comme une amélioration que si elle **améliore le comportement sur Goe sans dégrader les 8 autres matchs déjà validés** (Andrimont, Franchimont, MineroisSter, P1Minerois, Raeren, Stembert, Wanze, Juprelle).
- L'anomalie `43 vs 17` elle-même reste documentée comme connue et non résolue (§13.1 de `V4_CLUSTERING_ET_AUTOPSIE_RF.md`) — ce n'est plus un blocant pour avancer.

### 2.6. Statistiques de match — dernière étape, pas la première

**Objectif** : ne recalculer/améliorer les statistiques de match (xG/xA, heatmaps, réseaux de passes, classements) qu'une fois les cinq points précédents en place.

- Exclusion explicite des phases non pertinentes (pré-match, mi-temps, célébrations prolongées).
- Propagation des scores de confiance (point 2.4) jusqu'au rapport final.
- Signalement explicite, dans le rapport lui-même, des statistiques potentiellement contaminées — pas seulement en interne/logs.

**Exemple de restitution visée pour un match comme Goe :**

> 🟠 **Analyse partiellement validée**
> Début de match non confirmé avec un niveau de confiance suffisant.
> Les statistiques globales du match ne sont donc pas certifiées.
> Certaines statistiques individuelles restent disponibles avec leur niveau de fiabilité.

Un rapport qui dit clairement ce qu'il ne sait pas est jugé plus professionnel qu'un rapport complet aux chiffres précis mais faux.

---

## 3. Ordre d'exécution retenu

```
1. Sécuriser le début du match          (le plus critique, le plus indépendant du reste)
2. Réparer la chaîne équipe de bout en bout
3. Réparer l'identité joueur             (dépend de 2)
4. Ajouter le score de confiance joueur/statistique
5. Améliorer les statistiques de match   (dépend de 1-4)
```

Documenté avant tout code, précisément parce que plusieurs problèmes se croisent (tracking, équipe, identité, début de match, fiabilité) — commencer directement par le point 1 sans cette vue d'ensemble risquerait de corriger un symptôme sans que l'architecture cible soit fixée.

---

## 4. Ce que ce document ne couvre pas

- Le détail d'implémentation de chacun des 6 points — à spécifier au moment d'attaquer chaque point.
- La calibration précise des seuils de confiance (72% cité en exemple au point 2.1 est illustratif, pas validé).
- La cause exacte de l'écart `43 vs 17` sur Goe — délibérément laissée ouverte, voir §2.5.

---

## 5. Investigation RF / kickoff_score_hybride — état complet

### 5.1. Lien retrouvé entre les systèmes

`kickoff_score_hybride.py` (3 composantes indépendantes : géométrie, transition, ballon — aucun poids fixe) a évolué vers `ko_features.py`, dont les noms de features (`whistle_score`, `geo_separation`, `ball_score`) correspondent exactement à ceux des `*_df_mine_complet.csv` produits par `candidates_mining.py`. **Le RF analysé toute la journée est très probablement le descendant direct de cette exploration.**

`kickoff_detector.py` (l'heuristique `groups[-1]`, jugée obsolète — voir §2.1 mis à jour) est un système **séparé et plus ancien**, non lié à cette lignée.

### 5.2. Recall du mining vs qualité du classement RF — 9/9 matchs

| Match | Écart au vrai KO | Rang RF |
|---|---|---|
| MineroisSter | 1,0s | 3 |
| Stembert | 4,0s | 5 |
| Raeren | 0,0s | 10 |
| Andrimont | 0,0s | 11 |
| Wanze | 6,0s | 18 |
| Juprelle | 8,0s | 29 |
| Franchimont | 2,0s | 31 |
| P1Minerois | 9,8s | 31 |
| Goe | 11,0s | 48 |

🟢 **DÉMONTRÉ** : le recall du mining est excellent (écart toujours ≤11s). 🟢 **DÉMONTRÉ** : le classement RF est fiable (rang ≤10) sur seulement 3/9 matchs.

### 5.3. Cause des faux positifs mieux classés que le vrai KO

- 🟢 **DÉMONTRÉ** : 50% des candidats mieux classés sont des tirs connus (`label_auto=connu:shot`), avec forte variance par match (12,5% à 87,5%).
- 🟢 **DÉMONTRÉ**, sur 7-9/9 matchs : les 50% restants (`a_verifier`, sans événement connu) ont un `whistle_score` nettement plus élevé que les tirs et souvent que le vrai KO — suggère des sifflets isolés (fautes, hors-jeu) confondus avec un KO.
- 🟡 **HYPOTHÈSE non retenue après vérification** : "RF préfère systématiquement un instant légèrement après le KO" — infirmée en détail match par match (décalages temporels dispersés entre -362s et +297s, pas un motif cohérent de quelques secondes).
- 🟢 **DÉMONTRÉ** : `pct_immobile` corrèle positivement avec `proba_rf` sur 9/9 matchs globalement, mais vaut `0,000` au vrai KO sur 7/9 matchs — corrélation globale ≠ signal utile au moment précis recherché (piège méthodologique identifié et évité).

### 5.4. Résultats historiques `kickoff_score_hybride.py` (`synthese_indicateurs.csv`, 4 matchs : Andrimont, Juprelle, Franchimont, matchTest)

🔴 Aucune des 3 composantes (géométrie/transition/ballon) ne classe fiablement le vrai KO — rangs très inconsistants (2 à 94 sur ~100 candidats scannés).

🔴 **Transition = 0,000 au vrai KO sur 3/4 matchs** (Juprelle, Franchimont, matchTest) — `n_immobile_at_start=0`.

🔴 Contrôle négatif (fenêtre +400s, jeu normal) : seulement 6/12 comparaisons (exactement 50%) favorisent le vrai KO sur les 3 composantes — pas de spécificité démontrée sur cet échantillon de 4 matchs.

### 5.5. Re-test aujourd'hui avec le tracking corrigé (Andrimont, Raeren, Wanze)

🟢 **DÉMONTRÉ, hypothèse initiale réfutée** : le score `Transition=0,000` sur Andrimont n'est **pas** causé par la fragmentation de tracking (13 pistes non-fragmentées disponibles) — cause réelle : `immobile_threshold_px_s=8.0` beaucoup trop strict (aucune piste ne descend sous ce seuil avant le KO).

🟡 **HYPOTHÈSE, 2 points de données fiables seulement** : distribution des vitesses "avant KO" très différente entre matchs — Raeren `1,7-28,5 px/s`, Andrimont `13,9-72,6 px/s`. Pas de seuil fixe unique évident ; possible besoin de normalisation par match.

🔴 **Anomalie non expliquée** : Wanze montre des vitesses `58-317 px/s`, implausibles. Hypothèse "espacement de frames irrégulier" testée et **réfutée** (Raeren a le même motif d'espacement et donne un résultat plausible). Cause réelle non identifiée — possible changement de plan caméra à cet instant.

### 5.5bis. Anomalie Wanze — résolue

🟢 **DÉMONTRÉ** : cause identifiée — **zoom caméra** (pas un panoramique) entre t=31s et t=41s (local). Nombre de joueurs visibles chute de 27 à 12-13, portée horizontale des positions se rétrécit de 3437px à 1219px en 10s. La compensation de mouvement caméra actuelle (`_camera_motion_series`) ne corrige qu'une translation (panoramique) — un zoom déplace différemment les points selon leur distance au centre, ce que la correction par médiane ne peut pas capturer. Limite connue du prototype actuel, pas un bug de calcul. À traiter comme un chantier séparé (détection/exclusion des fenêtres de zoom, ou modèle de compensation plus riche) si cette piste est reprise.

Confirmé visuellement par 3 captures d'écran fournies : plan large avant/au KO (`16:26`), recadrage net 5s après (`16:31`) — cohérent avec un système de caméra automatique (VEO) qui suit le ballon dès la mise en jeu.

### 5.5ter. Hypothèse "rétrécissement de plan = signal de KO" — testée, résultat mitigé

🟡 **HYPOTHÈSE PARTIELLEMENT CONFIRMÉE, pas un signal fiable seul** : même métrique (portée horizontale des joueurs détectés) testée sur les 3 matchs autour de leur KO connu.

| Match | Portée avant KO | Portée après KO | Rétrécissement net ? |
|---|---|---|---|
| Wanze | 1963-3723px | 820-1579px | ✅ Très net |
| Andrimont | 1565-2023px | 1007-1275px | ✅ Modéré |
| Raeren | 1566-1856px | 1371-2659px | ❌ Absent |

Présent sur 2/3 matchs, absent sur Raeren. **Ne peut pas être utilisé comme signal unique** — cohérent avec la conclusion générale de tout `kickoff_score_hybride.py` : aucune composante isolée testée à ce jour n'est fiable seule sur l'échantillon disponible.

### 5.6. Décision actée

Ne pas modifier le RF ni `kickoff_score_hybride.py` avant d'avoir élucidé l'anomalie Wanze et calibré le seuil d'immobilité sur un échantillon plus large que 2-3 matchs.

---

## 6. Bug `tracker_id` — résolu (cause réelle enfin identifiée)

### 6.1. Le vrai bug, après des semaines d'investigation depuis Goe

🟢 **DÉMONTRÉ, corrigé, testé sur 2 matchs (Raeren, Andrimont)** : la collision de `tracker_id` observée depuis l'investigation Goe initiale (jusqu'à 65-93% des frames selon les matchs) **ne venait pas de `deep_sort_realtime`**, contrairement à ce qu'on pensait après déduction logique sur `is_confirmed()`.

**Cause réelle** : dans `vision/tracker.py::Tracker.update()`, après `reid.process()` (qui définit déjà correctement `tracker_id` = vrai id DeepSort dédupliqué), une boucle **écrasait** `tracker_id` avec l'`id` de fusion couleur de `PlayerReID` :
```python
r["tracker_id"] = r.get("id", r["player_id"])  # <- BUG : ecrase le bon tracker_id
r["id"]         = r["player_id"]
```
Si `PlayerReID` fusionnait à tort deux pistes DeepSort distinctes sous le même `reid_id` (couleurs de maillot similaires), cette ligne leur attribuait ensuite le **même** `tracker_id` — créant un faux doublon attribué à tort à DeepSort. Toute l'investigation précédente (Goe, "duplication de tracker_id", collisions 65-93%) mesurait ce champ **déjà corrompu** par ce bug.

**Correctif** : suppression de la ligne d'écrasement, `tracker_id` n'est plus modifié après `reid.process()`.

**Validation** : mesuré à 3 points du pipeline (compteur interne `get_dedup_stats()`, juste après `process_video()`, juste avant sauvegarde finale) sur Raeren et Andrimont — `0,0%` de doublons partout, sur les vraies données sauvegardées, pas seulement un compteur interne.

### 6.2. Méthode de diagnostic qui a permis de le trouver

Ajout d'un marqueur de version explicite (`version_marker.json`, écrit au même moment et dans le même dossier que les fichiers de données) pour éliminer toute ambiguïté sur la fraîcheur des fichiers analysés — recommandé de garder cette pratique pour tout futur diagnostic de ce type.

---

## 7. `kickoff_detector.py` — statut LEGACY / OBSOLETE, à ne plus appeler en production

### 7.1. Deuxième confirmation indépendante de l'échec sur Andrimont

| Run | KO réel | KO choisi par le resolver | Erreur |
|---|---|---|---|
| Andrimont #1 (45 min) | 307,5s | 1193,5s | +886s |
| Andrimont #2 (10 min, clip différent) | 307,5s | 644,6s | +337,1s |

🟢 **DÉMONTRÉ** : le comportement défaillant (choix systématique du "dernier groupe candidat", `groups[-1]`) n'est pas un artefact isolé — confirmé deux fois, sur deux extractions différentes du même match, jamais reproduit sur Raeren ni Wanze (où le même mécanisme fonctionne correctement).

### 7.2. Confirmation que `goal_posthoc.py` n'est pas en cause

Sur le run #2 (10 min) : `goal_posthoc` détecte 8 buts (0,8/min, cohérent avec Raeren 0,7/min — pas d'emballement). Validation Gemini conserve 7/14 candidats (taux normal). **La perte massive RAW→CLEAN (4034→40 événements) est causée à 99,5% par le kickoff resolver** (`4014 événements supprimés` comme "pré-match", d'après le log lui-même) — pas par `goal_posthoc`, contrairement à l'hypothèse formulée précédemment sur le run #1.

### 7.3. Décision actée

`kickoff_detector.py` est désormais catégorisé **LEGACY / OBSOLETE** pour l'architecture V5.2 — créé avant le système mining/RF/validation actuel, son heuristique `groups[-1]` est explicitement documentée comme provisoire dans le code lui-même. Le laisser agir en amont, de façon destructive (suppression physique des événements pré-offset), est incompatible avec V5.2 : il peut supprimer exactement les données dont le nouveau système a besoin pour trouver le vrai KO.

**Ne pas supprimer le code immédiatement** — d'abord localiser précisément ses points d'appel dans `process_video()`/`pipeline.py`, puis désactiver son effet destructif sans casser le reste du pipeline (§8).

---

## 8. Prochaine étape — désactiver `kickoff_detector.py` proprement, puis nouvelle sélection KO

Architecture cible : `tracking → événements bruts → mining des candidats → RF/clustering → validation → KO final`, en utilisant la vérité terrain des 9 matchs déjà collectée (voir `*_df_mine_complet.csv`, §5.2-5.3) pour améliorer la sélection — sans nouveaux runs coûteux, ces données sont déjà disponibles.

---

## 9. Audit `kickoff_detector.py` sur 6 matchs restants — état à date

### 9.1. Méthode

Notebook allégé (`notebook_6matchs_kickoff_seul.py`) : tracking + détection kickoff seule, sans Gemini/stats. `extraire_kickoff_candidats.py` parse le log pour obtenir tous les groupes candidats + décision finale.

### 9.2. Tableau consolidé (7/9 matchs tranchés, 2/9 incomplets)

| Match | Catégorie d'échec | KO réel | Candidat kickoff_detector | Écart | Note |
|---|---|---|---|---|---|
| Raeren | 🟢 Correct | 404,8s | ~404,8s | faible | mécanisme `early_high_guard` |
| Franchimont (run 1) | 🟢 Correct | 318,0s | 315,9s | 2,1s | 7 groupes, dernier choisi |
| Franchimont (run 2) | 🟡 Non-déterministe | 318,0s | — | — | 4 groupes cette fois, aucun accepté (même run, résultat différent) |
| **Wanze** | 🔴 **Mauvaise sélection** (reclassé — voir §9.3) | 986,0s | 1296,2s | **+310,2s** | confiance affichée 0,75, mais faux |
| Andrimont #1 | 🔴 Mauvaise sélection | 307,5s | 1193,5s | +886s | bon groupe jamais confirmé dans la liste |
| Andrimont #2 | 🔴 Mauvaise sélection | 307,5s | 644,6s | +337,1s | 2e run, même motif |
| P1Minerois | 🔴 Mauvaise sélection | 192,0s | 332,2s | +140,2s | bon groupe présent à ~178s (grp[2]/11), jamais choisi |
| Stembert | 🔴 Kickoff absent | 549-553s | — | — | 5 groupes, tous entre 73-186s, aucun proche du vrai KO |
| MineroisSter | ⚪ À refaire | 723,0s | — | — | clip testé à 600s au lieu de 900s, KO hors fenêtre |
| Juprelle | ⚪ Jamais testé | 548,0s | — | — | — |

### 9.3. Correction importante — Wanze reclassé

Initialement classé "🟢 Correct" par erreur de calcul (confusion dans l'addition offset+début de clip). Recalcul correct : `950 (début clip) + 346,2 (offset) = 1296,2s`, à comparer au KO réel confirmé `986,0s` → écart **+310,2s**, pas une réussite.

**Point clé retenu** : Wanze montre qu'une confiance affichée élevée (`players_conflict, conf=0,75`) ne garantit rien face à la vérité terrain — le resolver doit être jugé sur l'écart réel, jamais sur sa propre confiance déclarée.

### 9.4. Conclusion actée

Sur 7 matchs tranchés : **1 seul succès net (Raeren)**, 1 succès instable (Franchimont), **4 échecs** (Wanze, Andrimont ×2, P1Minerois, Stembert), dont 3 par "mauvaise sélection" (le bon candidat existe dans la liste mais `groups[-1]` ne le choisit pas) et 1 par "candidat absent" (aucun groupe ne se forme près du vrai KO).

**`kickoff_detector.py` confirmé LEGACY/OBSOLETE** — décision de §7.3 renforcée, pas remise en cause.

---

## 10. Investigation RF / whistle_score — contamination du dataset d'entraînement

### 10.1. Test décisif d'hier — `n_players` bat le RF, mais avec une réserve importante

Sur les 9 pools de mining actuels : trier naïvement par `n_players` croissant donne un rang moyen de **11,2**, contre **20,7** pour le RF entraîné. Tentative de combiner avec `activite_apres` : **11,8**, aucune amélioration — `n_players` porte l'essentiel du signal seul.

🟡 **IMPORTANT — ne pas conclure trop vite** : ce résultat prouve que `n_players` bat le RF *actuel* sur ces pools *actuels*. Il ne prouve pas que `n_players` est intrinsèquement le bon signal (voir §10.2, cette hypothèse est directement mise à mal).

### 10.2. Découverte majeure — contamination `v1_manuel` / `v2_mining` dans `features_ko_v2.csv`

**Structure du dataset** (533 lignes) :
```
source_dataset=v1_manuel  (25 lignes)  → 10 label=1, 15 label=0
source_dataset=v2_mining  (508 lignes) → 0 label=1 (aucun positif)
```

**100% des positifs historiques viennent de `v1_manuel`, 0% de `v2_mining`.** `source_dataset` est donc quasiment confondu avec `label` — contamination structurelle indépendante du RF lui-même.

**Test A (v1 vs v2, toutes lignes)** — séparation par feature (ratio écart/std) :

| Feature | Séparation v1/v2 |
|---|---|
| whistle_score | 🔴 1,56 (la plus forte) |
| n_players | 🔴 1,04 |
| pct_immobile | 🔴 0,81 |
| activite_apres | 🔴 0,69 |
| ball_speed_before | 🔴 0,68 |
| synchronisation_std_s | 🔴 0,67 |
| geo_separation | 🟠 0,29 |
| ball_speed_after | 🟠 0,22 |
| mean_speed_change_px_s | 🟠 0,21 |
| ball_score | 🟢 0,00 |
| stabilite_avant_s | 🟢 0,00 |

**Test B (KO vs non-KO, uniquement dans les 25 `v1_manuel`, contamination éliminée)** — AUC + rang :

| Feature | AUC | Rang moyen KO (/25) | Top-10 |
|---|---|---|---|
| **mean_speed_change_px_s** | **0,75** | 9,3 | 5/10 |
| pct_immobile | 0,62 | 12,0 | 5/10 |
| synchronisation_std_s | 0,60 | 12,5 | 4/10 |
| whistle_score | 0,53 (quasi aléatoire) | 12,5 | 4/10 |
| n_players | 0,32 (signal inversé, faible) | 15,7 | 1/10 |

**Réserve importante** : les valeurs individuelles des 10 KO pour `mean_speed_change_px_s` vont de **-1,5 (matchTest) à 16,9 (Raeren/Wanze)** — le signal n'est pas uniformément fort, porté par 3-4 valeurs extrêmes sur 10. `AUC=0,75` avec seulement 5/10 en top-10 = signal réel mais fragile, pas une solution propre à lui seul. n=25 = puissance statistique faible, "non significatif" ne veut pas dire "aucun effet".

**Test C (vrais KO 2026 vs distributions historiques v1/v2)** :

`whistle_score` du vrai KO actuel = **-0,04**, proche de `v2_mining` (0,21), très loin de `v1_manuel` (3,01) → confirme que le profil acoustique appris historiquement ne représente pas les KO actuels.

`n_players` du vrai KO actuel = 31,0, plus proche de `v1_manuel` (28,0) que `v2_mining` (49,0) → plus ambigu, ne pas trancher.

### 10.3. Cause racine retrouvée (recherche dans l'historique de conversations, session du 17/08)

**Comment `v1_manuel` a été constitué** : candidats choisis **à la main**, à l'origine du projet, **avant que `candidates_mining.py` existe**. Pas un algorithme — des timestamps pointés manuellement (probablement en regardant/écoutant la vidéo), puis les features ont été calculées rétroactivement sur ces instants.

**Cause exacte de l'écart `whistle_score`** : deux formules **toutes deux encore présentes dans le code actuel**, jamais harmonisées lors de la construction du CSV :
- `v1_manuel` → `whistle_score()` (précise, référence locale)
- `v2_mining` → `_whistle_score_at()` (approximative, référence globale)

Confirmé par un test de recalcul direct (session du 17/08) : le code actuel reproduit fidèlement les valeurs `v1_manuel` (ratios 0,92-1,21) mais pas du tout `v2_mining` (ratios 0,59 à -2,95, avec un changement de signe sur un candidat).

**Découverte supplémentaire, importante** : le mining historique (`v2_mining`, hors KO injectés) avait un recall mesuré comme mauvais à l'époque : `0/9 à ±3s`, `3/9 à ±5s`, `7/9 à ±10s`. Le mining **actuel** (nos 9 CSV du jour) a un recall quasi parfait (`0-11s` systématiquement). **`candidates_mining.py` a donc évolué entre la construction de `features_ko_v2.csv` et aujourd'hui** — le dataset d'entraînement est obsolète à deux niveaux indépendants : formule `whistle_score` non harmonisée ET version de mining dépassée.

### 10.4. Piste écartée — correction uniforme de `whistle_score`

Une correction (`refine_whistle_scores`) avait été testée sur 30 `random_state` différents (pas un seul) dans une investigation antérieure :
- Andrimont : amélioration robuste (30/30 tirages)
- Raeren : amélioration apparente avec 1 tirage, mais chevauchement total sur 30 tirages — **artefact de bruit, pas un effet réel**
- P1Minerois : négatif en moyenne sur 30 tirages (20/30 pires)

**Décision actée à l'époque, toujours valable** : ne pas appliquer cette correction uniformément.

### 10.5. État actuel et prochaine étape

🔴 **NE PAS réentraîner le RF sur `features_ko_v2.csv` tel quel** — dataset contaminé à deux niveaux (mélange de formules whistle_score + mining historique dépassé par rapport à aujourd'hui).

🟢 **Établi avec confiance** : le problème est dans la construction du dataset d'entraînement, pas uniquement dans le RandomForest lui-même.

🟡 **À faire avant tout réentraînement** : reconstruire un dataset propre avec (a) le mining actuel (`candidates_mining.py` version 2026, pas l'ancien), (b) une formule `whistle_score` unique et cohérente sur toutes les lignes, (c) idéalement plus de 10 positifs pour une puissance statistique correcte.

⚪ **Piste la plus prometteuse mais non confirmée** : `mean_speed_change_px_s` — AUC le plus élevé en environnement propre (0,75), mais fragile (n=25, signal porté par quelques valeurs extrêmes). À revalider sur un dataset reconstruit avant d'en faire une feature pivot.

### 10.6. CORRECTION IMPORTANTE — deux points non résolus, découverts après relecture croisée

**La conclusion "le problème est la contamination `whistle_score`" était prématurée.** Deux gaps sérieux, identifiés en confrontant ce document à une analyse indépendante :

**A. Jamais testé si la correction améliore réellement le classement.** Toute l'investigation §10.1-10.5 a *diagnostiqué* la contamination, mais n'a *jamais vérifié* que la retirer/corriger améliore le rang du vrai KO. Un test antérieur (autre conversation, sur Raeren) aurait montré que retirer les features contaminées (`whistle_score`, `stabilite_avant_s`) **dégrade** le classement (rangs 32, 32, 20) plutôt que l'améliorer — cohérent avec l'observation §10.5 que `mean_speed_change_px_s` seul n'est pas non plus une solution propre. **Interprétation prudente : même contaminé, `whistle_score` semble porter un peu de signal réel — le retirer brutalement perd plus qu'il ne gagne.** Ne pas conclure "corriger whistle_score" sans un test direct A/B sur les 9 pools actuels.

**B. Question plus fondamentale, non vérifiée : le dataset historique (`v1_manuel`/`v2_mining`) a-t-il lui-même été construit avec `kickoff_detector.py` actif et destructif ?**

`candidates_mining.py` dépend explicitement de `process_video()` en cache (confirmé dans son propre docstring, §25). Si les runs historiques qui ont produit `v2_mining` passaient par un `process_video()` où `kickoff_detector.py` supprimait déjà physiquement les événements/frames "pré-match" (exactement le comportement démontré aujourd'hui sur Andrimont — 16908 événements supprimés, incluant tout avant le faux offset de 893,5s) — **alors les candidats proches du vrai KO ont pu être détruits avant même que le mining ne les évalue, pour certains matchs historiques.**

Ceci offrirait une explication alternative, plus radicale, au mauvais recall du mining historique (`0/9 à ±3s, 3/9 à ±5s` — §10.3) : pas seulement "le mining a été amélioré depuis", mais possiblement "le mining historique n'a jamais eu la chance de voir le bon candidat sur certains matchs, parce qu'il était déjà supprimé en amont."

**Statut : mécanisme techniquement plausible et cohérent, mais non vérifié.** Nécessiterait de savoir précisément quand `v2_mining` a été construit et si `kickoff_detector.py` était actif/destructif à cette date — information non disponible actuellement.

**Priorité recommandée pour une prochaine session, avant toute reconstruction de dataset** : vérifier cette hypothèse B en premier — si confirmée, elle changerait fondamentalement l'interprétation de tout ce qui précède dans cette section.

---

## 11. Chemins vidéo des 9 matchs de référence — CONFIRMÉS, à réutiliser directement

```python
MATCHS = [
    {"nom": "Raeren", "video_source": "/kaggle/input/datasets/michaelziant/p4raeren/ster-b-raeren-b-2026-04-03_0.mp4", "ko_reel": 404.832},
    {"nom": "Wanze", "video_source": "/kaggle/input/datasets/michaelziant/wanzebas/ster-wanze-bas-oha-2026-02-28_0.mp4", "ko_reel": 986.0},
    {"nom": "Franchimont", "video_source": "/kaggle/input/datasets/michaelziant/p4franchimont/ster-b-franchimont-b-2026-04-09_0.mp4", "ko_reel": 318.0},
    {"nom": "Goe", "video_source": "/kaggle/input/datasets/michaelziant/stergoe/ster-b-goe-2026-03-20_0.mp4", "ko_reel": 1081.0},
    {"nom": "Andrimont", "video_source": "/kaggle/input/datasets/michaelziant/stertest/match-p4-andrimont-2026-04-26.mp4", "ko_reel": 307.5},
    {"nom": "MineroisSter", "video_source": "/kaggle/input/datasets/michaelziant/mineroisster/minerois-ster-2025-11-09_0.mp4", "ko_reel": 723.0},
    {"nom": "P1Minerois", "video_source": "/kaggle/input/datasets/michaelziant/p1minerois/p1-minerois-2026-03-28_0.mp4", "ko_reel": 192.0},
    {"nom": "Stembert", "video_source": "/kaggle/input/datasets/michaelziant/stembert/ster-b-stembert-2026-03-07_0.mp4", "ko_reel": 553.0},
    {"nom": "Juprelle", "video_source": "/kaggle/input/datasets/michaelziant/juprelle/ster-juprelle-2026-01-25_0.mp4", "ko_reel": 548.0},
]
```

Plus besoin de redemander ces chemins à chaque nouveau notebook — à copier directement.

---

## 12. Architecture expérimentale C — Recherche cascade Gemini direct sur frames vidéo

**Une TROISIÈME architecture, distincte des Architectures A (DINOv2+clustering, §-1) et B (mining+RF, §1-10).** Explorée en fin de session. **Statut final : VALIDÉE avec un résultat solide (9/9, écart max 5,2s)** — après plusieurs itérations ratées, documentées ci-dessous pour ne pas perdre le chemin parcouru.

### 12.1. Première tentative — machine à états A/B/C/X — ABANDONNÉE

**Principe testé** : classification A=pas prêt / B=potentiellement prêt / C=jeu en cours, en cascade de résolution `120s→60s→15s→3s→1s`.

**Résultat d'un scan observationnel pur (sans décision automatique) sur les 9 matchs** : le classificateur Gemini s'est montré fondamentalement instable sur cette tâche à état unique — `C` obtenu prématurément très loin de tout KO plausible (Franchimont : `C` dès `t=60s`, KO réel `318s`), `C` qui **régresse** vers `A` après plus d'une minute de jeu manifestement actif (Raeren), même timestamp donnant des réponses différentes selon le run malgré `temperature=0.0`.

**Décision actée à l'époque** : abandonner la machine à états à 3 états. Cette conclusion reste valable — ce n'est PAS la même architecture que celle validée en §12.2 ci-dessous.

### 12.2. Reconstruction — cascade à 2 questions séparées (Q1/Q2) — VALIDÉE

**Principe** : au lieu d'un état unique jugé de façon isolée, deux questions binaires séparées, chacune avec un rôle précis :
- **Q1** (« configuration compatible avec un coup d'envoi imminent ? ») — scan grossier `60s`, cherche une zone candidate
- **Q2** (« le match a-t-il déjà commencé ? ») — vérifie/confirme une fois `Q1` positif, puis recherche fine `15s→5s→1s` dans la fenêtre confirmée

**Modèle** : `gemini-3.1-pro-preview` (pas `gemini-2.5-flash`, utilisé par erreur dans toute la première tentative §12.1).

**Prompt Q1** : checklist stricte adaptée des principes déjà validés dans `detect_ko_vision.py` (vérifications préalables obligatoires, anti-hallucination géométrique, séparation par moitié de terrain), avec un **seuil** (3/4 critères, pas un ET strict — un ET strict à 5/5 donnait 0 `OUI` sur tout un match) plutôt qu'une exigence de perfection.

**Prompt Q2** — reconstruit après 2 régressions successives :
1. Version initiale (« jeu actif ? ») : trop stricte — un coup franc/corner (jeu réellement en cours, juste à l'arrêt) était classé comme "pas actif", faisant rater le KO de plus de 100s sur certains matchs.
2. Version corrigée trop largement (« configuration pas une formation de KO ? ») : trop permissive — acceptait à tort l'échauffement, l'entrée sur le terrain, l'animation d'avant-match (n'importe quelle scène de joueurs dispersés), avec une incohérence chronologique grave observée (MineroisSter : `OUI` à `420s` puis `NON` à `480s`, un retour en arrière logique impossible).
3. **Version finale** : base stricte de (1), avec corner/coup franc/touche/penalty **explicitement listés** comme comptant pour "match commencé", et un **signal du nombre de ballons** (plusieurs ballons visibles simultanément = échauffement, pas match réel — signal découvert comme très discriminant en pratique).

**Vote majoritaire** (jusqu'à 3 appels, arrêt anticipé) appliqué uniquement au point de confirmation `Q2` — pas à chaque checkpoint du scan grossier, pour maîtriser le coût. Rappel important : même le prompt `detect_ko_vision.py` très abouti montre une vraie variance d'échantillonnage sur images identiques — le vote n'est pas optionnel, même avec un bon prompt.

### 12.3. Bugs de robustesse trouvés et corrigés en cours de route

- **Trou de couverture** : après un candidat rejeté, reprendre le scan `60s` à `t_verif+60` sautait l'intervalle `[t_verif, t_verif+60]` — précisément là où tombait le vrai KO sur Raeren. Corrigé : reprise à `t_verif` exactement.
- **Aucun timeout configuré** sur l'appel Gemini — un appel bloqué (incident réseau) figeait le notebook sans aucun signe. Corrigé : timeout `60s` + retry généralisé (toute erreur, pas seulement timeout — `499 CANCELLED` observé en pratique, non couvert par un retry limité au seul `TimeoutError`).
- **`max_workers=1`** sur l'executor de threads : un appel bloqué monopolisait l'unique worker, faisant échouer en cascade tous les appels suivants même rapides. Corrigé à `8`.
- **Recherche fine dégénérée** : si le tout premier point testé dans la fenêtre confirmée était déjà `OUI`, la fenêtre s'effondrait à un seul point sans avoir cherché en arrière (résultats aberrants observés : `-303s`, `-373s`, `-836s` d'écart). Corrigé par une vérification de garde explicite avant l'affinage, qui abandonne proprement (signale une anomalie) plutôt que de rapporter un résultat silencieusement faux.
- **Redites de coût** : `t_bas` et `t_haut` étaient reredemandés à chaque palier de la recherche fine alors que leur valeur est déjà connue depuis le palier précédent (jusqu'à 3 appels économisés par match, sans changer le résultat).

### 12.4. Résultat final validé (9/9 matchs)

| Match | Écart |
|---|---|
| MineroisSter | +0,0s |
| Goe | +1,0s |
| P1Minerois | +1,0s |
| Stembert | +2,0s |
| Juprelle | +2,0s |
| Andrimont | +2,5s |
| Franchimont | +3,0s |
| Wanze | +5,0s |
| Raeren | +5,2s |

**9/9 détectés, écart maximal 5,2s, toujours positif (jamais en avance sur le vrai KO)** — comportement cohérent et sans exception. `225` appels Gemini au total pour les 9 matchs.

### 12.5. Avantage conceptuel par rapport à l'Architecture B (RF)

Ne demande jamais "lequel de ces N candidats est le KO ?" (tâche de classement difficile, où le RF échoue, rang moyen `20,7` — §10) — demande "sommes-nous avant ou après le début du match ?" à chaque instant testé (tâche binaire simple), en exploitant directement la chronologie plutôt qu'un score appris sur un dataset contaminé.

### 12.6. Stabilité confirmée — Run 2 (9/9, 7/9 strictement identiques au Run 1)

| Match | Run 1 | Run 2 | Identique ? |
|---|---|---|---|
| Raeren | 410s/+5,2s | 410s/+5,2s | ✅ |
| Wanze | 991s/+5,0s | 987s/+1,0s | écart 4s |
| Franchimont | 321s/+3,0s | 321s/+3,0s | ✅ |
| Goe | 1082s/+1,0s | 1082s/+1,0s | ✅ |
| Andrimont | 310s/+2,5s | 310s/+2,5s | ✅ |
| MineroisSter | 723s/+0,0s | 723s/+0,0s | ✅ |
| P1Minerois | 193s/+1,0s | 193s/+1,0s | ✅ |
| Stembert | 555s/+2,0s | 554s/+1,0s | écart 1s |
| Juprelle | 550s/+2,0s | 550s/+2,0s | ✅ |

**`9/9` détectés sur les deux runs, écart maximal `5,2s` identique, toujours positif (jamais en avance).** Les 2 seuls écarts entre runs (`Wanze` 4s, `Stembert` 1s) restent dans la même fenêtre de quelques secondes, cohérents avec la variance résiduelle de Gemini déjà documentée — pas un signe d'instabilité structurelle.

**Architecture considérée comme validée pour l'intégration.**

### 12.7. Optimisation de vitesse — parallélisation du scan Q1 — VALIDÉE après un bug d'interaction corrigé

**Piste retenue** : paralléliser le scan grossier `Q1` par lots de `6` checkpoints envoyés simultanément. Traitement des résultats toujours dans l'ordre chronologique — la logique de décision reste strictement identique, seul le temps d'attente change.

**Bug d'interaction trouvé au premier test** : combinée au retry robuste (§12.6bis — jusqu'à 6 tentatives possibles par checkpoint, sur 3 images différentes), la parallélisation pouvait demander jusqu'à `36` requêtes simultanées (6 checkpoints × 6 tentatives) contre seulement `8` workers disponibles. Résultat observé sur Goe : cascade de dizaines de timeouts en rafale, plusieurs checkpoints réellement perdus (`🛑 ÉCHEC TOTAL`) malgré le retry censé les protéger — le système s'auto-saturait.

**Correctif** : `max_workers` porté à `32` (au lieu de `8`), pour donner assez de marge même dans le pire cas. **Re-testé sur Goe et Andrimont (les 2 matchs qui avaient montré des timeouts) : résultat identique aux runs séquentiels précédents (`1082s`/`+1,0s` et `310s`/`+2,5s`), un seul timeout rencontré sur tout le test, absorbé proprement sans cascade.**

**Pistes écartées, avec justification** :
- Préfiltrer avec les données de tracking (Architecture B, mining) : rejeté car (a) le tracking complet est lui-même lent (`30+` minutes sur un clip de 10 min) s'il ne tourne pas déjà pour d'autres besoins, et (b) risque de perdre la garantie de couverture exhaustive qui fait la force du scan `Q1` actuel.
- Modifier les prompts/seuils : explicitement écarté sans preuve d'un problème réel.

**Fichier** : `notebook_test_rigoureux_rapide_9matchs.py` — parallélisation + `max_workers=32` + retry robuste, tous validés ensemble. Version de référence à jour.

### 12.6bis. Retry robuste — protection contre la perte silencieuse d'un checkpoint

Ajout suite à un timeout observé sur un checkpoint potentiellement critique (`t=240s`) : un échec réel (pas une vraie réponse `NON`) était auparavant traité comme `NON` silencieusement, ce qui aurait pu casser la garantie de couverture exhaustive du scan si le vrai KO tombait exactement sur un point qui échoue.

**Mécanisme** : si les 2 tentatives à l'instant `t` échouent, essaie `t+1s` puis `t-1s` (chacun avec ses propres 2 tentatives) avant d'abandonner réellement — jusqu'à `6` appels sur `3` images différentes. Un échec total désormais signalé bruyamment (`🛑 ÉCHEC TOTAL`), jamais silencieux.

### 12.9. Contrat de production V1 — figé

Décidé avant intégration, pour éviter de découvrir ces trous en production (comme le trou de couverture `Q1` découvert plus tôt, §12.3) :

| Élément | Décision V1 |
|---|---|
| Premier KO | ✅ inclus |
| Deuxième KO (reprise après mi-temps) | ❌ hors périmètre — module séparé à construire plus tard (`detect_period_kickoffs`) |
| `Q1` | scan `60s`, "configuration compatible avec un coup d'envoi imminent ?" |
| `Q2` | confirmation dès `Q1` positif, "le match a-t-il déjà commencé ?" |
| Recherche fine | `15s → 5s → 1s` |
| Limite de recherche | dérivée de `half_duration_min` (voir §12.10) |
| Budget d'appels Gemini | garde-fou fixe (`MAX_GEMINI_CALLS`, ex. `300`) — protège contre le retry robuste qui peut multiplier les appels par checkpoint |
| Limite de temps réel (wall-clock) | garde-fou fixe (`MAX_WALLCLOCK_S`, ex. `900`) — indépendant du budget d'appels, protège contre une vidéo lente qui reste dans son budget d'appels mais bloque le pipeline en pratique |
| KO trouvé | `status="AUTO_CONFIRMED"` |
| Recherche épuisée sans KO | `status="NOT_FOUND"`, `reason` explicite (`SEARCH_LIMIT_REACHED` / `WALLCLOCK_LIMIT_REACHED`) |
| Erreur technique persistante | `status="ERROR"` |
| `MANUAL_REQUIRED` | ❌ retiré du V1 — pas de cas métier clairement défini pour le distinguer de `NOT_FOUND` |
| `confidence` | ❌ retiré du V1 — jamais calculé ni propagé dans la cascade actuelle, ne pas promettre un champ qui n'existe pas |
| Suppression des événements pré-KO | **uniquement après `AUTO_CONFIRMED`** — jamais sur une hypothèse, jamais avant d'avoir déterminé le KO (contrairement à l'ancien `kickoff_detector.py`) |
| RF (Architecture B) | ❌ **retiré du chemin de production KO** (décision actée §12.20) — conservé dans le projet pour référence/audit uniquement |
| `kickoff_detector.py` (legacy) | reste désactivé, import conservé pour référence/rollback facile |
| `MATCH_END` | inchangé, hors périmètre de ce chantier |

**Principe architectural** : le détecteur ne décide pas seulement "quel est le KO" — il décide aussi s'il a assez d'information pour l'affirmer. D'où les statuts explicites plutôt qu'un simple `kickoff_s = detect_kickoff(...)` avec `None` en cas d'échec.

### 12.10. Limite de recherche dérivée de la durée de mi-temps

**Décision** : plutôt qu'une constante fixe ou un choix direct de limite de recherche par l'utilisateur, celui-ci indique la **durée de sa mi-temps** (`half_duration_min`, ex. `45` pour senior, `30` pour jeunes) avant l'analyse — plus naturel pour lui qu'un paramètre technique. Le pipeline en dérive la limite de recherche :

```
max_search_min = (half_duration_min / 3) * 2
```

Exemples : `45min → 30min` de recherche max ; `30min → 20min`.

**Objectif explicite de cette formule** : ne jamais chercher plus longtemps que la mi-temps elle-même n'est longue — un garde-fou de bon sens, pas une science exacte.

**⚠️ Formule non validée empiriquement, à ajuster si besoin.** Les 9 matchs de référence sont tous en mi-temps `45min` (KO le plus tardif observé : Goe, `18min`, marge de `12min` avec cette formule). Jamais testée sur un vrai match à mi-temps courte. Risque identifié : les événements très jeunes ont parfois des cérémonies d'avant-match **plus longues** proportionnellement (présentations, parents), pas plus courtes — l'hypothèse pourrait être fausse dans le sens le plus défavorable pour les plus jeunes catégories (ex. mi-temps `20min` → seulement `13,3min` de marge). **À surveiller en premier si des `NOT_FOUND` apparaissent sur des matchs jeunes.**

### 12.11. Intégration complète — `pipeline.py` + `kickoff_gemini_cascade.py` construits

**Empreinte minimale respectée dans `pipeline.py`** : deux changements seulement.

1. Nouveau paramètre `half_duration_min=45` sur `run_pipeline()`, même pattern que `video_end_s` déjà existant.
2. Le bloc `KICKOFF` (section `if _match_data is None:`) appelle `detect_kickoff_gemini(video_path, max_search_s=...)`, mappe son résultat vers `_kickoff_offset`/`_kickoff_conf` — le reste du bloc (suppression pré-match, `apply_kickoff_offset`, `apply_kickoff_offset_frames`, `reset_pre_kickoff_state`) fonctionne sans une seule ligne modifiée.

**Module `analysis/kickoff_gemini_cascade.py` construit**, directement dérivé de `notebook_test_rigoureux_rapide_9matchs.py` (prompts, seuils, mécanismes de robustesse identiques — retry `t±1s`, garde fenêtre dégénérée, correctifs `t_bas`/`t_haut`, parallélisation `max_workers=32`). Ajouts propres à la production :
- **Budget d'appels** (`MAX_GEMINI_CALLS_DEFAUT=300`) et **wall-clock** (`MAX_WALLCLOCK_S_DEFAUT=900s`), vérifiés avant chaque lot du scan `Q1` et après chaque vote `Q2` — déclenchent `NOT_FOUND`/`CALL_BUDGET_REACHED` ou `NOT_FOUND`/`WALLCLOCK_LIMIT_REACHED`.
- **`status=ERROR`** géré explicitement pour clé API absente (`MISSING_API_KEY`), échec d'initialisation du client, ou exception non prévue pendant la recherche — jamais de plantage silencieux du pipeline appelant.
- **État de recherche encapsulé** (`_EtatRecherche`, pas de variables globales) — sûr si plusieurs matchs sont traités en parallèle par le pipeline.

**Testé (avec fonctions Gemini simulées, sans appel réel)** avant livraison :
- Scénario multi-rejets type Stembert (plusieurs faux candidats avant le bon) → `AUTO_CONFIRMED` correct.
- Budget d'appels épuisé → `NOT_FOUND`/`CALL_BUDGET_REACHED` correct.
- Fenêtre dégénérée (`Q2` déjà vrai au point de départ) → `NOT_FOUND`/`DEGENERATE_WINDOW` correct, pas de faux timestamp.
- Clé API absente → `ERROR`/`MISSING_API_KEY` correct.

**Non testé avec de vrais appels Gemini dans ce contexte de production** — seule la logique est validée, pas encore un run bout-en-bout sur un vrai fichier vidéo via `pipeline.py`.

### 12.13. Retry automatique sur NOT_FOUND

**Observation qui motive cet ajout** : premier test de production sur Andrimont (§12.11), 2 appels consécutifs au module, même match, même code : `1×` `NOT_FOUND`/`DEGENERATE_WINDOW` (48 appels, 514,9s), `1×` `AUTO_CONFIRMED`/`310s` (43 appels, 296,5s). Cohérent avec la variance résiduelle de Gemini documentée tout au long de ce chapitre — pas un bug, mais un phénomène à absorber en production plutôt qu'à éliminer complètement.

**`detect_kickoff_gemini_avec_retry(video_path, max_search_s, max_retries=3)`** — nouvelle fonction, appelle `detect_kickoff_gemini()` jusqu'à `3` fois, s'arrête dès le premier `AUTO_CONFIRMED`. **Ne retente jamais `ERROR`** (conditions plus probablement structurelles — clé API manquante, client mal initialisé — que transitoires ; retenter ne les résoudrait pas). C'est cette fonction, pas `detect_kickoff_gemini()` directement, que `pipeline.py` appelle désormais.

**Testé (fonctions simulées)** : récupération au 2e essai (reproduit exactement le cas Andrimont observé), `ERROR` jamais retenté, échec `NOT_FOUND` persistant s'arrête bien après `3` tentatives.

**⚠️ Coût dans le pire cas** : jusqu'à `×3` le coût d'un essai normal (appels + temps) si `NOT_FOUND` se reproduit systématiquement. À surveiller en production — si `NOT_FOUND` s'avère fréquent, ce multiplicateur peut devenir significatif, et il faudrait alors chercher la cause plutôt que de compter sur le retry pour la masquer indéfiniment.

**Lacune corrigée** : logging ajouté directement dans `kickoff_gemini_cascade.py` (préfixe `[KICKOFF_GEMINI]`, cohérent avec le style `print()` déjà utilisé dans `pipeline.py`) — niveau candidat/vote/décision, pas le détail de chaque appel individuel (trop verbeux pour la production). Couvre `_rechercher_kickoff` et `detect_kickoff_gemini_avec_retry` (visibilité sur quelle tentative a réussi/échoué). Re-testé après ajout : logique inchangée (scénario multi-rejets toujours correct).

### 12.15. Repositionnement du curseur — éviter de tracker l'avant-match inutilement

**Problème identifié** : dans l'intégration initiale (§12.11), `process_video()` (le tracking complet, l'étape la plus lente — `30+` minutes observées) tournait sur **toute** la vidéo, y compris l'avant-match, qui était ensuite supprimé par `apply_kickoff_offset`/`apply_kickoff_offset_frames`. Gaspillage potentiel significatif — sur nos matchs de référence, le KO va de `192s` à `1081s` (`18min`), jusqu'à `~20%` de temps de tracking gaspillé sur un match de `90min`.

**Deux approches envisagées** :
1. Découper la vidéo avec `ffmpeg` en un clip commençant au KO, passer ce clip à `process_video()`. **Écartée après investigation** : `video_path` est utilisé à `25` endroits différents dans `run_pipeline()`, dont certains ne doivent PAS changer (`load_memory_for_video(video_path)` — cache indexé par le chemin original, casserait si on bascule vers un clip temporaire) et d'autres DOIVENT rester cohérents entre eux en numérotation de frame (`GeometryExtractor`, `read_jersey_numbers` — liraient des frames désynchronisées du tracking si l'un utilise le clip et l'autre l'original).
2. **Retenue** : repositionner le curseur de lecture dans `process_video()` lui-même, sans jamais changer `video_path`. `cv2.VideoCapture` sait se positionner directement sur une frame précise (`cap.set(cv2.CAP_PROP_POS_FRAMES, N)`) sans lire ni traiter ce qu'il y a avant. Élimine tous les problèmes de l'approche 1 : un seul fichier, une seule numérotation de frame, cohérente partout.

**Implémentation** :
- `process_video()` (`main.py`) : nouveau paramètre `start_time_s=0.0`. Si `>0`, positionne le curseur (`cap.set(...)`) juste après l'ouverture de la vidéo, et initialise `frame_id` à cette position (pas à `0`) — reste cohérent avec `apply_kickoff_offset_frames`, qui fait déjà `f - offset_frames` pour tout remettre à `0` après coup. Aucun changement nécessaire dans cette fonction de rebasage, déjà prête à recevoir des frames qui ne commencent pas à `0`.
- `pipeline.py` : la détection Gemini (`detect_kickoff_gemini_avec_retry`) est déplacée **avant** l'appel à `process_video()` dans le chemin `elif not candidate_segments:` (le chemin principal). `_kickoff_offset` calculé à cet endroit est passé comme `start_time_s` à `process_video()`. Un indicateur (`_kickoff_deja_detecte`) évite une double détection pour ce chemin, tout en préservant le comportement existant pour le chemin `candidate_segments`/`analyze_segments` (non touché par ce chantier — utilise déjà sa propre logique d'extraction de clips via `ffmpeg`, pour un usage différent).
- Comme demandé explicitement : si `NOT_FOUND`/`ERROR`, `start_time_s=0.0` → analyse depuis le tout début de la vidéo, comportement conservateur inchangé.
- Lecture légère des métadonnées (`fps`, durée) via un `cv2.VideoCapture` ouvert puis immédiatement relâché — nécessaire car `total_frames`/`fps` ne sont plus disponibles avant `process_video()` une fois la détection déplacée en amont.

**⚠️ Périmètre non couvert** : le chemin `candidate_segments` (scan grossier + extraction de segments via `ffmpeg`, mécanisme préexistant et indépendant) garde son ancien comportement — détection Gemini toujours après le traitement, pas de gain de temps sur ce chemin précis. À traiter séparément si ce chemin s'avère être le chemin principal en pratique.

**Précision du seek — VALIDÉE empiriquement sur un vrai fichier** (Andrimont, H.264, `1920×1080`, `~30fps`, `192548` frames, `6424,7s`). `notebook_test_seek_precision_andrimont.py` : `cap.set(CAP_PROP_POS_FRAMES, 9290)` (correspondant à `start_time_s=310.0s`) suivi d'un `cap.read()`, comparé pixel par pixel à la même frame obtenue par lecture séquentielle pure (sans seek) depuis `5s` avant. **Résultat : `+0` frame d'écart, différence pixel maximale `0/255` — identique.** Pas de snap sur une keyframe voisine sur ce fichier. Les deux questions distinctes sont maintenant tranchées : `frame_id` reste celui de la vidéo originale (démontré par lecture de code) **et** le seek est précis en pratique (démontré empiriquement).

**Reste non testé** : `process_video()` complet (tracking + application de l'offset en aval) avec `start_time_s` — ce test valide le mécanisme de seek isolément, pas l'intégration bout-en-bout, qui nécessite l'environnement `ProjectAIMika` complet (YOLO/tracker/OCR, indisponible dans ce contexte). Prochaine étape prioritaire.

**Test bout-en-bout réel effectué** (VS Code, environnement complet) : confirme l'ordre correct (détection Gemini avant tracking), confirme `frame 9290` identique au test de précision isolé, et **révèle un cas `DEGENERATE_WINDOW` réel absorbé par le retry automatique** (`§12.13`) — tentative 1 échouée, tentative 2 `AUTO_CONFIRMED` à `310s`, exactement le comportement voulu. Première preuve en conditions réelles (pas seulement simulée) que le retry apporte une vraie valeur.

**Découverte en cours de route, non anticipée** : `TeamColorDetector(sample_frames=60)` calibre les couleurs d'équipe sur les `60` premières frames *traitées* — avant ce chantier, ces frames venaient de `t=0` (souvent l'échauffement, joueurs dispersés). Avec `start_time_s` positionné au KO, ces mêmes frames tombent désormais juste après le coup d'envoi, où les joueurs sont encore relativement groupés — amélioration de calibration non planifiée, effet de bord du changement.

**Ajustement demandé et implémenté** : `start_time_s` positionné à `KO - 10s` (pas exactement au KO), pour capturer la formation statique des deux équipes alignées juste avant le coup d'envoi — meilleure fenêtre de calibration couleur que des joueurs déjà en mouvement. **`_kickoff_offset` (utilisé par `apply_kickoff_offset`/`apply_kickoff_offset_frames`) reste inchangé** — seule la position de lecture change, pas ce qui est retiré des événements finaux (toujours tout ce qui précède le vrai KO, marge de `2s`). Bénéfice secondaire identifié : garde la possibilité de montrer le coup d'envoi lui-même dans un futur résumé de match. Cas limite (`KO` proche de `t=0`, ou `NOT_FOUND`/`offset=0`) géré par `max(0, kickoff_offset - 10)` — jamais de valeur négative. Testé unitairement sur 3 cas (KO normal, `NOT_FOUND`, KO très proche de `0`) avant livraison.

**⚠️ Problème séparé identifié, non résolu** : la couleur d'équipe détectée pour l'équipe en jaune (`(83,128,147)` en BGR) est nettement plus terne que le jaune vif visible sur une image réelle du match (Andrimont) — suspicion que le recadrage `torse` inclut le short bleu de cette équipe en plus du maillot. Non lié au timing de calibration (persiste même avec la fenêtre améliorée). À investiguer séparément : voir le découpage de `torse` dans `rendering/overlay.py`.

### 12.17. Intégration à l'étape de preview du site — KO détecté avant l'analyse complète, fallback manuel

**Découverte importante en explorant les templates du site** : l'interface avait déjà une étape légère dédiée (`step-detecting` → `step-config` dans `dashboard.html`/`app.js`) qui détecte les couleurs d'équipe **avant** que l'utilisateur ne configure le reste (noms d'équipes, sport, mode, position joueur) et **avant** de lancer l'analyse complète. Le vrai endpoint (`/api/detect-teams/<upload_id>` dans `app.py`) appelle `analysis/detect_teams_preview.py`, qui contient même une fonction `detect_teams_preview_rapide()` déjà pensée dans un esprit proche de notre cascade (3 frames à des timestamps calculés, vote majoritaire Gemini) — mais **volontairement à distance du KO**, avec un raisonnement documenté explicite : les joueurs regroupés au centre au coup d'envoi seraient plus difficiles à distinguer par équipe pour Gemini qu'en plein jeu, où ils sont dispersés. **Question tranchée empiriquement en §12.18 — ce raisonnement ne se confirme pas.**

**Ajouté à cette même étape** : la détection KO tourne maintenant en parallèle de la détection des couleurs, dans un `try` séparé (un échec de l'un n'affecte jamais l'autre) :
- `AUTO_CONFIRMED` → affiché à l'utilisateur (`✅ Détecté automatiquement à 5:07`), avec un bouton "Corriger" s'il veut le changer
- `NOT_FOUND`/`ERROR` → **jamais de valeur devinée** — un champ de saisie manuelle (`mm:ss`) est proposé, vide = analyse depuis le début (comportement inchangé)

**Bénéfice supplémentaire, au-delà de l'UX** : la valeur (détectée ou saisie) est transmise via un champ caché (`kickoff_s_precalcule`) jusqu'à `run_pipeline()`, qui **saute entièrement sa propre détection Gemini** si une valeur est déjà fournie — élimine un appel Gemini complet en double entre la preview et l'analyse complète.

**Bug trouvé et corrigé pendant le développement** : `if kickoff_s_precalcule:` traiterait `0.0` (cas légitime : preview échouée, analyser depuis le début) de la même façon que `None` (aucune info, relancer la détection) — les deux sont "falsy" en Python. Corrigé en `is not None`, avec `app.py` convertissant explicitement un champ de formulaire vide en `0.0` (pas `None`), pour que ce cas déclenche bien "analyser depuis le début" sans jamais redéclencher Gemini inutilement. Testé unitairement sur les 4 cas (vide, valeur détectée, saisie manuelle, valeur corrompue) avant livraison.

**Fichiers modifiés** : `app.py` (endpoint `/api/detect-teams`, route `/upload`, `run_analysis()`), `app.js` (`handleKickoffResult`, `toggleKickoffManuel`, `updateKickoffFromManuel`), `dashboard.html` (bloc `kickoff-wrapper`), `pipeline.py` (paramètre `kickoff_s_precalcule` sur `run_pipeline()`).

**⚠️ Non couvert** : la voie Celery (`celery_run_analysis`, définie dans un fichier tasks non fourni) reçoit bien `kickoff_s_precalcule` dans son dict de kwargs, mais je n'ai pas pu vérifier/modifier sa définition elle-même (fichier non fourni) — à confirmer qu'elle transmet bien ce paramètre à `run_pipeline()` en interne.

### 12.18. KO vs milieu de match pour la calibration couleur — TRANCHÉ, le KO gagne

**Test comparatif** sur Andrimont (KO détecté à `310s`, confirmé `5×` aujourd'hui) : même prompt Gemini exact (copié mot pour mot depuis `detect_teams_preview.py::_ask_gemini_colors`, aucune reformulation), seule la sélection des `3` frames change entre les deux stratégies — comparaison équitable.

- **Stratégie A (autour du KO)** : `KO-5s`, `KO+10s`, `KO+30s` → `3/3` frames cohérentes, jamais de confusion entre équipes.
- **Stratégie B (milieu de match, formule existante de `detect_teams_preview_rapide`)** : `durée/2-5min`, `+20min`, `+40min` → **une inversion complète équipe A/équipe B** sur une des `3` frames, et erreur sur la couleur du short dans `2` cas sur `3`.

**Vérifié visuellement** sur les 6 images réelles (short blanc pour l'équipe rouge, confirmé sur toutes les photos) : la stratégie B se trompe sur le short **`2/3` du temps** (`rouge` au lieu de `blanc`), contre **`1/3`** pour la stratégie A. Cause probable : les frames de milieu de match montrent des joueurs en mouvement, plus distants, parfois qui se chevauchent — plus difficile pour Gemini de lire un détail comme la couleur du short que sur des joueurs immobiles et bien séparés au KO.

**Conclusion** : le raisonnement documenté dans `detect_teams_preview_rapide()` ("les joueurs regroupés au KO sont plus difficiles à distinguer") **ne se confirme pas empiriquement** — c'est l'inverse qui s'observe. **Décision actée : utiliser notre détection KO précise comme source d'échantillonnage pour la calibration des couleurs**, pas la logique milieu de match (jamais branchée en production de toute façon, §12.17).

**Prochaine étape immédiate** : adapter l'endpoint `/api/detect-teams` pour échantillonner les couleurs autour du KO déjà détecté (au lieu de `bootstrap_duration=30.0` fixe, qui sample encore l'avant-match/échauffement), en réutilisant le prompt `_ask_gemini_colors` déjà existant et validé — pas besoin d'en écrire un nouveau.

**✅ Confirmé en conditions réelles** (`test_detect_teams_colors_autour_ko.py`, vrais appels Gemini, environnement `ProjectAIMika` complet) : `170s` au total (KO + couleurs), `3/3` frames exploitées, résultat `team_0=rouge/blanc`, `team_1=jaune/bleu` — identique à la vérité terrain et au test manuel précédent. 6e confirmation indépendante du KO sur Andrimont (`310s`), toujours la même séquence de rejets. Format de sortie (`color_bgr`, `color_name`, `short_bgr`, `preview_frame`) directement exploitable par `app.js` sans changement frontend.

### 12.19. Bug critique trouvé et corrigé — `video_end_s` inefficace + confusion `frame_id`/`analyzed`

**Deux découvertes liées, faites lors du premier vrai test bout-en-bout complet** (`run_pipeline()` réel sur Andrimont, VS Code, avec `video_end_s=400` pour raccourcir le test).

**1. `video_end_s` ne coupait qu'*après* le traitement complet, pas avant** — même défaut symétrique à celui déjà corrigé pour le début (`§12.15`) : `pipeline.py` filtrait `events`/`frames_data` une fois le tracking terminé sur la vidéo **entière**, sans jamais raccourcir le tracking lui-même. **Corrigé** : `process_video()` accepte maintenant `end_time_s`, avec une vérification `frame_id >= end_frame` en tête de boucle, avant même `cap.read()` — même principe que `start_time_s`. `pipeline.py` transmet `video_end_s` à ce nouveau paramètre. Le filtrage après coup reste en place par sécurité, mais ne fait plus le travail seul.

**2. Bug plus sérieux, révélé par le test complet** : `718` events détectés par le tracking, puis **`0` restants après application de l'offset KO — tous classés "pré-match" et supprimés à tort**. Cause : dans `main.py`, `ball["frame"] = analyzed` (ligne du patch historique pour que `detect_events()` calcule `current_time`) utilisait `analyzed`, un **compteur relatif à la session** (toujours réinitialisé à `0`, indépendant de `start_time_s`) — pas `frame_id`, la position **absolue** dans la vidéo originale. Avec `start_time_s=300s`, la première frame analysée avait `frame_id≈8991` mais `analyzed≈0`, donnant `current_time≈0s` au lieu de `~300s`. `apply_kickoff_offset` (seuil `310s`) voyait alors tous les temps proches de `0`, bien en dessous du seuil → tout supprimé comme pré-match.

**Un second endroit avec exactement le même bug** : `ocr.read_all(frame_orig, tracked, frame_id=analyzed)` — même confusion, transmettait `analyzed` au lieu de `frame_id` à la lecture OCR des numéros de maillot.

**Bug préexistant, resté invisible jusqu'à `§12.15`** : avant l'introduction de `start_time_s`, `start_frame` valait toujours `0`, donc `frame_id` et `analyzed` coïncidaient par pure coïncidence — aucune conséquence visible. Notre optimisation n'a pas *créé* ce bug, elle l'a *révélé*.

**Corrigé** : `ball["frame"] = frame_id` et `ocr.read_all(..., frame_id=frame_id)` — les deux utilisent maintenant la position absolue, cohérente avec `apply_kickoff_offset`/`apply_kickoff_offset_frames`. `frames_data` lui-même utilisait déjà `frame_id` correctement (vérifié, pas de 3e occurrence du bug).

**Testé** : simulation numérique du scénario exact (`start_frame=8991`, `fps=30`) — avant le fix, `time=0,0s` (bug reproduit) ; après le fix, `time=299,7s` (cohérent avec `start_time_s=300s`).

**⚠️ Portée du bug** : concerne uniquement les runs avec `start_time_s>0`, donc tous les runs de production depuis l'intégration `§12.15` (tous les matchs, puisque `start_time_s` est systématiquement `>0` dès qu'un KO est détecté). **Tous les runs de production effectués entre l'intégration de `§12.15` et cette correction ont potentiellement perdu leurs events.** Aucun run de production réel n'avait eu lieu avant cette correction (seulement des tests), donc pas d'impact utilisateur réel à ce stade — mais confirme l'importance d'avoir fait ce test complet avant un vrai déploiement.

**✅ Correction validée en conditions réelles** (2e run complet `run_pipeline()` sur Andrimont, VS Code, même paramètres `start_time_s=300s`/`end_time_s=400s`) : `10 events après correction` (au lieu de `0` avant le fix) — le bug est bien résolu. `[VIDEO_END_S] Coupure forcée à 400s | events 10→10` confirme aussi que le fix `end_time_s` fonctionne correctement (aucune perte supplémentaire, tous les events étaient déjà dans la fenêtre voulue). Premier event survivant à `t=63,4s` (relatif au KO) — cohérent avec la fenêtre de tracking volontairement raccourcie `[KO-10s, KO+90s]` pour ce test, pas une anomalie. Le filtrage qualité en aval (`10→1` event final) reflète le comportement normal sur un échantillon aussi court, pas un problème lié à `§12.19`.

### 12.20. Décision architecturale finale — RF retiré du chemin de détection du KO

**Actée après validation complète de l'Architecture C (Gemini cascade)** : `9/9` × 2 runs indépendants, écart max `5,2s`, jamais en avance, confirmé en conditions réelles (KO + couleurs + tracking).

**Ce que le RF résolvait, et pourquoi ce n'est plus pertinent** : le RF classait une liste de candidats produits par le mining (`Mining → N candidats → RF → classement → choix`). Mais le mining trouve déjà un candidat proche du vrai KO sur les 9 matchs — c'est spécifiquement le **classement** par le RF qui échouait (rang moyen `20,7`, le bon candidat dans le top 10 seulement `3/9` fois, §10). **La cascade Gemini ne pose plus ce problème du tout** : elle n'a jamais besoin de classer une liste de candidats, elle exploite directement la chronologie de la vidéo (scan séquentiel, confirmation, affinage) — un problème structurellement différent et plus simple que celui pour lequel le RF avait été conçu.

**Décision** :

| Élément | Décision V5.2 |
|---|---|
| `kickoff_detector.py` | ❌ désactivé / legacy (déjà acté, §7) |
| `candidates_mining.py` | ❌ plus nécessaire **pour le KO** — réserve : pas nécessairement inutile pour tout le projet, aucune preuve actuelle qu'il sert ailleurs, mais pas de raison de le supprimer physiquement pour autant |
| `ko_features.py` | ❌ plus nécessaire pour le KO |
| RandomForest | ❌ retiré du chemin KO |
| Cascade Gemini Q1/Q2 | ✅ seul détecteur KO en production |
| `MATCH_END` | ✅ inchangé, hors périmètre |
| RF / mining historique | 📦 conservé dans le projet (fichiers non supprimés) pour référence/audit et comparaison future — juste retiré du **chemin de production** |

**Architecture simplifiée en résulte** :
```
AVANT (Architecture B)                    V5.2 (Architecture C)
Vidéo                                     Vidéo
  ↓                                         ↓
Tracking complet                          Gemini Q1/Q2
  ↓                                         ↓
Mining                                    KO AUTO_CONFIRMED ?
  ↓                                         ├── NOT_FOUND → analyse depuis 0
RF                                         ├── ERROR      → arrêt propre
  ↓                                         └── OUI        → démarrage au KO (§12.15)
KO                                              ↓
  ↓                                         Tracking
Nettoyage                                       ↓
  ↓                                         événements → identité → statistiques
Analyse
```

**Cohérent avec le principe central de V5.2** : ne pas fabriquer une réponse quand les preuves sont insuffisantes (`NOT_FOUND` explicite plutôt qu'un mauvais classement RF silencieusement accepté).

**Ce qui ne change pas** : aucun fichier RF/mining n'est supprimé physiquement — seulement retirés du chemin appelé par `pipeline.py`. Permet une comparaison future si besoin, sans code mort actif en production.

### 12.21. Prochaine étape

- **`§12.15`+`§12.19` désormais pleinement validés en conditions réelles** — KO, seek, couleurs, application de l'offset sur les events, tous confirmés sur Andrimont. Prochaine étape naturelle : un run complet **non tronqué** (sans `video_end_s`) sur un match entier, pour valider le comportement sur la durée complète (highlights, PDF, etc. — désactivés par `audit_mode` dans tous les tests jusqu'ici).
- Mettre à jour le contrat de production/documentation pour retirer formellement le RF du chemin principal (§12.20) — pas de changement de code supplémentaire nécessaire, juste refléter la décision dans les endroits qui mentionnaient encore le RF comme fallback.
- Étendre l'optimisation §12.15 au chemin `candidate_segments`, si ce chemin s'avère fréquent en pratique.
- Mesurer le taux réel de `NOT_FOUND` sur un échantillon plus large que 9 matchs, pour savoir si `max_retries=3` est suffisant ou s'il faut ajuster.
- Surveiller particulièrement les matchs à mi-temps courte (formule §12.10 non validée empiriquement) et les cas `NOT_FOUND` en général — vérifier que la raison (`reason`) correspond bien à un vrai cas limite, pas un bug.
- Fichiers de référence : `analysis/kickoff_gemini_cascade.py` (module de production, avec retry), `main.py` (`process_video()` avec `start_time_s`), `pipeline.py` (intégration, détection avant tracking, RF retiré du chemin), `notebook_test_rigoureux_rapide_9matchs.py` (base de développement/tests).

---

## 13. Extension — fin de 1ère mi-temps + KO2 (exploratoire, non intégrée en production)

### 13.1. Méthode — caractériser avant de coder, comme pour le KO1

Avant toute conception de fenêtre de recherche ou de prompt, vérité terrain rassemblée sur les 9 matchs de référence (fournie manuellement) : fin de 1ère mi-temps, KO2, fin de match.

| Match | Durée 1MT | Pause | Durée 2MT |
|---|---|---|---|
| Andrimont | 45,3min | **8,3min** | 43,7min |
| P1Minerois | 45,9min | 15,4min | 48,0min |
| Raeren | 45,8min | 11,9min | 45,3min |
| Franchimont | 46,3min | 10,3min | 46,9min |
| Goe | 46,7min | **8,3min** | 50,3min |
| Stembert | 48,0min | 11,4min | 50,9min |
| Wanze | 50,2min | 12,3min | 49,1min |
| Juprelle | 47,3min | 12,6min | 49,8min |
| MineroisSter | 45,9min | 13,2min | 48,7min |

**Découverte importante avant même de coder** : l'hypothèse initiale de fenêtre `KO2` (`Fin1MT+10min` à `+20min`) aurait **raté `2` matchs sur `9`** (Andrimont et Goe, tous deux à `8,3min` de pause — pas des anomalies isolées, un vrai motif). Fenêtre corrigée à **`Fin1MT+5min` à `+20min`**, validée `9/9` sur ces données. Fenêtre `Fin1MT` elle-même (`KO1+2200s` à `+3300s`, soit `~36,7` à `55min` après KO1) validée `9/9` également.

**Note** : une valeur initialement fournie pour Goe (`103:21`) contenait une coquille, corrigée en `123:21` — sans cette correction, la durée de 2e mi-temps de Goe semblait anormalement courte (`30,3min`) ; corrigée, elle rejoint la plage normale (`50,3min`).

### 13.2. Architecture testée — Q3 (nouvelle) + Q1/Q2 (réutilisées)

- **Fin de 1ère mi-temps** : nouvelle question `Q3` ("signes de fin de période"), scan + confirmation de persistance + recherche fine — **mais recherche fine affinée entre `dernier_non` et `premier_oui`** (pas entre `premier_oui` et `+60s` comme pour KO1/KO2, puisque `Q3` sert aux deux bouts, contrairement à `Q1`/`Q2` qui sont deux questions distinctes). Deux bugs de logique trouvés et corrigés pendant le développement (mauvais sens d'affinage ; risque de boucle infinie sur la reprise après rejet — reprise doit se faire au point de confirmation, pas au candidat lui-même).
- **KO2** : cascade `Q1`/`Q2` **existante, réutilisée sans modification**, appliquée sur la fenêtre `Fin1MT+5min` à `+20min`.

### 13.3. Test sur Andrimont — résultat mitigé, prompt Q3 non encore fiable

Premier test réel : `Q3` a rejeté `2` faux candidats (`3228s`, `3408s`) avant d'épuiser toute la fenêtre sans confirmer le vrai moment (`3028s` réel). Diagnostic avec raisonnement demandé : révèle que **des enfants jouent de façon informelle sur le terrain pendant la pause** (confirmé par l'utilisateur : `5` joueurs, pas la même équipe), confondus par `Q3` avec du jeu actif. Une version `v2` du prompt (exclusion explicite de l'activité informelle + échauffement de joueurs/remplaçants) a été testée sur les points problématiques : **résultat non concluant**, `Q3 v2` ne corrige pas clairement le point qui coinçait le plus (`3468s`).

### 13.4. Investigation précision `ffmpeg -ss` — fausse piste écartée

En creusant l'incohérence à `3468s` (le raisonnement textuel de Gemini ne correspondait pas à l'image partagée par l'utilisateur), hypothèse envisagée : `ffmpeg -ss` (méthode d'extraction utilisée dans tous les notebooks `Q1`/`Q2`/`Q3` de la journée) pourrait être imprécis, contrairement à `cv2.set()` (déjà validé au pixel près, `§12.15`).

**Test en deux passes** : balayage grossier (pas `1s`, `±5s`) puis fin (pas `0,1s`, `±2s`) autour de `t=3468s`, comparant `ffmpeg -ss` à la référence `cv2.set()`. **Résultat : convergence nette en forme de "V" exactement à `delta=0,0s`** (diff minimale `10,05`, remontant symétriquement des deux côtés) — signature claire d'un bon alignement temporel, pas d'une imprécision de timing. Le résidu (`diff=10,05`, jamais nul) s'explique plus probablement par une différence de pipeline de décodage/compression JPEG entre `ffmpeg` et `cv2`, pas par un désalignement temporel.

**Conclusion : `ffmpeg -ss` n'est pas en cause.** L'incohérence observée à `3468s` est une vraie variance de raisonnement de Gemini sur une scène ambiguë (joueurs qui traversent le terrain pendant la pause — repositionnement ou échauffement, visuellement proche d'une mise en place de coup d'envoi), pas un bug d'extraction. Aucun changement nécessaire à la méthode d'extraction utilisée dans `Q1`/`Q2`/`Q3`.

### 13.5. État actuel — non intégré en production

**Ce chantier reste exploratoire**, contrairement à la cascade KO1 (`§12`), validée `9/9` × 2 runs avant intégration. Rien de ce chapitre n'est branché dans `pipeline.py` ou `kickoff_gemini_cascade.py`.

**Prochaine étape** : le prompt `Q3` doit encore être affiné et testé plus largement (idéalement sur plusieurs des 9 matchs, comme cela a été fait pour `Q1`/`Q2`) avant d'envisager une intégration. La question ouverte n'est plus l'extraction de frame (tranchée, `§13.4`), mais la formulation de `Q3` elle-même face aux scènes ambiguës (échauffement, activité informelle) rencontrées pendant les pauses.
