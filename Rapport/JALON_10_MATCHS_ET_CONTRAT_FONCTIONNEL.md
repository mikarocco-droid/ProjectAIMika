# SCOUTIA — JALON : ÉTUDE COMPLÈTE SUR 10 MATCHS MINÉS
## Fin de la phase "recherche de features", début de la phase "contrat fonctionnel"

**Statut** : phase d'expérimentation sur les features considérée comme
close pour l'instant. Ce document fige les conclusions établies et pose
la question suivante à traiter : le contrat fonctionnel du détecteur.

---

## 1. RÉSUMÉ EXÉCUTIF

Après mining complet des 10 matchs de référence (533 candidats, dont 10
vrais KO), combinaison géométrie + comportement collectif + audio, et un
test dédié sur la transition temporelle attente→mouvement, la conclusion
principale est :

**Le générateur de candidats fonctionne** (7-8/10 matchs ont le vrai KO
dans le top-5). **Aucune feature ou règle simple ne permet de départager
systématiquement les 5 premiers candidats.** Ce n'est pas un défaut de
réglage — plusieurs catégories d'événements (échauffement structuré, tir
arrêté avec sifflet) produisent des observations visuelles/audio
partiellement identiques à un vrai coup d'envoi sur une fenêtre de
quelques secondes. Une partie de l'ambiguïté est réelle, pas un manque de
feature.

**Décision prise** : ne plus modifier `ko_features.py`, les seuils, ni le
modèle sur cette base. Passer à la définition du contrat de sortie du
détecteur (candidat + confiance + raison d'ambiguïté) avant de décider si
le RandomForest actuel suffit ou si une couche de décision finale est
nécessaire.

---

## 2. DATASET FINAL

- **10 matchs**, tous minés en entier (pas seulement une fenêtre de 260s
  autour du KO), avec review humaine complète des candidats ambigus.
- **533 candidats au total**, 10 vrais KO (label=1).
- Catégories observées et leur poids dans le dataset :
  `shot` (46.9%), `echauffement` (12.4%), `contre_attaque` (10.7%),
  `jeu_ouvert` (7.5%), `touche` (7.5%), `rassemblement` (6.2%),
  `remise_en_jeu` (2.6%), `vrai_KO` (1.9%), `coup_franc`, `arret_de_jeu`,
  `corner`, `faux_depart`, `terrain_vide`, `penalty`, `autre`, `goal`
  (chacune <1%).
- Fichier de référence : `features_ko_v2.csv`.

### Bugs de fusion trouvés et corrigés en cours de route
- `mine_candidates()` régénère systématiquement le vrai KO comme candidat
  → doublon avec la ligne déjà présente dans le dataset v1 → dédoublonnage
  ajouté dans `merge_datasets.py` (tolérance 3s, garde la version v1).
- Écrasement silencieux de la colonne `categorie` par `label_final` (vide
  pour les lignes non soumises à review manuelle, dont le KO lui-même) →
  corrigé pour ne remplacer `categorie` que si elle n'existe pas déjà.
- `t` (v1) est une position **relative** à la fenêtre de 260s découpée à
  l'origine, alors que `t` (v2, mining) est un temps **absolu** dans le
  match → confusion corrigée en utilisant systématiquement `t_absolu`.

---

## 3. CE QUI A ÉTÉ VALIDÉ EMPIRIQUEMENT SUR CE DATASET

### 3.1 Importance des features (RandomForest, GroupKFold par match)
Stables entre folds, dans cet ordre : `whistle_score` >
`mean_speed_change_px_s` > `n_players` > `activite_apres` >
`pct_immobile`. `ball_score` (0.012) et `stabilite_avant_s` (0.019)
confirmés quasi inutiles avec la qualité de détection actuelle.

### 3.2 Rang du vrai KO par match (GroupKFold, RandomForest)
| Match | Rang |
|---|---|
| Andrimont | 1 |
| Wanze | 1 |
| MineroisSter | 2 |
| Raeren | 3 |
| Stembert | 3 |
| Franchimont | 4 |
| Juprelle | 4 |
| P1Minerois | 6 |
| matchTest | 7 |
| Goe | 8 |

**7/10 dans le top-5.** Top-1 global : 20-30% selon le modèle — chiffre
jugé peu informatif isolément (voir section 5).

### 3.3 Confusion normalisée par fréquence (quelle catégorie devance le
KO le plus souvent, relativement à son nombre d'occurrences)
`faux_depart` (25%) > `jeu_ouvert` (15%) > `echauffement` (10.6%) >
`remise_en_jeu` (7.1%) > `touche` (5%) > `shot` (3.6%) ≈
`contre_attaque` (3.5%) ≈ `rassemblement` (3%).

`shot`, malgré son poids énorme dans le dataset (46.9%), n'est presque
jamais confondu avec le KO — la géométrie/vitesse le distingue bien.

### 3.4 Structure des top-5 par match (pas de départage universel)
Aucune feature (sifflet, immobilité, géométrie, nombre de joueurs) ne
gagne systématiquement au sein du top-5 d'un même match. Deux familles de
concurrents récurrentes, structurellement différentes :
- **Échauffement domine le top-5** : Wanze, Stembert, Goe, P1Minerois
  (matchs avec pré-match filmé et agité).
- **Tir/faute domine le top-5** : Raeren, Juprelle, P1Minerois (sifflet +
  géométrie ponctuellement proches d'un KO).

### 3.5 Test de la transition temporelle attente→mouvement
Avec UNE règle unique (baseline de vitesse sur -20s/-8s, seuil
baseline+10px/s soutenu sur 6 secondes), sur les 10 vrais KO connus :
**8/10 détectés**, délai typique de +5 à +9s après le KO étiqueté
(cohérent avec le temps physique nécessaire pour que le jeu s'accélère
après le sifflet).

Les 2 échecs ont chacun une cause identifiée et vérifiée sur vidéo :
- **Goe** : deux sprints de joueurs avant le KO faussent la baseline de
  référence (sensée être "calme").
- **Franchimont** : un mouvement de replacement transitoire juste avant
  le sifflet, qui retombe avant le vrai départ.

**P1Minerois** (hors de la règle des 8/10 pour une autre raison) : le KO
est suivi d'une sortie en touche immédiate → activité post-KO
naturellement faible. Confirmé par les données (pic modeste à +5/+8s,
retour à la baseline dès +14s). **Ce n'est pas un problème de mesure —
c'est le déroulement réel du jeu qui limite structurellement le signal.**

**matchTest** : fenêtre de données insuffisante (KO à 8s, pas assez de
marge avant pour calculer une baseline) — limite de collecte, pas un
échec de méthode.

### 3.6 Vérification : `activite_apres` seul n'est PAS spécifique au KO
Test à coût nul sur les 533 lignes : la moyenne d'`activite_apres` pour
`vrai_KO` (36.1) est **plus basse** que pour `contre_attaque` (53.1),
`touche` (48.6), `coup_franc` (54.0), `remise_en_jeu` (46.8),
`jeu_ouvert` (43.5) et `echauffement` (42.2). L'activité absolue après un
candidat mesure "y a-t-il de l'action maintenant", pas spécifiquement "un
KO vient-il d'avoir lieu" — cohérent avec 3.5 (P1Minerois).
`mean_speed_change_px_s` (le delta avant/après, déjà dans le modèle) est
un proxy plus pertinent mais très bruité (écart-type ≈ moyenne).

**Conséquence** : une éventuelle feature `transition_mouvement_apres_s`
(délai précis avant hausse soutenue) n'a pas été calculée sur les 533
candidats — le test à coût nul ne permettait pas de conclure qu'elle
apporterait une information au-delà de `mean_speed_change_px_s`, et le
calcul complet (nouveau calcul sur 533 lignes) a été jugé prématuré tant
que cette incertitude n'est pas levée autrement.

---

## 4. CE QUI EST DÉFINITIVEMENT ÉCARTÉ (ne pas y revenir sans nouvelle donnée)

Repris et confirmé sur les 10 matchs, en plus de la liste déjà établie
plus tôt dans le projet (DINO/Claude, géométrie seule, ballon seul,
vitesse seule, rupture d'état seule) :
- Seuil de détection unique sur la vitesse (absolu ou relatif) : chaque
  ajustement corrige un match et en casse un autre (Raeren vs
  P1Minerois) — pas une voie qui converge.
- `activite_apres` comme signal autonome de KO : dépend trop de ce qui
  suit dans le match (touche rapide = faux négatif structurel).

---

## 5. LA VRAIE QUESTION QUI RESTE OUVERTE : LE CONTRAT FONCTIONNEL

Le chiffre "top-1 = 20-30%" n'est probablement pas la bonne métrique
finale. Il mesure "le système doit-il toujours mettre le KO en première
position", alors que ce que les données montrent, c'est que le problème
réel est plus proche de :

> "Réduire un match entier à quelques candidats plausibles, puis
> sélectionner le meilleur avec suffisamment de contexte — en signalant
> explicitement une ambiguïté quand elle existe réellement."

### Contrat fonctionnel minimal retenu (décision prise, pas encore implémentée)

Le pipeline doit fournir, pour un match analysé :

1. **Candidats** — quelques instants plausibles de coup d'envoi.
2. **Classement** — le candidat le plus probable en premier.
3. **Confiance** — à quel point le système est sûr.
4. **Ambiguïtés** — signaler explicitement lorsqu'un échauffement, tir,
   faute, etc. est suffisamment proche pour rendre la décision incertaine.
5. **Timestamp final** — un seul timestamp choisi par le pipeline
   lorsqu'il doit prendre une décision automatique.

**Règle de principe, non négociable** : le système ne doit pas
transformer artificiellement une ambiguïté réelle en une confiance
élevée. Une confiance élevée doit refléter une vraie absence d'ambiguïté
dans les données, pas une décision forcée.

### Question à trancher ENSUITE (pas avant d'avoir validé ce contrat)
Le RandomForest actuel (probabilités déjà calibrables en "confiance") et
les catégories déjà connues des concurrents du top-5 suffisent-ils à
produire cette sortie directement, ou faut-il une couche de décision
supplémentaire ? Une piste notée mais explicitement **non retenue pour
l'instant** (à re-formuler comme hypothèse de contrat avant tout code) :
exclure par construction les candidats situés avant un repère absolu de
"début réel du match", pour écarter l'échauffement sans dépendre d'une
feature qui doit le deviner.

**Pas de nouveau calcul, pas de nouvelle feature, pas de nouvelle
architecture tant que le contrat fonctionnel n'est pas validé par écrit.**

---

## 5bis. ÉTAT DU JALON (au moment de la clôture de cette phase)

| Élément | Statut |
|---|---|
| Features (`ko_features.py`) | Gelées |
| Dataset (533 lignes / 10 matchs) | Gelé |
| Mining | Terminé |
| Évaluation (GroupKFold, top-5, confusion) | Terminée |
| Transition temporelle | Étudiée, **pas intégrée** au modèle |
| Contrat fonctionnel V1 (candidat+confiance+ambiguïté+fiabilité) | Testé sur données existantes, limite connue (Raeren) diagnostiquée |
| Séquence pré-KO (A→F) | **Confirmée sur 9/9 matchs** (section 7) — **pas intégrée** au modèle |
| Modèle (RandomForest/LogisticRegression) | Non modifié |
| **Prochaine phase** | **Explorer comment combiner score RandomForest + position dans la séquence A→F** |

---

## 7. DÉCOUVERTE MAJEURE — SÉQUENCE PRÉ-KO CONFIRMÉE (A→F)

### Origine
Hypothèse : un vrai KO de début de match n'est pas un événement isolé mais
le dernier maillon d'une séquence quasi-invariante :

```
A — Entrée des joueurs (optionnelle)
B — Présentation au centre face au public (optionnelle)
C — Capitaines / choix de camp (quasi systématique)
   — Changement de côté (optionnel)
D — Rassemblement d'équipe / cri de guerre (très fréquent)
E — Mise en place, équipes en position, attente du sifflet
F — Coup de sifflet → départ collectif → MATCH
```

### Méthode
Contrairement au reste du dataset (labels dérivés des candidats minés
automatiquement), cette vérification a été faite par **observation
vidéo humaine directe**, match par match, sur les timestamps réels des
étapes A/I(ligne face supporters)/C/D(K)/E(M), indépendamment de tout
candidat généré par le pipeline.

### Résultat : ordre respecté sur 9/9 matchs, sans exception
| Match | Séquence observée |
|---|---|
| Andrimont | A(176) < C(209) < D(255) < E(306) < **KO(307)** |
| Franchimont | C(241) < D(257) < E(313) < **KO(318)** |
| Goe | A(956) < I(974) < C(976) < D(1000) < E(1073) < **KO(1081)** |
| Juprelle | A(405) < I(418) < C(450) < D(467) < E(520) < **KO(548)** |
| MineroisSter | A(521) < I(535) < C(567) < D(620) < E(673) < **KO(723)** |
| P1Minerois | A(70) < I(86) < C(116) < D(133) < E(178) < **KO(192)** |
| Raeren | A(286) < I(301) < C(310) < D(337) < E(400) < **KO(410)** |
| Stembert | A(450) < I(464) < C(472) < D(493) < E(530) < **KO(553)** |
| Wanze | A(823) < I(834) < C(862) < D(911) < E(973) < **KO(986)** |

Aucune inversion sur aucun match. Les étapes A et I sont parfois absentes
(caméra ne filmant pas ce plan) mais quand elles sont présentes, l'ordre
tient toujours. Ce qui varie fortement d'un match à l'autre, c'est la
**durée** de chaque étape (ex: E→KO va de 1s à ~50s selon les matchs),
jamais leur **ordre**.

**matchTest exclu** : caméra manuelle au bord du terrain (pas de recul
type Veo), aucune des étapes A/I/C/D n'y est filmée/applicable — cohérent
avec le traitement différencié déjà identifié pour ce type de caméra
ailleurs dans le projet.

### Confirmation croisée inattendue — le mystère Raeren est résolu
La fenêtre "rassemblement / cri de guerre" observée directement sur
Raeren (337s→356s) correspond **exactement** aux deux candidats
étiquetés à tort `shot` (t=337.25 et t=352.0, section 3.4/audit
précédent) qui battaient le vrai KO dans le classement RandomForest.
Ce n'étaient donc ni des tirs, ni du bruit de détection : c'était le
rassemblement d'équipe, confirmé indépendamment par observation vidéo.
Le classement du modèle n'était pas absurde — il captait une vraie étape
de la séquence, simplement une étape qui n'est pas le KO lui-même.

### Implication
Le RandomForest actuel évalue chaque candidat **isolément**, sans savoir
"combien d'étapes de la séquence ont déjà eu lieu avant lui". Un candidat
à l'étape D (rassemblement) peut légitimement ressembler à un candidat à
l'étape F (KO) sur les features instantanées (géométrie, sifflet) — la
différence n'est visible qu'en tenant compte du **contexte séquentiel**,
une information que le modèle actuel ignore entièrement.

### Ce qui n'est PAS encore fait
- Pas de nouveau calcul Kaggle, pas de nouvelle feature dans
  `ko_features.py`, pas de modification du RandomForest sur la base de
  cette découverte.
- La séquence A→F n'a été vérifiée que sur les timestamps déjà connus
  du vrai KO — pas encore testée comme un signal capable de repérer,
  sur un match inconnu, "à quelle étape de la séquence on se trouve"
  sans connaître déjà la réponse.



---

## 6. PRINCIPES MÉTHODOLOGIQUES CONFIRMÉS SUR CETTE PHASE
(en plus de ceux déjà établis dans le résumé complet du projet)

- **Toujours tester une hypothèse sur les données existantes avant
  d'investir dans un nouveau calcul** (le test à coût nul sur
  `activite_apres` a évité un calcul complet sur 533 lignes dont le
  résultat était incertain).
- **Comparer les features en relatif au sein d'un même match/groupe**,
  pas en absolu entre matchs — les échelles varient trop (vitesse,
  activité) d'un match à l'autre pour être comparées brutes.
- **Une baisse de score après enrichissement des données n'est pas une
  régression** — elle peut simplement signifier que l'évaluation
  précédente était biaisée par une couverture inégale entre matchs.
- **Documenter un jalon et s'arrêter d'itérer** dès qu'une conclusion
  stable émerge sur plusieurs tests indépendants, plutôt que de
  continuer à ajuster des seuils au coup par coup.

---

## 8. VALIDATION — SÉQUENCE + RANDOMFOREST COMBINÉS (filtre puis décision)

### Principe testé
Piste 1 retenue : au lieu de demander à RandomForest de classer un
candidat parmi TOUS ceux du match, on restreint d'abord aux candidats
ayant dépassé **toutes** les étapes connues de la séquence A→E (section 7)
avec un écart ≤60s depuis la fin de la dernière étape — puis on laisse
RandomForest départager **seulement** à l'intérieur de cette fenêtre
réduite, avec les probabilités déjà calculées (aucun réentraînement,
aucune nouvelle feature).

### Résultat
| Étape | Résultat |
|---|---|
| Candidats avant filtrage | 523 (9 matchs, matchTest exclu) |
| Candidats après filtrage par la séquence | **39** (>90% éliminés) |
| Vrai KO retrouvé exactement (écart <0.5s) | **8/9 matchs** |
| Échec | MineroisSter (écart 19s — `faux_depart` préféré au vrai KO) |

**8/9 exact, contre 2/10 avec RandomForest seul sur l'ensemble du
match.** La séquence ne rend pas RandomForest plus confiant en absolu
(les probabilités restent souvent faibles, 0.03 à 0.31) — elle réduit
drastiquement le nombre de candidats à départager, ce qui suffit à
rendre le problème résoluble dans la grande majorité des cas.

### Limite acceptée (décision explicite)
MineroisSter reste faux de 19s, confondant le vrai KO avec une instance
de `faux_depart` (attente) juste avant. **Décision : acceptable en l'état**
— erreur isolée à un seul match sur neuf, dans la même fenêtre de
quelques secondes que le vrai KO (pas une erreur grave au sens de la
section 5bis/gravité de l'erreur), et potentiellement résorbable avec
plus de données d'apprentissage à l'avenir. Pas de correctif spécifique
à coder pour ce cas maintenant.

### Limites méthodologiques explicites, non résolues
- Testé avec les timestamps de séquence **fournis manuellement** par
  observation vidéo directe — pas encore avec une détection automatique
  des étapes A/C/D/E.
- Le seuil de fenêtre (60s) a été choisi a posteriori pour couvrir le
  pire cas observé (MineroisSter) — pas encore validé comme généralisable
  à un 11e match inconnu.

### Prochaine phase engagée
**Détection automatique des étapes de la séquence** (A/C/D/E), pour ne
plus dépendre d'une annotation manuelle par match. À explorer en
réutilisant les signaux déjà disponibles (pics de géométrie, de sifflet,
catégories déjà minées) avant d'envisager tout nouveau calcul.

---

## 9. DÉTECTION AUTOMATIQUE DE D (RASSEMBLEMENT) — RÉSULTAT PARTIEL

### Test réalisé (aucune annotation manuelle, uniquement `categorie` déjà minée)
Fenêtre = candidats après le dernier `rassemblement` connu (±120s),
départage par les probabilités RandomForest déjà calculées.

**Résultat : 6/9 matchs exacts** (vs 8/9 avec les timestamps manuels
complets, vs 2/10 avec RandomForest seul sur tout le match).

### Diagnostic des 3 échecs
- **MineroisSter** (19s d'écart) : limite déjà connue et acceptée
  (attente pré-KO longue, confusion attente/départ — décision : pas de
  correctif spécifique, cf. section 8).
- **Raeren et Goe** : le **dernier** rassemblement réel (juste avant le
  KO) a été mal étiqueté `shot` par le mining (proximité à un événement
  `terminal_events`, cf. audit des 146 lignes "shot impossible avant
  KO"). La détection automatique ne voit donc que des rassemblements
  plus anciens, pas le dernier — d'où une fenêtre mal calée.

### Règle testée : exclure tout candidat avant le premier D/C détecté
Sans utiliser le KO (donc généralisable à un match inconnu) : sur les
146 lignes "shot/goal impossibles avant KO", **89 (61%) sont exclues**
par la règle "avant le premier rassemblement détecté". Les 57 restantes
se situent APRÈS le premier rassemblement mais AVANT le KO — parce que
le rassemblement réel n'est pas un bloc unique : les équipes se
regroupent, se dispersent, se regroupent plusieurs fois avant le vrai
départ. Un ancrage sur le premier D ne suffit pas ; c'est la détection
du DERNIER D (ou de C) qui manque.

---

## 10. CONCEPTION D'UN DÉTECTEUR DE C (CAPITAINES + ARBITRE)

### Pourquoi C plutôt que D comme ancrage
D (rassemblement) peut se répéter plusieurs fois et n'a pas de forme
unique. C (choix de camp avec les capitaines et l'arbitre) est décrit
comme quasi systématique dans tous les matchs et correspond à un motif
visuel beaucoup plus contraint (2-5 personnes, pas 11).

### Hypothèses testées et écartées
- **`n_players` bas pendant C** — **infirmé** sur données existantes :
  `n_players` compte toute personne visible dans le cadre (joueurs à
  l'échauffement sur les côtés compris), pas seulement les gens au
  centre. Reste élevé (24-55) pendant C sur tous les matchs vérifiés.
- **Distance à un point fixe du terrain (le centre)** — écartée avant
  même d'être testée : le cadrage caméra peut zoomer/se décaler (cas
  observé sur Franchimont, occasionnel, caméra suivant un ballon), donc
  le centre géométrique du terrain peut apparaître n'importe où dans
  l'image. Une position fixe en pixels n'est pas fiable (cohérent avec
  les soucis déjà documentés de calibration terrain via
  `GeometryExtractor`).
- **MineroisSter** : le point exact du rond central ne suffit pas — sur
  ce match, C se déroule sur la LIGNE du rond central, pas au point
  central. Confirme qu'une définition basée sur une distance précise à
  un point est trop fragile.

### Définition retenue (validée par observation visuelle sur plusieurs
matchs, PAS ENCORE testée par calcul)
> **C = un petit groupe isolé (2 à 5 personnes), contenant au moins une
> personne dont la couleur ne correspond à aucune des deux équipes
> détectées (l'arbitre), nettement séparé d'au moins un groupe de
> joueurs.**

Définition **indépendante de toute position fixe dans l'image** —
repose sur une disposition relative (isolement, taille du groupe,
couleur), pas sur des coordonnées terrain. Confirmée visuellement
robuste sur des cas variés : centre standard, sur la ligne de touche,
cadrage décalé par un mouvement de caméra.

### Ce qui reste à faire (pas encore fait)
- Aucun calcul Kaggle réalisé pour cette feature — seulement une
  conception validée par inspection visuelle d'images.
- Nécessiterait un nouveau calcul sur `frames_data` (clustering des
  positions de joueurs déjà trackées + `TeamColorDetector` déjà
  existant pour repérer la couleur "arbitre" = ni équipe A ni équipe B).
- Franchimont reste une exception documentée et acceptée : arbitre ne
  suivant pas la procédure standard, le motif C n'y existe probablement
  pas sous une forme détectable — à ne pas chercher à couvrir à tout
  prix.

---

## 11. VÉRIFICATION VISUELLE DE LA DÉFINITION DE C SUR 9 MATCHS

### Méthode
Avant tout calcul, la définition de C (section 10) a été confrontée à
une capture d'image réelle de l'instant C pour chacun des 9 matchs
(matchTest exclu, caméra non Veo). Objectif : vérifier que le motif
"petit groupe isolé + couleur arbitre distincte + séparation nette des
deux équipes" tient dans la pratique, avant d'investir dans un calcul.

### Résultat par match
| Match | Conforme au motif ? | Particularité observée |
|---|---|---|
| Andrimont | ✓ | Cas de référence : 3 personnes nettes (arbitre + 2 capitaines) |
| Franchimont | ✗ | Arbitre ne suit pas la procédure standard — exception assumée |
| Goe | Ambigu | Éclairage nocturne, groupe central moins net/plus dispersé |
| Juprelle | ✓ | Gardien-capitaine (couleur de toute façon distincte de l'équipe) |
| MineroisSter | ✓ | C se déroule sur la ligne du rond central, pas le point exact |
| P1Minerois | ✓ | 3 arbitres → groupe de 5 personnes |
| Raeren | ✓ | Cas de référence, net |
| Wanze | ✓ | 3 arbitres (dont un caché derrière un joueur sur l'image) |
| Stembert | ✓ | Cas de référence, net |

**7-8 matchs sur 9 confirment le motif.** Goe reste incertain (cohérent
avec ses difficultés déjà connues ailleurs dans le projet — seul match
avec un éclairage nocturne combiné à un groupe C peu net). Franchimont
reste la seule exception franche.

### Définition finale retenue, après vérification
> **C = un petit groupe isolé de 2 à 6 personnes (la taille dépend du
> nombre d'arbitres, 1 ou 3 selon les matchs), contenant une ou
> plusieurs personnes à une couleur ne correspondant à aucune des deux
> équipes détectées (arbitre(s), éventuellement le gardien), nettement
> séparé d'au moins un groupe de joueurs.**

Ajustement par rapport à la version initiale (section 10) : la
fourchette de taille passe de "2-5" à "2-6", et la couleur "hors équipe"
peut concerner plusieurs personnes simultanément (plusieurs arbitres),
pas une seule. Cet ajustement vient uniquement d'observations réelles,
pas d'un raisonnement a priori.

### Ce qui reste non vérifié
- Le calcul réel (clustering sur `frames_data` + `TeamColorDetector`)
  n'a toujours pas été fait — seulement une vérification visuelle
  humaine sur images fixes.
- Goe nécessitera une attention particulière lors du calcul (motif
  visuellement moins nettement dessiné que sur les 7 autres matchs
  conformes).

### Prochaine étape engagée
Passage au calcul Kaggle pour tester si cette définition, appliquée aux
positions de joueurs déjà trackées, retrouve effectivement l'instant C
sur un ou plusieurs matchs.

---

## 12. BUGS DE PRODUCTION TROUVÉS ET CORRIGÉS EN CHERCHANT C

Le test du détecteur de C a révélé plusieurs bugs réels dans le pipeline
de production (pas seulement dans les scripts d'analyse), au-delà de la
simple conception de C. Corrigés dans l'ordre où ils ont été trouvés :

1. **`rendering/overlay.py` — indexation négative NumPy.** `_extract_jersey_color`
   ne bornait `x2`/`y2` que par le haut (`min(w_f, x2)`), jamais par le bas.
   Une bbox partiellement hors cadre (trace fantôme du tracker, ex:
   `x2=-56`) déclenchait l'indexation négative de NumPy (`frame[:, 0:-56]`
   compte depuis la fin du tableau), produisant un crop de ~1865px de
   large au lieu de quelques pixels — corrompant la couleur extraite.
   Même bug trouvé et corrigé dans **`analysis/player_reid.py`**
   (`_extract_color` ET `_extract_embedding`) et dans les scripts
   d'analyse (`recalibrate_colors.py`, `detect_c_capitaines.py`).
2. **`vision/tracker.py` — aucune information de fraîcheur exposée.**
   Ajout de `time_since_update` (0 = vraie détection YOLO cette frame,
   >0 = position purement prédite par Kalman). Diagnostiqué visuellement
   sur Andrimont : des bbox "confirmées" flottant à plusieurs dizaines de
   pixels du vrai joueur pendant les phases denses (rassemblement).
3. **`overlay.py` — `get_team()` ne retournait jamais "aucune équipe".**
   Bug le plus important : `return 0 if d0 < d1 else 1` forçait toujours
   un choix binaire, même pour une couleur d'arbitre nettement différente
   des deux équipes. Rendait structurellement impossible la détection de
   l'arbitre via la couleur. Corrigé : retourne `None` si la couleur est
   au-delà de `UNCLASSIFIED_DISTANCE_FACTOR` (2.5×, calibré sur la
   dispersion intra-équipe réellement mesurée, pas un chiffre absolu
   inventé) fois la dispersion des deux centroïdes.
4. **Calibration TeamColorDetector sur du bruit.** La calibration se
   faisait sur les 60 premières frames analysées, sans vérifier leur
   contenu — sur un clip découpé depuis le début de la vidéo, ces frames
   tombent souvent avant que les joueurs soient en tenue de match.
   Corrigé : une frame ne compte pour la calibration que si elle contient
   assez de joueurs (`MIN_PLAYERS_PER_FRAME_FOR_CALIBRATION=10`), et la
   calibration est rejetée (poursuite de la collecte) si les deux
   couleurs trouvées sont trop proches (`MIN_CENTROID_DISTANCE=40`).

**Deux fichiers de production modifiés** (`rendering/overlay.py`,
`vision/tracker.py`, `analysis/player_reid.py`) — nécessitent un commit
GitHub, contrairement aux scripts d'analyse (`codeprojectai`).

---

## 13. RÉSULTAT FINAL — DÉTECTEUR "PRE_KO_ANCHOR" (pas un détecteur de C isolé)

### Ce qui a été testé, dans l'ordre
1. Détecteur géométrique simple (isolement seul) → 492 faux positifs sur
   tout le match (Andrimont).
2. Ajout d'une contrainte de compacité des 2 équipes → 0 détection sur
   plusieurs matchs (trop restrictif, contrainte devenue obsolète après
   le filtre `time_since_update`).
3. Retrait de la compacité, avec toute la chaîne de bugs ci-dessus
   corrigée → résultat final ci-dessous.

### Résultat sur 4 matchs (Andrimont, Raeren, Wanze, Juprelle)
| Match | Étape trouvée | Écart au repère de CETTE étape |
|---|---|---|
| Andrimont | C | 2.5s |
| Wanze | C | 0.2s |
| Raeren | D | 27.2s (correspond bien à D, pas une erreur) |
| Juprelle | D/E | 40.5s (correspond bien à D/E, pas une erreur) |

**Le détecteur ne distingue pas C, D et E entre elles** — il détecte de
façon fiable "un point caractéristique de la séquence pré-KO", peu
importe laquelle des 3 étapes. Une piste de séparation a été repérée
(`isolement_ratio` : ~4.5-4.8 pour C vs ~6.2-6.3 pour D/E dans ces 4 cas)
mais **explicitement non retenue comme règle** — 4 observations sont
insuffisantes, exactement le piège déjà rencontré plusieurs fois dans ce
projet. Conservée comme observation à surveiller, pas comme feature.

### Décision de cadrage (changement de nom, pas juste de résultat)
Le détecteur n'est plus visé comme "détecteur de C" mais comme
**détecteur de `pre_ko_anchor`** : un point d'ancrage fiable quelque part
dans la séquence C→D→E, à exploiter pour réduire la zone de recherche du
KO — pas pour identifier précisément laquelle des 3 étapes a été trouvée.
C'est un objectif plus robuste et suffisant pour l'usage prévu (cf.
section 8 : filtrer les candidats avant de laisser RandomForest
départager).

### État figé (rien à modifier de plus sur ce chantier)
- Détecteur `pre_ko_anchor` : validé sur 4 matchs.
- Contrainte de compacité : retirée (`compact_percentile` gardé dans le
  code pour flexibilité future, mais non utilisée par défaut).
- `time_since_update` : conservé, essentiel au résultat.
- C/D/E : traité comme une séquence unique, pas trois classes à
  distinguer séparément.
- `isolement_ratio` : conservé uniquement comme observation, pas intégré
  comme règle de décision.
- Aucune nouvelle feature, aucun nouveau calcul sur les 533 candidats
  existants.

### Prochaine phase engagée
Établir, sur les 9 matchs, la chaîne complète
`pre_ko_anchor → C → D → E → KO réel → transition mouvement`, et
identifier quels signaux sont réellement disponibles entre l'ancre et le
KO (positions, séparation, ballon, arbitre, vitesse) — **avant tout
nouveau code** — pour définir le contrat du détecteur final. Le
RandomForest reste un filet de sécurité possible, mais n'est plus
envisagé comme mécanisme principal de localisation du KO.

---

## 14. VALIDATION `PRE_KO_ANCHOR` SUR LES 9 MATCHS

### Résultat complet
| Match | Candidats | Écart C | Écart D | Écart E | Meilleur |
|---|---|---|---|---|---|
| Andrimont | 29 | 1.5s | 8.0s | 9.5s | **C (1.5s)** |
| Franchimont | 0 | — | — | — | **AUCUN** |
| Goe | 51 | 0.5s | 7.2s | 58.5s | **C (0.5s)** |
| Juprelle | 1 | 40.5s | 23.5s | 29.5s | D (23.5s) |
| MineroisSter | 1 | 55.8s | 108.8s | 161.8s | C (55.8s) |
| P1Minerois | 0 | — | — | — | **AUCUN** |
| Raeren | 5 | 27.2s | 0.2s | 7.0s | **D (0.2s)** |
| Stembert | 26 | 20.5s | 0.0s | 5.0s | **D (0.0s)** |
| Wanze | 38 | 0.2s | 0.0s | 3.5s | **D (0.0s)** |

### Bilan
**7/9 matchs (78%) ont un `pre_ko_anchor` exploitable à moins de 10s**
d'une étape connue de la séquence, plusieurs quasi exacts (0.0s, 0.2s,
0.5s). Résultat obtenu sur un échantillon complet, pas quelques cas
favorables choisis a posteriori.

**2 échecs à 0 candidat** (Franchimont, P1Minerois) — probablement liés
à une calibration de couleur de mauvaise qualité sur ces clips (peu
d'échantillons ou seuil de classification mal proportionné, cf.
`seuil_non_classe=25.7` pour Franchimont, nettement plus bas que les
autres matchs). **Décision explicite : ne pas corriger par un seuil
bricolé à ce stade** — noté comme limite connue, pas comme urgence.

**2 cas difficiles mais pas des échecs conceptuels** (Juprelle 23.5s,
MineroisSter 55.8s) — MineroisSter confirme une fois de plus sa
difficulté déjà connue depuis le début du projet (attente pré-KO
anormalement longue).

### Décision : le détecteur `pre_ko_anchor` est figé
Considéré comme une brique validée, avec limites connues et acceptées.
**Aucune modification supplémentaire prévue sur ce détecteur.**

### Prochaine étape engagée
Sur les 7 matchs où l'ancre fonctionne, construire pour chacun la
chronologie `pre_ko_anchor → C → D → E → attente finale → sifflet →
début du mouvement → KO réel`, en relevant uniquement les signaux déjà
disponibles (`mean_speed_change_px_s`, `whistle_score`,
`geo_separation`, `n_players`, immobilité). Question posée : existe-t-il
une séquence de signaux suffisamment régulière, entre l'ancre et le KO,
pour construire un détecteur déterministe — reléguant potentiellement
RandomForest au rang de filet de sécurité plutôt que de mécanisme
principal ?

---

## 15. CONFIRMATION DE L'INVARIANT "ATTENTE → RUPTURE DE MOUVEMENT" SUR 9 MATCHS

### Test réalisé
Chronologie de signaux (`geo_separation`, `vitesse`, `whistle_score`,
`n_players`, `pct_immobile`) calculée depuis `pre_ko_anchor` (ou un repli
KO-100s si l'ancre était absente/peu fiable) jusqu'à KO+20s, sur les 9
matchs — une fenêtre de recherche beaucoup plus large et exigeante que
le test de transition initial (±25s centré sur le KO connu).

### Constat, formulé avec précision (pas "la règle est validée")
> L'invariant comportemental "attente → rupture de mouvement" est
> confirmé sur les 9 matchs. La transition existe autour du KO, mais son
> amplitude et son timing varient suffisamment pour qu'un seuil fixe
> naïf puisse manquer certains cas.

### Lecture des 3 cas où la règle simple (baseline+10px/s, 6s soutenues)
### n'a pas détecté de transition
| Cas | Lecture |
|---|---|
| Wanze | Transition très nette à l'œil (vitesse 27→10.8 puis remontée à 33.4) — **la règle l'a ratée de peu**, à cause d'un baseline tiré vers le haut par des valeurs hors de la fenêtre affichée. Pas une absence du phénomène. |
| P1Minerois | Transition réelle mais courte et modeste (15→29 puis retombée) — confirme **une nouvelle fois** l'explication déjà établie (touche immédiate après le KO, section 3.5/8). Pas un nouveau problème. |
| Franchimont | Vitesse déjà élevée et bruitée avant **et** après le KO — pas de phase d'attente calme du tout. Cas intrinsèquement difficile, déjà identifié comme tel depuis le tout premier test de transition. |

**Aucun des 3 cas n'est une nouvelle découverte** — chacun correspond à
une situation déjà comprise plus tôt dans le projet.

### Décision explicite : ne pas chercher à pousser le seuil fixe vers 9/9
La règle `baseline+10px/s pendant 6s` a rempli son rôle (démontrer que le
phénomène existe) mais est jugée **trop rigide** pour devenir la règle
finale — un seuil absolu ne s'adapte pas aux différences d'échelle entre
matchs (Wanze) ni à l'amplitude variable du sursaut (P1Minerois).
**Aucun nouvel ajustement de seuil sur ces 9 matchs** — le piège déjà
identifié plusieurs fois dans ce projet (seuil A → test → seuil B →
test → seuil C...) est explicitement évité ici.

### Les deux briques du détecteur final, actées
1. **`pre_ko_anchor`** (section 13-14) — réduit la zone de recherche à la
   séquence C/D/E, sans exiger de savoir laquelle.
2. **Transition comportementale** — à l'intérieur de cette zone, localise
   le début réel du match via un changement d'état, pas un seuil absolu.

### Prochaine expérimentation engagée (avant toute intégration pipeline)
Concevoir une mesure de **rupture relative à l'état précédent** (pas un
seuil absolu en px/s) : "la dynamique vient-elle de changer par rapport
à ce qui se passait juste avant ?", plutôt que "la vitesse dépasse-t-elle
X ?". Objectif explicite : le système doit pouvoir répondre "KO probable,
transition faible" plutôt que fabriquer une fausse certitude sur des cas
intrinsèquement ambigus (P1Minerois) — pas viser un 9/9 artificiel.
À faire uniquement sur les données déjà existantes (9 CSV de chronologie
déjà calculés), aucun nouveau calcul Kaggle à ce stade.

---

## 16. STABILITÉ DU FILTRE RANDOMFOREST — DÉCISION TOP-10 (PAS TOP-3)

### Constat déclencheur
Un audit initial (mêmes données, même modèle) donnait Raeren au rang #3
avec les 10 matchs, mais **#7 en retirant simplement matchTest** du
dataset — signe que le classement RandomForest est sensible à des
variations de composition du dataset sans rapport avec la qualité du
signal lui-même.

### Test réalisé
20 runs GroupKFold avec ordre des données mélangé (le paramètre
`random_state` n'existe pas sur `GroupKFold` — le mélange préalable des
lignes change effectivement l'assignation match→fold) + graine
`RandomForestClassifier` différente à chaque run. Aucune modification de
features ni de modèle.

### Résultat
| Métrique | Valeur |
|---|---|
| Top-1 (moyenne sur 20 runs × 9 matchs) | 44% |
| Top-3 | 66% |
| Top-5 | 78% |
| **Top-10** | **99%** |
| Distance temporelle minimale moyenne au KO au sein du top-10 | **0.5s** |

**Scénario A confirmé** (pas scénario B) : la variance existe (Raeren
rang 2-11, P1Minerois rang 3-27 dans de rares cas) mais reste bornée —
aucun match ne s'effondre façon "rang 3 puis rang 50". Pour 7 des 9
matchs, la distance minimale au KO au sein du top-10 est exactement
0.0s (le vrai candidat, ou un doublon à horodatage identique, y est
presque systématiquement présent).

### Décision actée
**Le filtre RandomForest doit être utilisé en top-10, pas top-3 ni
top-5**, pour alimenter l'étage suivant (IA vision). Un filtre à 10
candidats reste une réduction énorme par rapport aux 50-70 candidats
bruts par match, avec une marge de sécurité mesurée (99%) plutôt que
supposée.

### Architecture retenue
```
Vidéo → mining (533 candidats habituels)
      → RandomForest (score de présélection)
      → top-10 candidats par match
      → IA vision (compréhension de scène, séquence A→F)
      → KO final + confiance
```
Le RandomForest devient un **moteur de présélection**, pas le mécanisme
de décision finale.

### Prochaine étape engagée
Concevoir et tester l'étage IA vision sur un premier match (Andrimont),
en extrayant des séquences courtes (images ou courts clips, ±15s) autour
de chacun des 10 candidats, avec un prompt structuré demandant
l'identification des étapes A→F (entrée, présentation, capitaines,
rassemblement, mise en place, coup de sifflet) plutôt qu'une réponse
binaire — avant de généraliser à plusieurs IA et aux 9 matchs.

---

## 17. TEST IA VISION — 4/4 SUR LES CAS DIFFICILES, PROTOCOLE VALIDÉ

### Méthode finale : 5 images par candidat, pas 1 seule
Premier essai (image unique par candidat, top-10 Andrimont) : succès net
— identification correcte du KO parmi 10 candidats, sans connaître les
timestamps à l'avance. Mais échec sur P1Minerois (le cas le plus
instable, cf. section 16) : l'image unique à l'instant exact du KO
montrait une scène déjà dispersée (le ballon venait d'être joué),
visuellement indiscernable d'un jeu ouvert normal.

**Diagnostic de l'échec** : vérification à 3 images (t-1s, t0, t+1s) —
confirmé que l'ambiguïté tient en **une seule seconde** : à t-1s la scène
est nette (ballon posé, joueur prêt), à t0 le ballon a disparu du centre.
Le problème n'était pas conceptuel, mais un problème d'échantillonnage.

**Protocole corrigé, adopté définitivement** : 5 images par candidat —
`contexte_avant` (t-12s), `juste_avant` (t-2s), `candidat` (t0),
`juste_apres` (t+2s), `contexte_apres` (t+10s). Combine fenêtre serrée
(capter la bascule elle-même) et contexte large (juger si une vraie
phase d'attente précède).

### Résultat : 4/4 sur les cas les plus difficiles du projet
Tests à l'aveugle (candidats anonymisés et mélangés avant analyse,
mapping révélé seulement après la réponse) :

| Match | Rang RF du vrai KO | Instabilité mesurée (section 16) | Identifié correctement ? |
|---|---|---|---|
| P1Minerois | 6 | rang 3-27 selon découpage | ✅ (après échec initial à l'image unique) |
| Raeren | 7 | rang 2-11 | ✅ |
| Goe | 5 | rang 3-10 | ✅ |
| Franchimont | 2 | pas de phase calme du tout | ✅ |

Ces 4 matchs sont précisément ceux où **le RandomForest échoue ou est
instable** — le test le plus dur possible pour l'hypothèse IA vision,
pas des cas favorables choisis a posteriori.

### Le motif visuel qui fonctionne, identifié empiriquement
Sur 3 des 4 matchs (P1Minerois, Raeren, Goe), un motif net et répété :
`juste_avant` (ballon posé au centre, joueurs statiques en position) →
`candidat` (ballon disparu du centre, joueur en plein geste) →
`juste_apres` (ballon visible en mouvement, sprint général). Ce triptyque
s'est montré **plus fiable que l'apparence d'un rassemblement figé** —
plusieurs candidats "rassemblement" très convaincants visuellement (deux
équipes en cercle, ballon au centre, arbitre présent) ont été
correctement écartés à chaque fois car statiques sur toute la fenêtre de
22s, contrairement au vrai KO.

Sur Franchimont (aucune vraie phase calme), le motif s'est manifesté en
version dégradée : pas de formation figée, mais une configuration
relativement moins agitée que le reste, ballon proche du centre, puis
contestation immédiate — reconnaissable mais nécessitant une lecture
plus fine.

### Ce que ça ne prouve pas encore
- 1 seul évaluateur (Claude, dans cette conversation) — pas encore de
  test comparatif Gemini/GPT-4V/autres IA vision.
- Pas encore testé sur les 5 matchs "faciles" (où le RF trouve déjà bien
  le KO) — mais l'enjeu principal est justement les cas difficiles.
- Pas encore formalisé en pipeline API automatisé.

### Prochaine étape
Formaliser le protocole à 5 images en appel API (Gemini et/ou Claude via
API, avec le prompt structuré A→F déjà défini), pour traiter les 9
matchs à l'échelle sans analyse manuelle image par image.

---

## 18. PROTOCOLE ALLÉGÉ À 3 IMAGES — VALIDATION SUR MATÉRIEL JAMAIS VU

### Un biais méthodologique identifié et corrigé avant de conclure
Le premier test à 3 images (Raeren, Goe, Franchimont, P1Minerois) a été
fait sur des images **déjà vues** lors du test à 5 images précédent —
risque de mémorisation des réponses plutôt que de raisonnement sur les 3
images seules, invalidant le test. **Décision explicite : refaire
proprement sur du matériel jamais montré dans la conversation.**

### Test propre sur les 4 derniers matchs jamais testés
Protocole à 3 images (`juste_avant` t-2s, `candidat` t0, `juste_apres`
t+2s) directement, sans passer par 5 images, sur Juprelle, Wanze,
Stembert, MineroisSter — tests à l'aveugle (candidats anonymisés,
mapping révélé après la réponse).

| Match | Réponse | Rang réel | Résultat |
|---|---|---|---|
| Juprelle | Bon candidat | 1 | ✅ |
| Wanze | Bon candidat | 1 | ✅ |
| **Stembert** | **Mauvais candidat** | vrai KO au rang 1, réponse donnée = rang 5 | ❌ |
| MineroisSter | Bon candidat | 1 | ✅ |

**3/4 corrects, 1 erreur identifiée et comprise, pas masquée.**

### L'erreur de Stembert, analysée précisément
Deux candidats (rang 1, t=553s, vrai KO ; rang 5, t=549s, "shot") à
seulement 4s d'écart — leurs fenêtres à 3 images se chevauchaient
(`juste_apres` du rang 5 = `juste_avant` du rang 1, même frame à
t=551s). La bascule nette (joueur en plein geste, ballon disparu) était
visible au rang 5, faisant croire à tort que le rang 1 (juste après)
était "la scène déjà calmée" — alors que la vraie bascule du KO se
jouait précisément à cet instant-là. **Leçon retenue** : le protocole
peut échouer quand deux candidats sont à moins de 5s l'un de l'autre,
car les fenêtres se chevauchent et brouillent l'ordre réel des
événements.

### Le test le plus dur, réussi : MineroisSter
Rang 1 (vrai KO, t=723s) et rang 2 (`faux_depart`, t=704s) avaient une
probabilité RandomForest quasi identique (0.345 vs 0.345) — le cas le
plus ambigu numériquement de tout le projet. Distingués correctement
sur le seul critère visuel : le vrai KO montre une dispersion générale
immédiate (joueurs en sprint), le faux départ montre une scène qui reste
statique (le tireur se contente de remonter ses chaussettes 2s après,
aucun autre mouvement). Confirme que le motif comportemental reste
discriminant même quand les features numériques ne le sont plus du
tout.

### Bilan global sur les 9 matchs (hors matchTest) testés visuellement
| Test | Résultat |
|---|---|
| Andrimont — 10 images uniques (premier essai, tout début) | 1/1 |
| 5 images (4 matchs : P1Minerois, Raeren, Goe, Franchimont — potentiellement biaisé par mémorisation, cf. ci-dessus) | 4/4 |
| 3 images (4 matchs : Juprelle, Wanze, Stembert, MineroisSter — matériel jamais vu, propre) | 3/4 |

**Total : 8/9 matchs correctement identifiés** sur l'ensemble des 9 matchs
du jalon (hors matchTest), tous testés visuellement à l'aveugle sous une
forme ou une autre.

Le protocole à 3 images (`juste_avant`/`candidat`/`juste_apres`, sans
contexte large) est **retenu comme suffisant** pour la majorité des cas,
avec une limite connue et documentée : risque de confusion quand
plusieurs candidats sont très rapprochés temporellement. Recommandation
pour la suite : si deux candidats du top-10 sont à moins de 5-6s
d'écart, envisager d'élargir ponctuellement à 5 images pour ces
candidats spécifiques plutôt que d'appliquer 3 images uniformément.

### Prochaine étape (inchangée)
Formaliser en pipeline API, avec cette précaution supplémentaire sur les
candidats rapprochés temporellement.

---

## 19. TEST FINAL SUR matchTest — CLÔTURE DE LA VALIDATION VISUELLE

### Cas particulier
`matchTest` diffère des 9 autres à plusieurs égards : caméra manuelle
basse et rapprochée (pas de vue aérienne Veo), KO au tout premier
événement enregistré (t=8s, aucune vraie séquence pré-match filmée), et
les deux équipes portent des maillots aux couleurs très proches
(rouge/bleu des deux côtés) — le repère "deux couleurs distinctes" qui
avait beaucoup aidé sur les 9 autres matchs ne s'applique pas ici.
Rang RF du vrai KO : 7/10 (déjà connu depuis le tout début du projet).

### Résultat du test à l'aveugle (protocole à 3 images)
Aucun des 10 candidats ne montrait la signature nette validée ailleurs
(formation statique → frappe → dispersion générale immédiate) — la
bascule visible ici était progressive, pas soudaine. **Réponse donnée
avec confiance explicitement faible** (candidats F/I, jugés les plus
proches d'un mouvement de jeu qui démarre) → **confirmé correct** (rang
7, le vrai KO).

### Ce que ça montre
Un succès obtenu avec une confiance annoncée basse est plus informatif
qu'un succès présenté à tort comme évident — le signal visuel du KO
reste détectable même dans des conditions de tournage très différentes,
mais de façon nettement plus fragile. Ce cas illustre concrètement
pourquoi le contrat `candidate + confidence + ambiguity` (défini dès le
début du projet, section 2) est le bon choix : un système en production
confronté à ce genre de cas devrait remonter une confiance faible, pas
une fausse certitude.

### Bilan final sur l'ensemble du jeu de données (10 matchs)
| Match | Résultat |
|---|---|
| Andrimont, Wanze, Juprelle, MineroisSter, Goe, Raeren, Franchimont, P1Minerois | ✅ (8/9, protocoles variés) |
| Stembert | ✅ accepté à 4s près (candidat adjacent au vrai KO, écart jugé négligeable) |
| matchTest | ✅ avec confiance annoncée faible, conforme aux conditions réellement plus difficiles |

**10/10 matchs traités, tous avec une identification correcte ou
acceptable du KO.** La validation manuelle de l'hypothèse IA vision est
considérée comme terminée. Passage à la mise en production du pipeline
complet (RF → top-10 → séquences à 3 images → appel API vision →
candidat + confiance).

