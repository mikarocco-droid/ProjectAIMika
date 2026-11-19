# ScoutIA — Document de passation (30/10/2026)

Oct 30, 2026 · @Mika

## Objectif et état du pipeline

ScoutIA analyse automatiquement des vidéos de matchs amateurs (caméra Veo, vue panoramique) : buts, bornes du match, occasions, rapport PDF. Le produit doit rester entièrement automatique, sans action de l'utilisateur.

**Principe d'entrée :** ScoutIA part d'une vidéo brute, sans aucune donnée (ni événements, ni métadonnées Veo). Tout doit être trouvé à partir de l'image. Le client visé est celui qui n'a que le fichier vidéo : parent ou joueur à qui on l'a envoyé, club sans abonnement d'analyse Veo, autre caméra ou smartphone.

**Configuration validée, inchangée depuis la passation du 28/10 :** suivi en mode « historique » (`FILTRAGE_PISTES = ""`), `player_reid` version validée, voie « ballon » désactivée, réflexion réduite (validation KO et classification), modèle unique `gemini-3.8-flash`, second regard actif, occasions des descriptions KO actives dans le notebook de référence.

**Nouveau depuis le 28/10 :** `replay.py` conserve `time_since_update` dans le cache (0 = vraie détection, > 0 = position prédite). Le notebook de référence contrôle ce point au pré-vol.

**Résultats sur les buts :**

- jeu de référence (passation du 28/10) : 40/40 ;
- Minerois Ster, replay du 30/10 : 5/5, plus 2 faux buts « à vérifier ». Celui de 70:48 est très probablement la reprise de la 2e mi-temps : en replay, les bornes ne sont pas détectées, donc la règle de la pause ne s'applique pas (limite connue du mode replay).

**À vérifier :** le bilan du run complet de Franchimont en mode « historique » (buts, bornes, coût Gemini) n'a pas encore été consigné.

## Décisions

Les buts restent la fonctionnalité principale ; le suivi automatique du joueur est suspendu ; la recherche sur les occasions est close pour l'instant, avec le repérage « surface » v2 en réserve.

| Élément | Statut | Ce qui a été décidé |
| --- | --- | --- |
| Détection des buts | Validé | Fonctionnalité principale. Ne pas la modifier sans test complet sur le jeu de référence. |
| Occasions | Partiel | 7 à 11 occasions sur 21 retrouvées (Minerois Ster). À présenter comme indicatives, sans prétendre à l'exhaustivité. |
| Passe « occasions » à images denses sur la vue panoramique | Écarté | Test du 30/10 en échec (4/10 retrouvées, 3/10 témoins faux) : l'action est trop petite dans l'image. |
| Suivi automatique du joueur (montage individuel) | Suspendu | Test de décision du 29/10 en échec (critère C2). Les outils restent disponibles. |
| Montage assisté (clics de l'utilisateur) | Écarté | Choix produit : tout doit être automatique. |
| Ordre de travail | Validé | D'abord les actions chaudes du match, ensuite l'identification des joueurs dans ces actions. |
| Vue suivie de Veo | Vérifié | Le fichier téléchargé est déjà la vue qui suit l'action : pas de meilleur export (30/10). |
| Doublon d'occasions dans le rapport | À corriger | 111:35 et 111:36 sont la même action, comptée deux fois (« 16 occasions » affichées). |

**Mise à jour du 30/10, fin de journée :**

| Élément | Verdict |
| --- | --- |
| Détection des buts | À préserver |
| Occasions actuelles du pipeline | Indicatives (7 à 11 sur 21) |
| Repérage « surface » v2 | Prometteur mais trop large : en réserve, pas intégré |
| Analyse par séquences vidéo | Piste non validée |
| Rentabilité du repérage v2 | Non démontrée |
| Recherche sur les occasions | Close pour l'instant |

## Audit des occasions (Minerois Ster)

Le pipeline retrouve 7 à 11 des 21 occasions ; les images plus denses n'aident pas, car sur la vue panoramique le ballon et le tir ne se voient pas.

Vérité terrain : liste complète de 21 occasions et 4 non-occasions vérifiées, confirmée par l'utilisateur le 21/10 (`verite_terrain.json`). Heures dans la vidéo.

**Replay du 30/10** (notebook de référence, `REPLAY = True`, CPU, 14,9 min) :

| Résultat | Nombre | Occasions |
| --- | --- | --- |
| Trouvées | 7 | 15:17, 18:25, 23:28, 25:12, 42:45, 89:56, 106:53 |
| Probablement trouvées (même action, 10 à 30 s d'écart) | 4 | 50:45, 74:35, 91:08, 92:57 |
| Candidates mais rejetées par Gemini | 7 | 19:20 (bord de fenêtre), 22:33, 33:10, 48:41, 49:27, 80:13, 108:49 |
| Jamais candidates | 3 | 51:56, 53:01, 95:52 |

- Aucune des 4 non-occasions vérifiées n'a été signalée.
- 4 temps forts hors de la liste : 30:23, 81:51, 96:29 (peut-être le corner qui suit 95:52), 111:35.
- Les occasions sont une retombée de la détection des buts : elles réutilisent les fenêtres de validation des coups d'envoi (17 images sur 55 s, serrées au centre, espacées de 4 à 5 s ailleurs).

**Test à images denses du 30/10** (`test_occasions_denses.py`, 31 images à 1 par seconde, 20 cas, critères fixés avant) : **échec**. 4 occasions manquées retrouvées sur 10 (critère : 7), 3 témoins signalés à tort sur 10 (critère : 2 au plus), 0,53 $.

**Conclusion :** sur la planche, ni le ballon ni le tir ne sont visibles ; Gemini devine à partir du mouvement des joueurs et se trompe dans les deux sens. La limite est la taille de l'action dans l'image, pas la question posée ni la densité des images. Réserve : le test envoyait des images en 640 × 360, plus petites que celles du pipeline (960 × 540). Cette conclusion vaut pour la vue panoramique et pour ce match.

**Tests complémentaires du 30/10** (mêmes 21 occasions, critères fixés avant chaque test) :

| Test | Occasions | Autre critère | Verdict | Coût |
| --- | --- | --- | --- | --- |
| Images denses en 1920 × 1080 (`test_occasions_natif.py`) | 7/10 | 5/10 témoins faux (max 2) | Échec | 0,54 $ |
| Repérage « surface » v1 (`test_reperage_surface.py`) | 17/21 (min 18) | 23 % du temps (max 25 %) | Échec | 0,88 $ |
| Repérage « surface » v2, coups de pied arrêtés inclus (`test_reperage_surface_v2.py`) | 20/21 | 40 % du temps (max 25 %) | Échec | 0,68 $ |

- **Correction d'une erreur :** le fichier Veo téléchargé est déjà la vue qui suit l'action (zooms et dézooms), pas une vue panoramique fixe. Le test à 640 × 360 avait réduit les images 3 fois. En pleine résolution, le rappel passe de 4 à 7 sur 10, mais Gemini confond encore centre et tir (les 4 non-occasions vérifiées sont signalées) et invente un but à 49:27.
- **Coût des images :** environ 1 100 jetons par image, quelle que soit la taille envoyée (640 × 360, 960 × 540 ou 1920 × 1080 avec la résolution « haute »). Réduire les images n'économise rien.
- **Repérage v1 :** la consigne demandait « je ne sais pas » pour tout arrêt de jeu, donc pour les corners et les coups francs près de la surface. C'était une erreur de consigne, corrigée en v2.
- **Repérage v2 :** 20/21 et 5 buts sur 5. L'occasion manquée (106:53, coup franc tiré de loin dans le ciel) est hors de la définition et peu grave ; le pipeline actuel la trouve déjà. Mais le jeu est près d'une surface 44 % du temps de jeu : le critère ne trie pas assez.
- **Nuance importante :** avec la tolérance de ±10 s, les 89 phases du v2 couvrent 64 % du match. Des phases de même forme placées au hasard couvriraient environ 13 occasions sur 21 (estimation, pas un test). Gemini fait nettement mieux que le hasard, mais ne trie pas encore assez pour être rentable : qualifier 43 min à 1 image par seconde coûterait environ 2,15 $ de plus par match.
- **Analyse par séquences vidéo :** non testée. Moins chère par seconde, mais probablement en plus basse résolution, alors que le détail est ce qui manque. À comparer sur les mêmes séquences (images haute résolution contre vidéo, qualité, coût, distinction tir/centre/passe) avant toute conclusion.
- **Repérage par le ballon (182 candidats du pipeline) :** à ±5 s et ±10 s, il ne trouve pas les occasions mieux que le hasard. Piste écartée, comme « ballon proche du gardien » (gardien presque jamais identifié).

## Suivi automatique du joueur (suspendu)

Le test de décision du 29/10 a échoué : sur le cache « historique », le 10 rouge de Franchimont est retrouvé à 15:23 mais pas au moment de son dégagement raté de 49:37.

**Ce qui fonctionne** (scripts `kaggle_joueur_v2_8.py` et suivants, cache en mode « filtre » du 28/10) :

- lecture du numéro par Gemini sur des vignettes, avec couleur du maillot mesurée sur les pixels : 62 segments retenus, environ 5 à 6 min de présence du 10 identifiée, sans erreur visible sur les nouveaux segments ;
- consensus dans le temps (option, désactivée par défaut) : +67 s de présence, 20 groupes ajoutés sur 23 vérifiés nets ;
- touches de balle (Gemini, fenêtres de 2 s) : prudent, aucune fausse touche nette, mais seulement 4 touches trouvées sur le match.

**Ce qui ne fonctionne pas :**

- l'identité se perd au moment des actions : joueurs groupés, de face, pistes cassées en fragments de moins d'une seconde ;
- échanges entre coéquipiers collés (exemple : le 10 et le 11 au coup franc de 32:14-32:18) ;
- la reconnaissance par l'apparence (OSNet, poids ImageNet) ne distingue pas les coéquipiers : AUC de 0,54 à 0,59, proche du hasard ;
- position du ballon dans le cache peu fiable : le short blanc des joueurs rouges est pris pour le ballon.

**Test de décision du 29/10** (cache `cache_franchimont_historique`, critères fixés avant) :

| Critère | Résultat |
| --- | --- |
| C1 : 10 identifié à 15:23 | Oui (100 % de l'action) |
| C2 : 10 identifié à 49:37 | Non (14 % de l'action ; identifié à partir de 49:37,7) |
| C3 : aucun cadre sur pelouse vide | Oui, confirmé à l'œil |
| C4 : aucun réglage | Oui |

Acquis utile pour la suite : en mode « historique », 82 % des positions sont des prédictions ; une fois filtrées (`time_since_update == 0`), aucun cadre fantôme ne reste.

## Vérification Veo à faire (gratuite)

Tous les matchs de référence sont en vue panoramique ; il faut savoir si Veo propose aussi une vue suivie (caméra virtuelle qui cadre l'action). Le nom exact de l'option dans Veo n'est pas connu : « suivi », « follow-cam », « broadcast »…

**Fait le 30/10 :** les seules options de téléchargement de Veo sont « tous les clips sauvegardés » et « l'intégralité du match » (1920 × 1080, 29,97 images/s). Le fichier téléchargé est déjà la vue qui suit l'action, avec zooms et dézooms : il n'y a pas de meilleur export à chercher. L'étiquette « veo\_panoramique » de la vérité terrain décrit la caméra, pas l'export. Le lecteur Veo affiche aussi ses propres événements (tirs, coups francs, buts) ; leur comparaison aux 21 occasions reste à faire.

1. Dans Veo, ouvrir les options de téléchargement de Minerois Ster.
2. Si une vue suivie existe, la télécharger et regarder à l'œil trois moments (heures de la vidéo panoramique) :
   1. 19:20 : occasion manquée (tir sur la transversale) ;
   2. 55:42 : but ;
   3. 28:00 : témoin sans occasion.
3. Noter la résolution du fichier, si l'action est bien cadrée, si le ballon et les numéros sont lisibles, et si les heures correspondent à la vue panoramique (décalage ou découpage éventuels).

**Précautions :**

- ne pas basculer la détection des buts sur la vue suivie : elle est validée sur la vue panoramique. La vue suivie serait d'abord testée en plus, pour les occasions et les joueurs ;
- question produit à trancher ensuite : faut-il demander la vue suivie à l'envoi, ou accepter les deux formats ?

**À relever en même temps, dans l'éditeur Veo :**

- les événements « Tir » et « But » du fil d'événements de Minerois Ster, avec leurs minutes de jeu (pas l'heure de la vidéo), pour les comparer aux 21 occasions de la vérité terrain ;
- s'il existe une option pour exporter ou télécharger ces événements (aucune exportation officielle trouvée dans la documentation de Veo).

**Usage des événements Veo :** étalon de test uniquement, jamais une entrée du pipeline. S'ils retrouvent la plupart des 21 occasions, ils pourront servir de référence pour mesurer ScoutIA sur tous les matchs de test sans tout revisionner. Ils montrent aussi ce que les clubs équipés de Veo ont déjà : buts, tirs, coups de pied arrêtés.

## Coûts mesurés

Depuis le 28/10, environ 14 $ de Gemini ont été dépensés, dont 7,89 $ mesurés aux jetons et environ 6 $ estimés. Tarif de `gemini-3.8-flash` : 0,75 $ par million de jetons en entrée, 3,75 $ en sortie ; la réflexion est facturée comme de la sortie.

| Run ou test (date) | Questions | Coût |
| --- | --- | --- |
| Test occasions à images denses (30/10) | 20 | 0,53 $ |
| Replay Minerois Ster (30/10) | 190 payées, 13 relues | 2,43 $ |
| Test de décision, cache historique (29/10) | 136 | 0,31 $ |
| Lecture renforcée, 8 vignettes (29/10) | 250 | 0,84 $ |
| Touches v1.1, fenêtres du consensus (29/10) | 19 | 0,06 $ |
| Touches v1 (29/10) | 172 | 0,33 $ |
| Lecture v2.7, réflexion limitée (29/10) | 376 | 0,91 $ |
| Tests buteurs, touches, réflexion (29/10) | 75 | 0,44 $ |
| Lecture v2.5, réflexion normale (29/10) | 398 | 2,04 $ |
| Lectures v2.2 à v2.4 (29/10), non mesurées | environ 1 210 | environ 6 $ (estimé) |

| Test complémentaire (30/10) | Questions | Coût |
| --- | --- | --- |
| Occasions, images en 1920 × 1080 | 20 | 0,54 $ |
| Repérage « surface » v1 | 66 | 0,88 $ |
| Repérage « surface » v2 | 66 | 0,68 $ |

La recherche sur les occasions a coûté au total environ 2,60 $ en 4 tests.

**Repères :**

- lecture des numéros : 5,13 $ pour 1 000 questions en réflexion normale, 2,43 $ en réflexion limitée (lectures identiques à 98 %) ;
- questions à 8 vignettes : environ 3,40 $ pour 1 000 ;
- un replay n'est gratuit que si le cache Gemini a été produit avec les consignes actuelles. Celui de Minerois Ster datait du 20/10 : 190 questions ont été repayées ;
- non consigné : le coût du run complet de Franchimont en mode « historique » ;
- le journal `cout_gemini_franchimont.csv` est incomplet : des sessions Kaggle ont été perdues.

## Fichiers à conserver

Tout ce qui a été payé est enregistré dans des fichiers à garder dans le dataset `codeprojectai`, sous leur nom exact (sans « (1) »).

| Fichier | Contenu | Remarque |
| --- | --- | --- |
| `cache_franchimont_historique` | Cache de suivi de Franchimont en mode « historique », avec `time_since_update` | Cache de référence actuel |
| `cache_franchimont_reference` | Ancien cache, mode « filtre » (28/10) | À renommer `cache_franchimont_filtre_2810` : un replay des buts dessus serait faux |
| `cache_minerois_ster_reference` | Cache de suivi de Minerois Ster (20/10) |  |
| `gemini_cache_minerois_ster.jsonl` | 1 257 réponses Gemini, dont 190 du 30/10 | Remplacer l'ancien par celui du 30/10 |
| `verite_terrain.json` | Buts, bornes, occasions ; buteurs de Franchimont ajoutés le 29/10 |  |
| `lectures_v27_franchimont.json` | 1 794 segments lus (numéros du 10 rouge) | Base des lectures |
| `lectures_renforcees_franchimont.json` | Lectures à 8 vignettes, 3 actions de but |  |
| `lectures_decision_franchimont.json` | Lectures du test de décision (191 segments) |  |
| `touches_v1_franchimont_10_rouge.json` | 191 réponses « touche de balle » |  |
| `test_occasions_minerois_ster.json` | 20 réponses du test à images denses |  |
| `cout_gemini_franchimont.csv` | Journal des coûts | Incomplet |

**Scripts** (hors pipeline) : `kaggle_joueur_v2_8.py` (dernière version, consensus en option), `touches_v1_2.py`, `test_consensus_temporel.py`, `test_lecture_renforcee.py`, `test_decision_historique.py`, `test_occasions_denses.py`, `test_apparence_osnet.py`, `diag_touches_ballon.py`.

**Pipeline sur GitHub :** `replay.py` (corrigé), `pipeline.py`, `analysis/player_reid.py` (version restaurée), `analysis/kickoff_gemini_cascade.py`. Notebook de référence : sur Kaggle (avec le contrôle de pré-vol de `time_since_update`).

## Règles de travail

Chaque test a un critère fixé avant de regarder les résultats, et le verdict est appliqué tel qu'il tombe.

- Ne pas modifier le pipeline sans test ciblé ; toute modification des buts se vérifie d'abord par un replay, puis sur le jeu de référence complet.
- Ne jamais changer la configuration d'un run de référence.
- Critères fixés à l'avance ; aucun réglage après coup pour sauver un test.
- Aucun verdict si le test est incomplet (réponses manquantes, plafond atteint, clé absente) : on complète, puis on conclut.
- Consigner le coût réel (questions et jetons) de chaque run, dans le journal des coûts.
- Toute réponse Gemini payée est sauvegardée et réutilisée ; ne jamais repayer une question déjà posée.
- Avant chaque run : vérifier les noms exacts des fichiers en entrée et la version du dataset dans le notebook.
- Lancer les runs longs avec « Save & Run All (Commit) », pour ne pas perdre les fichiers produits.
- Distinguer ce qui est mesuré, ce qui est déduit et ce qui reste à vérifier ; une hypothèse n'est jamais présentée comme un résultat.
- Aucune clé API en clair dans un notebook ou un script : secrets Kaggle uniquement.

* **Nouvelle règle (30/10) :** aucune expérimentation lancée seulement parce qu'une technologie semble prometteuse. D'abord identifier le défaut concret du produit qu'elle résoudrait, puis définir un test qui permette de conclure.
* Relire la consigne envoyée à Gemini contre l'objectif du test : en v1 du repérage, elle excluait justement les coups de pied arrêtés.

## Prochaine étape et points ouverts

Prochaine étape : revenir aux défauts concrets du produit. La recherche sur les occasions est close pour l'instant ; le repérage « surface » v2 est en réserve.

- **Si la piste des occasions est reprise :** partir du repérage v2 (20/21), et résoudre d'abord le tri (44 % du temps de jeu retenu) et le coût de la qualification, avec un test à critères fixés avant.
- **Gratuit, quand tu as un moment :** relever les « Tir » du fil d'événements Veo de Minerois Ster et les comparer aux 21 occasions, pour savoir s'ils peuvent servir d'étalon de test.

**Points ouverts :**

- [ ] Consigner le bilan du run complet de Franchimont en mode « historique » (buts, bornes, coût).
- [ ] Révoquer la clé Gemini et la clé Claude (`sk-ant-…`) qui figuraient en clair dans d'anciens notebooks, si ce n'est pas déjà fait, et vérifier l'historique du dépôt GitHub `ProjectAIMika`.
- [ ] Renommer l'ancien cache en `cache_franchimont_filtre_2810`.
- [ ] Corriger le doublon d'occasions dans le rapport (111:35 et 111:36).
- [ ] Remplacer `gemini_cache_minerois_ster.jsonl` dans le dataset par celui du 30/10.
- [ ] Avant commercialisation : licence AGPL-3.0 d'`ultralytics` (point de la passation du 28/10).
- [ ] Prix : au coût actuel, le mode joueur (environ 4 à 5 $ de lecture par match) dépasserait les tarifs Pro (2,69 €) et Unique (5 €).
