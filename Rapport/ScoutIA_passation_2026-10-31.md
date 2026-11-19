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
| Doublon d'occasions dans le rapport | Corrigé (31/10) | 111:35 et 111:36 sont la même action, comptée deux fois (« 16 occasions » affichées). |

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

## Corrections et tests du 31/10

Deux défauts visibles par le client sont corrigés ; le faux but de 112:52 reste « à vérifier » après quatre pistes écartées.

**Corrigé et validé :**

- **Doublon d'occasions** (`pipeline.py`) : deux candidats voyant la même action avec deux issues différentes (Minerois Ster 111:35 arrêt, 111:36 but) donnaient deux occasions. Le dédoublonnage à 12 s ignore maintenant l'issue vue par Gemini. Replay de Minerois Ster : 15 occasions au lieu de 16, buts inchangés (5/5, mêmes faux « à vérifier »), 0 question payée, 9,1 min.
- **Frise du PDF** (`export/pdf.py`) : heures superposées (« 98:1100:49 ») et à moitié cachées par les repères. Étiquettes placées sous les repères, sur plusieurs lignes (jusqu'à 5) quand elles se chevaucheraient. Vérifié sur des PDF générés ; à confirmer par un replay.

**Faux but de 112:52 (changements pris pour un coup d'envoi) : pistes écartées**

| Piste | Résultat | Coût |
| --- | --- | --- |
| Séparation des équipes avec la couleur du cache | Échec : champ « team » vide pour la moitié des joueurs, équipe 0 deux fois plus fréquente | 0 $ |
| Écarter les buts sans heure vue (audit de 10 matchs) | 37 vrais buts avec heure vue, 2 sans (Spa 89:05, Andrimont 06:23 sur penalty), 22 faux candidats sans heure vue. Rejetée : ferait perdre les penalties | 0 $ |
| « Regard en arrière » (fenêtre de -60 s à 0 s) | Échec : Spa 89:05 retrouvé, mais Andrimont non, et Gemini invente un but à 112:15 pour le faux de 112:52 | 0,30 $ |
| Formation de coup d'envoi mesurée sur les pixels | Échec : 0/6 ; trop peu de joueurs visibles au coup d'envoi, couleurs regroupées par la lumière, une faute (83:03) ressemble à une formation | 0 $ |

**Ce qu'on en retient :**

- un penalty marqué est presque par nature un but « sans heure vue » (long arrêt avant, tir très court) ;
- Gemini affirme une formation de coup d'envoi même quand il n'y en a pas : il ne faut pas se fier à ce qu'il décrit pour écarter un but ;
- une vérification objective demanderait une reconnaissance fiable des équipes, qui n'existe pas encore. Ce chantier ne se justifierait que par un objectif produit (score par équipe, occasions par équipe), pas pour ce seul faux but ;
- le faux but de 70:48 (reprise de la 2e mi-temps) n'apparaît qu'en replay : en run complet, la règle de la pause l'écarte.

## Attribution des buts aux équipes (31/10)

Objectif produit : afficher le score par équipe. Une méthode à deux questions visuelles concordantes est prometteuse mais pas encore validée ; une seule question ne suffit pas.

Vérité de référence : équipe qui marque pour les 43 vrais buts, tenues des deux équipes (maillot et short) pour les 10 matchs, fournies par l'utilisateur. **Rectification Wanze** (correction de la vérité terrain, pas progrès de Gemini) : 22:50 marqué par le rouge et blanc sur corner (1-0), 65:46 par le bleu foncé sur penalty (1-1), score final 1-1.

| Test | Résultat | Verdict | Coût |
| --- | --- | --- | --- |
| Relecture des textes en cache par expressions régulières | 24 justes, 4 fausses, 71 % attribués | Échec | 0 $ |
| Relecture des textes par Gemini (texte seul), v1 | Textes de buts voisins mélangés (fenêtre de 60 s) | Échec (erreur de conception) | 0,01 $ |
| Relecture des textes par Gemini, v2 (fenêtre ±20 s) | 1 erreur, 72 % attribués : un quart des textes ne nomment pas l'équipe | Couverture insuffisante, méthode abandonnée | 0,03 $ |
| Visuel, une question (but + reprise, tenues fournies), v1 | 39 justes, 1 fausse (Wanze 22:50, maillots tous deux avec du rouge), 3 inconnus | Attribution indicative seulement | 0,70 $ |
| Visuel v2 : consigne « regarder le short » si les maillots partagent une couleur, reprise toujours après le but | 3 erreurs sur 42 (Wanze 22:50 juste, pour information) | Régression selon le critère | 0,70 $ |

**Découverte principale : Gemini n'est pas reproductible.** Pour Andrimont 48:58 et Stembert 110:45, les images et la question étaient identiques entre les deux runs visuels, et les réponses opposées (certitude « élevée » les deux fois). Environ une réponse sur vingt est fausse ; une seule réponse ne suffit donc pas pour un score.

**Simulation sur les deux runs (faite après coup, ne prouve rien) :** en n'acceptant une équipe que si les deux réponses concordent, 36 buts sur 42 justes, 0 erreur (86 %) ; en acceptant aussi une réponse seule quand l'autre est « inconnu », 39 sur 42, 0 erreur (93 %). Wanze 22:50 devient « inconnu ».

**Méthode à valider :** deux questions visuelles indépendantes par but (consigne du short si les maillots se ressemblent ; pas d'attribution si maillots et shorts se ressemblent), reprise toujours après le but, équipe acceptée seulement si les réponses concordent. Coût prévu : environ 0,03 $ par but. Validation sur de nouveaux matchs jamais utilisés, critères fixés avant : 0 erreur et au moins 80 % attribués.

**À vérifier avant l'intégration :** d'où viennent les tenues des équipes en production (pré-analyse du site ou question supplémentaire) — non vérifié dans le code.

## Travaux du 05/11 : seuil adopté, heure des buts, arrêt des occasions

**Seuil de détection adopté par défaut : 0,07 / 0,15** (`detect_kickoff_candidates.py` ; ancien réglage : SCOUTIA_KO_DELTA_MIN=0.10, SCOUTIA_KO_APRES_MIN=0.20). Replays comparés à l'ancien réglage : Raeren +1 vrai but (20:08) ; Minerois Ster, Franchimont, Spa et Andrimont (cache partiel, arrêté vers 61 min) : aucun vrai but perdu, aucun faux but ajouté. Les faux buts de Franchimont (07:41, 47:35, 68:49, 83:09) existaient déjà avec l'ancien réglage. Erreur de méthode notée : les références « 1 faux but » d'Andrimont, Franchimont et Spa avaient été posées sans mesure ; le notebook générique compare maintenant à un run à l'ancien réglage, fait juste avant.

**Heure des buts corrigée** (`gemini_validator.py`, `_ko_parse_goal_time`) : Gemini perd parfois un chiffre des minutes (« 06:09 » pour 64:09 à Spa, « 05:42 » pour 55:42 à Minerois Ster ; 3 réponses « but » sur 93). Récupération prudente : mêmes secondes, minutes commençant par celles annoncées, une seule heure possible dans la fenêtre. Sur 1 428 réponses des caches, seuls ces 3 cas changent, tous vers l'heure exacte. Spa devient 7/7 sans faux but.

**Occasions : recherche automatique ARRÊTÉE** (environ 4,30 $ depuis le 31/10). Minerois Ster (37 moments notés 1/2/3) : « lister toutes les tentatives » 13/21 mais 11 fausses alertes ; vérification à images serrées v1 71 % gardées / 46 % écartées, v2 50 % / 92 %. Gemini confond tirs et centres avec cette caméra, choisit le mauvais moment, invente (119:51) et n'est pas reproductible. Voie retenue : montage assisté.

**Règle d'affichage** : un faux but est entré dans le score estimé (Franchimont 47:35, « probable vert »). Libellé « action dangereuse » conservé.

**Compléments du 05/11 (après Spa)** : Spa 7/7 sans faux but avec `gemini_validator.py` corrigé. Attribution 6/7 justes ; 64:09 non attribué (question « coup d'envoi » fausse : B au lieu de A, probablement parce que le coup d'envoi détecté, 64:04, est AVANT le but et que la fenêtre d'images commence donc pendant l'action — hypothèse non testée). Règle d'affichage : premier vrai but affiché hors score (Spa 64:09) ; le score estimé n'en dépend pas (un but sans équipe n'est compté nulle part). **Libellé changé** : « but possible à vérifier » au lieu de « action dangereuse ». Faux buts de Franchimont vérifiés : 07:41 centre puis tir raté, 47:35 dribble mis en touche, 68:49 ballon trop long sorti en 6 m, 83:09 changement de joueur — aucun danger ; la validation KO prend des arrêts de jeu ordinaires pour des remises au centre.

**Deux pistes examinées le 05/11 (fin de journée)** :
- *Question « coup d'envoi » de l'attribution* : 39 justes sur 41 réponses (43 buts). Spa 64:10 faux dans deux runs indépendants : un joueur noir et bleu traverse le rond central pendant que le n°7 rouge attend au point (vérifié par l'utilisateur dans Veo, 64:31-64:42). Rien changé. Amélioration notée pour le prochain test de l'attribution : préciser que les joueurs de l'équipe qui vient de marquer traversent souvent le rond central, et regarder le joueur qui touche le ballon quand il se met en mouvement.
- *Faux buts nés d'arrêts de jeu* (`test_reprise_centre.py`, 1,88 $) : question « vraie remise en jeu au point central ? » sur +10/+90 s. ÉCHEC : 8 faux buts sur 9 écartés, mais 4 vrais sur 43 aussi (max 1) ; Gemini prend des coups d'envoi pour des coups francs avec une certitude « forte » (Stembert 110:45, Spa 49:20, Andrimont 48:58, Raeren 50:17). Combinaison avec la question « coup d'envoi » examinée après coup : garderait les 4 vrais buts, mais n'écarterait que des faux buts déjà sortis du score par la règle d'affichage — pas d'apport. Limite connue ; piste future hors Gemini : placement des joueurs par le suivi au moment de la reprise.

**À mettre sur GitHub** : `detect_kickoff_candidates.py`, `gemini_validator.py`, `pipeline.py` (règle de fin de match désactivée par défaut), `export/pdf.py`. **Dans codeprojectai** : `gemini_cache_spa.jsonl`, `gemini_cache_franchimont.jsonl`, `gemini_cache_andrimont.jsonl` des runs du 04-05/11.

## Travaux du 04/11 : test global, seuil, équipes, affichage du score

**Méthode d'attribution retenue (validée sur les 43 buts de référence)** : question « but » (8 images, qui attaque / célèbre) + question « coup d'envoi » précise (le joueur qui a le pied sur le ballon au point central et joue le premier ballon, maillot + short, 1 image/s du coup d'envoi détecté jusqu'à +45 s, sinon de +15 à +120 s, jamais au-delà du but suivant − 5 s) + arbitrage « short » si désaccord. Grille : but +2, coup d'envoi +2, arbitrage +1/+1 ; confirmé = but et coup d'envoi d'accord ; probable = écart ≥ 2 et au moins deux questions ; sinon non attribué. Référence : 37 confirmés justes (86 %), score estimé 41 justes / 1 faux, 9 matchs sur 10 au score estimé exact. Gemini n'est pas reproductible : jamais une équipe sur une seule réponse.

**Tenues** : détectées automatiquement par Gemini (16 images de la 1re mi-temps) — parfaites sur Raeren et Minerois Ster.

**Test global Raeren (run complet GPU, 95 min)** : 7/9 buts, 3 faux « à vérifier » (31:16 faute ; 100:48 et 105:26 = grosses occasions sauvées sur la ligne). Les 6 buts attribués sont justes.

**Seuil de détection des candidats** (`detect_kickoff_candidates.py`, SCOUTIA_KO_DELTA_MIN / SCOUTIA_KO_APRES_MIN, défaut inchangé 0,10 / 0,20) : à 0,07 / 0,15, Raeren retrouve 20:08 (8/9), Minerois Ster 5/5 sans faux but de plus (112:52 disparaît, 119:50 apparaît). À valider sur Andrimont, Franchimont, Spa (critères fixés : tous les vrais buts, au plus 2 faux hors reprise de 2e MT).

**Règle d'affichage (fixée avant les tests)** : un but « à vérifier » ET « sans équipe » sort du score et devient « action dangereuse, but possible non confirmé ». Raeren : 4 faux sortis, aucun vrai. Minerois Ster : 1 faux sorti, aucun vrai ; score estimé 2-3 exact.

**Règle de fin de match** (pipeline.py, désactivée par défaut, SCOUTIA_REGLE_FIN_MATCH=1) : but dont le coup d'envoi détecté tombe dans les 30 s avant la fin du match → « à vérifier » (jamais supprimé). Cause : Minerois 119:48, dernier corner en 6 m puis coup de sifflet final. À valider en run complet.

**Penalties** : écartés. Raeren 69:12 reste manqué (déduplication à 10 s = +30 candidats par match). Question explicite « penalty ? » : 1/4 reconnu, 0 fausse alerte ; penalty inventé par Gemini à Stembert 35:49 (grosse occasion). Limite connue : 1 but sur 43.

**Reste à faire avant l'intégration** : valider le seuil sur 3 matchs ; tester la règle de fin de match en run complet ; réessai automatique en cas d'erreur 503 ; temps de calcul (95 min GPU, 79 % pour le suivi).

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
* **Règle (31/10) :** on ne lance pas un chantier technique parce qu'il pourrait résoudre un cas limite ; on le lance quand il apporte une amélioration mesurable au produit.

## Prochaine étape et points ouverts

Prochaine étape : revenir aux défauts concrets du produit. La recherche sur les occasions est close pour l'instant ; le repérage « surface » v2 est en réserve.

- **Si la piste des occasions est reprise :** partir du repérage v2 (20/21), et résoudre d'abord le tri (44 % du temps de jeu retenu) et le coût de la qualification, avec un test à critères fixés avant.
- **Gratuit, quand tu as un moment :** relever les « Tir » du fil d'événements Veo de Minerois Ster et les comparer aux 21 occasions, pour savoir s'ils peuvent servir d'étalon de test.

**Points ouverts :**

- [ ] Consigner le bilan du run complet de Franchimont en mode « historique » (buts, bornes, coût).
- [ ] Révoquer la clé Gemini et la clé Claude (`sk-ant-…`) qui figuraient en clair dans d'anciens notebooks, si ce n'est pas déjà fait, et vérifier l'historique du dépôt GitHub `ProjectAIMika`.
- [ ] Renommer l'ancien cache en `cache_franchimont_filtre_2810`.
- [ ] Remplacer export/pdf.py sur GitHub et vérifier la nouvelle frise du PDF par un replay de Minerois Ster (gratuit). Le doublon d'occasions est corrigé et validé (31/10).
- [ ] Remplacer `gemini_cache_minerois_ster.jsonl` dans le dataset par celui du 30/10.
- [ ] Avant commercialisation : licence AGPL-3.0 d'`ultralytics` (point de la passation du 28/10).
- [ ] Prix : au coût actuel, le mode joueur (environ 4 à 5 $ de lecture par match) dépasserait les tarifs Pro (2,69 €) et Unique (5 €).
