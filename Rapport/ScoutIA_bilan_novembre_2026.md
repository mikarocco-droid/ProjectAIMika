# ScoutIA — Bilan de novembre 2026 : travaux et décisions

13 novembre 2026

## En résumé

En novembre, le pipeline est passé d'environ 3,50 à 4 $ de Gemini par match à 1,40 à 2 $ en replay (équipes comprises, bornes du match non comprises ; 2,87 $ pour le run complet du 11/11), sans perdre de vrai but, et une analyse en replay prend désormais 11 à 15 min de questions Gemini. Le principal frein au lancement n'est plus technique : c'est la licence Ultralytics (YOLO, AGPL).

Les décisions qui structurent la suite :

- **Détection des buts** : seuil des candidats coups d'envoi abaissé à 0,07 / 0,15 (validé 05-06/11) ; les buts viennent uniquement de la validation des coups d'envoi.
- **Résolution des images** : coups d'envoi en moyenne, temps forts en basse, escalade en résolution normale des désaccords (validé 09/11 sur 10 matchs : 41/43 buts, faux buts 24 → 14).
- **Équipes des buts** : attribution intégrée au pipeline, en résolution normale, 1 image par seconde pour Q2 (11-12/11) ; aucune attribution fausse sur les tests de référence.
- **Modèle** : Gemini 3.8 Flash seul, verrouillé et surveillé ; Gemini 3 Flash Preview et la réflexion réduite des bornes ont échoué.
- **Vitesse** : questions posées « d'un coup » (128 à la fois) et attribution en parallèle, validées le 13/11 sur 3 matchs.
- **Flex (−50 %)** : compatible et facturé au bon tarif, mais délais irréguliers ; réservé à une formule « sous 48 h », avec repli au tarif standard après 30 min d'attente (à tester).
- **Abandonnés** : recherche des occasions par Gemini, lecture du buteur, mise en cache des consignes, traitement par lots (Batch), alias « dernière version » du modèle.
- **Business** : jeton à 5 € minimum ; statut envisagé d'indépendant complémentaire en franchise de TVA, seulement si la licence YOLO est abordable.

## Détection des buts et montage joueur (1er-10 novembre)

La détection des buts repose désormais entièrement sur la validation des coups d'envoi : c'est le choix qui donne 40 à 41 vrais buts trouvés sur le jeu de référence. Le reste du mois a servi à en baisser le coût, pas à changer la méthode.

| Date | Sujet | Résultat mesuré | Décision |
| --- | --- | --- | --- |
| 06/11 | Montage joueur : lecture des numéros par PARSeq affiné (v3.0) | Même résultat que Gemini (4 participations sur 5 du n°31), 0,003 $ au lieu de 0,68 $ | PARSeq affiné retenu pour le montage joueur |
| 05-06/11 | Lecture du buteur par Gemini | Presque toujours inutile, environ 0,15 $ par match | Désactivée (SCOUTIA_LECTURE_BUTEUR=0) |
| 05-06/11 | Recherche des occasions (tests « danger » v1 et v2, Minerois Ster) | 24/24 occasions gardées mais 0/13 non-occasions écartées ; 2 erreurs de vision (tir pris pour un centre) | Recherche des occasions arrêtée |
| 05/11 | Faux but de fin de match (Minerois Ster) | Corner pris pour un but vers 119:48 | Règle de fin de match ajoutée au pipeline |
| 04-06/11 | Seuil des candidats coups d'envoi abaissé à 0,07 / 0,15 | Raeren +1 vrai but ; Minerois, Franchimont, Spa, Andrimont : aucun vrai but perdu, aucun faux en plus | Réglage par défaut |
| 04/11 | Reconnaissance des penalties par une question dédiée | 1 penalty reconnu sur 4 (critère 3), 0 fausse alerte | Échec : écarté |
| 04/11 | Équipe du but par vision Gemini, tenues détectées automatiquement | Minerois Ster : 5/5 justes, score exact 2-3 | Chantier lancé, devenu l'attribution des équipes |
| 01-04/11 | Faux buts : séparation des équipes, « regard en arrière », formation en pixels | Aucune piste ne règle le cas 112:52 | Pistes écartées ; un chantier n'est lancé que s'il améliore le produit de façon mesurable |
| 01/11 | Montage joueur v2.8 : identification automatique du n°10 (Franchimont) | Critère C2 en échec (14 % de l'action couverte, il fallait 50 %) | Arrêt du suivi automatique sous cette forme ; priorité aux actions chaudes |

Un décalage d'heure est resté inexpliqué sur Spa : le but de 64:10 annoncé à 63:49, soit 21 s trop tôt ; l'heure des buts a été corrigée dans gemini_validator.py le 06/11.

## Résolution des images et équipes des buts (6-12 novembre)

La résolution réduite divise par deux le coût des coups d'envoi et des temps forts, à condition de reposer les cas litigieux en résolution normale ; pour l'attribution des équipes, seule la résolution normale tient le critère « 0 équipe fausse ».

**Détection des buts** (coups d'envoi et temps forts) :

| Date | Réglage testé | Vrais buts | Faux buts | Coût de ces deux étapes | Verdict |
| --- | --- | --- | --- | --- | --- |
| 09/11 | Coups d'envoi moyenne, temps forts basse, escalade en normale (10 matchs) | 41 / 43 | 24 → 14 | 10,27 $ pour 10 matchs | **Retenu par défaut** |
| 08/11 | Tout en moyenne (10 matchs) | 40 / 41 évaluables | — | environ 1,20 $ par match | Échec (Franchimont 79:42 perdu) |
| 07/11 | Tout en moyenne (Spa) | 7 / 7 | 0 | 1,46 $ (−47 %) | Réussi sur un match |
| 07/11 | Tout en basse (Spa) | 5 / 7 | 2 | 0,86 $ | Échec |

Les 2 buts manquants du 09/11 sont les mêmes qu'en résolution normale : le penalty de Raeren et Andrimont 84:06 (cache incomplet).

**Équipes des buts** (Q1 « qui a marqué », Q2 « qui engage », Q3 arbitrage) :

| Date | Réglage testé (43 vrais buts) | Justes | Faux | Non attribués | Verdict |
| --- | --- | --- | --- | --- | --- |
| 12/11 | Normale, Q2 à 1 image toutes les 3 s | 36 | 1 | — | Échec |
| 12/11 | Normale, Q2 à 1 image toutes les 2 s | 37 | 1 | — | Échec |
| 11/11 | Moyenne + escalade, dans le pipeline (4 matchs) | 21 | 2 | 1 | Échec : arrêté |
| 09/11 | Moyenne | 38 | 0 | 5 | Échec de justesse (il fallait 39) |
| 09/11 | **Normale, Q2 à 1 image par seconde** | **40 à 41** | **0** | **2 à 3** | **Retenu par défaut** |

Décisions associées :

- L'attribution est intégrée au pipeline (étape 11b) et au rapport PDF ; les tenues sont reconnues 10 fois sur 10.
- Une requête au-delà de la limite de 30 Mo de Gemini (jusqu'à 106 images en Q2) est recompressée en JPEG 70, puis 55, puis 40 ; plusieurs runs l'ont fait sans échec.
- Un but « à vérifier » sans équipe reste hors du score ; le rapport affiche un score confirmé et un score estimé.

## Modèle Gemini, réflexion et surveillance de version

Gemini 3.8 Flash reste le seul modèle du pipeline : les deux tentatives d'économie sur le modèle lui-même ont dégradé les résultats.

| Date | Essai | Résultat | Décision |
| --- | --- | --- | --- |
| 12/11 | Gemini 3 Flash Preview (moins cher, Spa) | 7 vrais buts mais 7 faux, dont 3 « confirmés » ; 1 équipe fausse ; même coût (1,89 $) | Échec : on reste sur 3.8 Flash |
| 12/11 | Réflexion limitée pour les bornes du match | −42 % sur ces questions, mais fin du match de Spa annoncée 6 min trop tôt, fins de mi-temps décalées de 11 à 34 s | Échec : réglage actuel conservé |
| 12/11 | Alias « dernière version » du modèle | Risque de changement silencieux ; « 3.7 Flash » demandé a été servi par 3.8 Flash | Refusé ; surveillance ajoutée |

Ce qui est en place :

- **Verrou** : toutes les questions du projet passent par gemini-3.8-flash, avec le cache des réponses.
- **Surveillance** : chaque réponse note la version réelle du modèle ; une autre version déclenche une « ALERTE MODÈLE » dans le bilan du run.
- **Réflexion réduite** (budget nul) pour la validation des coups d'envoi et la classification des temps forts, choix d'octobre inchangé ; les bornes restent en réflexion par défaut.
- **Adresse officielle de Google imposée** (13/11) : la mise à jour de google-genai faisait passer les questions par un relais Kaggle.

## Vitesse : questions « d'un coup » et attribution en parallèle (13 novembre)

Poser jusqu'à 128 questions à la fois, au lieu de 4, ramène les questions Gemini d'un match en replay à 11 à 15 min, sans changer ni les résultats ni le coût.

| Match | Vrais buts | Faux buts (référence) | Équipes justes / fausses | Durée du replay | Coût Gemini |
| --- | --- | --- | --- | --- | --- |
| Minerois Ster | 5 / 5 | 2 (1) | 5 / 0 | 11,0 min | 1,85 $ |
| Spa | 7 / 7 | 0 (0) | 6 / 0 (1 non attribué) | 14,7 min | 1,42 $ |
| Franchimont | 5 / 5 | 4 (4) | 5 / 0 | 14,9 min | 2,00 $ |

- **Réglage par défaut : 128 questions à la fois** (GEMINI_MAX_PARALLEL), en tarif standard comme en Flex. Le quota du compte (niveau 3 : 20 000 questions et 20 millions de jetons par minute) laisse une large marge : le pic observé sur 28 jours est de 57 questions et 618 000 jetons par minute.
- **Attribution en parallèle** : tous les buts en même temps, Q1 et Q2 ensemble pour chaque but ; mêmes questions, affichage dans l'ordre des buts (SCOUTIA_ATTRIBUTION_PARALLELE=1 rétablit l'ancien fonctionnement).
- Les bornes du match restent posées une par une : chaque question dépend de la précédente.
- La durée totale en production reste dominée par le suivi GPU (environ 86 min sur 115 le 11/11) ; le gain de 10 à 15 min est surtout utile à une formule « rapide ».

## Mode Flex : tarif −50 %, délais irréguliers (12-13 novembre)

Flex fonctionne et est facturé à moitié prix, mais Google répond par rafales : la plupart des questions reviennent en quelques minutes, quelques-unes après plusieurs heures. Il ne convient qu'à une formule « sous 48 h ».

| Date | Essai | Résultat |
| --- | --- | --- |
| 13/11 | Match complet, 128 questions à la fois (Minerois Ster) | Après 3 h 36 : 96 validations sur 100, les 5 vrais buts trouvés, 1 question perdue sur une erreur « 500 » ; quelques questions bloquent la vague pendant des heures. Run en cours. |
| 13/11 | 20 questions réelles, en direct chez Google (soirée) | 19/20 obtenues en moins de 45 min ; délai médian 2,8 min, maximum 35 min ; 62 refus temporaires ; standard : 7 s |
| 12-13/11 | Match complet, 4 puis 16 questions à la fois, via le relais Kaggle | 100 validations en environ 8 h 25 ; trop lent |
| 12/11 | Compatibilité avec 3.8 Flash | Accepté (en-tête « flex » confirmé) ; délais de 4 s à 10 min ; 4 refus sur 7 essais |
| 12/11 | Facture Google | Lignes Flex distinctes : 0,337 € par million de jetons en entrée, 1,684 € en sortie, soit la moitié du standard |

Réglages en place dans gemini_validator.py :

- **Activation** par SCOUTIA_GEMINI_TIER = « flex » ; désactivé par défaut.
- **Les bornes du match restent au tarif standard** : le suivi GPU en dépend.
- **Repli au tarif standard** : passé l'échéance du match (SCOUTIA_FLEX_ECHEANCE, fixée par le site à commande + 36 h pour une promesse de 48 h) ou après l'attente maximale d'une question (SCOUTIA_FLEX_ATTENTE_MAX_S ; 30 min dans le prochain test).
- **Nouveaux essais automatiques** sur les refus temporaires (503, 429, délais, et « 500 INTERNAL » depuis le 13/11), espacés de 20 s à 2 min.
- **Délai maximal de 10 min par requête** (13/11), pour que l'attente maximale puisse jouer.
- Prérequis : google-genai 3.2.0 ou plus récent.

Le traitement par lots (Batch), au même tarif mais jusqu'à 24 h par lot et plusieurs lots enchaînés, a été écarté au profit de Flex.

## Coûts mesurés, facture Google et mise en cache

Un match coûte aujourd'hui 1,40 à 2 $ de Gemini en replay et 2,87 $ en run complet avec les bornes (mesure du 11/11), et ce tarif doublera au 1er janvier 2027 ; le coût vient presque entièrement des images envoyées.

**Coût d'un match, mesuré** :

| Mesure | Questions | Coût Gemini |
| --- | --- | --- |
| Run complet « comme en production », Minerois Ster, 11/11 (analyse + montage joueur) | 388 + relecture joueur | 2,87 $ |
| dont validation des coups d'envoi | 110 | 0,99 $ |
| dont équipes des buts | 20 | 0,73 $ |
| dont bornes du match | 140 | 0,66 $ |
| dont classification des temps forts | 101 | 0,29 $ |
| Replays du 13/11 (sans les bornes), par match | 199 à 227 | 1,42 à 2,00 $ |

Le run du 11/11 a duré 150 min (analyse 115 min, montage du joueur n°16 35 min) pour 5/5 vrais buts et 5/5 équipes justes.

**Facture Google** (30 jours, du 13/10 au 12/11) : 266,84 € net, dont 260,15 € pour Gemini 3.8 Flash (328,8 millions de jetons, environ 0,79 € par million). Le taux de change retenu dans les calculs est de 0,90 € pour 1 $.

**Mise en cache des consignes : abandonnée** (13/11). Google n'a rien mis en cache automatiquement (0 % sur 3 matchs). Les consignes font 150 à 750 jetons, sous le minimum de mise en cache (1 024 jetons pour 2.5 Flash, non vérifié pour 3.8), et changent dès leur première phrase (heure du candidat). Une question en compte environ 9 600 au total : le gain maximal serait de 0,05 à 0,10 $ par match (estimation).

À vérifier :

- 6,55 € facturés pour Gemini 3.0 Pro, et 255 questions Gemini 3.1 Pro visibles sur 28 jours, alors que le verrou impose 3.8 Flash depuis le 25/10 : origine non identifiée.
- Tarif de Flex en 2027 : doublement annoncé, source non vérifiée.
- Coût GPU par match : environ 1 € retenu par hypothèse, sans devis.

## Licences : Ultralytics bloque le lancement (12-13 novembre)

La détection des joueurs et du ballon utilise Ultralytics YOLO, sous licence AGPL-3.0 : un service commercial doit soit publier tout son code, soit acheter la licence Enterprise, dont le prix est sur devis.

Ce que confirment les pages Ultralytics consultées le 13/11 :

- Les offres Free (0 $) et Pro (29 $ par siège et par mois) restent sous AGPL : elles ne règlent pas le problème.
- Seule l'offre Enterprise couvre l'usage commercial, le déploiement privé, l'entraînement de modèles commerciaux et l'usage interne propriétaire ; c'est un abonnement annuel.
- Selon Ultralytics, même un usage interne sans publication demande la licence ; la licence est détenue par l'entité qui développe le produit.

Décisions :

- **Demande de devis Enterprise** préparée (anglais), à envoyer par le bouton « Demande un devis » ; l'adresse licensing@ultralytics.com renvoie une erreur selon leur forum.
- **Pas de lancement commercial ni de pilote avec des clubs avant la réponse** ; les tests sur nos propres vidéos restent possibles.
- **Plan B** si le prix est trop élevé : remplacer le détecteur par un modèle sous licence Apache 2.0 (RT-DETR, YOLOX ou RF-DETR, licences à vérifier), avec revalidation complète sur les 10 matchs de référence.

Autres points de l'inventaire des licences, à faire trancher par un juriste :

| Composant | Licence | Point à vérifier |
| --- | --- | --- |
| Modèles maison (yolov8_football, basketball, handball) | Considérés AGPL par Ultralytics | Couverts par la licence Enterprise ? |
| fpdf2 | LGPL-3.0 | Conditions d'usage |
| FFmpeg (version Kaggle) | GPL | Usage comme programme séparé, sans distribution |
| Poids OSNet (ré-identification) | Entraînés sur MSMT17 | Conditions du jeu de données |
| Poids PARSeq maison | Code Apache-2.0 | Origine des données d'entraînement |
| DeepSort, ByteTrack, torchreid | MIT présumé | Non installés lors de l'inventaire |
| API Gemini, Anthropic, Stripe, Cloudflare R2, RunPod | Conditions d'utilisation | Âge minimum, données, usage commercial |

## Business plan, prix, TVA et statut

Le jeton à 5 € est le prix minimum retenu ; le projet ne deviendra une activité d'indépendant complémentaire que si la licence YOLO est compatible avec une petite activité pilote.

**Formules et coûts par match** (0,90 € pour 1 $, GPU environ 1 € par hypothèse) :

| Formule | Jetons | Coût 2026 | Coût 2027 | Marge 2027 avec TVA 21 % | Marge 2027 en franchise de TVA |
| --- | --- | --- | --- | --- | --- |
| Match sous 48 h (Flex) | 2 | environ 2,60 € | environ 4,10 € | environ +3,20 € | environ +4,80 € |
| Match rapide (standard) | 3 | environ 3,50 € | environ 6,00 € | environ +4,90 € | environ +7,30 € |
| Joueur en plus | 1 | environ 0,30 € | environ 0,35 € | environ +3,30 € | — |

Toutes ces marges sont des estimations, au jeton de 5 € et frais Stripe compris (1,5 % + 0,25 €). La colonne « franchise » compte par prudence 21 % de TVA belge en plus sur les achats étrangers (Gemini, GPU).

Décisions et hypothèses :

- **Jetons vendus par lots** ; les remises de volume coûtent plus qu'elles ne rapportent avec Stripe.
- **Piste de modèle** : abonnement club (par exemple 40 à 60 € par mois et par équipe, environ 4 matchs) ; montage joueur en option pour les familles ; arrêt probable de juin à août.
- **Statut envisagé** : indépendant complémentaire, sous le régime de franchise de TVA (seuil de 25 000 € de chiffre d'affaires annuel selon les sources consultées le 13/11, à confirmer par un comptable).
- **Démarrer sans licence YOLO** : déconseillé, car une licence achetée plus tard ne couvre pas forcément l'usage passé.
- **Retour sur 1 000 € investis** (frais fixes environ 70 € par mois hors licence, marge moyenne 4 € par match) : environ 20 mois à 30 matchs par mois, 4 mois à 80, moins de 2 mois à 200.

À obtenir avant de décider : le devis Ultralytics ; les cotisations sociales d'indépendant complémentaire ; la règle de TVA sur les services achetés à l'étranger en franchise ; l'avis d'un juriste sur le RGPD (images de mineurs) ; l'avis de 2 ou 3 clubs sur le prix.

## Fichiers modifiés et état de GitHub

Deux fichiers du dépôt ont changé le 13/11 et doivent être remplacés sur GitHub avant le prochain run ; pipeline.py et export/pdf.py sont à jour depuis le 11/11.

| Fichier | Où | Dernière modification | À faire |
| --- | --- | --- | --- |
| ai/gemini_validator.py | GitHub | 13/11 : 128 questions par défaut, adresse officielle de Google, délai de 10 min par requête, reprise des erreurs « 500 », Flex | Remplacer |
| ai/attribution_equipes.py | GitHub | 13/11 : buts en parallèle, adresse officielle, délai de 10 min | Remplacer |
| requirements.txt | GitHub | — | Mettre google-genai>=3.2.0 (fichier non disponible ici) |
| pipeline.py, export/pdf.py | GitHub | 11/11 : attribution des équipes (étape 11b) et rapport | À jour |
| scoutia_prod_un_match.py | Notebook de production | 13/11 : 128 questions à la fois | Remplacer la cellule |
| test_resolution_gemini_cellule.py | Entrée des tests | 13/11 : 128 questions à la fois | Remplacer |
| test_flex_matchs.py | Notebook de test | 13/11 : 128 questions, attente Flex de 30 min | Prêt à lancer |
| test_parallele_standard.py | Notebook de test | 13/11 | Validé (3 matchs) |
| test_flex_latence.py | Notebook de test | 13/11 | Exécuté |
| inventaire_licences.py | Notebook | 12/11 | Exécuté ; trois défauts d'affichage connus |

Réglages par défaut en production : résolution moyenne pour les coups d'envoi, basse pour les temps forts, escalade en normale ; attribution en normale, Q2 à 1 image par seconde ; 128 questions à la fois ; tarif standard ; lecture des maillots, lecture du buteur, voie « ballon » et résumé IA désactivés.

## Points ouverts et prochaines étapes

La prochaine décision dépend du devis Ultralytics ; côté technique, il reste à valider Flex sur un match complet avec le repli à 30 minutes.

**Technique**

- [ ] Envoyer la fin du journal du run Flex en cours (128 questions à la fois, sans repli) pour mesurer la durée de la classification et des vagues suivantes.
- [ ] Mettre à jour GitHub (ai/gemini_validator.py, ai/attribution_equipes.py, requirements.txt), puis lancer test_flex_matchs.py avec l'attente Flex de 30 min, si possible à une autre heure.
- [ ] Vérifier sur la facture du jour du run que les questions Flex sont bien facturées au tarif Flex.
- [ ] Identifier l'origine des questions Gemini 3.0 et 3.1 Pro (graphique de la ligne « Gemini 3.1 Pro » dans AI Studio) ; si elles continuent après le 25/10, trouver le script qui contourne le verrou.
- [ ] Confirmer la révocation des clés Gemini et Claude écrites en clair dans d'anciens notebooks (signalé le 01/11) et leur passage en secrets Kaggle.

**Licence et juridique**

- [ ] Envoyer la demande de devis Enterprise à Ultralytics, avec les questions sur la licence au nom d'un indépendant et sur une offre « licence seule ».
- [ ] Faire trancher l'inventaire des licences par un juriste (voir la section Licences), ainsi que le RGPD pour les images de mineurs.

**Business**

- [ ] Demander à un comptable : régime de franchise de TVA, TVA sur les achats étrangers, cotisations d'indépendant complémentaire.
- [ ] Une fois la licence réglée : essai gratuit d'un mois avec 2 ou 3 clubs pour valider le prix de 40 à 60 € par mois.
- [ ] Refaire le plan d'affaires avec le prix réel de la licence et le régime de TVA.
