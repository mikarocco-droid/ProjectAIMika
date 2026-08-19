# analysis/detect_ko_vision.py
# -*- coding: utf-8 -*-
#
# Détection du coup d'envoi par IA vision — étage final du pipeline.
#
# Architecture validée sur les 10 matchs de test (cf. JALON, sections 13
# à 19) :
#
#   RandomForest (ko_features.py, déjà entraîné)
#         │  filtre à top-10 candidats par match (recall mesuré : 99%,
#         │  cf. jalon section 16 — top-3/top-5 insuffisants et instables)
#         ▼
#   Pour chaque candidat : extraction de 3 images
#   (juste_avant t-2s, candidat t0, juste_apres t+2s)
#         │  motif recherché : formation statique → frappe →
#         │  dispersion générale immédiate (validé 8/9 sur matchs Veo,
#         │  1/1 sur caméra manuelle avec confiance dégradée)
#         ▼
#   IA vision (Gemini, même moteur que detect_teams_preview.py)
#         │  compare les 3 images, juge si la bascule est visible
#         ▼
#   Sortie : candidate (t) + confidence + ambiguity
#   (contrat fonctionnel V1, cf. jalon section 2 — ne jamais transformer
#   une ambiguïté réelle en confiance artificielle)
#
# IMPORTANT : ce module ne remplace pas le RandomForest, il le complète.
# Le RF reste responsable de réduire l'espace de recherche (533 candidats
# → 10) ; ce module départage les 10 candidats retenus.

import os
import cv2
import base64
import json


# ─────────────────────────────────────────
# EXTRACTION DES SÉQUENCES À 3 IMAGES
# ─────────────────────────────────────────

def extraire_sequence_candidat(video_path, t_candidat, fps_source=None, offset_s=2.0):
    """
    Extrait 3 images autour d'un instant candidat :
      juste_avant (t-offset_s), candidat (t0), juste_apres (t+offset_s)

    Protocole validé (jalon section 18) avec offset_s=2.0 (valeur par
    défaut, comportement de production INCHANGÉ) : suffisant dans la
    grande majorité des cas. Limite connue : deux candidats à moins de
    5-6s l'un de l'autre peuvent avoir des fenêtres qui se chevauchent
    (cf. l'erreur sur Stembert) — à surveiller si le top-10 contient
    des candidats très rapprochés temporellement.

    offset_s : PARAMÈTRE DE TEST — cf. cas Raeren où le candidat retenu
    (404.8s) était décalé d'environ 4s par rapport au vrai KO (~409s,
    pas 410s comme indiqué dans le référentiel historique). Avec
    offset_s=2.0, la fenêtre (402.8/404.8/406.8) ratait presque
    entièrement le plateau de mouvement réel (406-410s). Un offset plus
    large (ex. 4.0) pourrait capter la bascule même avec ce genre de
    décalage — À VALIDER par benchmark contrôlé avant tout changement
    du défaut de production (cf. jalon, benchmark ±2 vs ±4).

    Retourne un dict {"juste_avant": np.array, "candidat": np.array,
    "juste_apres": np.array} (frames BGR, format OpenCV), ou None pour
    les images qui n'ont pas pu être lues.
    """
    cap = cv2.VideoCapture(video_path)
    if fps_source is None:
        fps_source = cap.get(cv2.CAP_PROP_FPS)

    offsets = {"juste_avant": -offset_s, "candidat": 0.0, "juste_apres": offset_s}
    frames = {}
    for label, offset in offsets.items():
        t = max(0, t_candidat + offset)
        frame_num = int(t * fps_source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        frames[label] = frame if ret else None

    cap.release()
    return frames


def _frame_to_base64(frame):
    """Encode une frame OpenCV (BGR) en base64 JPEG, pour l'API Gemini."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


# ─────────────────────────────────────────
# APPEL IA VISION (GEMINI)
# ─────────────────────────────────────────
#
# ⚠️ HISTORIQUE CRITIQUE — NE JAMAIS REFORMULER CE PROMPT EN "HISTOIRE
# GLOBALE" SANS RE-VALIDER SUR LES 9 MATCHS AVEC gemini-2.5-flash ⚠️
#
# Ce prompt "checklist" (vérification image par image : image1=statique,
# image2=frappe, image3=dispersion) A DÉJÀ ÉTÉ reformulé une fois en
# version "histoire globale" (jugement holistique des 3 images comme un
# seul récit, plutôt qu'une vérification stricte critère par critère).
# Testé empiriquement sur gemini-2.5-flash (session antérieure), le
# résultat était netament PIRE, pas meilleur :
#
#   Match          | Checklist (ancien) | Histoire globale (testé)
#   Andrimont      | ~2-3 faux positifs/10 | 5 faux positifs/10
#   Stembert       | ~2 faux positifs/10   | 4 faux positifs/10 (+ vrai KO raté)
#   MineroisSter   | ~3 faux positifs/10   | 7 faux positifs/10
#
# Cause : assouplir le critère donne au modèle la permission implicite
# de rationaliser presque n'importe quelle action de jeu comme
# "cohérente avec un début de match", puisque la plupart des
# tirs/contre-attaques ont un moment où plusieurs joueurs avancent
# ensemble. Sur gemini-3.1-pro-preview, le même prompt "histoire
# globale" donnait un résultat different (trop conservateur, pas trop
# permissif) — donc l'effet dépend fortement du modèle, jamais neutre.
#
# CETTE LEÇON A ÉTÉ OUBLIÉE UNE FOIS DÉJÀ pendant cette investigation
# (la version "histoire globale" s'est retrouvée réintroduite dans ce
# fichier sans que quiconque se souvienne du test qui l'avait invalidée
# — découvert seulement après coup, en fouillant les transcripts d'une
# session antérieure). Le prompt "checklist" a été restauré comme
# référence stable.
#
# ITÉRATION SUIVANTE (celle ci-dessous, actuellement en place) :
# structure "OBLIGATOIRE vs SECONDAIRE" explicite, différente à la fois
# du "checklist" original ET de la version "histoire globale" qui avait
# échoué. Motivée par une confabulation précise observée (Andrimont
# t=194.9s : Gemini affirmait "les deux équipes se dispersent" alors
# qu'une seule équipe était présente, probablement un regroupement
# après un but). Distinction clé avec la version qui avait échoué :
# celle-ci donnait la permission d'IGNORER un détail peu clair
# ("ne rejette pas... parce qu'un détail n'est pas parfaitement
# visible") ; celle-ci au contraire liste explicitement 3 faits
# OBLIGATOIRES (deux équipes distinctes, chacune organisée séparément
# en image 1, vraie mise en mouvement des DEUX équipes en image 3) à
# répondre "non" si non vérifiables, et seulement 5 détails SECONDAIRES
# (ballon, couleurs imparfaites, arbitre) qui ne doivent jamais bloquer
# un "oui". Le critère "franchir la ligne médiane" testé dans une
# itération intermédiaire a été retiré (jugé possiblement trop strict
# pour une fenêtre de 2s) au profit d'un critère de mise en mouvement
# réelle des deux équipes, plus réaliste.
#
# ⚠️ CETTE ITÉRATION N'A PAS ENCORE ÉTÉ VALIDÉE EMPIRIQUEMENT — testée
# uniquement par relecture, pas encore relancée sur les 18 candidats de
# référence (10 faux positifs connus avec le prompt checklist simple).
# NE PAS considérer comme acquise avant ce test.
# ─────────────────────────────────────────

PROMPT_KO_VISION = """Tu vas analyser 3 images consécutives extraites d'une vidéo d'un match de football amateur.

Les images sont espacées de 2 secondes :

* IMAGE 1 = environ t-2 secondes
* IMAGE 2 = instant candidat
* IMAGE 3 = environ t+2 secondes

═══════════════════════════════════════════════════
VÉRIFICATIONS PRÉALABLES OBLIGATOIRES — À FAIRE EN PREMIER, AVANT TOUTE AUTRE ANALYSE
═══════════════════════════════════════════════════

Avant de lire le reste des instructions, réponds à ces deux questions simples. Si l'une des deux donne "oui", la séquence est automatiquement REJETÉE (avant_compatible=false, transition_visible=false) — inutile d'analyser quoi que ce soit d'autre, n'essaie pas de compenser par le reste de la scène.

QUESTION PRÉALABLE 1 — Un but est-il visible sur IMAGE 1 OU IMAGE 2 ?
Regarde IMAGE 1 ET IMAGE 2 séparément. Si tu vois un cadre de but (poteaux verticaux, barre transversale, filet) dans L'UNE OU L'AUTRE de ces deux images, peu importe sa taille ou sa distance apparente, cela signifie que la scène se déroule près d'un but. La ligne au sol que tu pourrais prendre pour la ligne médiane est alors presque certainement la ligne de la surface de réparation ou de but — PAS la ligne médiane. Un vrai coup d'envoi ne montre JAMAIS un but dans le cadre, quel que soit l'angle de caméra, car il est filmé depuis le centre du terrain, loin des deux buts. Si un but est visible sur IMAGE 1 ou IMAGE 2 → réponds "oui" à cette question → rejet automatique.

QUESTION PRÉALABLE 2 — Les joueurs sont-ils mélangés sur IMAGE 1 ?
Regarde uniquement IMAGE 1. Repère les couleurs de maillots des deux équipes. Les joueurs des deux couleurs apparaissent-ils mélangés ou entremêlés entre eux (proches les uns des autres sans regroupement nettement séparé par équipe et par côté du terrain), plutôt que clairement regroupés chacun de leur côté ? Un vrai coup d'envoi montre TOUJOURS chaque équipe regroupée distinctement de son côté — jamais des joueurs des deux couleurs mélangés ensemble au même endroit. Si les maillots des deux couleurs sont mélangés sur IMAGE 1 → réponds "oui" à cette question → rejet automatique.

Ces deux vérifications priment sur tout le reste de l'analyse ci-dessous. Un ballon au centre, un arbitre présent, ou une frappe visible ne compensent JAMAIS un "oui" à l'une de ces deux questions.

═══════════════════════════════════════════════════

OBJECTIF :
Déterminer si ces 3 images montrent réellement le moment où le ballon est mis en jeu depuis le centre du terrain, c'est-à-dire le début effectif d'une séquence de jeu après une situation de préparation au coup d'envoi.

IMPORTANT :
Ne cherche PAS simplement une scène qui "ressemble" à un coup d'envoi.
Nous voulons distinguer :
A) une véritable mise en jeu : joueurs déjà prêts → ballon joué → joueurs commencent réellement à jouer
de
B) une préparation au coup d'envoi : joueurs qui arrivent encore, se rassemblent, font un cri de guerre, discutent, se placent progressivement, puis seulement plus tard le vrai coup d'envoi
de
C) une autre reprise de jeu ou une phase de jeu déjà commencée.
D) la géométrie et la position réelle sur le terrain
Lis les 3 images comme UNE SEULE séquence temporelle continue.

Chaque affirmation doit être fondée uniquement sur ce qui est réellement visible dans les images. Ne déduis jamais un événement qui n'est pas observable.
Ne suppose jamais qu'un élément est présent parce qu'il serait normalement
présent dans cette situation.													 

════════════════════════════════════

1. IMAGE 1 — LA SITUATION EST-ELLE DÉJÀ PRÊTE ?
   ════════════════════════════════════

Avant de considérer la séquence comme compatible avec une mise en jeu, examine attentivement l'état des joueurs dans IMAGE 1.

Le ballon au centre et la présence de l'arbitre ne suffisent PAS.

IMAGE 1 doit montrer une situation déjà suffisamment stabilisée et immédiatement compatible avec une mise en jeu.

ACCEPTE comme "avant_compatible" si :

* le ballon est visible au sol dans ou très près de la zone du point central ;
* les joueurs des deux équipes sont déjà majoritairement en position sur le terrain ;
* les deux équipes sont clairement distinctes ;
* la disposition générale est compatible avec une mise en jeu depuis le centre ;
* il n'y a pas de signe évident que les joueurs sont encore en train d'arriver ou de se préparer.

REJETTE "avant_compatible" si IMAGE 1 montre clairement :

* des joueurs qui entrent encore sur le terrain ;
* plusieurs joueurs qui rejoignent encore leurs coéquipiers ;
* une équipe qui forme encore un cercle ou un cri de guerre ;
* une équipe regroupée pour une photo ;
* une discussion collective ou une préparation tactique ;
* des joueurs encore en train de s'échauffer ;
* une mise en place progressive clairement visible ;
* une équipe dont une partie importante n'est pas encore en position ;
* une situation où le match semble encore en phase de préparation plutôt qu'en attente immédiate de la mise en jeu.

POINT TRÈS IMPORTANT :

Le fait que le ballon soit déjà au centre ne signifie PAS que le coup d'envoi est imminent.

Le ballon peut être placé au centre alors que les joueurs sont encore en train de se rassembler, de faire un cri de guerre ou de se préparer.

Exemple :
Si une équipe est encore regroupée pour son cri de guerre et que plusieurs joueurs de l'autre équipe arrivent encore sur le terrain, alors IMAGE 1 n'est PAS une situation immédiatement prête au coup d'envoi, même si le ballon et l'arbitre sont déjà au centre.

ATTENTION AU VOCABULAIRE :
"Regroupée" et "stabilisée" ne sont PAS compatibles entre eux. Si tu décris une équipe comme "regroupée", "en cercle", "rassemblée" ou "en cluster", le fait d'ajouter qu'elle est "stable" ou "stabilisée" à ce moment précis NE COMPENSE PAS le rejet — cela reste un REJET de avant_compatible. Une équipe prête pour un coup d'envoi est RÉPARTIE sur sa moitié de terrain (joueurs à des positions distinctes), jamais regroupée en un seul point, peu importe à quel point ce regroupement semble immobile ou stable.

CRITÈRE DE SÉPARATION PAR MOITIÉ DE TERRAIN, TRÈS IMPORTANT :
Les règles du jeu imposent que, lors d'un coup d'envoi, TOUS les joueurs de l'équipe qui ne botte pas soient dans LEUR PROPRE moitié de terrain, et que l'équipe qui botte ait également ses joueurs dans sa propre moitié (à l'exception du joueur qui s'apprête à jouer le ballon). Une formation de coup d'envoi valide montre donc une séparation nette de part et d'autre de la ligne médiane.

Si tu observes un ou plusieurs joueurs d'une équipe positionnés dans la moitié de terrain de l'équipe adverse (au-delà de la ligne médiane, mélangés avec des joueurs de l'autre équipe, ou dispersés sur l'ensemble du terrain sans respecter cette séparation), ce N'EST PAS une formation de coup d'envoi valide. Réponds "non" à "avant_compatible" dans ce cas, même si un rond central est visible, même si le ballon semble proche du centre, et même si les deux équipes sont par ailleurs visuellement distinctes par leurs couleurs.

Ne confonds jamais "les deux équipes sont visibles et distinctes" (couleurs différentes) avec "les deux équipes sont correctement séparées par moitié de terrain" (positionnement). Les deux conditions sont nécessaires, ni l'une ni l'autre ne suffit seule.

CAS PARTICULIER, TRÈS FRÉQUENT — LES DEUX ÉQUIPES DU MÊME CÔTÉ :
Vérifie spécifiquement si les DEUX équipes (ou la majorité de leurs joueurs visibles) se trouvent du MÊME côté de la ligne médiane, c'est-à-dire dans la MÊME moitié de terrain plutôt que chacune dans sa propre moitié. Ceci est différent du cas d'un simple joueur isolé qui dépasse la ligne : ici, c'est la majorité ou la totalité des deux équipes qui est concentrée d'un seul côté.

Ce cas de figure N'EST JAMAIS une formation de coup d'envoi, même si :
- un rond central est visible ;
- le ballon semble proche du centre ou de la ligne médiane ;
- les deux équipes restent visuellement distinctes par leurs couleurs ;
- la situation paraît stable ou immobile.

Une telle disposition (les deux équipes groupées du même côté) indique une phase de jeu normale — par exemple une équipe qui vient de récupérer le ballon près de sa propre surface, une remise en jeu, ou toute autre situation où le jeu s'est simplement déplacé vers un côté du terrain — jamais un coup d'envoi.

Pour vérifier ce point, essaie de compter, même approximativement, combien de joueurs de chaque équipe se trouvent de chaque côté de la ligne médiane. Si l'écrasante majorité des joueurs des DEUX équipes se trouve du même côté, réponds "non" à "avant_compatible", quelle que soit la position exacte du ballon.
════════════════════════════════════════
ÉTAPE 0 — IDENTIFIER LA GÉOMÉTRIE DU TERRAIN
════════════════════════════════════════

Avant de décider si la séquence est un coup d'envoi, identifie autant que
possible la géométrie réelle du terrain.

Le centre du terrain correspond à l'intersection :
- de la ligne médiane ;
- et de l'axe longitudinal du terrain.

Il est normalement matérialisé par le rond central.

Pour localiser le centre, utilise en priorité :
- le rond central ;
- la ligne médiane ;
- les lignes de touche ;
- les lignes de but ;
- les surfaces de réparation ;
- les autres lignes du terrain permettant de reconstruire la géométrie.

IMPORTANT :
																						
												
																				  
																								   

Le centre de l'IMAGE n'est PAS nécessairement le centre du TERRAIN.

La caméra peut filmer le terrain depuis un angle, une extrémité ou une
															  
															  
										 
														  
											   
													
position latérale.
																													   

Ne considère donc JAMAIS qu'un ballon est au centre simplement parce qu'il
est au milieu de l'image.

Si le rond central est visible, utilise-le comme référence prioritaire.

Si le rond central n'est pas visible, utilise les autres lignes du terrain
pour estimer sa position uniquement si cette estimation est suffisamment
fiable.

Si la géométrie visible ne permet pas de déterminer avec suffisamment de
certitude où se trouve le centre du terrain, ne suppose pas que le ballon
est au centre.

════════════════════════════════════════
RÈGLE ANTI-HALLUCINATION, TRÈS IMPORTANTE
════════════════════════════════════════

Un but visible dans l'image, une surface de réparation, ou simplement un
espace dégagé d'herbe NE SONT PAS le rond central. Le rond central est
une ligne courbe blanche spécifique, visible au sol, formant un cercle
autour du point central — pas n'importe quelle zone vide du terrain.

Avant d'affirmer "le rond central est visible" ou "les joueurs sont
positionnés autour du centre", vérifie explicitement :
- Vois-tu une ligne courbe blanche caractéristique du rond central ?
- Vois-tu la ligne médiane droite qui traverse tout le terrain ?
- Si un but (cage, filet, poteaux) est visible dans l'image, le point
  jugé "central" n'est presque certainement PAS le centre du terrain,
  mais une zone proche du but (surface de réparation, dégagement,
  remise en jeu du gardien) — dans ce cas, réponds "non" à toute la
  séquence, quelle que soit l'organisation apparente des joueurs.

Si tu ne peux pas identifier avec certitude le rond central OU la ligne
médiane dans IMAGE 1, tu DOIS répondre que la géométrie ne permet pas de
confirmer un coup d'envoi central — ne complète jamais ce doute par une
supposition sur la disposition des joueurs.

════════════════════════════════════════
ÉTAPE 1 — EXCLURE LES AUTRES REMISES EN JEU
════════════════════════════════════════

Le fait qu'un ballon soit immobile et que des joueurs soient organisés ne
suffit PAS.

La séquence doit correspondre à une mise en jeu depuis LE CENTRE DU TERRAIN.

Rejette la séquence si elle correspond visuellement à une autre situation,
notamment :
												

- corner ;
- coup franc ;
- penalty ;
- sortie de but ;
- touche ;
- reprise de jeu depuis une autre zone ;
- ballon situé dans ou près d'une surface de réparation ;
- ballon situé près d'une ligne de touche ;
- ballon situé près d'un coin du terrain ;
- jeu déjà en cours.

CAS TRÈS IMPORTANT — CORNER :

Si la géométrie du terrain montre que le ballon est situé près d'un angle
du terrain ou d'un drapeau de corner, ce n'est PAS un coup d'envoi.
																											

Même si :
- le ballon est immobile ;
- l'arbitre est présent ;
- plusieurs joueurs sont organisés ;
- les joueurs commencent ensuite à courir ;
- la séquence ressemble visuellement à une mise en jeu ;

le résultat doit être false si le ballon est dans la zone d'un corner.

Ne confonds jamais une zone rectangulaire, une surface de réparation ou une
autre partie du terrain avec le centre du terrain.

CAS IMPORTANT — SURFACE DE RÉPARATION :
																			 
																						 

Une zone qui semble être "au milieu" de l'image peut en réalité être une
surface de réparation.

Utilise les lignes du terrain pour déterminer sa position réelle.
				
════════════════════════════════════
2. IMAGE 1 — POSITION DU BALLON
════════════════════════════════════

Vérifie séparément la position du ballon.

Le ballon doit être :

* visible ;
* posé au sol ou clairement placé dans la zone centrale ;
* à proximité du point central / rond central.

Si le ballon est clairement ailleurs sur le terrain, ou si sa position ne peut pas être déterminée, la séquence n'est pas compatible avec une mise en jeu depuis le centre.

Cependant, ne rejette PAS uniquement parce que le ballon est difficile à voir si sa position centrale est clairement identifiable autrement.

════════════════════════════════════
3. IMAGE 2 — LA MISE EN JEU EST-ELLE RÉELLEMENT VISIBLE ?
════════════════════════════════════

IMAGE 2 correspond à l'instant candidat.

Cherche des preuves visuelles que le ballon est effectivement mis en jeu depuis le centre.

Une mise en jeu visible peut être indiquée par :

* un joueur qui frappe clairement le ballon ;
* le ballon qui commence clairement à se déplacer depuis le point central ;
* une action immédiatement identifiable comme le déclenchement du jeu depuis le centre.

Ne considère PAS comme preuve suffisante :

* simplement un joueur proche du ballon ;
* un joueur qui semble prêt à frapper ;
* un ballon posé au centre sans mouvement observable ;
* une simple modification de position des joueurs.

Si aucune mise en jeu réelle n'est observable dans les 3 images, "mise_en_jeu_visible" doit être false.

IMPORTANT :
Ne suppose jamais qu'un joueur a frappé le ballon uniquement parce que le ballon est ensuite ailleurs.

PRÉCISION IMPORTANTE SUR LE TIMING :
La frappe ne tombe pas forcément exactement au moment de l'IMAGE 2 — les 3 images sont espacées de 2 secondes, et l'instant candidat n'est qu'une estimation, pas un instant exact au trentième de seconde. Si l'IMAGE 1 montre une situation "avant_compatible" claire, et que l'IMAGE 3 montre sans ambiguïté un jeu déjà engagé (ballon déplacé depuis le centre, joueurs des deux équipes en mouvement), alors "mise_en_jeu_visible" doit être considéré comme true MÊME SI le ballon apparaît encore immobile sur l'IMAGE 2 elle-même — la mise en jeu a simplement eu lieu entre l'IMAGE 2 et l'IMAGE 3. Ne rejette pas uniquement parce que l'action de frappe elle-même n'est pas figée exactement sur l'IMAGE 2 ; ce qui compte est qu'une progression réelle et cohérente existe entre IMAGE 1 (prêt) et IMAGE 3 (jeu engagé).

════════════════════════════════════
4. IMAGE 3 — LE JEU COMMENCE-T-IL RÉELLEMENT ?
════════════════════════════════════

IMAGE 3 doit confirmer la transition vers le jeu actif.

Cherche :

* le ballon désormais en mouvement ou déplacé depuis le centre ;
* plusieurs joueurs qui commencent réellement à se déplacer ;
* une évolution cohérente de la position des joueurs des deux équipes ;
* une transition entre une situation préparée et une situation de jeu actif.

Un simple changement de position d'un seul joueur ne suffit PAS.

Une image 3 différente de l'image 1 ne prouve PAS à elle seule qu'il s'agit du coup d'envoi.

════════════════════════════════════
5. DIFFÉRENCE ESSENTIELLE : PRÉPARATION VS VRAI COUP D'ENVOI
════════════════════════════════════

C'est le point le plus important de l'analyse.

Tu dois distinguer :

SCÉNARIO A — VRAI COUP D'ENVOI

IMAGE 1 :
Les joueurs sont déjà prêts et la situation est stabilisée.

IMAGE 2 :
Le ballon est effectivement joué depuis le centre.

IMAGE 3 :
Les joueurs commencent réellement à jouer.

→ Compatible avec un vrai coup d'envoi.

SCÉNARIO B — PRÉPARATION AU COUP D'ENVOI

IMAGE 1 :
Les joueurs arrivent encore, se regroupent, font un cri de guerre, discutent ou se mettent encore en place.

IMAGE 2 :
La situation évolue.

IMAGE 3 :
Les joueurs commencent à se disperser ou à jouer.

→ REJETER.

Même si le ballon était déjà au centre.

SCÉNARIO C — AUTRE REPRISE / JEU DÉJÀ EN COURS

IMAGE 1 :
Le ballon est déjà en jeu, les joueurs sont déjà engagés dans une action ou la configuration n'est pas celle d'une mise en jeu depuis le centre.

→ REJETER.

════════════════════════════════════
6. NE PAS INVENTER DE CONTEXTE
════════════════════════════════════

Tu ne connais pas le chronomètre réel de la vidéo.

Tu ne sais pas si cette mise en jeu correspond :

* au début du match ;
* au début de la deuxième mi-temps ;
* à une reprise après un but.

Tu ne dois donc PAS essayer de déterminer cela à partir d'hypothèses.

L'objectif est uniquement de déterminer si les images montrent une véritable mise en jeu depuis le centre après une situation de préparation stabilisée.

Ne dis jamais :
"c'est probablement le début du match"
si cette information n'est pas visible.

Ne dis jamais :
"c'est probablement une reprise après un but"
si cette information n'est pas visible.

════════════════════════════════════
7. RÈGLE DE DÉCISION
════════════════════════════════════

Pour répondre TRUE, les conditions suivantes doivent être réunies :

1. avant_compatible = true
2. mise_en_jeu_visible = true
3. transition_apres = true

Si une condition essentielle n'est pas clairement vérifiable dans les images, réponds false.

Ne compense jamais une condition manquante par une forte impression générale.

Par exemple :

* ballon au centre + arbitre + joueurs = PAS automatiquement true ;
* ballon au centre + joueurs encore en cri de guerre = false ;
* joueurs prêts + frappe clairement visible + transition collective = true.

════════════════════════════════════
8. CONFIANCE
════════════════════════════════════

La confiance doit représenter la confiance dans la décision globale basée sur les preuves visuelles.

Elle ne doit PAS être systématiquement élevée.

Utilise approximativement :

* 0.90–1.00 : les trois conditions sont clairement visibles ;
* 0.70–0.89 : la décision est assez claire mais certains éléments sont partiellement visibles ;
* 0.50–0.69 : plusieurs éléments sont ambigus ;
* < 0.50 : les images ne permettent pas une décision fiable.

Si une condition essentielle manque clairement, réponds false et explique précisément laquelle manque.

════════════════════════════════════
9. RAISONNEMENT
════════════════════════════════════

Le raisonnement doit être court et factuel.

Ne répète pas un texte générique du type :
"les joueurs se dispersent, indiquant le début du jeu."

Indique précisément :

* l'état observable des joueurs dans IMAGE 1 ;
* la position observable du ballon ;
* ce qui est réellement observable dans IMAGE 2 ;
* ce qui change réellement dans IMAGE 3 ;
* et, si rejet, la condition précise qui échoue.

Ne mentionne jamais une information qui n'est pas visible dans les images.

Réponds UNIQUEMENT avec ce JSON valide, sans texte avant ni après :

{
"avant_compatible": true,
"mise_en_jeu_visible": true,
"transition_apres": true,
"transition_visible": true,
"confidence": 0.92,
"raisonnement": "Description courte et factuelle des éléments réellement observables dans les trois images."
}

RÈGLE FINALE :
Ne réponds jamais true simplement parce que la scène ressemble à un coup d'envoi.
Pour répondre true, il faut voir une séquence cohérente :
FORMATION DÉJÀ PRÊTE → MISE EN JEU RÉELLEMENT OBSERVABLE → DÉBUT DU JEU.
Si l'image 1 montre encore une phase de préparation, d'arrivée, de regroupement ou de cri de guerre, rejette la séquence.
"""


# ─────────────────────────────────────────
# PRÉ-FILTRE RAPIDE (1 image, avant le vote à 3 images)
# ─────────────────────────────────────────
# Objectif de coût : rejeter à bas prix (1 seul appel, pas 2-3) les
# candidats clairement hors sujet (tir, remise en jeu localisée,
# célébration, plan de banc de touche...), sans jamais risquer de
# rejeter à tort un vrai KO. Conçu délibérément PERMISSIF — un faux
# positif ici ne coûte que 2-3 appels de vote en plus (le vote
# tranchera correctement) ; un faux négatif ferait perdre le candidat
# définitivement, sans recours. Asymétrie de coût assumée.
#
# Vérifié visuellement (jalon) : le ballon peut être invisible même sur
# un vrai KO (cas Raeren, candidat t=408s) — donc le ballon n'est
# JAMAIS un critère bloquant ici, exactement comme pour le vote à 3
# images.

PROMPT_PREFILTRE_RAPIDE = """Tu vas voir UNE SEULE image d'un match de football amateur. C'est un premier tri RAPIDE, PAS une décision finale — en cas de doute, réponds "oui" (un examen plus approfondi sur plusieurs images suivra de toute façon).

Question : cette image pourrait-elle correspondre à un instant proche d'un coup d'envoi (début de match, début de mi-temps, ou reprise après un but) ?

Signal typique à rechercher : les joueurs des deux équipes semblent organisés de part et d'autre d'une zone centrale du terrain (pas besoin d'un alignement parfait), et/ou un arbitre est visible sur le terrain. Les joueurs sont généralement immobiles ou en attente.

IMPORTANT : le ballon n'a PAS besoin d'être visible — il est souvent petit, caché par un joueur, ou peu contrasté (éclairage nocturne, plan large). Son absence NE DOIT JAMAIS, à elle seule, faire répondre "non".

Ce filtre doit être PERMISSIF : en cas de doute, d'image ambiguë, de mauvais angle, de mauvaise qualité, ou de joueurs partiellement visibles, réponds "oui". Réponds "non" SEULEMENT si l'image montre CLAIREMENT autre chose : une seule équipe visible (pas de formation à deux camps), une célébration de but, un arrêt de jeu localisé dans un coin du terrain (touche, corner, coup franc excentré), un plan de banc de touche ou de tribune sans vue d'ensemble du terrain, etc.

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après :
{
  "formation_plausible": true/false,
  "raisonnement": "1 phrase courte"
}
"""


def _appeler_gemini_avec_retry(client, model, parts, max_tentatives=3, delai_base_s=2.0):
    """
    Appelle client.models.generate_content avec retry + backoff exponentiel
    sur les erreurs TRANSITOIRES (503 UNAVAILABLE, timeout, erreurs réseau).

    Ne retente PAS sur les erreurs qui ne se résoudront jamais toutes seules
    (JSON invalide, réponse vide) - celles-là sont gérées par l'appelant,
    pas ici, puisque retenter avec les mêmes images ne changerait rien.

    Lève l'exception d'origine si toutes les tentatives échouent - à
    l'appelant de l'attraper (comportement inchangé pour lui).

    TEMPÉRATURE = 0.0 (nouveau) : aucune configuration de génération
    n'était appliquée auparavant - l'API utilisait donc son défaut
    (proche de 1.0, mode "créatif"/probabiliste). Suspecté comme cause
    possible à la fois de la confabulation observée (Gemini invente des
    détails précis et absents de l'image, ex: Wanze t=89.25 - scène
    d'échauffement décrite comme un coup d'envoi organisé) ET de la
    variance inter-runs mesurée (mêmes images, même prompt, résultats
    différents selon le tirage). Une tâche d'analyse factuelle d'image
    n'a besoin d'aucune créativité - la température la plus basse
    possible est appropriée ici, contrairement à une tâche générative.
    À VALIDER par un nouveau test sur les mêmes 18 candidats : si la
    confabulation persiste à température 0, la cause est ailleurs
    (limite de perception du modèle sur l'image elle-même, pas un
    effet d'échantillonnage) - distinction importante pour la suite.
    """
    import time
    try:
        from google.genai import types
        config = types.GenerateContentConfig(temperature=0.0)
    except Exception:
        config = None  # si le SDK ne supporte pas ce type, on continue sans

    codes_transitoires = ("503", "UNAVAILABLE", "timeout", "timed out",
                          "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")

    derniere_erreur = None
    for tentative in range(max_tentatives):
        try:
            if config is not None:
                return client.models.generate_content(model=model, contents=[{"parts": parts}], config=config)
            return client.models.generate_content(model=model, contents=[{"parts": parts}])
        except Exception as e:
            derniere_erreur = e
            message = str(e)
            est_transitoire = any(code in message for code in codes_transitoires)

            if not est_transitoire or tentative == max_tentatives - 1:
                raise  # erreur définitive, ou plus de tentatives -> on relance telle quelle

            delai = delai_base_s * (2 ** tentative)  # 2s, 4s, 8s...
            print(f"  [detect_ko_vision] Erreur transitoire ({message[:80]}...) - "
                  f"retry {tentative + 1}/{max_tentatives - 1} dans {delai:.0f}s")
            time.sleep(delai)

    raise derniere_erreur


def demander_gemini_prefiltre(image, client, model="gemini-3.1-pro-preview"):
    """
    Envoie UNE SEULE image (typiquement "juste_avant") à Gemini pour un
    tri rapide et permissif, avant le vote complet à 3 images.

    image : frame OpenCV BGR unique (pas un dict, contrairement à
    demander_gemini_ko).

    Retourne {"formation_plausible": bool, "raisonnement": str}, ou
    None si l'appel échoue — à traiter comme "pas d'avis", donc ne
    JAMAIS bloquer sur un None (cf. utilisation dans _evaluer :
    en cas de None, on procède quand même au vote complet, jamais
    l'inverse).
    """
    if image is None:
        return None

    parts = [{"text": PROMPT_PREFILTRE_RAPIDE}]
    img_b64 = _frame_to_base64(image)
    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    try:
        response = _appeler_gemini_avec_retry(client, model, parts)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        return {
            "formation_plausible": bool(result.get("formation_plausible", True)),  # défaut permissif si clé manquante
            "raisonnement": result.get("raisonnement", ""),
        }
    except Exception as e:
        print(f"  [detect_ko_vision] Erreur Gemini (pré-filtre) : {e}")
        return None


def demander_gemini_ko(images_dict, client, model="gemini-3.1-pro-preview"):
    """
    Envoie les 3 images à Gemini avec le prompt structuré, retourne le
    jugement (transition visible ou non, confiance, raisonnement).

    images_dict : {"juste_avant": frame, "candidat": frame, "juste_apres": frame}
    (frames OpenCV BGR, comme retourné par extraire_sequence_candidat)

    Retourne None si un appel échoue (image manquante, erreur API, JSON
    invalide) — à traiter comme "pas d'avis" par l'appelant, pas comme
    un refus catégorique.
    """
    if any(images_dict.get(k) is None for k in ("juste_avant", "candidat", "juste_apres")):
        return None

    parts = [{"text": PROMPT_KO_VISION}]
    for label in ("juste_avant", "candidat", "juste_apres"):
        img_b64 = _frame_to_base64(images_dict[label])
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    try:
        response = _appeler_gemini_avec_retry(client, model, parts)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        avant = bool(result.get("avant_compatible", False))
        mise = bool(result.get("mise_en_jeu_visible", False))
        transition = bool(result.get("transition_apres", False))
        return {
            # Décision STRICTE recalculée en code — ne fait plus confiance
            # au transition_visible auto-rapporté par Gemini seul, qui peut
            # se contredire avec ses propres sous-critères (cas observé :
            # Andrimont 937/1262/1273/1322 où Gemini identifie lui-même
            # une géométrie incompatible mais votait quand même True).
            "avant_compatible": avant,
            "mise_en_jeu_visible": mise,
            "transition_apres": transition,
            "transition_visible": avant and mise and transition,
            "transition_visible_brute_gemini": bool(result.get("transition_visible", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "raisonnement": result.get("raisonnement", ""),
        }
    except Exception as e:
        print(f"  [detect_ko_vision] Erreur Gemini : {e}")
        return None


# ─────────────────────────────────────────
# VOTE MAJORITAIRE (jusqu'à 5 appels, avec arrêt anticipé généralisé)
# ─────────────────────────────────────────
# MISE A JOUR (suite au test de stabilité inter-runs sur 18 candidats,
# 4 runs identiques) : contrairement à l'hypothèse initiale ("92% des
# candidats stables 3/3 ou 0/3"), plusieurs candidats montrent une
# vraie variance d'échantillonnage avec le prompt actuel (ex: 25%,
# 33%, 67%, 75% de True selon le tirage, sur des images ET un prompt
# strictement identiques). Un cas est particulièrement critique :
# Wanze t=89.25 (33% True sur 3 runs) est chronologiquement AVANT le
# vrai KO du match (980s) - si validé à tort, le pipeline verrouille
# dessus (arrêt anticipé, cf. detecter_ko_par_vision) et ne teste
# JAMAIS le vrai KO, produisant un résultat final faux avec une
# confiance artificiellement élevée.
#
# max_appels passé de 3 à 5, ET la condition d'arrêt anticipé
# généralisée : l'ancienne version s'arrêtait TOUJOURS après 2 votes
# en accord, quelle que soit la valeur de max_appels - augmenter
# max_appels seul n'aurait donc rien changé (le tirage malchanceux
# vecteur du problème est justement "2 True d'affilée dès le début").
# Nouvelle règle : s'arrêter uniquement quand le camp en tête a déjà
# mathématiquement gagné la majorité, compte tenu des appels restants.

def voter_transition(images_dict, client, model="gemini-3.1-pro-preview", max_appels=5):
    """
    Fait jusqu'à max_appels appels indépendants à demander_gemini_ko sur
    les MÊMES images, avec arrêt anticipé dès que la majorité est
    MATHÉMATIQUEMENT acquise (impossible pour les votes restants de
    changer l'issue) - pas simplement "2 votes d'accord", qui ne
    protège pas contre un tirage malchanceux initial.

    Retourne {"decision": bool, "confidence": float, "n_appels": int,
    "votes": [...détail de chaque appel...]} ou None si le premier appel
    échoue déjà (image manquante notamment).
    """
    votes = []
    for _ in range(max_appels):
        jugement = demander_gemini_ko(images_dict, client, model=model)
        if jugement is None:
            if not votes:
                return None  # échec dès le 1er appel -> pas d'avis du tout
            break  # on garde ce qu'on a déjà obtenu
        votes.append(jugement)

        # Arrêt anticipé GÉNÉRALISÉ : s'arrêter seulement si le nombre de
        # votes restants ne peut plus faire basculer la majorité.
        n_true = sum(1 for v in votes if v["transition_visible"])
        n_false = len(votes) - n_true
        appels_restants = max_appels - len(votes)
        if n_true > n_false + appels_restants or n_false > n_true + appels_restants:
            break

    n_true = sum(1 for v in votes if v["transition_visible"])
    n_total = len(votes)
    decision = n_true > n_total / 2  # majorité stricte (2/3, 2/2, ou 1/1 en cas d'échec partiel)

    # Confiance retenue : moyenne des votes allant dans le sens de la décision finale
    confs_majoritaires = [v["confidence"] for v in votes if v["transition_visible"] == decision]
    confidence_moyenne = sum(confs_majoritaires) / len(confs_majoritaires) if confs_majoritaires else 0.0

    return {
        "decision": decision,
        "confidence": confidence_moyenne,
        "n_appels": n_total,
        "votes": votes,
    }


# ─────────────────────────────────────────
# FONCTION PRINCIPALE — CONTRAT candidate + confidence + ambiguity
# ─────────────────────────────────────────

def detecter_ko_par_vision(video_path, top_n_candidats, fps_source=None,
                             model="gemini-3.1-pro-preview", seuil_ambiguite_secondes=10.0,
                             max_appels_par_candidat=5, offset_s=2.0,
                             candidats_deja_evalues=None, utiliser_prefiltre=False,
                             desactiver_arret_anticipe=False):
    """
    Point d'entrée principal. Prend le top-N candidats déjà produit par
    le RandomForest (ko_features.py + le modèle entraîné), départage-les
    par IA vision (vote majoritaire jusqu'à 3 appels par candidat),
    retourne le contrat fonctionnel V1.

    Objectif du projet : le PREMIER coup d'envoi du match (pas "tous les
    KO", pas "le candidat le plus confiant"). Sur une vidéo qui couvre
    plus que la seule phase pré-match (ex. match complet avec but(s)
    marqué(s)), plusieurs candidats peuvent légitimement être de vrais
    coups d'envoi (l'initial + des reprises après but) — dans ce cas,
    seul le plus ancien temporellement est retenu comme timestamp_final.

    TRI CHRONOLOGIQUE PUR (pas par proba RF) : les candidats sont traités
    par ORDRE CHRONOLOGIQUE CROISSANT. Une version hybride (tri par proba
    RF décroissante, avec vérification des antérieurs) a été testée et
    ABANDONNÉE : plus chère (16-20 appels/match contre 8-23 en
    chronologique) sans gain de qualité, car la probabilité RF ne
    corrèle pas avec la position temporelle (jalon section 23). Dès
    qu'un candidat positif est trouvé, seuls les candidats dans la
    fenêtre d'ambiguïté (seuil_ambiguite_secondes) sont encore vérifiés,
    le reste n'est jamais évalué.

    top_n_candidats : liste de dicts [{"t": float, "categorie": str,
    "proba": float}, ...] — l'ordre d'entrée n'importe pas, la fonction
    trie elle-même par t croissant.

    seuil_ambiguite_secondes : si un 2e candidat positif existe à moins
    de ce nombre de secondes du premier retenu, l'ambiguïté est signalée.

    offset_s : décalage (en secondes) des images juste_avant/juste_apres
    par rapport au candidat, transmis à extraire_sequence_candidat.
    Défaut 2.0 = comportement de production validé (jalon section 18-23).
    Valeur PLUS GRANDE en test uniquement (cf. cas Raeren, jalon section
    26) — ne pas changer le défaut avant validation par benchmark
    contrôlé ±2 vs ±4 sur plusieurs matchs.

    candidats_deja_evalues : dict optionnel {t: résultat_évaluation}
    (même structure qu'une entrée de "candidats_evalues" en sortie).
    Si fourni, les candidats déjà présents dans ce dict ne sont PAS
    ré-évalués par Gemini — leur résultat est simplement réutilisé.
    Sert à l'élargissement adaptatif du pool sans jamais payer deux fois
    le même appel (cf. detecter_ko_par_vision_adaptatif ci-dessous).

    utiliser_prefiltre : bool, défaut False (comportement de production
    INCHANGÉ si non spécifié). Si True, un pré-filtre à 1 image (image
    "juste_avant" seule) est appelé AVANT le vote complet à 3 images,
    pour rejeter à bas coût (1 appel au lieu de 2-3) les candidats
    clairement hors sujet. Conçu délibérément permissif (cf.
    PROMPT_PREFILTRE_RAPIDE) — en cas de doute ou d'échec technique,
    procède quand même au vote complet, jamais l'inverse. À valider par
    benchmark avant adoption en défaut de production.

    desactiver_arret_anticipe : bool, défaut False (comportement de
    production INCHANGÉ). Si True, DIAGNOSTIC UNIQUEMENT — évalue TOUS
    les candidats du pool, même après qu'un premier positif ait été
    trouvé (ignore la fenêtre d'ambiguïté pour le SKIP, pas pour le
    choix final). Découverte majeure (jalon section 18) : sur 3 matchs
    testés, le vrai KO n'a JAMAIS été atteint par la politique
    "premier positif = stop", à cause de faux positifs précoces
    (probablement des formations d'avant-match) — ce mode permet de
    vérifier si le vrai KO aurait été correctement accepté s'il avait
    eu l'occasion d'être examiné. Coût : jusqu'à N appels au lieu de
    l'arrêt anticipé habituel, NE PAS utiliser en production.

    Retourne :
    {
        "timestamp_final": float ou None,
        "confidence": "élevée" / "moyenne" / "faible",
        "ambiguity": bool,
        "candidats_evalues": [...détail des candidats RÉELLEMENT évalués,
                               les autres sont listés avec evalue=False...],
    }

    Ne transforme jamais une ambiguïté réelle en confiance élevée
    (contrat fonctionnel V1, jalon section 2).
    """
    try:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("  [detect_ko_vision] GEMINI_API_KEY manquante — impossible de continuer")
            return {
                "timestamp_final": None, "confidence": "faible",
                "ambiguity": True, "candidats_evalues": [],
                "erreur": "GEMINI_API_KEY manquante",
            }
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"  [detect_ko_vision] Gemini indisponible : {e}")
        return {
            "timestamp_final": None, "confidence": "faible",
            "ambiguity": True, "candidats_evalues": [],
            "erreur": str(e),
        }

    candidats_deja_evalues = candidats_deja_evalues or {}

    def _evaluer(c):
        # Si ce candidat a déjà été évalué lors d'une passe précédente
        # (mode adaptatif), réutiliser le résultat SANS rappeler Gemini —
        # évite tout appel redondant lors d'un élargissement de pool.
        if c["t"] in candidats_deja_evalues:
            return candidats_deja_evalues[c["t"]]

        images = extraire_sequence_candidat(video_path, c["t"], fps_source=fps_source, offset_s=offset_s)

        # Pré-filtre optionnel (1 appel), pour rejeter à bas coût les
        # candidats clairement hors sujet avant le vote complet (2-3
        # appels). Permissif par construction : un échec technique ou un
        # doute ("formation_plausible" absent/True) laisse TOUJOURS
        # passer au vote complet — jamais de rejet basé sur ce seul
        # pré-filtre.
        if utiliser_prefiltre and images.get("juste_avant") is not None:
            prefiltre = demander_gemini_prefiltre(images["juste_avant"], client, model=model)
            if prefiltre is not None and prefiltre["formation_plausible"] is False:
                return {
                    "t": c["t"], "categorie_rf": c.get("categorie"), "proba_rf": c.get("proba"),
                    "evalue": True, "vote": None,
                    "jugement_vision": {
                        "transition_visible": False, "confidence": 0.9,
                        "raisonnement": f"[pré-filtre] {prefiltre['raisonnement']}",
                    },
                }

        vote = voter_transition(images, client, model=model, max_appels=max_appels_par_candidat)
        return {
            "t": c["t"], "categorie_rf": c.get("categorie"), "proba_rf": c.get("proba"),
            "evalue": True, "vote": vote,
            "jugement_vision": ({
                "transition_visible": vote["decision"],
                "confidence": vote["confidence"],
                "raisonnement": f"vote majoritaire {sum(1 for v in vote['votes'] if v['transition_visible'])}/{vote['n_appels']}",
            } if vote else None),
        }

    # Tri chronologique croissant — condition nécessaire à l'arrêt anticipé
    candidats_tries = sorted(top_n_candidats, key=lambda c: c["t"])

    resultats = []
    candidats_positifs = []
    premier_positif_t = None

    for c in candidats_tries:
        # Arrêt anticipé : au-delà de la fenêtre d'ambiguïté autour du
        # premier positif trouvé, plus rien ne peut changer la décision.
        # DESACTIVABLE en mode diagnostic (desactiver_arret_anticipe=True).
        if (not desactiver_arret_anticipe and premier_positif_t is not None
                and (c["t"] - premier_positif_t) >= seuil_ambiguite_secondes):
            resultats.append({
                "t": c["t"], "categorie_rf": c.get("categorie"), "proba_rf": c.get("proba"),
                "evalue": False, "vote": None,
            })
            continue

        resultats.append(_evaluer(c))

        if resultats[-1]["vote"] is not None and resultats[-1]["vote"]["decision"]:
            candidats_positifs.append(resultats[-1])
            if premier_positif_t is None:
                premier_positif_t = c["t"]

    if not candidats_positifs:
        meilleur_rf = top_n_candidats[0] if top_n_candidats else None
        return {
            "timestamp_final": meilleur_rf["t"] if meilleur_rf else None,
            "confidence": "faible",
            "ambiguity": True,
            "candidats_evalues": resultats,
            "note": "Aucune transition claire détectée par l'IA vision — repli sur le meilleur candidat RF, à traiter avec prudence.",
        }

    # Déjà trié par t croissant -> le premier positif trouvé est le bon
    meilleur = candidats_positifs[0]

    ambiguity = False
    if len(candidats_positifs) > 1:
        ecart_temporel = candidats_positifs[1]["t"] - meilleur["t"]
        ambiguity = ecart_temporel < seuil_ambiguite_secondes

    conf_valeur = meilleur["vote"]["confidence"]
    if ambiguity:
        confidence_label = "moyenne"
    elif conf_valeur >= 0.75:
        confidence_label = "élevée"
    elif conf_valeur >= 0.4:
        confidence_label = "moyenne"
    else:
        confidence_label = "faible"

    return {
        "timestamp_final": meilleur["t"],
        "confidence": confidence_label,
        "ambiguity": ambiguity,
        "candidats_evalues": resultats,
    }


def detecter_ko_par_vision_adaptatif(video_path, top_n_complet, fps_source=None,
                                       model="gemini-3.1-pro-preview", seuil_ambiguite_secondes=10.0,
                                       max_appels_par_candidat=5, offset_s=2.0,
                                       taille_pool_initial=10, taille_pool_max=30):
    """
    ASTUCE DE COÛT : mode adaptatif à 2 passes, pour élargir le pool de
    candidats (top-10 -> top-30) SANS payer le surcoût dans les cas où
    le top-10 suffit déjà (la majorité des matchs, cf. jalon).

    Passe 1 : detecter_ko_par_vision() sur les `taille_pool_initial`
    premiers candidats (défaut 10, comportement de production actuel,
    coût inchangé).

    Passe 2 : SEULEMENT SI la passe 1 échoue (ambiguity=True et
    confidence='faible', c'est-à-dire aucun positif franc trouvé),
    élargir aux candidats classés jusqu'à `taille_pool_max` (défaut 30).
    Les candidats déjà évalués en passe 1 sont réutilisés tels quels
    (via candidats_deja_evalues) — AUCUN appel Gemini n'est répété.

    Le tri chronologique + arrêt anticipé déjà présents dans
    detecter_ko_par_vision() s'appliquent normalement sur le pool
    élargi : si le bon candidat est proche chronologiquement du début,
    le coût de la passe 2 reste faible même avec un pool à 30.

    top_n_complet : liste COMPLÈTE des candidats disponibles, DÉJÀ
    triée par proba RF décroissante (comme la sortie habituelle du
    scoring RF) — au moins `taille_pool_max` éléments si possible.

    Retourne le même contrat que detecter_ko_par_vision(), avec deux
    clés ajoutées pour la traçabilité :
      "pool_final_utilise" : taille_pool_initial ou taille_pool_max
      "passe2_declenchee" : bool
    """
    pool_initial = top_n_complet[:taille_pool_initial]

    resultat = detecter_ko_par_vision(
        video_path, pool_initial, fps_source=fps_source, model=model,
        seuil_ambiguite_secondes=seuil_ambiguite_secondes,
        max_appels_par_candidat=max_appels_par_candidat, offset_s=offset_s,
    )

    echec_passe1 = (resultat.get("confidence") == "faible" and resultat.get("ambiguity") is True)

    if not echec_passe1 or len(top_n_complet) <= taille_pool_initial:
        resultat["pool_final_utilise"] = len(pool_initial)
        resultat["passe2_declenchee"] = False
        return resultat

    print(f"  [adaptatif] Passe 1 (top-{taille_pool_initial}) non concluante — "
          f"élargissement à top-{taille_pool_max}...")

    # Réutiliser les résultats déjà obtenus en passe 1, indexés par t,
    # pour ne jamais rappeler Gemini sur les mêmes candidats.
    cache = {r["t"]: r for r in resultat["candidats_evalues"] if r.get("evalue")}

    pool_elargi = top_n_complet[:taille_pool_max]
    resultat_final = detecter_ko_par_vision(
        video_path, pool_elargi, fps_source=fps_source, model=model,
        seuil_ambiguite_secondes=seuil_ambiguite_secondes,
        max_appels_par_candidat=max_appels_par_candidat, offset_s=offset_s,
        candidats_deja_evalues=cache,
    )
    resultat_final["pool_final_utilise"] = len(pool_elargi)
    resultat_final["passe2_declenchee"] = True
    return resultat_final