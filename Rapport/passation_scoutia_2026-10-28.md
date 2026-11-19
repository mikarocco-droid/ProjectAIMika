# ScoutIA — Document de passation (28/10/2026)

À joindre au premier message d'une nouvelle conversation. Il résume l'état du projet, les fichiers à jour, les décisions prises et les prochaines étapes. Les conversations précédentes du projet restent consultables par la recherche.

## 1. Configuration de production VALIDÉE (ne pas modifier sans test complet)

- Suivi en mode **« historique »** (`FILTRAGE_PISTES = ""`). Le mode « filtre » (`"3"`) a été testé le 28/10 : il **dégrade la détection des buts** (Franchimont 2/5, 12 faux buts, heures décalées). Interdit pour le pipeline des buts.
- **Ancienne version** de `analysis/player_reid.py` (restaurée le 28/10). La version à attribution exclusive (`player_reid_exclusif.py`) est **expérimentale** et n'est utilisée par aucun run.
- Voie « ballon » désactivée, réflexion réduite (validation KO et classification), lecture des maillots désactivée (réactivée en mode joueur), résumé par l'IA désactivé.
- **Modèle Gemini unique : `gemini-3.8-flash`**, imposé par le verrou de `ai/gemini_validator.py`. Depuis le 25/10, `app.py` charge ce verrou **au démarrage** (avant, la pré-analyse du site appelait `gemini-3.1-pro-preview` et `gemini-2.5-flash`, d'où des coûts imprévus).
- Les notebooks Kaggle font un **pré-vol** au démarrage (60+ contrôles), dont « règle de la pause SÛRE » et « player_reid : version VALIDÉE ».

## 2. Jeu de référence

| Match | Vrais buts | Faux buts « confirmés » | Bornes vérifiées |
|---|---|---|---|
| Raeren (portion) | 7/7 | 0 | — |
| Andrimont (portion) | 5/5 | 0 | — |
| Franchimont | 5/5 | 0 | oui |
| Spa | 7/7 | 0 | oui |
| Minerois Ster | 5/5 | 0 | — |
| P1 Minerois | 3/3 | 0 | oui |
| Goe | 2/2 | 0 | — |
| Stembert | 3/3 | 0 | oui |
| Wanze | 2/2 | 0 | oui |
| Juprelle | 1/1 | 0 | oui |
| **Total** | **40/40** | **0** | 23 bornes justes sur 24 |

- Faux buts « à vérifier » restants : environ **un pour trois matchs** (ex. Juprelle 30:14, un six mètres). Avec la réflexion réduite, beaucoup de vrais buts sont aussi « à vérifier » (Spa : 7/7).
- Vérité terrain : `verite_terrain.json` (buts, bornes vérifiées, faux positifs connus).
- Seule borne fausse : fin de 1re mi-temps de Wanze (−44 s), couverte par la marge de 90 s.

## 3. Coûts mesurés (gemini-3.8-flash, 0,75 $/M entrée, 3,75 $/M sortie)

- **Moyenne ≈ 2,96 $ par match (≈ 2,73 €)**, de 1,42 $ (Goe) à environ 4 $ (Franchimont, estimation incertaine).
- Le coût dépend surtout du **nombre de candidats** de coups d'envoi (≈ 0,026 $ chacun : validation + classification). Bornes : 0,50 à 1 $.
- Facture réelle de Juprelle : 1,79 € (cohérent avec l'estimation).
- Prix : Starter (3,95 €/vidéo) à l'équilibre, **Pro (2,69 €) à perte**, Unique (5 €) avec marge.

## 4. Fichiers à jour (dossier des sorties de la conversation du 28/10)

| Fichier | Emplacement | Remarque |
|---|---|---|
| `pipeline.py` (v46) | racine | règle de la pause corrigée (`_MARGE_PAUSE_MI_TEMPS_S`) |
| `gemini_validator.py` (v28) | `ai/` | verrou, cache, réflexion réduite |
| `main.py`, `app.py`, `tasks.py`, `config.py` | racine | app.py : verrou au démarrage, aperçu pendant l'envoi, quota, pré-analyses limitées |
| `post_processing.py` | `analysis/` | priorité aux buts validés |
| `kickoff_gemini_cascade.py` | `analysis/` | « seconde chance » de la reprise, `/tmp` → dossier système, recherche fine sûre, modèle 3.8-flash |
| `detect_teams_preview.py` | `analysis/` | modèle 3.8-flash |
| `player_reid.py` | `analysis/` | **version validée restaurée** |
| `player_reid_exclusif.py` | `analysis/` | expérimental, ne pas utiliser dans le pipeline |
| `export/pdf.py`, `static/app.js`, `templates/*` | — | rapport V1, app.js : coup d'envoi, couleurs, aperçu MP4 (classique et fragmenté) |
| `kaggle_match_reference.py` | Kaggle | notebook de référence (pré-vol à jour) |
| `kaggle_joueur_v2.py` | Kaggle (CPU) | montage d'un joueur, identités reconstruites après l'analyse |
| `kaggle_diag_identites.py` | Kaggle (CPU) | diagnostic des conflits d'identité sur tous les caches |
| `kaggle_terrain2d.py`, `kaggle_terrain2d_v2.py` | Kaggle | expérimentation du terrain 2D (en pause) |

**À pousser sur GitHub si ce n'est pas fait :** `pipeline.py`, `analysis/player_reid.py` (version restaurée), `analysis/kickoff_gemini_cascade.py`.

## 5. Corrections importantes des 25-28/10

- **Bug critique corrigé (v46)** : la règle de la pause de mi-temps utilisait une variable redéfinie localement dans `run_pipeline` → « cannot access local variable » → **toute la détection des buts interrompue** (Franchimont 0/5). Leçon : contrôler la portée des variables avec `symtable`, pas seulement avec le banc de test.
- **Règle de la pause** : aucun but entre la fin de la 1re mi-temps (+90 s) et la reprise (corrige le faux but de Juprelle 68:28). Inactive en replay (bornes inconnues).
- **Marges de fin de mi-temps** : 30 → 90 s.
- **Libellé « Occasion »** pour tous les temps forts (plus de « But possible »).
- **Reprise manquée par « fenêtre dégénérée »** (Franchimont 28/10) : seconde chance vérifiée (recul de 30 s, puis recherche fine). Testée, pas encore vérifiée en run réel.
- **Site** : pré-analyse pendant l'envoi (début de la vidéo + index MP4, aussi pour MP4 fragmentés Veo), quota vérifié à l'envoi, 5 pré-analyses par jour maximum, champ vide → le pipeline détecte lui-même le coup d'envoi.

## 6. Chantier des identités (analyse individuelle)

**Constat (diagnostic sur 12 caches)** : conflits d'identité dans ≈ 100 % des images, sur ≈ 80 % des identifiants. Deux causes :
1. `player_reid` attribue le même identifiant à plusieurs joueurs d'une même image (attribution non exclusive) ;
2. le mode « historique » transmet des **pistes fantômes** (positions prédites jusqu'à ~12 s), d'où ~40 positions par image.

**Décision** : **ne plus toucher au suivi ni à la ré-identification du pipeline des buts.** Les identités sont reconstruites **après l'analyse**, à partir du cache, pour l'analyse individuelle uniquement.

**`kaggle_joueur_v2.py` (premier résultat réel, Franchimont, 10 rouge)** :
- morceaux de trajectoire = identifiant DeepSort (`tracker_id`, jamais partagé dans une image) ;
- couleurs des équipes par Gemini juste après le coup d'envoi (équipe 0 verte, équipe 1 rouge) ;
- numéro relu sur 4 images par morceau ; retenu si lu ≥ 2 fois sans contradiction ;
- 400 morceaux lus (plafond atteint) : **65 % avec un autre numéro**, 31 % illisibles, 3 retenus pour le 10, 9 lus une seule fois ;
- montage de 1,1 min (08:33, 15:21, 17:07) ; durée 24 min (allers-retours dans la vidéo).
- **En attente** : vérification par l'utilisateur de `verification_lectures.jpg` et du montage (le repère est-il sur le 10 rouge ?).
- **Ensuite** : relever le plafond (lire tous les morceaux, 1 à 2 $ par joueur), lire les images dans l'ordre chronologique (vitesse), raccorder les morceaux sans numéro aux morceaux voisins du 10.

**Mode joueur du site** : l'actuel ne garde que les temps forts attribués au joueur (montage quasi vide) ; à remplacer par la méthode de `kaggle_joueur_v2.py`, avec repère jaune.

## 7. Terrain 2D (en pause)

- Modèle public Roboflow (`football-pitch-detection`, 32 points clés, licence MIT pour le code).
- v2 (RANSAC + contrôle de qualité) : projection acceptée sur 53 % (Franchimont) / 27 % (Spa) des images, erreur 7-9 px quand acceptée ; trop peu de points sur terrains amateurs.
- Coût : 73 ms/image sur T4 → ~7 min par match à 1 image/s.
- Usage futur : raccorder des trajectoires après une perte, pas réparer des identités fusionnées.
- Attention licence : `ultralytics` est sous **AGPL-3.0** (concerne déjà la détection YOLO de ScoutIA) : vérifier avant commercialisation.

## 8. Prochaines étapes (par priorité)

1. Pousser les 3 fichiers (section 4).
2. Vérifier le montage v2 du 10 rouge (planche + montage).
3. **Run complet de Franchimont** dans la configuration validée (notebook de référence tel quel) : attendu 5/5, bornes trouvées (vérifie la seconde chance), **coût Gemini réel sur une journée**, et cache propre à remettre dans le dataset `codeprojectai`.
4. **Score par équipe** : besoin de la vérité terrain (équipe qui marque chaque but) sur 3-4 matchs (Spa, Franchimont, Minerois Ster, Stembert) ; méthode : couleur de l'équipe qui célèbre, recoupée par celle qui engage ensuite.
5. Décision produit : compter tous les buts trouvés comme buts (précision ≈ 10/11 mesurée, rappel 100 %), à confirmer sur 3-4 matchs en replay ; puis score dans le PDF et la page de résultats.
6. Avant production : Stripe en mode test, Celery avec docker-compose, aperçu pendant l'envoi sur une vraie vidéo Veo (`boites.py` si erreur), limite Celery de 2 h sur RTX 4090, licence ultralytics.
7. Plus tard : vérifier le journal Gemini (corners, six mètres…), ménage du code mort.

## 9. Règles de travail retenues

- Toujours vérifier une modification du pipeline par un **replay** (rapide) avant un run complet sur GPU.
- Ne jamais changer la configuration du suivi dans un run de référence.
- Pour mesurer un coût exact : un seul run ce jour-là, rien d'autre (ni site en local, ni autre notebook).
- Les messages torchreid / cuDNN / cuBLAS au démarrage de Kaggle sont normaux.
