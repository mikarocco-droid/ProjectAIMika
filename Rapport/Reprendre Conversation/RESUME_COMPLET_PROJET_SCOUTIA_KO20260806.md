# SCOUTIA — DÉTECTION DU COUP D'ENVOI — RÉSUMÉ COMPLET DU PROJET

**Objectif du document** : permettre de reprendre ce travail dans une
nouvelle conversation sans rien reperdre. Tout ce qui a été testé, trouvé,
corrigé, abandonné, ou validé est ici, dans l'ordre chronologique.

---

## 1. CONTEXTE DU PROJET

**Scoutia** est un pipeline d'analyse vidéo de football (repo GitHub
`https://github.com/mikarocco-droid/ProjectAIMika.git`). Objectif de cette
partie du projet : détecter automatiquement l'instant du coup d'envoi (KO)
dans des vidéos de matchs amateurs filmées par une caméra fixe/panoramique
(type Veo).

### Les 10 matchs de référence, avec leur TRUE_KICKOFF (en secondes)

| Match | TRUE_KICKOFF | Chemin vidéo Kaggle |
|---|---|---|
| Franchimont | 318 | `/kaggle/input/datasets/michaelziant/p4franchimont/ster-b-franchimont-b-2026-04-09_0.mp4` |
| Andrimont | 307 | `/kaggle/input/datasets/michaelziant/stertest/match-p4-andrimont-2026-04-26.mp4` |
| Raeren | 410 | `/kaggle/input/datasets/michaelziant/p4raeren/ster-b-raeren-b-2026-04-03_0.mp4` |
| Stembert | 553 | `/kaggle/input/datasets/michaelziant/stembert/ster-b-stembert-2026-03-07_0.mp4` |
| Goe | 1081 | `/kaggle/input/datasets/michaelziant/stergoe/ster-b-goe-2026-03-20_0.mp4` |
| Wanze | 986 | `/kaggle/input/datasets/michaelziant/wanzebas/ster-wanze-bas-oha-2026-02-28_0.mp4` |
| Juprelle | 548 | `/kaggle/input/datasets/michaelziant/juprelle/ster-juprelle-2026-01-25_0.mp4` |
| P1Minerois | 192 | `/kaggle/input/datasets/michaelziant/p1minerois/p1-minerois-2026-03-28_0.mp4` |
| MineroisSter | 723 | `/kaggle/input/datasets/michaelziant/mineroisster/minerois-ster-2025-11-09_0.mp4` |
| matchTest | 8 | `/kaggle/input/datasets/michaelziant/matchcomplet/match_testcomplet.mp4` |

**Cas particulièrement difficiles à connaître** :
- **Andrimont** : contient 3 "faux départs" avant le vrai KO (185-191s,
  219-225s, 271-275s) — les joueurs se placent en configuration de KO
  (arbitre + ballon au centre, équipes séparées), mais une ou les deux
  équipes interrompent pour un rituel de groupe (cri de guerre en cercle).
  Le vrai KO n'arrive qu'après, à 307s.
- **matchTest** : signal visuel intrinsèquement faible (confirmé par
  data leakage — voir section 3), mais un des cas les plus "faciles" une
  fois le bon instant identifié.

### Infrastructure technique établie

- **Environnement de test** : Kaggle Notebooks (GPU T4). Le repo Scoutia
  est cloné à chaque run via `git clone --depth 1
  https://github.com/mikarocco-droid/ProjectAIMika.git`.
- **Dataset Kaggle compagnon** : `codeprojectai`
  (`/kaggle/input/datasets/michaelziant/codeprojectai`) — contient les
  fichiers `.py` qu'on développe nous-mêmes (pas dans le repo GitHub),
  ajoutés manuellement par l'utilisateur.
- **Piège récurrent rencontré plusieurs fois** : Kaggle garde en cache
  l'ancienne version d'un module Python déjà importé dans la session —
  après avoir mis à jour un fichier du dataset, il faut **redémarrer
  complètement la session Kaggle**, pas juste re-ajouter le dataset.
- **Boilerplate de setup répété dans chaque script Kaggle** : clone du
  repo, `pip install ultralytics opencv-python-headless anthropic
  python-dotenv werkzeug scikit-learn joblib numpy scipy psutil
  deep-sort-realtime`, `apt-get install tesseract-ocr ffmpeg`, création
  des dossiers `__init__.py` pour les packages du repo, écriture d'un
  `param.env` minimal, puis `from main import process_video`.
- **Découpe vidéo** : `ffmpeg -y -ss <start> -i <video> -t <duration> -vf
  fps=4 -c:v libx264 -preset ultrafast -crf 23 -an <out>` — `-an` (sans
  audio) sauf pour la piste sifflet, où il faut garder l'audio (`-c:a aac
  -b:a 64k` à la place de `-an`).
- **`process_video()`** (dans `main.py` du repo) retourne `(events,
  jersey_map, fps, total_frames, frames_data)` avec `return_frames=True`.
  `frames_data` est une liste de dicts, un par frame analysée :
  `{"frame": int, "frame_w":, "frame_h":, "players": [{"id":,
  "team":, "center":[x,y], ...}], "ball": {"center":[x,y],
  "interpolated": bool, "conf":}}`.
- **Coût en temps** : `process_video()` sur une fenêtre de 960s (16 min) à
  4fps prend ~150-900s selon le nombre d'événements détectés. Sur une
  fenêtre de 260s, ~130-170s. C'est le principal poste de temps de tout
  run Kaggle.

---

## 2. CE QUI A ÉTÉ TESTÉ ET DÉFINITIVEMENT ÉCARTÉ (ne pas y revenir)

| Approche | Résultat | Pourquoi elle échoue |
|---|---|---|
| **DINO v2 (embeddings) + LogisticRegression** | Baseline v2 : 8/10 | Reconnaît une apparence, pas fiable sur cas ambigus |
| **DINO générateur d'événements + Claude Vision (scoring)** | 9/10 en Niveau 1, mais Niveau 3 avec faux positifs | Claude confond parfois reprise de mi-temps / rassemblement avec KO |
| **Géométrie seule** (séparation des équipes) | 0/10 en décision unique (audit A0 sur `kickoff_detector.py`) | Vraie aussi pendant un rassemblement, une pause, du jeu ouvert équilibré |
| **Ballon seul** (position/vitesse) | Détection très instable | YOLO générique confond le ballon avec des faux positifs de décor (confirmé visuellement) ; même le `BallTracker` Scoutia produit des vitesses aberrantes sans compensation caméra |
| **Vitesse moyenne des joueurs seule** | Contre-attaques produisent un pic plus fort que le KO | Une contre-attaque part d'un état déjà actif, avec une accélération collective parfois plus intense qu'un KO |
| **Rupture de régime (changement d'état) seule** | Détecte aussi pressing, récupération haute, coup franc joué vite | Le foot est plein de ruptures, pas seulement le KO |
| **`kickoff_detector.py` (système de production Scoutia existant)** | 0/10 en A0 audit | Stratégie 1 (plus longue séquence stable = KO) mesure une condition nécessaire mais pas suffisante ; `EARLY_HIGH_GUARD` rejette parfois de bons candidats |

**Conclusion établie et validée plusieurs fois** : chaque signal pris seul
est **nécessaire mais jamais suffisant**. Le KO n'est pas un pic ni une
image — c'est une **séquence d'états** (voir section 5).

---

## 3. DÉCOUVERTE MÉTHODOLOGIQUE MAJEURE — FUITE DE DONNÉES (matchTest)

Un diagnostic très ancien (`diagnostic_matchtest_cluster1_peaks.py`) avait
montré un "plateau parfait" de score DINO (0.93-1.00) sur 0-13s pour
matchTest, ce qui avait motivé toute une refonte (NMS temporel, re-scan
DINO). **Ce diagnostic contenait une fuite de données** : le classifieur
avait été entraîné en incluant les propres annotations de matchTest
(leave-one-out cassé). Une fois corrigé, le score réel au voisinage du KO
tombe à ~0.40, signal beaucoup plus faible et bruité.

**Audit complet effectué sur 16 scripts** : cette fuite était **isolée à
ce seul script diagnostic**, jamais utilisée dans un résultat de
production retenu. Les Expériences 1-6 et l'audit niveau3_v4 restent valides.

**Leçon générale retenue** : toujours vérifier qu'un classifieur exclut
bien le match sur lequel on l'évalue (leave-one-out réel, pas supposé).

---

## 4. LA DÉCOUVERTE DE `kickoff_detector.py` (SYSTÈME EXISTANT)

En cours de route, l'utilisateur a partagé le vrai pipeline de production
Scoutia (`pipeline.py`, `main.py`, `kickoff_detector.py`,
`terminal_events.py`) — révélant qu'un détecteur géométrique **existait
déjà**, indépendant de tout ce qu'on avait construit (DINO/Claude).

### Fonctionnement de `kickoff_detector.py`
- **Stratégie 1** : regroupe les candidats à séparation d'équipes élevée
  par proximité temporelle, choisit le groupe **le plus long** parmi les
  3 plus longs (pas le mieux noté).
- **Affinage ballon** (`_detect_ball_played_in_center`) : détecteur
  OpenCV (seuillage/contours) calibré pour le **fps natif** (25-30fps) —
  exige `PRESENT_THRESH=4` frames consécutives. À 4fps (notre
  échantillonnage réduit), 4 images = 1 seconde entière → quasi impossible
  à satisfaire → échec systématique de cette étape.
- **`EARLY_HIGH_GUARD`** (seuil 60s) : si `_p1_early_high=True` (activité
  déjà élevée dans les 60 premières secondes de la vidéo), **rejette tout
  offset > 60s même si l'affinage a réussi avec un bon résultat** — bug
  confirmé (contredit son propre CAS B, censé chercher une transition
  échauffement→match au-delà de 60s).

### Audit A0 (10 matchs, patch de repli géométrique appliqué au clone Kaggle
uniquement, jamais au vrai repo)
**Résultat : 0/10 corrects.** Diagnostic causal (tableau de vérité, logique
de transition d'état — un composant n'est responsable que s'il transforme
un succès en échec) :
- **Stratégie 1 responsable de 9/10 échecs** — ne trouve jamais le bon
  candidat en premier lieu.
- **`EARLY_HIGH_GUARD` responsable de 1/10** (P1Minerois : l'affinage
  ballon avait trouvé 189.8s, à 2.2s du vrai KO 192s — un succès quasi
  parfait, détruit par le garde-fou).
- **Affinage ballon jamais testé dans de bonnes conditions** (0 candidat
  correct reçu de Stratégie 1 pour le tester).

**Conclusion** : le problème est à la **source** (génération du candidat
par Stratégie 1), pas dans le raffinement. Ce système a été laissé de côté
(pas supprimé, juste plus utilisé comme base) au profit d'une approche
reconstruite sur les données réelles de tracking Scoutia.

---

## 5. PIVOT — RECONSTRUCTION SUR `frames_data` RÉEL (pas YOLO générique)

Découverte clé : `process_video()` produit déjà `frames_data` avec du vrai
tracking DeepSort (IDs stables) et un vrai `BallTracker` Scoutia — pas
besoin de reconstruire un tracker maison avec YOLO COCO générique (ce qui
avait été tenté brièvement avec un prototype fragile, abandonné).

### Bugs de compensation caméra trouvés et corrigés (`kickoff_score_hybride.py`)
1. **Compensation ponctuelle insuffisante** : la médiane des déplacements
   joueurs entre deux instants ne corrigeait qu'**un seul pas de la grille
   temporelle**, alors que le ballon (détection éparse) peut avoir des
   écarts de plusieurs pas entre deux détections réelles. **Corrigé** par
   un décalage **cumulatif** (`_camera_motion_series` retourne un
   cumul depuis le début de la fenêtre, `_camera_offset_between(cumulative,
   t0, t1)` calcule le bon décalage entre deux instants QUELCONQUES).
2. **Filtrage des positions interpolées** (`exclude_interpolated=True`
   dans `_ball_at`) — le ballon "interpolé" par `BallTracker` n'est pas une
   vraie mesure.
3. **Filtre de plausibilité géométrique** (`max_dist_to_player_px=300`) —
   confirmé nécessaire par diagnostic visuel (voir section 6).

### Découverte visuelle du faux positif ballon récurrent (Andrimont)
Diagnostic par accélération (7/7 images inspectées = faux positif, jamais
le vrai ballon). Motif clair : avant le KO, l'objet confondu est **fixe**
(près de la clôture/décor) ; après le KO, il varie mais reste **toujours
isolé, loin de tout joueur**. D'où le filtre de plausibilité (distance min
à un joueur).

---

## 6. L'OBSERVATOIRE — mean_speed_change_px_s identifié comme signal le plus prometteur

Approche : mesurer ~10 signaux indépendants sur des candidats étiquetés
(vrai_KO / rassemblement / arrêt_de_jeu / jeu_ouvert), **sans combiner ni
décider**, pour voir quel(s) signal(aux) séparent naturellement les
catégories.

**Résultat sur 4 matchs (16 candidats)** :
| Catégorie | mean_speed_change_px_s (moyenne) |
|---|---|
| vrai_KO | **13.52** |
| jeu_ouvert | 5.20 |
| arret_de_jeu | 3.58 |
| rassemblement | **-0.30** |

C'était le signal le plus net trouvé jusqu'ici — calculé simplement
(médiane vitesse après moins médiane vitesse avant, compensée caméra),
**sans seuil arbitraire**.

**Signaux confirmés non informatifs en l'état** :
- `transition_rate` : toujours 1.00 (seuil d'immobilité 8px/s trop strict,
  ne laisse passer qu'1 joueur en moyenne).
- `ball_near_frame_center` : le centre du cadre caméra ne correspond pas
  au rond central du terrain (pas de calibration terrain) — signal
  trompeur (valeur la PLUS BASSE pour vrai_KO dans un test).

---

## 7. TENTATIVES DE DÉTECTEUR — deux échecs informatifs

### Détecteur "argmax global" (`detecteur_rupture.py`)
Cherche le maximum de `mean_speed_change_px_s` sur tout le match. **1/4
corrects** (Franchimont : écart 0.0s). Piégé par les contre-attaques :
Andrimont (écart 199s), Raeren (écart 164s), Stembert (écart 405s) — le
maximum global tombe sur une contre-attaque, pas le KO.

### Détecteur "rupture d'état" (`detecteur_rupture_etat.py`)
Vecteur d'état [vitesse, dispersion, géométrie] par fenêtre de 6s, distance
euclidienne normalisée entre fenêtres consécutives, seuil auto-calibré
(médiane + 2×MAD), cherche la **première** rupture importante ET
**durable** (pas un pic isolé qui retombe). Validé sur données synthétiques
(ignore correctement une contre-attaque brève). Jamais testé sur les 10
matchs réels (le projet a bifurqué vers la définition avant ce test).

---

## 8. PIVOT CONCEPTUEL — LE DOCUMENT DE DÉFINITION

**Changement de philosophie explicite** : arrêter de chercher "quel est LE
signal", et définir d'abord ce qu'est un KO indépendamment de la détection.

### `DEFINITION_COUP_ENVOI_SCOUTIA.md` — contenu clé
- Le KO est **la première transition irréversible entre "avant-match" et
  "match"**, pas une image ni un score.
- Machine à états : **A0 (formation en cours) ↔ A1 (formation stable) →
  B (départ collectif) → C (jeu établi)**. A0↔A1 forme une **boucle** — le
  système peut faire des faux départs (retour A1→A0) plusieurs fois avant
  la vraie transition vers B.
- **Le ballon est un indice bonus, jamais une condition** (trop peu
  fiable pour faire partie de la définition).
- **Stabilité = moyenne ET variance** (un signal faible mais oscillant
  n'est pas un état stable).
- **Question ouverte identifiée comme la plus importante** : comment
  distinguer présentation / échauffement / discussion / attente arbitre
  de la vraie formation de KO ? (frontière de l'état A1)

### Réponse empirique partielle (diagnostic visuel Andrimont)
3 faux départs confirmés visuellement (185-191s, 219-225s, 271-275s,
chacun 4-6s, interrompus par un rituel de groupe/cri de guerre), suivis
d'un état stable de ~32s menant directement au KO (275-307s).

**⚠️ Mise en garde ajoutée après contre-vérification sur 6 matchs** : la
durée de l'épisode stable **ne généralise pas** comme critère de
décision :
- Andrimont : épisode menant au KO = 12s — **identique** à un faux départ
  antérieur (212-222s, aussi 12s).
- Juprelle : épisode menant au KO = seulement 4s — **plus court** que
  plusieurs faux départs (14s, 12s, 6s).
- Franchimont et Goe : **aucun épisode détecté près du KO du tout**
  (dernier épisode 15-25s avant le KO) — possible biais de la fenêtre de
  mesure de 6s qui masquerait les phases calmes courtes (jamais tranché,
  voir section 9).

**Signal plus robuste identifié** : la séparation géométrique moyenne de
l'épisode menant au KO est, dans 4/4 cas testés, la **plus élevée parmi
tous les épisodes précédents** (critère causal, pas besoin de connaître le
futur) — contrairement à la durée.

---

## 9. QUESTION NON TRANCHÉE — Franchimont et Goe (biais de fenêtre ?)

Un script de courbes brutes (`courbes_brutes_ko.py`, sans fenêtre de
moyennage de 6s, geo_separation instantané + vitesse point-à-point) a été
lancé sur Franchimont et Goe pour vérifier si une phase calme existe
vraiment. **Résultat obtenu mais pas encore interprété en profondeur** :
- Franchimont : vitesse entre 18.2 et 43.0 px/s sur toute la fenêtre
  20-42s avant KO (aucun creux net vers 0).
- Goe : vitesse entre 12.9 et 53.1 px/s, pareil, aucun creux net.

**Ni Cas A (phase longue masquée) ni Cas B (phase courte) clairement** —
un possible 3e cas : le bruit de détection (gigue de boîte englobante)
empêche la vitesse de descendre sous ~15-20px/s même à l'arrêt réel. **Ce
point n'a jamais été confirmé par observation humaine directe de la
vidéo** (voir section 10) — c'était l'étape suivante prévue avant le pivot
vers l'audio (section 11).

---

## 10. PAUSE SUR LE CODE — GRILLE D'OBSERVATION HUMAINE

Document créé (`GRILLE_OBSERVATION_HUMAINE_KO.md`) : 9 questions à remplir
en **regardant les vidéos directement, sans une ligne de code** (quand les
joueurs s'arrêtent-ils vraiment, combien de temps, que fait l'arbitre, le
ballon est-il déjà au centre, qui démarre en premier, existe-t-il des faux
départs visuellement identiques au vrai KO...). Andrimont et matchTest
partiellement pré-remplis à partir des images déjà examinées. **Les 8
autres matchs restent vierges — à remplir si cette piste est reprise.**

---

## 11. SIGNAL AUDIO (SIFFLET) — confirmé prometteur

Idée de l'utilisateur : le sifflet comme **vérité terrain** pour mesurer
les écarts entre signaux (pas comme détecteur autonome, car un arbitre
siffle aussi pour touches, fautes, buts, mi-temps).

**Vérification (pas un détecteur) sur 3 matchs** (`verification_sifflet.py`,
spectrogramme STFT, bande 2-4.2kHz) :
- Andrimont, Goe, Juprelle : **piste audio présente**, signature nette
  (bande fine, haute énergie) visible sur les 3 spectrogrammes, à moins de
  1-2s du KO documenté dans chaque cas.

**Attention** : les vidéos **trimées avec `-an`** (utilisées dans presque
tous les scripts précédents) n'ont PAS d'audio — il faut re-découper avec
`-c:a aac -b:a 64k` pour toute utilisation future du son.

---

## 12. ÉTAT ACTUEL — COMBINAISON MULTI-SIGNAUX (dernière étape avant la pause)

**Changement de philosophie final** : au lieu de choisir des poids à la
main pour combiner les signaux, **laisser un modèle (régression logistique
/ random forest) les apprendre** à partir d'un tableau de features
étiquetées (vrai_KO=1 / autre=0).

### Fichiers livrés (derniers en date, PAS ENCORE EXÉCUTÉS)

1. **`ko_features.py`** — module **autonome** (aucune dépendance aux
   fichiers expérimentaux précédents, aucune dépendance Kaggle), pensé
   pour s'intégrer directement dans `analysis/` du repo Scoutia. Fonction
   principale : `extract_features(frames_data, fps, frame_w, video_path,
   t, label=None)` → un dict avec `geo_separation, mean_speed_change_px_s,
   pct_immobile, synchronisation_std_s, ball_score, ball_speed_before,
   ball_speed_after, stabilite_avant_s, activite_apres, whistle_score,
   whistle_peak_t`. Contient aussi `whistle_score(video_path, t, ...)`
   (spectrogramme, z-score local, pas de seuil absolu).

2. **`collecte_features_kaggle.py`** — script Kaggle **allégé** (fenêtres
   de 260s par match au lieu de 960s, ~4-5 candidats par match au lieu de
   dizaines, conserve l'audio avec `-c:a aac`). Candidats négatifs déjà
   choisis à partir des données connues de ce projet (buts rejetés par xG
   dans les logs précédents = arrêts de jeu réels ; rassemblement connu
   sur Andrimont). **Estimation : ~25-30 minutes pour les 10 matchs**
   (jamais lancé). Produit `features_ko.csv`.

3. **`entrainement_modele_ko.py`** — charge `features_ko.csv`, entraîne
   `LogisticRegression` (coefficients interprétables) et
   `RandomForestClassifier` (importances de variables), avec validation
   croisée **par match** (`GroupKFold` — un match entier exclu à chaque
   fold, pas juste des lignes au hasard, pour éviter le sur-apprentissage
   sur un match déjà vu). Testé sur données synthétiques : identifie
   correctement les signaux injectés comme discriminants.

### PROCHAINE ACTION CONCRÈTE (pas encore faite)

1. Uploader `ko_features.py` dans le dataset Kaggle `codeprojectai`
   (remplace tous les anciens fichiers expérimentaux, qui ne sont plus
   nécessaires).
2. Lancer `collecte_features_kaggle.py` sur Kaggle (~25-30 min).
3. Télécharger `features_ko.csv`.
4. Lancer `entrainement_modele_ko.py features_ko.csv` (peut tourner
   localement, pas besoin de Kaggle — juste pandas/sklearn).
5. Regarder les poids appris : le sifflet et la géométrie sont les
   candidats les plus probables à ressortir comme dominants, d'après tout
   ce qui a été observé jusqu'ici — mais **à confirmer par les données
   réelles, pas supposer**.
6. Une fois validé, `ko_features.py` peut être copié dans `analysis/` du
   repo Scoutia et appelé directement depuis `pipeline.py` /
   `kickoff_detector.py`, en réutilisant le `frames_data` déjà produit par
   `process_video()` en production (pas de second passage de tracking
   nécessaire).

---

## 13. PRINCIPES MÉTHODOLOGIQUES ÉTABLIS (à ne pas oublier)

- **Ne jamais fixer un seuil/poids à la main sans l'avoir mesuré** sur
  plusieurs matchs — erreur déjà commise avec `kickoff_detector.py`
  d'origine, à ne pas reproduire.
- **Toujours vérifier visuellement** (images/spectrogrammes) avant de
  conclure qu'un signal est bon ou mauvais — plusieurs "bugs" se sont
  révélés être des vrais problèmes de mesure (compensation caméra,
  fenêtre de moyennage) plutôt que des limites du phénomène observé.
- **Généraliser à partir d'un seul match est le piège le plus fréquent
  de ce projet** — presque chaque motif "confirmé" sur Andrimont s'est
  révélé partiellement ou totalement invalidé en le testant sur 3-6
  matchs de plus.
- **La validation croisée doit exclure des matchs entiers**, pas des
  lignes au hasard, sinon le modèle "reconnaît" un match déjà vu plutôt
  que d'apprendre un phénomène général.
