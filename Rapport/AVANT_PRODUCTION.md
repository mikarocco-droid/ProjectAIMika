# À faire avant la mise en production — ScoutIA

Ce document liste tout ce qui doit être vérifié, changé ou sécurisé avant de
déployer en production. Il est mis à jour à chaque session, dès qu'un point
pertinent est identifié — à relire intégralement avant le déploiement.

---

## 🔴 CRITIQUE — Sécurité

- [ ] **`requirements.txt` : `deep-sort-realtime` manquant**
  Découvert lors du premier test complet du pipeline sur `Kaggle`
  (11/09/2026) : `vision/tracker.py` importe `deep_sort_realtime`
  (`from deep_sort_realtime.deepsort_tracker import DeepSort`), mais ce
  paquet n'apparaît pas dans `requirements.txt`. Un vrai déploiement
  (`pip install -r requirements.txt`) planterait de la même façon dès le
  premier appel à `Tracker()` — **aucune analyse ne pourrait fonctionner**.
  À ajouter à `requirements.txt` avant tout déploiement.
  *(Identifié le 11/09/2026)*

- [x] **`Set_Admin.py` : route `/make-admin` sans aucune protection**
  Corrigé le 11/09/2026. Route retirée entièrement — le script garde
  uniquement l'usage en ligne de commande (`python Set_Admin.py`),
  jamais exposé via `HTTP`.

- [ ] **`Set_Admin.py` : email admin codé en dur** (`mikarocco@hotmail.com`)
  Non traité — le script en ligne de commande reste utile en dev, mais
  vérifier qu'il n'est pas exécuté automatiquement au démarrage et n'est
  pas inclus dans le déploiement final.
  *(Identifié le 11/09/2026)*

- [x] **Base de données de production (`SQLite`)**
  Corrigé le 11/09/2026 — avertissement explicite et impossible à
  manquer ajouté dans `config.py` : si `SQLite` reste actif alors que
  `DEBUG=False` (donc probablement en production), un message d'alerte
  bien visible s'affiche au démarrage. **N'empêche pas le démarrage**
  (éviterait de casser un déploiement en cours de configuration) — mais
  rend le risque impossible à manquer dans les logs. `DATABASE_URI` doit
  quand même être positionnée vers `PostgreSQL`/`MySQL` avant un vrai
  déploiement multi-utilisateurs — l'avertissement ne fait qu'éviter que
  ça passe inaperçu.

- [x] **`SECRET_KEY` codée en dur par défaut** (`"dev-only-change-me"`)
  Corrigé le 11/09/2026 — détection élargie dans `config.py` (pas
  seulement la valeur par défaut exacte, aussi les valeurs de dev
  courantes comme `"local-dev"` et toute clé trop courte `<24`
  caractères), avec avertissement fort et visible si `DEBUG=False`.
  **Avant un vrai déploiement**, générer une vraie valeur :
  `python -c "import secrets; print(secrets.token_hex(32))"` — jamais
  la même clé qu'en dev, jamais commitée dans `Git`.

---

## 🟡 IMPORTANT — Fonctionnalités non finalisées

- [ ] **Détection des buts — trop imprécise sur certaines caméras (`153` buts sur `Franchimont`)**
  Trouvé le 11/09/2026 lors du premier vrai test complet. L'heuristique
  actuelle (`analysis/terminal_events.py`, trajectoire du ballon
  traversant la ligne de but) génère un nombre de faux positifs
  absurdement élevé sur caméra `low_side_zoom` (`134` candidats bruts,
  `153` après tout le pipeline de filtrage/validation `Gemini` — un vrai
  match n'a jamais autant de buts). Seuil `goal_y_min=0.25` calibré sur
  un seul match (`andrimont_0`, mentionné en commentaire dans le code)
  qui ne généralise pas à `Franchimont`. **Lié au problème de vitesse** :
  chaque candidat déclenche une relecture dense de la vidéo (`skip=1`)
  AVANT le filtrage — `156` zones relues inutilement ici.

  **Piste proposée par l'utilisateur, à garder en tête** : plutôt que de
  détecter le but lui-même (ballon traversant la ligne — sensible à
  l'angle de caméra), détecter la **reprise au centre du terrain après
  un but** — un motif visuel stable et reconnaissable, similaire à ce
  que `KO1`/`KO2` détectent déjà de façon fiable avec `Gemini` vision
  (formation statique, joueurs alignés au centre). Avantages
  potentiels : `(1)` motif visuel plus stable, moins sensible à l'angle
  de caméra que la trajectoire du ballon ; `(2)` pourrait réduire le
  nombre de zones à analyser densément (chercher le but seulement dans
  une petite fenêtre juste avant chaque reprise trouvée, plutôt que
  scanner tout le match) — réglerait potentiellement la précision ET la
  vitesse en même temps. Nécessiterait de distinguer une reprise
  après-but d'une reprise après mi-temps/début de match (déjà couvertes
  par `KO1`/`KO2`).

  **Ne pas corriger à la légère** — le seuil actuel interagit avec
  plusieurs autres paramètres (`LINE_MARGIN`, `STUCK_MIN`, `SPEED_MIN`,
  `REBOUND_DROP`, `FAST_SHOT_SPEED`) probablement calibrés ensemble sur
  plusieurs matchs de test. Mérite une session dédiée avec tests sur
  l'ensemble des `11` matchs de référence, pas une correction rapide en
  fin de session.

- [ ] **Suivi individuel d'un joueur — plan en 2 étapes (Re-ID d'abord, déclencheurs ensuite)**
  Discuté le 11/09/2026, suite à une proposition détaillée de
  l'utilisateur pour le mode "joueur ciblé" (ex. suivre uniquement le
  `n°6` rouge). Décision : **ne pas construire les déclencheurs avant
  d'avoir fiabilisé l'identification continue d'un joueur donné** —
  bloquer le développement tant que cette base n'est pas solide.

  **Constat sur l'existant** : `analysis/player_identity.py` (mode
  `"joueur"` actuel) ne fait qu'un filtrage/réconciliation des
  `events`/`jersey_map` déjà calculés — aucune logique dédiée de
  déclenchement ou de classification d'action par joueur.
  `analysis/player_reid.py` a déjà un `Re-ID` basique
  (`_extract_embedding` : histogramme de couleur `+` position), mais
  **pas d'embedding d'apparence profond** — `reidentify_players()`
  utilise explicitement "pas d'embedding visuel". La lecture des
  numéros de maillot par `Gemini` ne réussit qu'à `~13-15%` des
  `track_id` tentés (mesuré ce soir : `179`/`1385`, `217`/`1544`).

  **`3` problèmes techniques identifiés** (par l'utilisateur) qui
  cassent le suivi continu d'un joueur en vidéo broadcast :
  `1)` taux de lecture de maillot trop faible ; `2)` changements d'`ID`
  du tracker lors de regroupements/occlusions (risque général connu
  pour `SORT`/`DeepSort`, pas mesuré spécifiquement sur nos données) ;
  `3)` homogénéité visuelle entre coéquipiers (même maillot → histogramme
  de couleur peu discriminant entre deux joueurs de la même équipe).

  **Étape `1` (prérequis, à faire avant tout)** :
  - Propagation temporelle du numéro de maillot : une lecture fiable
    toutes les `~2` min suffit si le `track_id` ne saute pas — propager
    l'identité rétroactivement et vers l'avenir sur tout le segment de
    trajectoire continu.
  - Embeddings d'apparence profonds (`OSNet`/`TorchreID`, local, `0€`
    d'`API`) pour ré-identifier un joueur après un saut d'`ID`, en
    complément/remplacement de l'histogramme de couleur actuel —
    résoudrait spécifiquement le problème d'homogénéité visuelle entre
    coéquipiers.
  - Mémoire spatiale (position relative sur le terrain) pour empêcher
    un `ID` de "téléporter" après un croisement de joueurs.

  **Étape `2` (fonctionnalité, seulement après l'étape `1` validée)** :
  - Déclencheurs locaux légers sur la télémétrie du joueur suivi
    (sprint, entrée dans la surface, proximité du ballon) — `0€`
    d'`API`, calcul local uniquement.
  - Appel `Gemini` ciblé uniquement aux moments déclenchés (`~20-50`
    par match estimé), pour classifier l'action (passe/tir/tacle/
    dribble) — coût estimé très faible (`<0.05€`/match, à valider
    précisément comme pour les autres estimations de coût ce soir).

  **À valider avant d'implémenter quoi que ce soit** : le taux réel de
  changement d'`ID` du tracker de joueurs sur nos propres matchs (pas
  mesuré ce soir — seul le tracker de ballon a montré des resets
  fréquents dans les logs).

- [ ] **Segmentation par bornes de match — ACTIF PAR DÉFAUT, NON TESTÉ EN CONDITIONS RÉELLES**
  `use_match_boundary_segments=True` partout (`pipeline.py`, `app.py`,
  `tasks.py`) depuis le 11/09/2026, sur demande explicite.

  **Détection des 4 bornes** : `KO1` → `KO2` (séquentiels, dépendance
  obligatoire — `KO2` a besoin de `KO1` comme ancrage) → **`Fin1MT` et
  `FinMatch` en parallèle** (`ThreadPoolExecutor`, `2` workers — les deux
  dépendent uniquement de `KO2`, aucune dépendance entre elles).

  **Analyse `YOLO` des 2 mi-temps EN PARALLÈLE** (`analyze_segments()`,
  `ProcessPoolExecutor`) — plus rapide que du séquentiel. **Ordre final
  toujours garanti correct** (`1ère` MT avant `2e` MT dans les données
  produites) même si la `2e` mi-temps finit de calculer avant la `1ère` :
  `analyze_segments()` trie le résultat combiné par `time`/`frame` avant
  de le retourner, indépendamment de l'ordre de complétion.

  **Logs explicites à chaque étape** (bannière de démarrage, chaque borne
  confirmée individuellement avec son timestamp, plan d'analyse avec
  durées et `%` de vidéo économisé, légende `"Segment 1/2 = 1ère MI-TEMPS"`
  avant l'analyse parallèle, bannière de fin avec compte `events`/`frames`)
  **et une vérification a posteriori** (compte les frames réellement
  retournées dans chaque fenêtre temporelle attendue à partir des DONNÉES
  RÉELLEMENT PRODUITES, signale explicitement si des frames se retrouvent
  hors zone — ne se contente jamais de supposer que ça a fonctionné).

  **Repli de sécurité intégré** : si une étape échoue, retombe
  automatiquement sur le pipeline complet existant, sans casser
  l'analyse.

  **⚠️ Jamais vu tourner sur un vrai match** — à surveiller de près au
  premier vrai test (regarder précisément les logs `[BOUNDARY_SEG]` pour
  confirmer que les `4` bornes se détectent bien, que le pourcentage de
  vidéo économisé est cohérent, et qu'aucune frame ne se retrouve hors
  zone).
  *(Implémenté le 11/09/2026, rendu actif par défaut le 11/09/2026,
  parallélisation ajoutée le 11/09/2026)*

- [ ] **Architecture `Fin1MT`/`FinMatch` (`Gemini` vision) — INTÉGRÉE avec repli de sécurité, NON RETESTÉE EN CONTEXTE**
  `find_fin1mt_gemini`/`find_finmatch_gemini` sont maintenant les
  méthodes **principales** dans le flux normal (11/09/2026) —
  `find_fin1mt_audio`/`find_match_end()` (anciennes méthodes) restent en
  **repli automatique** si la détection `Gemini` échoue (résultat
  `None`), donc aucune perte de détection possible en cas de souci.
  ⚠️ Chaque fonction individuellement validée aujourd'hui (via notebooks
  isolés, `KO2` fourni en vérité terrain) mais **jamais testées ensemble
  dans le flux réel du pipeline** (avec un `KO2` lui-même détecté, pas
  fourni) — à valider sur au moins quelques matchs de référence avant
  de considérer cette intégration pleinement fiable.
  *(Intégré le 11/09/2026)*

- [ ] **Seuil de confiance `team_color_matching.py`** (`DISTANCE_MAX_CONFIANCE
  = 30.0`) — valeur de départ choisie arbitrairement, **jamais calibrée
  empiriquement** (aucun appel `Gemini` réel testé dessus). À ajuster
  avec de vrais tests avant de compter dessus en production.
  *(Identifié le 11/09/2026)*

- [ ] **`FinMatch`/`Andrimont` : écart `+175s` reproductible**
  Diagnostiqué (bascule réelle à `6170s` ratée par la grille `pas=60s`),
  mais non résolu. Pistes documentées mais non testées.
  *(Identifié le 11/09/2026, session précédente)*

---

## 🟢 MINEUR — Nettoyage / robustesse

- [ ] **Réduction `max_output_tokens` / format `JSON` strict — appel par appel, pas en masse**
  Discuté le 11/09/2026. Principe légitime (tokens de sortie plus chers
  que les tokens d'entrée) mais **pas applicable en aveugle** à tous les
  appels `Gemini` existants. Vérifié : `ai/gemini_validator.py` (validation
  buts, `find_goal_after_shot`) analyse activement le texte narratif
  `"evidence"` par recherche de mots-clés (`_has_net_signal`,
  `_has_walking_back`, `_has_celebration`, recherche de `"+5"` dans le
  texte) — une troncature de `max_output_tokens` risquerait de couper ce
  texte et de casser cette logique déjà calibrée. À traiter au cas par
  cas : appels à état simple (ex. classification `NORMAL`/`ATTACK` d'une
  future architecture par échantillonnage) → sûr à limiter ; appels où
  le texte libre est lui-même parsé pour la décision → laisser tel quel
  ou restructurer en champs `JSON` séparés plutôt qu'en texte à parser.

- [ ] **Code mort à supprimer** — voir `CODE_MORT_A_SUPPRIMER.md` (document
  séparé, alimenté au fil des balayages). `9` fichiers confirmés morts à
  ce stade (doublons racine de fichiers réels dans `analysis/`/`ai/`,
  scripts `RunPod`/`Celery` legacy, ancienne détection `KO1` par
  `RandomForest`). Balayage pas encore terminé sur tout le projet.
  *(Identifié le 11/09/2026)*

- [ ] **`dashboard.html` : champ caché `user_team_id` dupliqué**
  Deux `<input type="hidden" name="user_team_id" id="user-team-id">`
  identiques à la suite — invalide (`ID` dupliqué), probablement un
  copier-coller accidentel. Inoffensif en pratique mais à nettoyer.
  *(Identifié le 11/09/2026)*

- [ ] **Nettoyage automatique des fichiers de pré-analyse** —
  `cleanup_orphaned_previews()` ajoutée (purge après `24h`), à vérifier
  qu'elle tourne bien en continu une fois en production (le scheduler
  `start_cleanup_scheduler()` doit être actif).
  *(Corrigé le 11/09/2026 — à vérifier au déploiement)*

- [ ] **Timeout détection équipes/`KO1`** (`app.js`, `TIMEOUT_MS`)
  Porté à `8` minutes suite à un cas réel à `267s` — à surveiller en
  production sur des vidéos plus longues/complexes, pourrait nécessiter
  d'être encore augmenté selon les cas réels rencontrés.
  *(Corrigé le 11/09/2026 — à surveiller)*

---

## ✅ RÉSOLU

- [x] **`STEP 0a` : `frame_skip`/`batch_size` ignoraient la config — analyse `>7h` au lieu de `1-2h`**
  Trouvé le 11/09/2026 suite à un vrai run mesuré à plus de `7h`
  (attendu : `1-2h` pour un match complet, selon la `FAQ` du site).
  `analyze_segments()` (`segment_extractor.py`) a ses propres valeurs
  par défaut codées en dur (`frame_skip=2`, `batch_size=8`), qui
  ignoraient complètement : `(a)` le calcul `AUTO SKIP` fait en tête de
  `run_pipeline()` (vise `~6.25fps` analysés quelle que soit la source —
  ici calculait `5` pour une vidéo à `30fps`, ignoré, tournait à `15fps`,
  plus de `2x` de frames en trop) ; `(b)` `config.YOLO_BATCH_SIZE=4`
  (ignoré, tournait à `8`). Le chemin `candidate_segments`/`coarse_scan`
  préexistant transmettait déjà ces valeurs correctement — uniquement
  mon nouveau chemin de segmentation (`STEP 0a`) avait cet oubli.
  Corrigé : `frame_skip=config.FRAME_SKIP_EVERY` et
  `batch_size=config.YOLO_BATCH_SIZE` transmis explicitement.
  **Non encore revalidé avec un nouveau run complet** pour confirmer le
  temps réellement gagné.

- [x] **`STEP 0a` : `2` autres incohérences trouvées via vérification systématique (`FinMatch` + `KO1=0.0`)**
  Suite à la correction `Fin1MT` ci-dessous, une comparaison systématique
  des `4` bornes entre `STEP 0a` et le flux normal a révélé **`2` autres**
  divergences du même type :
  - **`FinMatch`** : `STEP 0a` n'avait **aucun repli** si `Gemini` échouait
    (abandon immédiat de toute la segmentation), contrairement au flux
    normal (repli sur `find_match_end()`). Corrigé : repli sur
    `find_finmatch_audio()` ajouté dans `STEP 0a` (`find_match_end()`
    non utilisable ici — dépend de `frames_data`, indisponible avant le
    passage `YOLO`).
  - **`kickoff_s_precalcule=0.0`** : signal explicite signifiant *"KO1
    inconnu, ne pas redétecter"* — `STEP 0a` le traitait par erreur
    comme "non fourni" et tentait quand même une détection fraîche,
    contrairement au flux normal qui respecte ce signal. Corrigé, sur
    principe explicite de l'utilisateur (*"en cas de doute, mieux vaut
    que l'analyse parte de `0.0` que de tenter autre chose d'incertain"*) :
    `STEP 0a` abandonne maintenant proprement la segmentation dans ce
    cas, laissant le repli normal (déjà éprouvé) prendre le relais.

- [x] **`STEP 0a` : incohérence `Fin1MT` audio vs `Gemini` — corrigée**
  Trouvé par l'utilisateur le 11/09/2026 : `STEP 0a` (segmentation par
  bornes) utilisait encore `find_fin1mt_audio` (ancienne méthode)
  directement pour `Fin1MT`, sans jamais essayer `find_fin1mt_gemini`
  d'abord — contrairement au flux normal, mis à jour plus tôt dans la
  soirée pour utiliser `Gemini` en priorité avec l'audio en repli.
  Incohérence introduite car `STEP 0a` avait été écrit **avant** cette
  décision. Corrigé : `STEP 0a` utilise maintenant `find_fin1mt_gemini`
  en priorité (en parallèle avec `FinMatch`), avec repli automatique sur
  `find_fin1mt_audio` si `Gemini` échoue — cohérent avec le flux normal.
  **Découvert via un vrai écart mesuré sur `Franchimont`** : la méthode
  audio seule avait donné `2777s` au lieu du vrai `3095s` (`-318s`) —
  diagnostic confirmé (`score audio` au mauvais candidat `4.25`, au bon
  candidat seulement `1.22` — l'algorithme audio a un vrai faux positif
  sur ce match précis). À revalider : si `find_fin1mt_gemini` fait
  mieux sur ce même match une fois testé.

- [x] **Comptage complet des appels Gemini/Claude — pour calibrer un prix utilisateur**
  Ajouté le 11/09/2026, à la demande explicite de l'utilisateur. Balayage
  complet du projet effectué pour trouver **toutes** les sources de coût
  (`4` sources distinctes, `2` `API` différentes) :
  - `Gemini` : `KO1`/`KO2`/`Fin1MT`/`FinMatch` (détection des bornes)
  - `Gemini` : validation buts/tirs `+` lecture numéros de maillot
    (`ai/gemini_validator.py`, fichier volumineux jamais examiné avant
    ce soir — probablement la plus grosse source de coût réelle)
  - `Gemini` : détection des couleurs (pré-analyse, `app.py`, client
    `Gemini` séparé)
  - `Claude` : résumé du match (`ai/claude.py`) — **`API` différente,
    tarification différente**, à ne jamais additionner aveuglément avec
    les appels `Gemini`

  Chaque source instrumentée à son point d'entrée réel (pas de
  suppositions) : `_call_gemini()` dans `ai/gemini_validator.py` (point
  d'entrée unique confirmé, y compris un second appel direct qui le
  contournait, trouvé et instrumenté séparément), `_ask_gemini_colors()`
  dans `analysis/detect_teams_preview.py`, `messages.create()` dans
  `ai/claude.py`. `run_pipeline()` retourne désormais
  `gemini_calls_breakdown`, `gemini_calls_total`, `claude_calls_total`
  avec affichage clair en fin d'analyse.

  **⚠️ Jamais mesuré avec un vrai run complet** — notebook de test
  construit (`notebook_test_avec_couts.py`), en attente d'exécution
  réelle pour obtenir les vrais chiffres à utiliser pour la tarification.

  **Bug corrigé au passage** : `STEP 0a` (segmentation par bornes)
  redétectait `KO1` inutilement même quand déjà fourni via
  `kickoff_s_precalcule` — corrigé, économise maintenant un appel
  `Gemini` complet par analyse.

- [x] **Fichiers temporaires : risque de collision entre analyses simultanées**
  Corrigé le 11/09/2026 (à la demande explicite de l'utilisateur, en
  prévision de la production avec plusieurs analyses en parallèle).

  **Partie 1 — dossiers temporaires `Gemini`/audio** : un dossier
  temporaire **unique par analyse** (`gemini_tmp_YYYYMMDD_HHMMSS`, sous
  `output_dir` — déjà unique par `analysis_id`) est créé en début de
  `run_pipeline()` et transmis explicitement à **tous** les appels de
  détection `Gemini`/audio (`KO1`, `KO2`, `Fin1MT`, `FinMatch` — `9`
  points d'appel à travers tout le fichier, flux normal et segmentation
  par bornes). Avant ce correctif, ces fonctions utilisaient toutes le
  même dossier `/tmp` partagé par défaut — `find_fin1mt_audio`/
  `find_finmatch_audio` avec des noms de fichiers fixes, et
  `_extraire_frame` avec des noms basés uniquement sur le timestamp
  vidéo (`ko_frame_120.0.jpg`) — risque réel de collision si `2`
  analyses différentes interrogeaient coïncidemment le même instant.

  **Partie 2 — audit exhaustif de tout le projet** (à la demande
  explicite de l'utilisateur, "c'est très important") : vérification
  fichier par fichier de tout ce qui écrit sur disque —
  `segment_extractor.py`, `detect_teams_preview.py`, `player_reid.py`,
  `vision/tracker.py`, `team_color_matching.py`, `main.py`, et toutes
  les routes d'upload d'`app.py`. Tous déjà protégés par un identifiant
  unique (`upload_id`, `analysis_id`, `uuid`), **sauf un** :

  **Bug préexistant trouvé et corrigé** (présent avant même les
  changements de ce soir, pas introduit par la session) : la route
  `/upload` (soumission finale du formulaire, fichier vidéo principal)
  nommait le fichier final uniquement à partir du nom **original**
  fourni par l'utilisateur (`secure_filename(f.filename)`), jamais rendu
  unique. Si `2` utilisateurs (ou le même utilisateur `2` fois)
  uploadaient un fichier de même nom (très courant, ex. `"match.mp4"`),
  le second écrasait le fichier du premier sur le disque — potentiellement
  **pendant que le premier était encore en cours d'analyse** (asynchrone,
  `Celery`/thread). Corrigé : chaque upload génère maintenant un
  identifiant unique (date+heure à la seconde `+` suffixe aléatoire
  court), préfixé au nom de fichier — nom original conservé en suffixe
  pour rester lisible.
  *(Identifié et corrigé le 11/09/2026)*

- [x] **`KO2` pas adaptatif (`20s→10s`) + `marge_avant_min=6`**
  Intégré en production le 11/09/2026 (`match_boundaries_v2.py` :
  nouveaux défauts `marge_avant_min=6`, `pas_scan_fin=10` ; mécanisme
  adaptatif porté dans `kickoff_gemini_cascade.py`, `KO1` non affecté
  car n'utilise jamais ce paramètre). Validé `11/11` avec vrais `KO1`
  détectés avant intégration.

- [x] **`app.py` : `run_pipeline` jamais importé — `NameError` systématique**
  `run_analysis()` (chemin fallback threading, sans `Celery`) appelait
  `run_pipeline(...)` sans jamais l'importer nulle part dans le fichier —
  échec systématique de toute analyse lancée par ce chemin. Corrigé le
  11/09/2026 (ajout de `from pipeline import run_pipeline`).

- [x] **`app.py` : erreurs d'analyse invisibles dans le terminal**
  Le `except` de `run_analysis()` stockait l'erreur tronquée (`200`
  caractères) dans l'interface mais n'affichait jamais la trace complète
  dans le terminal — rendait le diagnostic très difficile. Corrigé le
  11/09/2026 (trace complète désormais affichée).

- [x] **`tasks.py` : `team_names`/`kickoff_s_precalcule` non acceptés**
  Bug préexistant (signature de tâche `Celery` incomplète, aurait fait
  planter toute soumission avec noms d'équipe). Corrigé le 11/09/2026.

- [x] **Modèle `Analysis` : colonnes `team_name_0`/`team_name_1` manquantes**
  `TypeError` systématique à la soumission. Corrigé le 11/09/2026.

- [x] **Upload vidéo transmis deux fois** (preview + soumission finale)
  Gaspillage bande passante/disque significatif sur de grosses vidéos.
  Corrigé le 11/09/2026 (réutilisation du fichier de preview).

- [x] **`KO1` : vérification préalable à `t=0`**
  Corrige les matchs démarrant avant `60s`. Intégré en production le
  11/09/2026.