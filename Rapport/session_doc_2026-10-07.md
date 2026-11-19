# Session Scoutia — 29-30 septembre 2026
## Migration YOLO26m complète + détection buts high_side

---

## CONTEXTE

Suite aux sessions précédentes (28-29 sept) qui avaient :
- Validé YOLO26m pour la détection ballon (3.4× plus de détections, 2× plus rapide)
- Corrigé les timestamps events (bug `time=0`)
- Corrigé les fixes KO2/`_video_duration_s`

Cette session a complété la migration et résolu la chaîne de détection buts sur caméra `high_side`.

---

## 1. TEST YOLO26m SUR LES JOUEURS

### Protocole
Script `test_yolo26_joueurs.py` — 500 frames après KO (t=350s), conf=0.4, même résolution que pipeline (960×540).

### Résultats

| Métrique | YOLO11m | YOLO26m | Delta |
|----------|---------|---------|-------|
| Joueurs/frame médiane | 10 | 12 | +2 |
| Frames ≥10 joueurs | 66% | 87.8% | +21.8% |
| Frames ≥15 joueurs | 1.6% | 11.6% | ×7 |
| Frames sans joueur | 0% | 0% | = |
| Confiance médiane | 0.654 | 0.690 | +0.036 |
| Inférence (ms/frame) | 31.6 | **16.7** | -14.9ms |

### Décision
Migration YOLO26m justifiée pour joueurs ET ballon — une seule inférence par frame.

---

## 2. ARCHITECTURE SINGLE YOLO26m

### Avant (double inférence)
```
Frame → YOLO11m(classes=[0]) → joueurs
Frame → YOLO26m(classes=[32]) → ballon
```

### Après (single inférence)
```
Frame → YOLO26m(classes=[0,32]) → joueurs + ballon
```

### Gain mesuré sur pipeline complet
- `total_ids` : 260 → 252 (moins de tracks fantômes)
- `distance_centroides` : 1.106 → 1.184 (meilleure séparation équipes)
- `joueurs identifiés` : 34 → 34 (identique)
- `compacité kmeans` : 173.4 → 162.1 (clusters plus compacts)

### Fichiers modifiés
- `vision/detector.py` — paramètre `use_single_yolo26`, skip `load_player_model` si activé
- `main.py` — bloc single YOLO26m dans `process_batch`, suppression double inférence
- `pipeline.py` — `use_single_yolo26` dans signature `run_pipeline` et `process_video`

### Paramètre
```python
run_pipeline(
    ball_model_name   = "yolo26m.pt",
    use_single_yolo26 = True,   # défaut en production
)
```

---

## 3. DÉTECTION BUTS HIGH_SIDE — CHAÎNE COMPLÈTE

### Problème initial
Sur caméra `high_side`, les tirs ne sont jamais détectés (joueurs trop petits). Le pipeline exigeait un tir préalable avant d'accepter un but → tous les buts rejetés avec `[buffer vide]`.

### Solution : HIGH_SIDE_NO_SHOT (events.py)

Nouveau chemin dans `detect_events` :
```python
elif (camera_type == "high_side"
      and (_vitesse_recente_elevee or _trajectoire_compatible)
      and _pas_evenement_contradictoire
      and state.get("goal_cd", 0) == 0):
    events.append({
        "source": "high_side_no_shot",
        "lost_frames": _bt.lost_frames,
        ...
    })
```

Log : `✅ goal CONFIRMÉ (HIGH_SIDE_NO_SHOT) à t=384.3s — vitesse=False trajectoire=True goal_cd=0 x=1691(88.1%) y=524(48.5%) lost_frames=0`

### Filtres pré-Gemini (pipeline.py)

**Filtre HS_NO_SHOT_PRE** — élimine les faux positifs évidents avant Gemini :
- `x < 1% ou x > 99%` → sortie de cadre (élimine t=576s x=0%, t=1798s x=0%)
- `lost_frames >= 2` → signal non fiable (élimine t=734s lost=2)

**Fix CENTRAL_BX** — le filtre de zone centrale utilisait `bx` absent pour `high_side_no_shot`. Correction : utiliser `x_norm` comme `bx` pour ce type d'event (x=88.1% passe le filtre).

**Filtre TC_PRE** — `terminal_clearance` éloignés de >120s d'un but → ignorés avant Gemini. Au run précédent : 79/82 rejetés, gaspillage massif. Après : ~10 appels au lieu de 82.

### Filtres goal_posthoc (goal_posthoc.py)

Bug critique : `peak_before` utilisé avant sa définition dans `valid_high_side` → `goal_posthoc` complètement ignoré.

Fix : `peak_before` calculé avant le bloc `if not recent_shot_strict`.

Filtres F1/F2/F3 intégrés dans `valid_high_side` :
```python
_hs_is_border_exit = rebound and (_hs_x_norm < 0.08 or _hs_x_norm > 0.92)  # F1
_hs_out_of_frame   = _hs_x_norm < 0.03 or _hs_x_norm > 0.97                # F2
_hs_fast_exit      = peak_before > 300 and (_hs_x_norm < 0.12 or _hs_x_norm > 0.88)  # F3
```

Validé sur 6 candidats (5 FP, 1 vrai but) — F3 élimine 4/5 FP sans toucher au vrai but (peak=96 < 300).

---

## 4. BUGS CORRIGÉS

| Bug | Fichier | Fix |
|-----|---------|-----|
| `UnboundLocalError: find_ko2_gemini` | pipeline.py | Import redondant ligne 466 supprimé |
| `UnboundLocalError: _video_duration_s` | pipeline.py | Défini à `None` avant `_skip_ko2`, recalculé après `process_video` |
| `_skip_ko2` absent | pipeline.py | Réintégré — skip KO2 si `video_end_s < KO1 + 45min` |
| `peak_before` non défini | goal_posthoc.py | Déplacé avant `valid_high_side` |
| `NameError: _single_yolo26_conf_ball` | main.py | Variable définie dans les 3 branches du if/elif/else |
| `ball["frame"] = frame_id` manquant | main.py | Injecté dans tous les events après detect_events |
| `time=0` pour events sans ballon | main.py | `e["time"] = round(frame_id/fps)` en post-traitement |
| `FALLBACK SUPPRIMÉ` verbeux | events.py | Passé en `logging.debug` |
| Double logging StreamHandler | notebook | StreamHandler retiré |
| Bruit numba/librosa | notebook | `logging.getLogger("numba").setLevel(WARNING)` |
| `Detector.__init__` manque paramètres | detector.py | `ball_model_name`, `ball_conf`, `use_single_yolo26` ajoutés |
| YOLO11m chargé inutilement | detector.py | Skip `load_player_model` si `use_single_yolo26=True` |
| PDF `—` Helvetica | export/pdf.py | `cell()`/`multi_cell()` surchargés avec `clean()` automatique |
| TypeError possession sort | notebook | `key=lambda x: str(x[0])` |

---

## 5. ÉTAT FINAL PIPELINE

### Paramètres de production (Andrimont high_side)
```python
run_pipeline(
    video_path           = VIDEO,
    sport                = "football",
    camera_type          = "high_side",
    ball_model_name      = "yolo26m.pt",
    ball_conf            = 0.15,
    use_single_yolo26    = True,
    kickoff_s_precalcule = 310.0,
    video_end_s          = 3700,
)
```

### Architecture détection buts high_side
```
YOLO26m(classes=[0,32])
  ├── joueurs → DeepSort → OSNet → ReID → équipes
  └── ballon (conf≥0.15) → BallTracker → HSV
                                ↓
                         detect_events()
                         camera_type="high_side"
                                ↓
                    HIGH_SIDE_NO_SHOT si pas de tir
                                ↓
                    HS_NO_SHOT_PRE (x<1% ou lost≥2)
                                ↓
                         goal_posthoc (F1/F2/F3)
                                ↓
                    TC_PRE (terminal_clearance éloignés)
                                ↓
                            Gemini
                                ↓
                          But confirmé
```

### Ce qui reste à valider
- [ ] Recharger crédit Anthropic (console.anthropic.com)
- [ ] Run complet `video_end_s=3700` sur les 5 buts connus (383, 1831, 2554, 2938, 3582s)
- [ ] Valider que goal_posthoc détecte les buts 2-5
- [ ] FP résiduel t=1523.7s (même profil que vrai but) → laisser à Gemini

### Fichiers pushés sur GitHub
- `vision/detector.py`
- `main.py`
- `pipeline.py`
- `analysis/events.py`
- `analysis/goal_posthoc.py`
- `export/pdf.py`
- `kaggle_notebook.py`

---

## 6. TEST YOLO26m JOUEURS — DONNÉES BRUTES

Script : `test_yolo26_joueurs.py`
Vidéo : match-p4-andrimont-2026-04-26.mp4
Fenêtre : t=350s → t=417s (jeu actif après KO)

```
YOLO11m : 10.4 joueurs/frame | conf médiane 0.654 | 31.6ms | 66% frames ≥10 joueurs
YOLO26m : 11.9 joueurs/frame | conf médiane 0.690 | 16.7ms | 87.8% frames ≥10 joueurs
```

Pipeline complet (600s, CPU) :
```
YOLO11m double : total_ids=260 | distance_centroides=1.106 | 34 joueurs
YOLO26m single : total_ids=252 | distance_centroides=1.184 | 34 joueurs
```

---

# Session Scoutia — 6-7 octobre 2026
## KO_DETECTION + validation Gemini pour buts high_side

---

## CONTEXTE

Suite de la session 29-30 sept. Les bugs principaux étaient corrigés mais 3/5 vrais buts restaient non détectés (1831s, 2938s, 3582s).

---

## 1. DIAGNOSTIC `_goal_already_added`

**Bug identifié :** `_goal_already_added = any(e.get("type") == "goal" for e in events)` bloquait TOUS les buts après le premier du match (t=384s).

**Fix :** Fenêtre glissante `goal_cd_max / fps` secondes :
```python
_goal_cd_max_s = goal_cd_max / max(fps, 1)
_goal_already_added = any(
    e.get("type") == "goal"
    and 0 <= current_time - e.get("time", 0) <= _goal_cd_max_s
    for e in events
)
```

**Impact :** 1831s désormais détecté via `HIGH_SIDE_NO_SHOT` conf=0.95 ✅

---

## 2. TEST YOLO26m BRUT — 3 buts manqués

Script : `test_yolo_buts.py` — conf=0.05, ±5s, frame_skip=4

| But | YOLO voit ballon | Frames ±2s | Diagnostic |
|-----|-----------------|------------|------------|
| 1831s | 25% (19/75) | ✅ 2 détections conf=0.13-0.22 | CAS A — ballon visible mais x=31-39% (pas en zone de but au moment exact) |
| 2938s | 1% (1/75) | ❌ 0 détections | CAS B — ballon masqué dans mêlée |
| 3582s | 1% (1/75) | ❌ 0 détections | CAS B — ballon masqué par joueurs |

**Conclusion :** 2938s et 3582s ne peuvent pas être détectés par le ballon — limite physique. → Signal alternatif nécessaire.

---

## 3. KO_DETECTION — Architecture

### Principe
```
frames_data (joueurs déjà trackés)
      ↓
detect_kickoff_candidates() — signal géométrique
      ↓
KO candidats (balance équipes monte)
      ↓
validate_kickoff_context_gemini() — fenêtre [-25s, +5s]
      ↓
GOAL / NO_GOAL / UNCERTAIN
```

### Fichiers créés
- `analysis/detect_kickoff_candidates.py` — détecteur géométrique
- `ai/gemini_validator.py` — `validate_kickoff_context_gemini()`
- `pipeline.py` — étape 2c `KO_DETECTION` pour `camera_type=high_side`

### Détecteur géométrique
Signal principal : **balance équipes** = `min(n_team0, n_team1) / max(n_team0, n_team1)`

Seuils calibrés sur match Andrimont :
```python
balance_delta_min = 0.10   # hausse minimale
balance_after_min = 0.20   # niveau après
min_gap_s         = 30.0   # distance minimale entre candidats
```

Rappel mesuré : **5/5 vrais buts** couverts avec 40 candidats sur match complet.

| But | Candidat KO | Délai | Score |
|-----|------------|-------|-------|
| 383s | t=399s | +16s | 0.152 |
| 1831s | t=1836s | +5s | 0.249 |
| 2554s | t=2556s | +2s | 0.263 |
| 2938s | t=2951s | +13s | 0.300 |
| 3582s | t=3603s | +21s | 0.450 |

### Validation Gemini
Prompt `validate_kickoff_context_gemini` :
- Fenêtre [-25s, +5s] autour du candidat KO
- 11 frames espacées
- KO visible = signal fort sur high_side
- Exclusions explicites : dégagement gardien, tête hors cadre, touche/dégagement, coup de pied de but, occasion dégagée

Seuils : `GOAL conf>=0.70` ou `UNCERTAIN conf>=0.65`

### Résultats finaux
| Candidat | Décision | Conf | Vrai but ? |
|----------|----------|------|------------|
| t=49:11 (2951s) | GOAL | 0.95 | ✅ 2938s |
| t=60:03 (3603s) | GOAL | 0.95 | ✅ 3582s |
| t=36:39 (2199s) | GOAL | 0.95 | ❌ FP — jeu normal au centre |
| 32 autres | NO_GOAL | 0.95-1.00 | ✅ rejetés |

**Bilan : +2 vrais buts, 1 FP résiduel**

---

## 4. BILAN DÉTECTION GLOBALE — Match Andrimont

| But | Via ballon | Via KO | Confirmé |
|-----|-----------|--------|---------|
| 383s | ✅ HIGH_SIDE_NO_SHOT | skip (déjà confirmé) | ✅ |
| 1831s | ✅ HIGH_SIDE_NO_SHOT (fix _goal_already_added) | skip | ✅ |
| 2554s | ✅ HIGH_SIDE_NO_SHOT | skip | ✅ |
| 2938s | ❌ ballon masqué | ✅ KO_DETECTION | ✅ |
| 3582s | ❌ ballon masqué | ✅ KO_DETECTION | ✅ |

**5/5 vrais buts couverts par au moins une voie de détection.**

---

## 5. BUGS CORRIGÉS (cette session)

| Bug | Fichier | Fix |
|-----|---------|-----|
| `_goal_already_added` bloquait tous buts après le premier | events.py | Fenêtre `goal_cd_max/fps` secondes |
| `base64` non importé dans gemini_validator | gemini_validator.py | `import base64` ajouté |
| `.env` absent dans notebook replay | kaggle_replay_ko.py | `os.makedirs(".env", exist_ok=True)` |
| `cross_frame` utilisait `c_prev` non défini | events.py | Remplacé par `state["last_ball_pos"]` |
| `n_validated_goals` utilisait `events_validated` périmé | pipeline.py | Utilise `events` (liste finale) |
| Modèles Gemini obsolètes (2.0-flash, 2.5-flash) | gemini_validator.py | Tout migré vers `gemini-3.8-flash` |

---

## 6. FICHIERS À PUSHER

| Fichier | Emplacement | Contenu |
|---------|------------|---------|
| `events.py` | `analysis/` | fix `_goal_already_added`, `cross_frame/cross_side/bx_prev/bx_cross` dans dict |
| `detect_kickoff_candidates.py` | `analysis/` | nouveau — détecteur KO géométrique |
| `gemini_validator.py` | `ai/` | `validate_kickoff_context_gemini`, modèles `gemini-3.8-flash` |
| `pipeline.py` | racine | étape 2c `KO_DETECTION`, fix `n_validated_goals` |

---

## 7. PENDING

1. Intégrer `KO_DETECTION` dans le run complet 3700s (GPU nécessaire)
2. Valider que le FP t=36:39 est bien filtré par absence de `HIGH_SIDE_NO_SHOT` voisin
3. Vérifier 1831s sur run complet (HSV non déterministe)
4. Tester sur d'autres matchs high_side pour valider la généralisation
