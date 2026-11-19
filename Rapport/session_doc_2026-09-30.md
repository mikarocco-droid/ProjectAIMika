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
