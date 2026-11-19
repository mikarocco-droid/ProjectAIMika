# PROJET SCOUTIA — RÉSUMÉ COMPLET POUR NOUVELLE CONVERSATION
## Date : 2026-06-14

---

## CONTEXTE GÉNÉRAL

Pipeline d'analyse vidéo football automatisé.
**Stack** : Flask + YOLO11m + DeepSort FP16 GPU + Gemini 2.5 Flash + Claude.
**Tests** : Kaggle T4x2. **Prod cible** : Hetzner CX22 + RunPod RTX 4090 + Cloudflare R2.
**Repo GitHub** : ProjectAIMika (https://github.com/mikarocco-droid/ProjectAIMika)
**Vidéo de test principale** : match_test.mp4 (15 min, buts réels à **02:20** et **09:47**)

---

## ARCHITECTURE PIPELINE (état actuel)

```
terminal_events (goal/save/clearance) — détecte 6 buts côté gauche ET droit
    + goal_posthoc fallback (filtre offensif : ±20s goal, ±15s offensif, ±5s défensif score≥8)
    + Gemini validation (3 offsets [0,-5,5])
    + SHOT→GOAL (find_goal_after_shot, parallèle, seuils 6/4)
    + HIGHLIGHT VIS (lecture numéros par description visuelle)
    + player_identity_memory (PIM) — mémoire joueurs inter-matchs
    + geometry/ — extraction géométrie terrain (Sprint BC)
    + kickoff_detector — détection coup d'envoi
    + camera_profile adaptatif
    + A/B test V1/V2 sur SHOT→GOAL
```

---

## RÉSULTATS RUNS

| Run | Gemini | Temps | Buts |
|-----|--------|-------|------|
| V21 | 29 appels | 43.2 min | 2/2 ✅ |
| V22 | 29 appels | 44.2 min | 1/2 ⚠️ |
| V23 | 54 appels | 61.7 min | 1/2 + 1 FP ⚠️ |
| V24 | EN COURS | — | ? |

**V21 = meilleur résultat historique** (2/2 buts, 0 FP).

---

## ÉTAT ACTUEL — V23 ANALYSÉ

### Buts V23
- ✅ **10:03** → VRAI but 09:47 (shot=580.6s, score=12, conf=0.95)
- ❌ **04:21** → FP (shot=255.7s, score=8 — V1 garde, V2 rejette)
- ❌ **02:20** → MANQUANT

### Cause du but 02:20 manquant
- `shot=129.5s` → `find_goal_after_shot` → Gemini voit "beside post" à 02:11
- Score = **-3** dans le repo GitHub (avant fix) → rejeté
- Le fix **-3 → -1** a été commité dans GitHub pendant cette session
- **V24 est en train de tourner** pour confirmer que le fix récupère le but
- Note : même avec -1, le ballon est à bout portant → le filet peut ne pas être visible
  Le coup d'envoi à off=+18s devrait scorer +5 et compenser

### FP 04:21
- V1 (find_goal_after_shot) : score=8 → goal
- V2 (nouvelle logique observation) : score=0 → pas goal
- **Décision : ne pas toucher l'A/B test** tant que stats insuffisantes (besoin 10-20 divergences)

### HIGHLIGHT VIS — 41 appels inutiles
- 1 appel Gemini par tir même si jersey_map connaît déjà le joueur
- PIM non consulté avant chaque appel → 0 cache hit
- Fix dans kaggle_notebook_patched.py : skip si player_id dans jersey_map
- **Économie attendue : ~33 appels → ~14 min récupérées**

---

## PARAMÈTRES CLÉS (état actuel dans GitHub)

```python
OFFSETS_POSTHOC     = [0, -5, 5]
OFFSETS_EVENTS      = [-1, 0, 2]          # ← POTENTIEL FIX : ajouter +10 ou +12
find_goal_after_shot offsets = [0, 2, 4, 7, 12, 18, 25, window=30]
context_before      = 35s (buts gemini) / 25s (buts normal) / 12s (tirs)
context_goal        = 8s
goal_cooldown       = 45s
speed_base          = max(3.0, min(x, 4.5))
SHOT→GOAL seuils    : goal_score>=6 pour 2 votes, >=4 pour 1 vote
beside_post         = -1 (fix commité)
XG_MIN_FOR_HIGHLIGHT = 0.50
beside_post dans find_goal_after_shot : -1 ✅
```

---

## NOUVEAUX MODULES DEPUIS V21

- `player_identity_memory.py` — PIM : registre 31 joueurs, conf≥0.6
- `geometry/` — GeometryExtractor, FieldModel, GeometryDiagnostics, FieldAnchorModel
- `kickoff_detector.py` — détection coup d'envoi
- `analysis/detect_teams_preview.py` — détection équipes via Gemini vision
- `analysis/player_entity.py` — entités joueurs enrichies
- SHOT→GOAL V2 — A/B test actif
- HIGHLIGHT VIS — 41 appels/match → trop cher, fix en cours

---

## SPRINT BC — GÉOMÉTRIE TERRAIN

### État actuel (run V23)
```
Vidéo 1 (avec les vrais buts) :
  goal_box      : left=0.052 right=0.085 (106/150 obs) ✅ signal fort
  penalty_line  : 0.054 (103 obs) ← INCOHÉRENT (devrait être > goal_line)
  sideline      : top=0.169 bot=0.430
  goal_left σ   : 0.1341 (INSTABLE ⚠️)
  Ready         : False
  goal_prior    : None
  px_per_m      : None
```

### Problème fondamental identifié
**GeometryExtractor ne sait pas ce qu'il regarde.**
- Il détecte des structures verticales mais les étiquette incorrectement
- `obs.goal_left_x` ment : ce n'est pas un poteau, c'est une "ligne verticale importante"
- `penalty_line=0.054` < `goal_box_left=0.052` → physiquement impossible
- L'approche Hough a atteint ses limites sur ce type de caméra amateur

### Roadmap BC décidée

**BC.3 — FieldAnchorModel contraint** (prochaine étape)
- Remplacer `obs.goal_left_x` par `obs.vertical_anchor_x` (arrêter le mensonge sémantique)
- Créer un modèle de **contraintes** pas de **positions fixes** :
  ```
  goal_line_x < penalty_line_x < halfway_x  (toujours vrai)
  ```
- Système hypothèses → validation (pas détection → interprétation)
- Fusion : camera_profile + observations + contraintes FIFA
- Sortir `confidence_geometry` par hypothèse

**BC.4** — Accumulation multi-frames (200 frames → terrain moyen du match)

**BC.5** — Projection terrain (x,y métriques pour events)

**BC.6** — Mini-terrain Scoutia (heatmaps, zones de tirs en coordonnées réelles)

### Fichiers geometry/ à uploader pour BC.3
- `geometry/field_anchor_model.py`
- `geometry/geometry_extractor.py`
(demandés mais pas encore uploadés — à faire en début de prochaine session)

---

## LEARNING

- 9 matchs | 813 events | xG avancé 43 samples | sklearn actif ✅
- FP zones : non actif (< 25 matchs)
- SPATIAL_MAX_DIST = 226px

---

## PENDING / ACTIONS EN COURS

| Priorité | Action | État |
|----------|--------|------|
| 🔴 | V24 run Kaggle — tester fix beside_post -1 | EN COURS |
| 🟠 | Si V24 toujours 1/2 → ajouter off=+10 dans find_goal_after_shot | Prêt |
| 🟡 | HIGHLIGHT VIS skip PIM (kaggle_notebook_patched.py) | Prêt à commiter |
| 🟡 | AB STATS agrégées | Prêt à commiter |
| 🟢 | BC.3 — FieldAnchorModel contraint | Démarrer |
| ⬜ | Crédits Anthropic API (erreur 400) | Pending |
| ⬜ | 9 matchs → objectif 25 | Pending |
| ⬜ | Qualité clips : copy stream vs CRF 18 vs CRF 23 | Pending |

---

## FICHIERS MODIFIÉS CETTE SESSION

- `gemini_validator.py` — beside_post -3→-1, prompt filet déformé, early stop prompt
- `video_utils.py` — context_before adaptatif (penalty=45s, gemini=35s, tir=12s)
- `kaggle_notebook.py` — check video_utils corrigé + 3 nouveaux patches runtime :
  - HIGHLIGHT VIS skip PIM
  - AB STATS agrégées
  - video_utils adaptatif auto-patch
- Tout commité dans GitHub ✅

---

## POUR REPRENDRE LA SESSION

Commencer par :
1. Résultats V24 (log Kaggle) — le but 02:20 est-il récupéré ?
2. Si oui → commiter kaggle_notebook_patched.py (HIGHLIGHT VIS + AB STATS)
3. Si non → ajouter off=+10 dans OFFSETS_EVENTS ou find_goal_after_shot
4. Uploader geometry/field_anchor_model.py et geometry/geometry_extractor.py → démarrer BC.3

