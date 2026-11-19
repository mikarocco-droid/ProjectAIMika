# PROJET SCOUTIA — Résumé session 2026-07-12
# Coller au début d'une nouvelle conversation pour reprendre

---

## CONTEXTE GÉNÉRAL

Projet Scoutia : pipeline analyse vidéo football automatique.
Stack : Flask + YOLO11m + DeepSort FP16 + Gemini 2.5 Flash
GitHub : https://github.com/mikarocco-droid/ProjectAIMika

---

## CE QUI A ÉTÉ FAIT DANS CETTE SESSION

### 1. Fix notebook_kickoff_from_cache.py
- Bug : `builtins.print = _orig_print` était exécuté AVANT la lecture du fichier log
- Fix : fsync → lecture `log_text` → PUIS restauration de print
- Résultat : le profiler trouve maintenant les 19 groupes sur andrimont YOLO ✅

### 2. Dataset consolidé mis à jour
- andrimont_full remplacé par version YOLO (19 groupes au lieu de 6 proxy)
- Dataset final : 55 lignes, 4 matchs ok, 5 vrais KO, 32 faux
- Fichier : `kickoff_groups_combined.csv`

### 3. player_color_profiler.py créé (nouveau module)
- 404 lignes, syntaxe OK
- Compare 4 méthodes d'extraction couleur par joueur/frame :
  - `original` : moyenne RGB sur tout le crop
  - `centered_crop` : médiane HSV sur torse (25%-75% vertical)
  - `centered_crop_no_grass` : idem + filtre pixels pelouse/ombre
  - `hs_histogram` : pic histogramme HS sans pelouse
- Produit CSV + rapport de synthèse
- **À COMMITTER** sur GitHub : `analysis/player_color_profiler.py`

### 4. notebook_color_profiler.py créé
- Notebook Kaggle pour lancer le profiler sur une vidéo
- Config : VIDEO + MATCH_ID à changer selon le match
- Prérequis : `player_color_profiler.py` commité dans `analysis/`

---

## ÉTAT DU DATASET

### kickoff_groups_combined.csv — 55 lignes, 44 colonnes

| match | groupes | ±30s | status |
|-------|---------|------|--------|
| andrimont_full (YOLO) | 19 | 2 | ok |
| p1-minerois | 6 | 1 | ok |
| ster-b-franchimont | 6 | 1 | ok |
| ster-b-raeren | 6 | 0 | recall_fail |
| ster-b-stembert | 6 | 1 | ok |
| ster-b-goe | 6 | 0 | recall_fail_proxy (nuit) |
| ster-wanze | 6 | 0 | recall_fail_proxy (nuit) |

**Matchs ok : 4, 37 groupes, 5 vrais KO, 32 faux**

### EDA features clés (matchs ok seulement)

| feature | True (n=5) | False (n=32) | diff |
|---------|-----------|-------------|------|
| sep_avg | 0.516 | 0.479 | +0.037 |
| duration | 14.4s | 17.1s | -2.7s |
| **delta_dur_prev** | **5.000** | **-1.966** | **+6.966** ← meilleure |
| gap_before | 57.5s | 44.8s | +12.7s |
| rank_from_start | 3.000 | 6.281 | -3.3 |

---

## VRAIS KICKOFFS PAR MATCH

| match | KO réel (s) | KO fmt |
|-------|------------|--------|
| andrimont_full | 307.0s | 5:07 |
| p1-minerois-2026-03-28 | 192.0s | 3:12 |
| ster-b-franchimont-b-2026-04-09 | 318.0s | 5:18 |
| ster-b-raeren-b-2026-04-03 | 410.0s | 6:50 |
| ster-b-stembert-2026-03-07 | 553.0s | 9:13 |
| ster-b-goe-2026-03-20 | 1081.0s | 18:01 |
| ster-wanze-bas-oha-2026-02-28 | 986.0s | 16:26 |
| ster-juprelle-2026-01-25 | 548.0s | 9:08 |

---

## FICHIERS DISPONIBLES (outputs)

- `kickoff_detector.py` — 1541 lignes, corrections échelle, figé ✅
- `kickoff_group_candidate_dataset.py` — 43 colonnes avec features temporelles ✅
- `kickoff_profiler.py` — inchangé ✅
- `notebook_kickoff_only.py` — pipeline vidéo → CSV, avec PROXY DIAG + frames JPEG
- `notebook_kickoff_from_cache.py` — pipeline cache YOLO → CSV (fix log appliqué)
- `kickoff_groups_combined.csv` — dataset consolidé 55 lignes
- **`player_color_profiler.py`** — nouveau, à committer ⚠️
- **`notebook_color_profiler.py`** — nouveau, à lancer sur Kaggle

---

## COMMITS GITHUB EN ATTENTE

```
analysis/kickoff_detector.py
  "Kickoff detector: fix coordinate-scale bugs and add profiling instrumentation"

analysis/kickoff_group_candidate_dataset.py
  "Add kickoff group candidate dataset exporter with temporal features"

analysis/player_color_profiler.py
  "Add player color profiler diagnostic tool"
```

---

## DIAGNOSTIC PROXY COULEUR — RÉSULTATS

### Ce qui est établi
- `total` élevé (20-30) partout sur ster-wanze et ster-goe
- `team` quasi nul (ratio 0.00-0.06) → CAS A : joueurs détectés, équipes non assignées
- 95% des blobs = ombres de pelouse (H=28-33, S=28-35) que le masque rate
- Bleu marine (H=110-125, S=50-70, V=45-55) : partiellement détecté
- Terrain hivernal (ster-juprelle) : total=0 partout → masque exclut la pelouse brune

### Conclusion
Le proxy morphologique est structurellement limité sur :
- Matchs de nuit
- Terrain hivernal
**Ce n'est PAS un bug de kickoff_detector.py**

---

## PROCHAINES ÉTAPES — ORDRE PRIORITAIRE

### ÉTAPE 1 (immédiat) — Commits GitHub
Les 3 fichiers ci-dessus.

### ÉTAPE 2 (court terme) — Lancer color profiler
```
1. Committer player_color_profiler.py sur GitHub
2. Lancer notebook_color_profiler.py sur Kaggle
   → sur ster-wanze (nuit)
   → sur ster-juprelle (hivernal) ou ster-b-raeren
3. Récupérer les CSV + rapports
4. Analyser : valid_pixels_ratio, hue_diff, agreement entre méthodes
```

**Questions auxquelles le profiler doit répondre :**

| total | team | Conclusion |
|-------|------|-----------|
| élevé | élevé | bug logique détecteur |
| élevé | faible | → proxy couleur à améliorer |
| faible | faible | caméra hors terrain |

**Tableau comparatif à construire :**

| Méthode | Valid pixels | Dom. fraction | Agreement | Stabilité |
|---------|-------------|--------------|-----------|-----------|
| original | ? | ? | ? | ? |
| centered_crop | ? | ? | ? | ? |
| centered_crop_no_grass | ? | ? | ? | ? |
| hs_histogram | ? | ? | ? | ? |

**Cas problématiques à isoler :**
- `cng_valid_ratio < 0.1` → crop trop petit ou trop de pelouse
- `hue_diff_orig_cng > 20` → méthodes divergent, lequel est juste ?
- `agreement_orig_cng = 0` → l'équipe change selon la méthode

### ÉTAPE 3 (selon résultats profiler)
**Si extraction bonne → passer directement au KMeans**
**Si extraction mauvaise → améliorer l'extraction d'abord**

KMeans prévu :
- Fenêtre glissante 30s
- Clustering k=2 sur couleurs dominantes filtrées
- Robuste : rouge/bleu, jaune/noir, blanc/vert, etc.
- Cas difficile blanc/noir → séparer par luminosité V

### ÉTAPE 4 (après proxy stabilisé)
- Compléter dataset jusqu'à 8-10 matchs ok
- DecisionTree(max_depth=3) sur features temporelles
- Questions : P(KO | rank), delta_dur_prev > 4 discrimine-t-il ?

---

## DÉCISIONS ARCHITECTURALES FIGÉES

1. Ne plus modifier kickoff_detector.py (hors bugs prouvés)
2. Le proxy couleur est dans les notebooks, PAS dans kickoff_detector.py
3. La solution proxy = KMeans, pas des seuils RGB supplémentaires
4. Pour matchs difficiles → utiliser notebook_kickoff_from_cache.py avec YOLO
5. match_status = "ok" | "recall_fail" | "recall_fail_proxy" | "hors_fenetre"
6. Le color profiler ne modifie RIEN au pipeline — observation uniquement

---

## NOTES TECHNIQUES

- andrimont cache Kaggle : `/kaggle/input/datasets/michaelziant/andrimont-cache/cache`
- notebook_kickoff_from_cache.py : fix log appliqué (log_text lu AVANT restauration print)
- notebook_kickoff_only.py : masque original restauré + _is_green_pixel v2 + crop torse
- Transcripts : /mnt/transcripts/2026-07-07-08-43-24-scoutia-kickoff-dataset-proxy-diag.txt