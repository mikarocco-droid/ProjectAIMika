# PROJET SCOUTIA — Résumé complet session 2026-07-07
# À coller au début d'une nouvelle conversation

---

## CONTEXTE GÉNÉRAL

Projet Scoutia : pipeline d'analyse vidéo football automatique.
Stack : Flask + YOLO11m + DeepSort FP16 + Gemini 2.5 Flash
GitHub : https://github.com/mikarocco-droid/ProjectAIMika

Le pipeline détecte automatiquement :
- Les buts (pipeline BC4 + Gemini validator)
- Le coup d'envoi (kickoff_detector.py)
- Les possessions, stats, replay engine

---

## PHASE ACTUELLE : KICKOFF_GROUP_CANDIDATE DATASET

### Contexte
On a arrêté de modifier kickoff_detector.py à la main (tuning de seuils).
On construit un dataset pour comprendre objectivement le comportement
et éventuellement entraîner un classifieur simple (DecisionTree depth=3).

---

## ÉTAT DES FICHIERS (commitables sur GitHub)

### analysis/kickoff_detector.py — FIGÉ ✅
- 1541 lignes
- Corrections d'échelle appliquées :
  - `_post_activity` normalisée [0-1], seuil `200` → `0.10`
  - `p_motion` calculé en coordonnées normalisées (division par frame_w)
- Logique `[-1]` restaurée (dernier groupe = candidat kickoff)
- Instrumentation complète : [KICKOFF PLAYERS], [KICKOFF GROUPS], [KICKOFF PHYS]
- Aucune heuristique expérimentale (pas de sep×dur, KP_MAX_T, [0], max(sep))
- Message de commit : "Kickoff detector: fix coordinate-scale bugs and add profiling instrumentation"

### analysis/kickoff_group_candidate_dataset.py — FIGÉ ✅
- 43 colonnes dont features temporelles
- Colonnes clés :
  - Identité : match_id, candidate_id, video_name, camera_type
  - Temps : t_start, t_end, duration, t_start_fmt
  - Joueurs : sep_avg, sep_max, players_avg, players_max
  - Mouvement : p_motion_avg, p_motion_before, p_motion_after
  - Ballon : activity_after, ball_visible_ratio
  - Décision : selected_original, selected_rank, score_original
  - Annotation : distance_to_real_kickoff, is_within_30s, verified_label
  - Features temporelles : rank_from_start, rank_from_end, gap_before, gap_after,
    delta_sep_prev, delta_dur_prev, rolling_sep_3, rolling_players_3,
    is_local_sep_max, t_relative
  - Status : match_status
- Message de commit : "Add kickoff group candidate dataset exporter with temporal features"

### analysis/kickoff_profiler.py — inchangé ✅

---

## NOTEBOOKS KAGGLE (dans outputs)

### notebook_kickoff_only.py
- Pipeline vidéo → CSV en ~5min
- Avec diagnostic proxy [PROXY DIAG] toutes les 30s
- Avec frames annotées JPEG (proxy_diag_1m00s.jpg etc.)
- Avec _extract_player_proxies amélioré :
  - Crop torse (25%-75% vertical)
  - Filtre pixels pelouse (_is_green_pixel v2 : pelouse + ombres)
  - Extraction couleur dominante HSV par joueur

### notebook_kickoff_from_cache.py
- Pour matchs avec cache YOLO pkl existant (~2min)
- Même output que notebook_kickoff_only.py

---

## DATASET kickoff_groups_combined.csv — ÉTAT ACTUEL

### 9 matchs collectés

| match | groupes | KO dans ±30s | selected | status |
|-------|---------|-------------|---------|--------|
| andrimont_full | 6 | ✅ rank=3,4 | ❌ | ok |
| p1-minerois | 6 | ✅ rank=3 | ❌ | ok |
| ster-b-franchimont | 6 | ✅ rank=0 | ❌ | ok |
| ster-b-raeren | 6 | ❌ absent | — | recall_fail |
| ster-b-stembert | 6 | ✅ rank=5 | ✅ SEL | ok |
| ster-b-goe | 6 | ❌ hors_fenetre | — | recall_fail_proxy (nuit) |
| ster-wanze | 6 | ❌ recall_fail | — | recall_fail_proxy (nuit) |
| ster-juprelle | 0 | ❌ | — | recall_fail_proxy (terrain hivernal) |
| (1 dernier à intégrer) | | | | |

**Matchs exploitables (ok) : 4 matchs, 24 groupes, 5 vrais KO**

### EDA sur matchs "ok" (features clés)

| feature | True (n=5) | False (n=19) | diff |
|---------|-----------|-------------|------|
| sep_avg | 0.516 | 0.471 | +0.045 |
| duration | 14.4s | 20.6s | -2.8s |
| delta_dur_prev | 5.000 | -3.500 | **+8.500** ← meilleure feature |
| gap_before | 57.5s | 49.9s | +7.6s |
| rank_from_start | 3.000 | 2.368 | +0.632 |

---

## DIAGNOSTIC PROXY COULEUR — PROBLÈME IDENTIFIÉ

### Ce qui a été fait
- Ajout [PROXY DIAG] : total/team/ratio/ball/green toutes les 30s
- Ajout frames annotées JPEG pour diagnostic visuel
- Amélioration _extract_player_proxies :
  - Crop torse au lieu du blob entier
  - Filtre pixels pelouse avant calcul couleur
  - Attribution équipe élargie (rouge + bleu marine + orange)

### Diagnostic ster-wanze et ster-b-goe (matchs de NUIT)
```
total = 20-30 (joueurs bien détectés)
team  = 0-2   (ratio 0.00-0.06)
→ CAS A : joueurs détectés mais équipes non assignées
→ Cause : proxy couleur rouge/bleu trop spécifique
→ Maillots bleu marine (H=110-125, S=50-70, V=45-55) partiellement détectés
→ 95% des blobs = ombres de pelouse (H=28-33, S=28-35, V=130)
   masque anti-pelouse rate ces ombres peu saturées
```

### Diagnostic ster-juprelle (terrain HIVERNAL)
```
total = 0.0 partout — VIDE
→ Terrain brun-vert hivernal : la pelouse est dans la plage H=30-90 du masque
   → tout le terrain est masqué → zéro blob détecté
```

### Conclusion
**Le proxy morphologique est structurellement limité sur :**
- Matchs de nuit (ombres de pelouse comme faux blobs, bleu marine manqué)
- Terrain hivernal (pelouse brune exclue par masque)

**Ce n'est PAS un problème de kickoff_detector.py.**
C'est une limite du proxy couleur dans notebook_kickoff_only.py.

---

## PROBLÈMES À DEUX NIVEAUX

### Problème A — Recall (génération des groupes)
Sur certains matchs, le vrai KO n'est dans aucun groupe généré.
Causes identifiées :
- Proxy couleur trop spécialisé rouge/bleu → matchs de nuit, terrain hivernal
- Solution structurelle : passer à KMeans clustering sur couleurs dominantes
  (robuste à toute combinaison rouge/bleu, jaune/noir, blanc/vert etc.)

### Problème B — Ranking (sélection du bon groupe)
Sur les matchs ok, le vrai KO est dans les candidats mais pas sélectionné.
L'algorithme [-1] sélectionne le dernier groupe, souvent faux.
Solution : DecisionTree(max_depth=3) sur les features du dataset.

---

## PROCHAINES ÉTAPES — PAR PRIORITÉ

### IMMÉDIAT : Commits GitHub
```
analysis/kickoff_detector.py
  "Kickoff detector: fix coordinate-scale bugs and add profiling instrumentation"

analysis/kickoff_group_candidate_dataset.py
  "Add kickoff group candidate dataset exporter with temporal features"
```

### COURT TERME : Améliorer le proxy couleur

Feuille de route validée (conversation) :

**Étape 1** — Remplacer _extract_player_proxies par version sans dépendance pelouse
- Détection par contraste local (pas de masque couleur pelouse)
- Ou : utiliser le masque mais ne pas filtrer sur couleur pelouse pour le crop
- Extraire H, S, V du torse pour chaque blob

**Étape 2** — player_color_profiler.py (nouveau module)
- Visualiser distribution des couleurs sur quelques matchs
- Vérifier que deux clusters existent réellement
- Choisir espace colorimétrique : HSV ou Lab

**Étape 3** — KMeans k=2 sur fenêtre glissante 30s
- Deux équipes automatiquement sans règle codée en dur
- Robuste : rouge/bleu, jaune/noir, blanc/vert, etc.
- Cas difficile : blanc/noir → séparer par luminosité V

### MOYEN TERME : Dataset + Modélisation

Objectif : 8-10 matchs exploitables (status=ok)
→ Actuellement 4 matchs ok, 24 groupes, 5 vrais KO

Après 8-10 matchs :
- DecisionTree(max_depth=3) pour règles lisibles
- Questions à répondre :
  - Le vrai KO est-il toujours dans les 4 premiers groupes ?
  - delta_dur_prev > 4 discrimine-t-il réellement ?
  - Existe-t-il plusieurs familles de kickoff ?

### Dataset B (futur) — Recall
Ligne par fenêtre temporelle.
Question : le groupe existe-t-il dans cette fenêtre ?
Séparé du Dataset A (ranking).

---

## DÉCISIONS ARCHITECTURALES FIGÉES

1. Ne plus modifier kickoff_detector.py (hors corrections de bugs prouvés)
2. Le proxy couleur est dans notebook_kickoff_only.py, pas dans kickoff_detector.py
3. La solution proxy = KMeans, pas des seuils RGB supplémentaires
4. Pour matchs difficiles (nuit, hivernal) → utiliser notebook_kickoff_from_cache.py
   avec le vrai pipeline YOLO qui a son propre tracking couleur
5. match_status = "ok" | "recall_fail" | "recall_fail_proxy" | "hors_fenetre"

---

## FICHIERS DISPONIBLES DANS OUTPUTS

- kickoff_detector.py (1541 lignes)
- kickoff_group_candidate_dataset.py (43 colonnes)
- kickoff_profiler.py
- notebook_kickoff_only.py (avec PROXY DIAG + frames annotées)
- notebook_kickoff_from_cache.py
- kickoff_groups_combined.csv (dataset actuel)

---

## NOTES TECHNIQUES

- KO andrimont_full = 307s, cache existant sur Kaggle `andrimont-cache`
- notebook_kickoff_only.py : MAX_MINUTES=30, SAMPLE_FPS=2.0
- Fix boucle infinie logging : _print appelle _log.info() seulement
- Fix lecture log : fsync + lecture fichier après find_kickoff_offset()
- match_status "ok" uniquement pour matchs de jour avec proxy fonctionnel
- Transcripts session : /mnt/transcripts/2026-07-07-08-43-24-scoutia-kickoff-dataset-proxy-diag.txt