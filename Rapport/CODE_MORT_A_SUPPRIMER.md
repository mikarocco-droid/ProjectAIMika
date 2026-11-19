# Code mort à supprimer — ScoutIA

Liste des fichiers/fonctions confirmés **non utilisés nulle part dans le
projet** (aucune importation, aucune référence), découverts au fil des
audits. Vérifié systématiquement avant d'ajouter un élément ici — ne
contient que ce qui est confirmé mort, pas de suppositions.

**Ne pas supprimer avant la fin du balayage complet** — cette liste sera
complétée au fur et à mesure.

---

## Fichiers entiers, jamais importés nulle part

- [ ] **`scout.py`** (racine du projet)
  Contient plusieurs appels `Claude` (`get_summary`, `analyze_frames`,
  `refine_highlight`, `refine_all_highlights`, `cut_highlights`,
  `create_highlight_reel`) — semble être une **ancienne version** du
  pipeline d'analyse, avant `pipeline.py`/`main.py`. Utilisé uniquement
  par `celery_worker.py` (lui-même mort, voir ci-dessous) — aucune autre
  référence trouvée dans tout le projet.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`celery_worker.py`** (racine du projet)
  Définit une tâche `Celery` alternative à `tasks.py` (celle réellement
  utilisée par `app.py`). Importe `scout` (ci-dessus). **Jamais importé
  ni référencé nulle part** dans le reste du projet — `app.py` utilise
  exclusivement `tasks.py` pour ses tâches `Celery`.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`runpod_client.py`** (racine du projet)
  Aucune référence trouvée ailleurs dans le projet.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`runpod_worker.py`** (racine du projet)
  Aucune référence trouvée ailleurs dans le projet.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`webhook_handler.py`** (racine du projet)
  Gère un webhook `RunPod` (`handle_runpod_webhook`) — jamais câblé dans
  `app.py` (le décorateur `@app.route` n'apparaît qu'en commentaire dans
  ce fichier, jamais réellement enregistré). `RunPod` lui-même déjà
  confirmé mort ci-dessus. **Le webhook `Stripe`, lui, est bien actif**
  — géré séparément et directement dans `app.py`
  (`@app.route("/webhook/stripe")`), donc les paiements ne sont pas
  affectés par la suppression de ce fichier.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`detect_ko_vision.py`** (racine du projet)
  Ancienne approche de détection `KO1` par `RandomForest` (`ko_features.py`)
  — antérieure à l'architecture `Gemini` actuelle
  (`kickoff_gemini_cascade.py`, celle utilisée et améliorée ce soir).
  Son en-tête indique `# analysis/detect_ko_vision.py`, suggérant qu'il
  a été déplacé/remplacé sans être supprimé. Aucune référence ailleurs.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`events.py`** (racine du projet — doublon)
  En-tête indique `# analysis/events.py` — c'est la **vraie** version
  (`analysis/events.py`) qui est importée par `main.py` et `pipeline.py`
  (`from analysis.events import ...`). Cette copie à la racine n'est
  jamais importée par rien.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`highlight_scorer.py`** (racine du projet — doublon)
  En-tête indique `# ai/highlight_scorer.py` — c'est la **vraie** version
  (`ai/highlight_scorer.py`) qui est importée par `pipeline.py`
  (`from ai.highlight_scorer import score_all_highlights`). Cette copie
  à la racine n'est jamais importée par rien.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`match_boundaries_v2.py`** (racine du projet — doublon)
  La vraie version utilisée est `analysis/match_boundaries_v2.py`
  (celle sur laquelle on a travaillé toute la soirée — pas adaptatif
  `KO2`, etc.). Cette copie à la racine n'est jamais importée par rien.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`detect_teams_preview.py`** (racine du projet — doublon)
  La vraie version utilisée est `analysis/detect_teams_preview.py`
  (celle utilisée par `app.py` pour la détection des couleurs). Cette
  copie à la racine n'est jamais importée par rien.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`goal_posthoc.py`** (racine du projet)
  Aucune instruction `import goal_posthoc`/`from goal_posthoc import`
  trouvée nulle part dans le projet.
  *(Confirmé mort le 11/09/2026)*

- [ ] **`gemini_validator.py`** (racine du projet — doublon)
  La vraie version utilisée est `ai/gemini_validator.py` (celle
  instrumentée ce soir pour le compteur de coûts). Cette copie à la
  racine n'est jamais importée par rien.
  *(Confirmé mort le 11/09/2026)*

---

## Scripts autonomes — confirmés à supprimer par l'utilisateur (11/09/2026)

*(scripts de test ponctuels, jamais importés nulle part — l'utilisateur
confirme qu'il ne s'en sert plus, ajoutent du volume au projet pour rien)*

- [ ] **`analyze_log.py`** — outil `CLI` d'analyse de `pipeline.log`
- [ ] **`calibrate_xg.py`** — script de calibration du modèle `xG`
- [ ] **`player_confidence.py`** — script de test autonome
- [ ] **`validate_profiler.py`** — script de validation autonome
- [ ] **`kaggle_test.py`** — script de test manuel du pipeline sur `Kaggle`
- [ ] **`debug_preview.py`** — scratchpad de debug (à coller dans `Colab`)

---

## Fonctions/code mort à l'intérieur de fichiers actifs

*(rien identifié pour l'instant à ce niveau de granularité — sera
complété si le balayage en trouve)*

---

## Notes

- **`Set_Admin.py` n'est PAS dans cette liste** — il reste nécessaire
  (seul moyen de se donner les droits admin depuis que `/make-admin` a
  été retiré) — à garder, contrairement aux scripts de test ci-dessus.
- Cette liste ne couvre que ce qui a été **activement vérifié** (recherche
  de toutes les références/imports dans le projet) — l'absence d'un
  fichier ici ne garantit pas qu'il est utilisé, juste qu'il n'a pas
  encore été vérifié.
- Avant de supprimer, un dernier `grep` de vérification est recommandé
  au cas où quelque chose aurait changé depuis la découverte.

