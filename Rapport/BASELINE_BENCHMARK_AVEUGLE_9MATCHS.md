# BASELINE DE RÉFÉRENCE — BENCHMARK AVEUGLE 9 MATCHS

**BASELINE V1 — FIGÉE.** À partir de cette version, aucune modification (code, prompt, ou paramètre) ne doit être jugée « meilleure » sans comparaison explicite, chiffrée, contre ce document.

**Statut : référence figée.** Aucune modification de code, de prompt, ou de paramètre n'a été faite entre la collecte de ces résultats et la rédaction de ce document. C'est la première mesure de bout en bout de la chaîne complète en conditions de production, jamais faite avant cette phase.

---

## 1. Protocole

- **Chaîne testée** : vidéo source complète → `process_video()` (YOLO/DeepSort) → `mine_candidates(true_kickoff_t=None)` (aveugle, sans injection) → RF (entraîné en leave-one-match-out sur `features_ko_v2.csv`) → pool top-30 par proba RF décroissante → `detecter_ko_par_vision_adaptatif()` (baseline KO Vision V1 figée : `gemini-3.1-pro-preview`, `temperature=0.0`, `max_appels=5`, `offset_s=2.0`, préfiltre désactivé) → `{timestamp_final, confidence, ambiguity, candidats_evalues}`.
- **Aveugle réel** : le timestamp du vrai KO n'apparaît nulle part dans le code exécuté avant la production du résultat. La comparaison avec la vérité terrain se fait dans un second temps, sur fichier séparé.
- **9 matchs** : Andrimont, Franchimont, Goe, Juprelle, MineroisSter, P1Minerois, Raeren, Stembert, Wanze — les 9 matchs de référence de tout le projet.
- Pour 5 matchs (MineroisSter, P1Minerois, Raeren, Stembert, Wanze), une **seconde exécution** de l'étape vision a été faite après une perte de session (récupération sans refaire le mining, à partir des `df_mine_complet.csv` déjà calculés) — ce qui a involontairement produit une mesure de stabilité inter-runs en plus du résultat lui-même.

## 2. Résultat central — ne pas résumer par un seul chiffre

**Ne pas présenter "3/9" comme le taux de réussite de la baseline.** Les runs répétés montrent que ce chiffre dépend du tirage Gemini, pas uniquement de la configuration :

> **1 succès au premier passage (Andrimont). Deux matchs supplémentaires (MineroisSter et Wanze) ont réussi lors d'un second passage, sans aucune modification de la baseline. Au total, 3 succès distincts ont donc été observés sur les 9 matchs au cours des runs réalisés, mais ce chiffre ne constitue pas un taux de réussite de la baseline. La stabilité du pipeline de bout en bout reste à mesurer correctement par plusieurs runs indépendants par match.**

## 3. Tableau détaillé — les 9 matchs

| Match | KO réel | Rang meilleur candidat | Dans le pool testé (≤30) | Run 1 | Run 2 (si dispo) | Mode d'échec / statut |
|---|--:|--:|:-:|---|---|---|
| **Andrimont** | 307,0 | 11/69 | ✅ | **307,5 (succès, écart 0,5s)** | — | ✅ Extension adaptative a sauvé le match |
| **Franchimont** | 318,0 | 31/68 | ❌ (juste hors) | 1132,0 (échec) | 1132,0 (échec, confirmé identique) | Faux positif confiant → passe 2 jamais déclenchée |
| **Goe** | 1081,0 | 48/60 | ❌ | 890,48 (échec, repli) | — | Mining réellement mauvais (même le pool complet à 60 ne contient rien d'exploitable) |
| **Juprelle** | 548,0 | 29/59 | ✅ | 224,25 (échec, repli) | 282,5 (échec, faux positif précoce, écart ≈ −265,5s avant le KO) | Faux positif précoce bloque l'accès au bon candidat, *à l'intérieur même* du pool étendu |
| **MineroisSter** | 723,0 | 3/62 | ✅ (top-10) | 704,0 (échec, écart 19s, mauvais candidat) | **722,032 (succès, écart 0,97s, 3/3 votes)** | Instable d'un run à l'autre — succès et échec observés sur config identique |
| **P1Minerois** | 192,0 | 31/71 | ❌ (juste hors) | 42,88 (échec, repli) | 42,88 (échec, confirmé identique) | Pool insuffisant |
| **Raeren** | 410,0 | 10/65 | ✅ (top-10) | 352,0 (échec, repli) | 352,0 (échec, confirmé identique) | Bon candidat (404,83s) évalué et rejeté — probable lien avec `offset_s` (cf. §4) |
| **Stembert** | 553,0 | 5/66 | ✅ (top-10) | 541,75 (échec, écart 11s, mauvais candidat) | 1264,0 (échec, mauvais candidat différent ; bon candidat 549,0 rejeté sur "mise en jeu non captée") | Confirme le lien avec `offset_s` (§4) + vote serré (3 vs 2) sur le mauvais candidat retenu |
| **Wanze** | 986,0 | 18/60 | ✅ | 660,176 (échec, repli) | **980,0 (succès, écart 6,0s, 3/4 votes)** | Instable d'un run à l'autre — succès et échec observés sur config identique |

## 4. Modes d'échec identifiés — classification

### A. Pool insuffisant
**P1Minerois** (rang 31/71). Le bon candidat n'atteint jamais la vision, quel que soit le comportement de cette dernière. Pas un problème de vision ni de vote.

### B. Mining réellement en échec
**Goe** (rang 48/60). Même le pool complet ne contient rien d'exploitable à moins de 15s du vrai KO. Confirmé indépendamment par calcul de rang seul (sans vision).

### C. Faux positif confiant bloque l'accès au bon candidat
**Franchimont, Juprelle.** Le bon candidat est pourtant dans le pool testé (rang 18 et 29 respectivement), mais un candidat antérieur, chronologiquement plus tôt, est validé par la vision avec une confiance élevée — déclenchant l'arrêt anticipé avant que le bon candidat n'ait été évalué. Pour Franchimont, ceci empêche même le déclenchement de la passe 2 (le top-10 seul "réussit" du point de vue du code). Pour Juprelle, le blocage se produit à l'intérieur du pool déjà étendu à 30.

### D. Rejet vision sur candidat par ailleurs correct — probable lien avec `offset_s`
**Raeren, Stembert.** Dans les deux cas, le candidat proche du vrai KO est bien évalué, la formation initiale (`avant_compatible`) est reconnue comme correcte par au moins une partie des votes, mais la mise en jeu elle-même (`mise_en_jeu_visible`/`transition_apres`) n'est pas clairement observable dans la fenêtre ±2s. Ceci correspond exactement à une limite **déjà documentée dans le code AVANT ce benchmark** (docstring `offset_s` de `detecter_ko_par_vision`, cas Raeren 404,8s vs vrai KO ~409s) — pas une découverte de ce benchmark, mais sa première confirmation empirique en conditions de production.

### E. Instabilité bout-en-bout malgré `temperature=0.0`
**MineroisSter, Wanze.** Sur config strictement identique (même pool de candidats, mêmes paramètres), le résultat final bascule entre échec net et succès net d'un run à l'autre. La variance déjà documentée au niveau d'un seul vote (ex. Franchimont, 4 votes non-unanimes) se propage donc jusqu'au résultat final du pipeline — pas seulement au détail interne des votes.

### F. Extension adaptative qui fonctionne comme prévu
**Andrimont.** Le seul cas où le mécanisme conçu pour ça (`detecter_ko_par_vision_adaptatif`, passe 1 échoue totalement → extension à 30) a effectivement sauvé un match qui aurait échoué en top-10 seul.

## 5. Ce que ce benchmark ne permet PAS encore de conclure

- La proportion réelle de matchs "récupérables sur un second run" parmi les 6 en échec (on n'a testé qu'un 2ᵉ passage, pas 3 ou plus).
- Si l'hypothèse `offset_s` (mode D) se généraliserait à d'autres matchs non testés sous cet angle.
- Si le mécanisme de blocage par faux positif précoce (mode C) est plus fréquent que les 2 cas observés, sur un échantillon plus large.

---

## 6. Prochaines étapes proposées

### Phase 1 — ce document (terminée)
Aucune modification. Synthèse, classification des modes d'échec, conservation de tous les résultats bruts (`*_resultat_aveugle.json`, `*_df_mine_complet.csv`).

### Phase 2 — mesurer avant de corriger
- Plusieurs runs complets par match (pas juste 1 ou 2), même pool, même configuration figée.
- Mesurer un **taux de succès par match** (pas juste succès/échec binaire) et un taux global.
- Mesurer la **variance du timestamp final** sur les matchs qui réussissent (à quel point l'écart au vrai KO varie-t-il d'un run à l'autre ?).
- Distinguer explicitement, pour chaque échec, laquelle des 4 causes (mining / orchestration / `offset_s` / vision) est responsable — pas un score global.

### Phase 3 — seulement ensuite, tester une correction
Toute modification (notamment `offset_s=4.0`, ou une révision de la condition de déclenchement de la passe 2 adaptative) doit être testée **contre cette baseline figée**, sans mélanger la mesure de généralisation et la mesure d'effet d'une correction — exactement la discipline suivie jusqu'ici pour chaque itération du prompt.

## 7. Hypothèses de correction non encore testées

*(Aucune de ces pistes n'a été implémentée ni testée à ce stade — listées ici pour mémoire, à valider en Phase 3.)*

1. **`offset_s=4.0`** (au lieu de 2.0) — déjà suggéré dans le code avant ce benchmark, maintenant appuyé par 2 cas empiriques (Raeren, Stembert) au lieu d'1 seul mentionné historiquement.
2. **Revoir la condition de déclenchement de la passe 2 adaptative** — actuellement elle ne se déclenche que si la passe 1 échoue *totalement* (`confidence='faible'` ET `ambiguity=True`). Les cas Franchimont/Juprelle suggèrent qu'un critère supplémentaire serait nécessaire pour détecter "passe 1 a trouvé quelque chose, mais ça pourrait être le mauvais candidat" — reste à définir ce critère sans simplement déplacer le problème (toujours étendre à 30 ferait perdre l'économie de coût qui justifie le mode adaptatif).
3. **Mesurer si l'augmentation de `max_appels` au-delà de 5, ou un changement de méthode d'agrégation du vote, réduirait l'instabilité bout-en-bout** (mode E) — non testé, seulement observé.
4. **Un protocole de vote à l'échelle du pipeline entier** (pas seulement par candidat) — faire tourner le pipeline complet plusieurs fois et prendre le résultat majoritaire — coûteux, à évaluer seulement si la Phase 2 confirme que l'instabilité bout-en-bout est fréquente.
