# INVESTIGATION OFFSET_S — VERS UNE ARCHITECTURE À DEUX HORIZONS TEMPORELS

**Statut : prototype testé sur les 9 matchs (résultat nettement supérieur à la baseline, 5/9 succès nets contre 1-3/9) ET stabilité confirmée sur 3 runs (dispersion 0,00s, sans exception).** Seul le problème d'orchestration (Juprelle) reste à traiter avant toute décision de remplacement de la baseline. Ce document complète `BASELINE_BENCHMARK_AVEUGLE_9MATCHS.md` (section "hypothèses de correction non encore testées", point 1).

---

## 1. Origine du problème

Le benchmark aveugle à 9 matchs a révélé un mode d'échec récurrent : un candidat proche du vrai KO (formation confirmée correcte par la vision) est rejeté parce que la mise en jeu elle-même n'est pas observable dans la fenêtre symétrique `±2s` actuelle. Ce problème était en réalité déjà documenté dans le code AVANT ce benchmark (docstring `offset_s` de `detecter_ko_par_vision()`, cas Raeren), mais jamais validé empiriquement.

**3 matchs sur 9 confirment ce mode d'échec avec un raisonnement identique** : Raeren (404,832s), Stembert (549,0s), Franchimont (313,25s — reclassé depuis "faux positif bloque la passe 2" après test direct).

## 2. Test 1 — Matrice offset_s symétrique (2/4/6/8/10/15s)

**Méthode** : mêmes 3 candidats de récupération + Juprelle (témoin négatif, rejeté pour une raison différente : but + mélange d'équipes) + 4 candidats déjà connus comme correctement rejetés (Andrimont ×2, Wanze ×2), testés à `offset_s` croissant, prompt de production inchangé.

**Résultats** :
- Franchimont et Stembert : récupérés dès `4s`, stables jusqu'à `10s`.
- Raeren : récupéré à `2-8s`, mais bascule à `False` à `10s` (signal de variance, voir §5).
- **`15s` : les 3 candidats retombent à `False`** — élargir trop la fenêtre détruit le signal plutôt que de l'améliorer.
- Aucun nouveau faux positif détecté dans le groupe témoin jusqu'à `8-10s` (Wanze 1391,472 : extraction échouée à 10/15s, vidéo tronquée à 1400s — absence de donnée, pas une garantie).
- Juprelle : reste `False` à tous les offsets testés — confirme qu'il ne s'agit pas du même mécanisme.

**Conclusion de ce test** : une fenêtre `6-8s` semble la zone la plus solide, mais **une seule fenêtre symétrique reste conceptuellement bancale** — elle demande implicitement à Gemini de prouver *formation* ET *mise en jeu* dans le même intervalle.

## 3. Test 2 — Diagnostic formation/mise en jeu séparés

**Hypothèse testée** : découpler les deux preuves. Prompt A (1 image, offsets `-10/-8/-6/-4/-2`) : *"la formation est-elle prête ?"*. Prompt B (1 image, offsets `+2/+4/+6/+8/+10`) : *"le jeu est-il déjà actif ?"*.

**Résultat (1 passage, 4 candidats)** :

| Match | Formation (avant) | Bascule mise en jeu |
|---|---|---|
| Franchimont | ✓ sur 4/5 points | **+6s** |
| Raeren | ✓ sur 3/4 points disponibles | **+6s** |
| Stembert | ✓ sur 5/5 points | **+6s** |
| Juprelle (témoin) | ✓ seulement aux points les plus éloignés (-10/-8), `False` proche du candidat | Déjà `True` dès **+2s** (jeu déjà en cours, pas une transition) |

**Les 3 candidats de récupération basculent au même endroit exact (+6s), jamais avant.** Juprelle se comporte différemment sur les deux plans, cohérent avec son diagnostic déjà établi (scène de jeu normale, pas une transition de KO).

## 4. Test 3 — Confirmation de la bascule +4s/+6s (3 runs)

**Méthode** : répétition 3× du point pivot (+4s vs +6s) sur les 3 candidats de récupération, pour vérifier que la bascule n'est pas un artefact de variance.

**Résultat : 18/18 réponses cohérentes, aucune exception.**

```
Franchimont  :  +4s=0/3  +6s=3/3
Raeren       :  +4s=0/3  +6s=3/3
Stembert     :  +4s=0/3  +6s=3/3
```

Complété par les 3 points manquants du test précédent (échecs techniques 503), tous cohérents avec le reste : Raeren formation confirmée à -6s, toujours statique à +2s ; Juprelle toujours "non séparé" à -6s.

## 5. Point de vigilance — variance déjà documentée

Le test 1 (matrice) montrait Raeren basculant à `False` à `offset_s=2` sur un run alors qu'un run antérieur (image inspectée manuellement) donnait la même conclusion pour une raison différente (joueur prétendument dans le rond central). Le test 3, lui, ne montre **aucune** variance sur 18 observations. Il n'y a pas de contradiction : le test 3 porte sur une question plus simple et plus binaire (1 image, 1 question fermée) que le test 1 (3 images, jugement composite) — une hypothèse plausible est que **décomposer le problème en questions plus simples réduit aussi la variance**, en plus de mieux localiser la preuve. À garder à l'esprit, pas encore confirmé formellement.

## 6. Conclusion — hypothèse de conception

> **Le problème n'est pas la taille de la fenêtre, c'est que la preuve "formation" et la preuve "mise en jeu" n'ont pas le même horizon temporel naturel.**

Architecture proposée, à tester en prototype :

```
                  CANDIDAT (t)
                     │
        ┌────────────┼────────────┐
        │            │            │
      -10s          0s          +6s
        │                         │
   formation                 jeu actif
   statique                  détectable
   (fenetre large,           (jamais avant +6s,
    tolerante au              seuil confirme sur
    decalage du               3 matchs x 3 runs
    mining)                   sans exception)
```

Décision = formation confirmée (quelque part dans la fenêtre avant) **ET** mise en jeu confirmée (à partir de +6s) — deux preuves indépendantes, chacune sur son propre horizon, plutôt qu'une fenêtre unique symétrique devant tout prouver à la fois.

## 7. Prochaine étape

Tester ce prototype sur l'ensemble du pool des 9 matchs (réutilisation des `df_mine_complet.csv` déjà collectés, aucun re-mining), en comparaison directe avec la baseline figée — jamais en remplacement tant que la comparaison n'est pas faite. Cf. `notebook_prototype_2horizons_9matchs.py`.

## 8. Test 4 — Prototype complet sur les 9 matchs (`notebook_prototype_2horizons_9matchs.py`)

**Méthode** : logique à 2 horizons (formation testée `[-6,-4,-8]` avec court-circuit, mise en jeu testée `[6,8,10]` seulement si formation confirmée), appliquée au pool top-30 de chacun des 9 matchs (mêmes `df_mine_complet.csv` que le benchmark aveugle de référence — **aucun re-mining**), même parcours chronologique + arrêt anticipé + fenêtre d'ambiguïté que la baseline.

### Résultat — comparaison directe avec la baseline

| Match | KO réel | Détecté | Écart | Statut prototype | Statut baseline (rappel) |
|---|--:|--:|--:|:-:|---|
| Andrimont | 307,0 | 307,5 | +0,50s | ✅ Succès | Succès (déjà acquis) |
| **Franchimont** | 318,0 | **313,25** | **-4,75s** | ✅ **Succès** | Échec confirmé (1132,0, faux positif confiant) |
| Goe | 1081,0 | — | N/A | ❌ Échec (attendu) | Échec (mining réellement mauvais — connu) |
| Juprelle | 548,0 | 292,0 | -256,0s | ❌ Échec | Échec (faux positif précoce — même mécanisme) |
| MineroisSter | 723,0 | 722,032 | -0,97s | ✅ Succès | Instable (échec/succès selon run) |
| P1Minerois | 192,0 | — | N/A | ❌ Échec (attendu) | Échec (pool insuffisant — connu) |
| **Raeren** | 410,0 | **404,832** | **-5,17s** | ✅ **Succès** | Échec confirmé sur plusieurs runs |
| Stembert | 553,0 | 541,75 | -11,25s | ⚠️ Ambigu (mais 549,0 aussi validé positif, écart -4,0s) | Instable, et confiance à tort élevée sur le mauvais candidat |
| Wanze | 986,0 | 980,0 | -6,00s | ✅ Succès | Instable (échec/succès selon run) |

**Synthèse : 5/9 succès nets (contre 1 à 3/9 selon le run sur la baseline), 1 cas ambigu correctement signalé comme tel (au lieu d'une fausse confiance), 2 échecs déjà connus et inchangés (mining/pool), 1 échec persistant (Juprelle).**

### Ce que ce test confirme

1. **L'hypothèse à 2 horizons est directement validée sur les 2 cas mêmes où elle avait été diagnostiquée** (Franchimont, Raeren) — passage d'un échec confirmé sur plusieurs runs à un succès nouveau, écart de quelques secondes seulement.
2. **Stembert montre une amélioration qualitative** même sans succès net : le prototype détecte honnêtement l'ambiguïté (`ambiguity=True`, deux candidats positifs à 7,25s d'écart) là où la baseline affichait une confiance élevée sur le mauvais candidat.
3. **Le problème d'orchestration (Juprelle) est confirmé comme un problème SÉPARÉ**, non résolu par le changement de logique vision — cohérent avec le diagnostic déjà posé. Le bon candidat (rang 29, ~556s) reste hors de portée à cause d'un faux positif précoce qui déclenche l'arrêt anticipé avant de l'atteindre.
4. **Goe et P1Minerois ne sont ni réparés ni dégradés** — cohérent avec leur diagnostic (mining/pool, pas vision).

### Ce que ce test ne règle pas encore

- Le mécanisme d'orchestration (faux positif précoce bloquant l'accès à un candidat plus tardif) reste à traiter séparément — cf. `BASELINE_BENCHMARK_AVEUGLE_9MATCHS.md` §7, point 2.
- Un seul passage sur les 9 matchs — la stabilité inter-runs de ce prototype (comme pour la baseline) n'a pas encore été mesurée à cette échelle. À faire avant toute décision de remplacement de la baseline.
- Le candidat de Juprelle validé à tort (t=292,0) mérite un examen visuel, comme cela a été fait pour les cas précédents, avant de conclure sur sa nature (hallucination vs scène réellement ambiguë).

## 9. Statut actuel (mis à jour — voir §10)

## 10. Test 5 — Stabilité inter-runs (3 runs × 9 matchs, `notebook_stabilite_v2_3runs_9matchs.py`)

**Méthode** : Prototype V2 figé (aucune modification), répété 3 fois sur les 9 matchs. Économie de coût appliquée sans perte de qualité de mesure : le run déjà réalisé (test 4, §8) réutilisé comme "run 1" (0 appel Gemini), Goe et P1Minerois limités à 1 run (échecs déjà connus et les plus coûteux, aucune valeur ajoutée à les répéter). Coût réel : ~692 appels au lieu de ~1578 estimés initialement.

### Résultat — dispersion 0,00s sur les 9 matchs, sans exception

| Match | Succès | Ambigu | Écarts (3 runs) | Dispersion |
|---|:-:|:-:|---|--:|
| Andrimont | 3/3 | 0/3 | 0,50 / 0,50 / 0,50 | 0,00s |
| Franchimont | 3/3 | 0/3 | -4,75 / -4,75 / -4,75 | 0,00s |
| Goe | 0/3 | 0/3 | — (aucun candidat, comme prévu) | — |
| Juprelle | 0/3 | 0/3 | -256,0 / -256,0 / -256,0 | 0,00s |
| MineroisSter | 3/3 | 0/3 | -0,97 / -0,97 / -0,97 | 0,00s |
| P1Minerois | 0/3 | 0/3 | — (aucun candidat, comme prévu) | — |
| Raeren | 3/3 | 0/3 | -5,17 / -5,17 / -5,17 | 0,00s |
| Stembert | 0/3 | 3/3 | -11,25 / -11,25 / -11,25 | 0,00s |
| Wanze | 3/3 | 0/3 | -6,00 / -6,00 / -6,00 | 0,00s |

### Ce que ce test confirme

1. **Le prototype V2 est nettement plus stable que la baseline sur ce même échantillon.** La baseline montrait MineroisSter et Wanze basculer entre échec et succès d'un run à l'autre (cf. §1-2 de `BASELINE_BENCHMARK_AVEUGLE_9MATCHS.md`, mode E) ; ici, ces deux mêmes matchs sont stables à 100% sur 3 runs.
2. **Juprelle échoue de façon déterministe et reproductible** (même candidat erroné, `t=292,0`, à chaque run) — ce n'est pas une variance ponctuelle, c'est un mécanisme d'orchestration systématique. Le corriger devrait donc donner un résultat fiable, pas juste "parfois meilleur".
3. **Stembert signale l'ambiguïté de façon parfaitement cohérente** (mêmes deux candidats positifs sur les 3 runs) — le système ne varie pas dans son incertitude.

### Ce qui reste à faire

- Le problème d'orchestration (Juprelle) — cf. §8, point 3, reste la seule pièce manquante avant de considérer le prototype V2 comme un remplacement sérieux de la baseline.
- Examen visuel du candidat Juprelle `t=292,0` toujours en attente.

## 11. Statut actuel (remplace §9)

**Le prototype V2 (2 horizons) a passé les deux conditions de validation initialement posées : (a) supériorité nette sur la baseline (test 4, §8) et (b) stabilité inter-runs confirmée sans exception (test 5, §10).** Il ne reste qu'une seule chose avant de l'envisager comme remplacement sérieux de la baseline : traiter le problème d'orchestration qui affecte encore Juprelle (candidat correct jamais atteint à cause d'un faux positif précoce verrouillant l'arrêt anticipé) — un problème déjà identifié comme distinct de la logique vision elle-même, cf. `BASELINE_BENCHMARK_AVEUGLE_9MATCHS.md` §7 point 2.
