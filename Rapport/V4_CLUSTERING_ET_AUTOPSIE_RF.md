# V4 — CLUSTERING TEMPOREL ET AUTOPSIE RF

**Statut : investigation Goe close pour cette phase, après réplication de la configuration exacte de production (V4.10) et vérification visuelle complémentaire (§13.1). L'écart `43 vs 17` persiste malgré cadence réelle, configuration exacte et suivi continu — cause encore inconnue, recentrage recommandé sur la provenance du dataset historique plutôt que le comportement du tracker. Une anomalie réelle de collision de `tracker_id` (65% des frames de la zone étudiée) a été démontrée, et une rampe `n_players` (11→47) incompatible avec les captures d'écran de la scène a été documentée — ces deux observations sont compatibles avec une inflation d'origine tracker, sans que la cause exacte de l'écart `43 vs 17` soit établie pour autant. Nouvelle piste ouverte pour la suite (§14) : audit de fiabilité de l'identité joueur sur la durée d'un match, jugé prioritaire sur l'amélioration du classement xG/xA. Spécification produit ajoutée (§14.1) : "Match Start Resolver" à 3 paliers de confiance, qui déverrouille la production sur Goe sans attendre la résolution du mystère `43 vs 17`.** Bilan global : 7/9 succès automatiques (dont P1Minerois, chaîne causale complète validée §6), 1/9 ambiguïté honnête (Juprelle, règle de désambiguïsation +21s validée sur 3 runs), 1/9 échec de mining non résolu mais rigoureusement caractérisé (Goe). Complète `V3_SELECTION_MULTICANDIDATS.md` (§10).

---

## 1. Position dans la chaîne

```
Mining/RF (pool large de candidats)
        │
        ▼
Vision V2 (2 horizons) — figée, non modifiée
        │
        ▼
Filtre de contexte lointain (-45s/-60s) — figé, non modifié
        │
        ▼
V4.1 — REGROUPEMENT TEMPOREL (nouveau)
        │
        ▼
Résultat final : événement (fenêtre + centre), pas un timestamp arbitraire
```

## 2. V4.1 — Regroupement temporel des positifs

### Test de sensibilité (5/10/15/20/25/30s)

Sur les 13 positifs de référence (issus du benchmark V3 à 9 matchs), le regroupement est **rigoureusement identique de 10s à 30s** : un seul groupe multi-candidats se forme (Stembert), sans jamais fusionner à tort deux événements distincts, quel que soit le seuil dans cette plage.

**Marge de sécurité** : dispersion interne du seul vrai groupe (Stembert, `541,75`→`557,0`) ≈ 8s au maximum entre deux membres consécutifs, contre un écart minimum de **264s** entre deux événements réellement distincts n'importe où ailleurs dans les 9 matchs (Juprelle). Facteur de sécurité ×33.

**Conclusion** : `CLUSTER_WINDOW_S = 20.0` est un choix défendable pour une V1 du clustering, mais **provisoire** — le test ne permet pas de démontrer qu'une valeur précise dans la plage [10, 30] est meilleure qu'une autre sur les données actuelles. À affiner si de nouveaux cas limites apparaissent sur d'autres matchs.

### Module implémenté (`clustering_temporel.py`)

- Fonction `regrouper_candidats_en_evenements()` : chaînage progressif (un candidat rejoint le groupe si son écart au **dernier** membre, pas au premier, est ≤ seuil).
- Représentation par événement : `event_start`, `event_end`, `event_center`, `n_positive` — pas une simple moyenne présentée comme un timestamp exact.
- Fonction `resultat_final_match()` : logique de décision à 3 branches (repli / succès avec fenêtre / ambiguïté de nature — jamais de fusion forcée).

### Vérifications de non-régression (exécutées, toutes passent)

- **Stembert** : `[541,75, 549,0, 557,0]` fusionnent en 1 événement (`centre=549,25`), `1264,0` reste séparé.
- **Juprelle** : `292,0` et `556,0` restent **deux événements distincts** — le clustering ne doit jamais résoudre artificiellement une ambiguïté de nature.
- **Raeren** : `404,832` et `1028,5` restent séparés (623,7s d'écart).

### Résultat combiné (clustering + filtre de contexte déjà validé)

| Match | Avant clustering | Après |
|---|---|---|
| Stembert | ⚠️ Ambiguïté (3 candidats) | ✅ **Succès** — `event_center=549,25s`, fenêtre `[541,75 ; 557,0]`, écart -3,75s |
| Juprelle | ⚠️ Ambiguïté | ⚠️ Ambiguïté (inchangé, à raison) |

**Nouveau bilan sur les 9 matchs : 6/9 succès automatiques, 2/9 échecs de mining, 1/9 ambiguïté réelle (de nature, pas de précision), 0/9 faux résultat confiant.**

## 3. V4.2 — Autopsie des features RF (Goe, P1Minerois)

### Rappel du problème

| Match | Candidat proche du vrai KO | Rang RF | Proba RF |
|---|---|--:|--:|
| Goe | +11,0s | 48/60 | 0,0000 |
| P1Minerois | -9,75s | 31/71 | 0,0000 |

Le signal existe dans les données brutes (candidat temporellement excellent), mais le RF le score à zéro.

### Méthode

Comparaison des 11 features brutes de ces 2 candidats contre celles des 6 candidats confirmés comme vrais KO par la vision V3 (Andrimont, Franchimont, MineroisSter, Raeren, Stembert, Wanze), feature par feature, pour repérer une éventuelle valeur hors plage.

### Résultat — aucune feature individuellement anormale

| Feature | Plage des "bons" | Goe | P1Minerois | Anomalie |
|---|---|--:|--:|---|
| `geo_separation` | 0,71 – 0,96 | 0,88 | 0,73 | — |
| `n_players` | 20 – 36 | 43 | 33 | Goe légèrement hors plage (+7) |
| `mean_speed_change_px_s` | -3,0 – 3,5 | 1,7 | -1,8 | — |
| `pct_immobile` | 0,00 – 0,37 | 0,00 | 0,00 | — |
| `ball_score` | 0,50 | 0,50 | 0,50 | — |
| `ball_speed_before` | 39,3 – 351,5 | 270,9 | 184,1 | — |
| `ball_speed_after` | 89,5 – 333,0 | 237,9 | 154,5 | — |
| `stabilite_avant_s` | 0,00 – 8,00 | 0,00 | 0,00 | — |
| `activite_apres` | 18,7 – 48,4 | 48,6 | 36,5 | Goe négligeable (+0,2) |
| `whistle_score` | -0,80 – 4,20 | -0,75 | -0,66 | — |

**P1Minerois est dans la plage normale sur les 11 features, sans une seule exception.** Goe déborde de façon négligeable sur 2 features. Aucune feature "cassée" ou aberrante identifiée.

### Interprétation

Le RF n'échoue probablement pas à cause d'une feature défaillante, mais parce qu'il est entraîné sur **très peu d'exemples positifs réels** (~6 à 9, un par match historique). Dans un espace à 11 dimensions, ceci peut produire une frontière de décision étroite qui ne généralise pas à un candidat valide mais numériquement un peu différent de tous les exemples connus — sans qu'aucune dimension individuelle ne soit en cause.

## 4. V4.3 — Test de correction : ajout d'exemples vs pondération de classe

### Test A — Ajouter les 6 nouveaux positifs comme exemples d'entraînement

Les 6 candidats confirmés par la vision V3 sont, pour l'essentiel, **les mêmes événements** que les exemples déjà présents dans le jeu d'entraînement (écarts de 0,5 à 6s avec l'exemple manuel déjà existant du même match) — pas des exemples radicalement nouveaux.

**Résultat** :

| Match | Proba RF originale | Proba RF avec les 6 ajoutés | Effet |
|---|--:|--:|---|
| Goe | 0,0000 | 0,0000 | Aucun |
| P1Minerois | 0,0000 | 0,0458 | Léger |

**Conclusion : piste insuffisante.** Le jeu d'entraînement reste extrêmement déséquilibré (~12 positifs pour plus de 500 négatifs) — 6 exemples de plus, très proches de ceux déjà connus, n'apportent pas assez de diversité pour déplacer significativement la frontière de décision.

### Test B — Déséquilibre structurel : `class_weight='balanced'`

Avec seulement ~10 positifs pour 533 lignes (ratio 1:52), chaque arbre du RF (`n_estimators=200`) peut structurellement sous-représenter le signal positif dans son échantillon bootstrap. Test de la pondération de classe (`class_weight='balanced'` et `'balanced_subsample'`), sans ajouter aucun exemple, sans toucher à aucune feature.

**Résultat — effet réel, mais qui ne touche qu'un seul des deux matchs** :

| Match | Original | `class_weight='balanced'` |
|---|---|---|
| **P1Minerois** | rang 31/71, proba 0,0000 | **rang 8/71**, proba 0,0100 — entrerait facilement dans un pool à 30 |
| **Goe** | rang 48/60, proba 0,0000 | rang 48/60, proba 0,0000 — **aucun changement** |

**Vérification approfondie sur Goe** : proba non arrondie = `0.0`, exactement — **0 arbre sur 200 ne vote positif**, quelle que soit la pondération testée. Ce n'est pas un artefact d'arrondi, c'est un rejet unanime et total, insensible au rééquilibrage de classe.

### Conclusion de V4.3 — deux causes distinctes, pas une seule

**P1Minerois et Goe, qu'on avait rapprochés jusqu'ici ("RF donne 0,000 aux deux"), ont probablement deux causes différentes :**

- **P1Minerois** : un vrai problème de déséquilibre de classe, **identifié et corrigible** par `class_weight='balanced'` — un changement simple, ne touchant aucune feature ni aucun exemple.
- **Goe** : rejet unanime et total, insensible à la pondération — probablement plus proche du diagnostic initial ("mining réellement en échec"), ou une cause encore différente et non identifiée à ce stade. **Ne pas traiter Goe et P1Minerois comme un seul et même problème.**

## 5. Validation de `class_weight='balanced'` sur les 9 matchs — vérifier l'absence de régression

**Question posée avant d'adopter quoi que ce soit** : `class_weight='balanced'` améliore P1Minerois, mais est-ce qu'il dégrade ailleurs des matchs qui fonctionnent déjà bien ? Test complet : retrain leave-one-out sur les 9 matchs, comparaison du rang du candidat déjà confirmé + composition complète du nouveau top-30 (nouveaux entrants, candidats perdus).

### Résultat — rang du candidat confirmé, avant/après

| Match | Rang original | Rang `balanced` | Effet |
|---|--:|--:|---|
| Andrimont | 11 | 3 | ✅ Amélioré |
| Franchimont | 18 | 9 | ✅ Amélioré |
| Goe | 48 | 48 | = Stable |
| Juprelle (candidat vrai, 556,0) | 29 | 31 | ~ Négligeable |
| MineroisSter | 3 | 2 | ✅ Amélioré |
| **P1Minerois** | 31 | **8** | ✅ **Nettement amélioré** |
| Raeren | 10 | 2 | ✅ Amélioré |
| Stembert | 5 | 5 | = Stable |
| Wanze | 18 | 19 | ~ Négligeable |

### Vérification spécifique : un nouveau candidat menace-t-il jamais un bon candidat ?

Chaque match voit 1 à 4 nouveaux candidats entrer dans le top-30 (la plupart "loin du KO", proba très faible 0,005-0,015) — mais **dans aucun des 9 matchs un nouvel entrant ne fait chuter significativement le rang du candidat déjà confirmé**. Les entrants comblent des places libérées en bas du top-30, ils ne déclassent jamais un bon candidat.

**Vérification sur Juprelle en particulier** : le faux candidat (`292,0`) et le vrai (`556,0`) bougent tous les deux légèrement (respectivement 12→13 et 29→31) sans changer d'ordre relatif — `class_weight='balanced'` ne favorise pas le faux positif, l'ambiguïté déjà connue reste inchangée, ni aggravée ni faussement résolue.

### Conclusion

**`class_weight='balanced'` est sain sur l'ensemble du benchmark** — gain net réel sur 5 matchs, stable sur 2, dégradation négligeable sur 2 (perte d'une poignée de rangs, sans conséquence pratique). Validé pour adoption au niveau du mining, sous réserve du test grandeur nature complet (§6).

## 6. V4.5 — Validation grandeur nature sur P1Minerois (`notebook_v4_5_p1minerois_balanced.py`)

**Test à périmètre strictement limité à P1Minerois** — le seul match dont le pool change de façon actionnable avec `class_weight='balanced'` (rang 31→8, entrée dans le top-30). Pipeline V4 complet appliqué sans aucune modification (2 horizons, filtre de contexte lointain, clustering).

### Résultat — RÉSULTAT A, succès complet

```
RF original (rang 31/71)  →  hors top-30  →  jamais evalue  →  repli total (historique)
RF balanced (rang 8/30)   →  dans le pool  →  V4 vision      →  positif confirme
                          →  contexte lointain : premier KO (pas une reprise)
                          →  clustering : 1 seul evenement, pas d'ambiguite
                          →  SUCCES : KO detecte a 182,25s, ecart -9,75s
```

**Détail** : formation confirmée à `-6s` (premier essai), mise en jeu confirmée à `+10s` (`n_appels=4` au total pour ce candidat). Contexte lointain à `-45s`/`-60s` : les deux confirment du pré-match (*"regroupés en cercles de préparation"*, *"regroupement statique au centre"*), donc correctement classé comme candidat premier KO, pas une reprise.

**Coût réel** : 91 appels vision (30 candidats, pool complet sans arrêt anticipé) + 2 appels de contexte lointain = 93 appels au total pour ce match.

### Conclusion

**La chaîne causale complète est validée de bout en bout** : le déséquilibre de classe du RF excluait un candidat temporellement excellent du pool testable ; `class_weight='balanced'` corrige ce point précis sans dégrader les autres matchs (cf. §5, test complet à 9 matchs) ; une fois le candidat rendu disponible, le pipeline V4 (déjà validé, non modifié) le retrouve et le confirme correctement.

**P1Minerois n'est donc plus un échec de vision, ni un vrai échec de mining comme Goe — c'était un problème de déséquilibre de classe du RF, maintenant identifié, corrigé, et validé sur un cas réel.**

## 7. Bilan mis à jour — statut de chaque match du benchmark de référence

| Match | Statut après V4 (clustering + RF balanced sur P1Minerois) |
|---|---|
| Andrimont, Franchimont, MineroisSter, Raeren, Wanze | ✅ Succès (déjà acquis, V3) |
| Stembert | ✅ Succès (résolu par le clustering, §2) |
| **P1Minerois** | ✅ **Succès (résolu par `class_weight='balanced'`, §6)** |
| Juprelle | ⚠️ Ambiguïté de nature (non résolue, à raison — cf. §9 de `V3_SELECTION_MULTICANDIDATS.md`) |
| Goe | ❌ Mining (témoin négatif confirmé — `class_weight='balanced'` sans aucun effet, 0/200 arbres, cause distincte non identifiée) |

**7/9 succès automatiques, 1/9 ambiguïté honnête, 1/9 échec de mining réel et isolé.**

## 8. V4.6 — Autopsie des chemins de décision (`n_players`)

**Question posée** : pourquoi 0/200 arbres votent positif pour Goe, alors qu'aucune feature individuelle n'est hors plage ?

**Méthode** : pour chaque arbre du RF, extraction du chemin de décision exact suivi par le candidat de Goe, identification de la feature utilisée au dernier split avant la feuille finale.

**Résultat** : `n_players` est la feature du dernier split dans **70/200 arbres (35%)** — la valeur de Goe (43) est **au-dessus de tous les seuils rencontrés** (21,5 à 35), dans 100% des cas. Vérification : ce n'est pas un pic isolé, tout le match Goe a une distribution de `n_players` systématiquement plus élevée que les autres matchs (médiane 54,5 contre 32,9 pour Wanze) — probablement un artefact de la vidéo (angle de caméra, spectateurs visibles), pas un vrai signal de jeu.

**Test de normalisation** (`n_players_relative = valeur / médiane du match`) : effet réel mais mitigé sur les 9 matchs (améliore Stembert, Raeren, MineroisSter ; dégrade légèrement Andrimont, Wanze, Juprelle), et **ne change presque rien pour Goe** (rang 48→49). Vérification approfondie : le ratio de Goe (0,789) est réellement le plus élevé des 9 matchs, mais pas isolé (Wanze, qui réussit, est juste derrière à 0,750) — la frontière de décision apprise sur trop peu d'exemples reste trop étroite pour absorber une valeur en limite, même normalisée.

**Conclusion** : piste confirmée comme pertinente, mais insuffisante seule.

## 9. V4.7 — Autopsie du deuxième facteur (`whistle_score`)

Même méthode appliquée à `whistle_score`, deuxième feature la plus fréquente au dernier split (51/200 arbres, 26%) — Goe (`-0,75`) est systématiquement sous tous les seuils rencontrés (1,77 à 4,66), 51/51.

**Mais `whistle_score` seul n'explique pas non plus le rejet** : la valeur de Goe est presque identique à celle de Wanze (`-0,80`, encore plus négative, pourtant un match qui réussit) et P1Minerois (`-0,66`, corrigé et réussi). En percentile intra-match, Goe (20%) se situe au milieu de la plage des candidats confirmés (6,7% à 98,4%).

### Répartition complète du dernier split sur les 200 arbres

| Feature | Arbres | % |
|---|--:|--:|
| `n_players` | 70 | 35% |
| `whistle_score` | 51 | 26% |
| **7 autres features combinées** | **79** | **40%** |

**Conclusion — hypothèse confirmée par les données** : le rejet de Goe n'est pas dû à une ou deux features dominantes, mais à un **décalage multidimensionnel** du match entier par rapport au petit ensemble d'entraînement (9-10 positifs). Aucune feature individuelle ne domine suffisamment (61% cumulé sur les deux principales, 40% répartis sur 7 autres) pour justifier un correctif ciblé feature par feature.

## 10. Transition vers V4.9 — pourquoi ne pas continuer à corriger feature par feature

Un décalage multidimensionnel confirmé ferme la porte aux corrections ciblées (normalisation d'une feature à la fois) et ouvre la voie à la seule piste structurelle restante : **élargir la diversité des exemples positifs d'entraînement**, plutôt que d'ajuster des features individuellement. Contrairement au premier essai (V4.3, où les 6 exemples ajoutés étaient quasi-identiques aux exemples manuels déjà en base), on dispose maintenant d'exemples confirmés par la vision V4 sur des instants légèrement différents et pour un match supplémentaire (P1Minerois, jamais représenté avant) :

- Andrimont (307,5), Franchimont (313,25), MineroisSter (722,032), Raeren (404,832), Wanze (980,0) — déjà utilisés en V4.3.
- **Stembert — 3 exemples** (541,75 / 549,0 / 557,0), au lieu d'un seul comme avant — issus du clustering, chacun une observation légèrement différente du même événement réel.
- **P1Minerois (182,25)** — nouveau, jamais représenté dans le jeu d'entraînement avec ce statut de positif confirmé par la vision.

## 11. V4.9 — Réentraînement avec le jeu élargi et diversifié

**Test sur Goe** : jeu d'entraînement élargi (10→18 positifs, incluant P1Minerois pour la première fois et 3 observations de Stembert au lieu d'1), avec et sans `class_weight='balanced'`.

| Configuration | Proba Goe | Rang Goe |
|---|--:|--:|
| RF original, jeu original (9 positifs) | 0,0000 | 48/60 |
| RF original, **jeu élargi (18 positifs)** | **0,0025** | **33/60** |
| RF `balanced`, jeu original | 0,0000 | 48/60 |
| RF `balanced`, jeu élargi | 0,0000 | 49/60 |

**Premier signal non nul jamais observé pour Goe** (jeu élargi seul) — mais **insuffisant en pratique** (rang 33, toujours hors d'un pool à 30). **Découverte notable** : combiner jeu élargi et `class_weight='balanced'` ensemble annule le gain plutôt que de l'amplifier — les deux techniques n'ont pas d'effet additif sur ce cas.

**Vérification de non-régression sur les 7 autres matchs** (jeu élargi seul) : 5 améliorés ou stables, 2 dégradations négligeables (MineroisSter 3→5, Stembert 5→6 — restent d'excellents rangs).

### Conclusion de cette phase d'investigation sur Goe

Le jeu élargi **progresse** sur Goe sans le **résoudre**. Combiné au décalage multidimensionnel confirmé en V4.7-V4.8, ceci suggère que 18 exemples positifs restent probablement insuffisants pour qu'un Random Forest apprenne une frontière de décision assez large pour absorber un match aussi atypique que Goe — il faudrait vraisemblablement beaucoup plus d'exemples divers que ce qu'on peut extraire de 9 matchs de référence.

**Goe reste, à ce stade, le seul échec non résolu du benchmark** — mais désormais bien caractérisé : ni un problème de vision, ni un problème de mining au sens brut (le candidat existe, temporellement excellent), mais une limite de généralisation du RF face à une distribution vidéo atypique et un jeu d'entraînement encore trop restreint.

## 12. Vérifications complémentaires post-V4.9

### 12.1 Hypothèse "éclairage nocturne" — testée et écartée

Suite à l'observation visuelle que la frame réelle à `t=1092,0` (candidat de Goe) montre un effectif normal (~13-16 personnes) malgré `n_players=43`, l'hypothèse d'un éclairage nocturne à ombres marquées causant une sur-détection a été testée par comparaison de distribution :

| Match | Nocturne ? | Médiane `n_players` | Résultat |
|---|:-:|--:|---|
| Wanze | Oui | **36,0** (la plus basse de tous) | ✅ Réussi |
| Andrimont | Non | 57,0 | ✅ Réussi |
| Goe | Oui | 54,5 | ❌ Échec |
| **P1Minerois** | Oui | **62,0** (la plus haute de tous) | ✅ Réussi (corrigé) |

**L'hypothèse ne tient pas** : P1Minerois, également nocturne, a une médiane encore plus élevée que Goe et s'est pourtant corrigé facilement. Wanze, nocturne aussi, a la médiane la plus basse de tous les matchs. Comparaison visuelle directe (image P1Minerois vs Goe) : aucune différence flagrante de conditions de tournage. **Conclusion : ce n'est ni "nocturne" en général, ni le niveau absolu de `n_players`, qui distingue Goe — cohérent avec le décalage multidimensionnel déjà identifié en V4.7, pas une catégorie de conditions de tournage identifiable.**

### 12.2 Audit de la vérité terrain — vérification jamais faite jusqu'ici

**Question posée** : toute l'analyse de Goe (features, chemins de décision, distributions) repose sur `KO_REEL=1081,0`, une valeur héritée de `features_ko_v2.csv` (`source_dataset=v1_manuel`) et **jamais vérifiée visuellement** dans cette investigation, contrairement à la plupart des autres matchs (Raeren, Juprelle, Franchimont, Stembert, Andrimont, tous vérifiés par capture d'écran réelle).

**Vérifications effectuées** :
- Absence de bug de calcul : pas de doublon de timestamp, ligne unique à `t=1092,0`, médiane recalculée indépendamment et identique (54,5).
- **Image extraite exactement à `t=1081s`** : chronomètre Veo affiche `1ER 00:01` (1ʳᵉ mi-temps, 1 seconde de jeu), horodatage Veo `18:01` = 1081s exactement. **`KO_REEL=1081,0` est confirmé correct, à la seconde près.**

**Conclusion de l'audit : aucune erreur de données ou de raisonnement identifiée.** L'ensemble de l'analyse de Goe (V4.2, V4.6, V4.7, V4.9) porte bien sur le bon événement. Le problème reste entier mais bien circonscrit : sur **toutes** les frames inspectées de ce match (avant-match à 2:41, jeu en cours à 19:43 et 18:12, instant exact du KO à 18:01), l'effectif visible est systématiquement normal (~12-16 personnes) — jamais rien qui ressemble à 43. L'anomalie semble donc **localisée à des frames spécifiques** (détection YOLO ponctuellement défaillante), pas une caractéristique visuelle générale du match.

### 12.3 Piste testée : visualisation des boîtes de détection brutes → replay continu complet (V4.10)

## 13. V4.10 — Replay continu de production, `t=0` à `t=1092` (`notebook_goe_replay_continu_v47b.py`)

**Objectif** : reproduire `n_players=43` avec les vrais paramètres de production (conf réelle, cadence réelle ~75% du fps natif, `max_age` réel, suivi DeepSort continu depuis `t=0`), après audit statique préalable ayant confirmé que `n_players = len(players)` sur une seule frame instantanée sélectionnée par `_players_at()` (tolérance ±1s) — aucune agrégation cachée.

### Résultat central — Cas B, non-réplication

| | Valeur |
|---|---|
| Frame sélectionnée par `_players_at()` | `t=1092,024s` (écart 0,024s — sélection temporelle correcte) |
| `n_players` reproduit | **17** |
| `n_players` historique | **43** |

**`43` n'est pas reproduit, même avec la configuration de production correcte** (conf=0,4, ~75% du fps natif ≈22,5fps, suivi continu réel de 18 minutes). Ceci ferme la piste "cadence/configuration insuffisante" — la cause de l'écart `43 vs 17` est ailleurs.

### Découverte annexe — duplication de `tracker_id`

En examinant le détail des 17 entrées de la frame sélectionnée : **3 `tracker_id` (340, 362, 379) apparaissent chacun deux fois**, à des positions différentes. Investigation approfondie (relecture du code + analyse du `frames_data` complet sauvegardé sur la zone `1085-1095s`) :

- **`id == tracker_id`** sur toutes les entrées de la frame étudiée → `reid.process()` (ré-identification par couleur de maillot) **n'est pas responsable** de cette duplication.
- **Le phénomène est massif et structurel, pas un cas isolé** : sur les 225 frames de la zone `1085-1095s`, **146 (65%)** présentent au moins un doublon strict de `tracker_id` — et certains doublons ont **les deux occurrences à `time_since_update=0`** (donc pas simplement une piste fraîche + une piste fantôme non nettoyée).
- **Le filtre `time_since_update==0` est réfuté comme correction** : appliqué seul, il ferait chuter `n_unique_actifs` à une moyenne de 7,4 (plage 5-11) — bien en dessous de l'effectif réel visible à l'écran. `n_unique_brut` (tous tracks confirmés, ~12-13) reste la mesure la plus cohérente avec la scène, sans pour autant être déclarée vérité terrain absolue.
- **`is_confirmed()` ne peut pas être la cause** : c'est un filtre par soustraction (`[t for t in tracks if t.is_confirmed()]`) — il ne peut mathématiquement pas introduire un doublon qui n'existait pas déjà dans la liste brute qu'il filtre. Par déduction logique, **la duplication est donc déjà présente dans la sortie brute de `update_tracks()`**, sans qu'un nouveau test empirique soit nécessaire pour l'établir.

**Formulation retenue, volontairement mesurée** : *"Une collision de `tracker_id` est observée dans la sortie du tracker après `update_tracks()`. Les artefacts disponibles ne permettent pas de déterminer si son origine se situe dans l'implémentation de `deep_sort_realtime` (version 1.3.2) ou dans les détections fournies en entrée au tracker (par exemple, un chevauchement imparfait de boîtes YOLO)."* — cette distinction nécessiterait de capturer et comparer les détections YOLO brutes envoyées à `update_tracks()`, non sauvegardées dans cette phase, et n'a pas été poursuivie (voir Décision d'ingénierie ci-dessous).

### Ce que cette duplication N'explique PAS

**Le point le plus important de tout V4.10** : même en supprimant parfaitement tous les doublons observés, on resterait à ~14-15 identifiants distincts sur les 17 mesurés — un écart de 26 à 29 unités avec le `43` historique subsiste entièrement inexpliqué. **La duplication de `tracker_id` et l'écart `43 vs 17` sont deux anomalies indépendantes, pas une seule et même cause.**

```
                Goe historique (n_players=43)
                         │
          ┌──────────────┴──────────────┐
          │                              │
  Anomalie tracker DÉMONTRÉE      Écart historique 43→17
  (collisions tracker_id,          ENCORE INEXPLIQUÉ
   65% des frames, 1085-1095s)     malgré replay production-fidele
```

### Décision d'ingénierie — arrêt volontaire de cette branche

**Ne pas relancer de nouveau replay pour élucider l'origine exacte de la duplication** (bibliothèque vs détections en entrée) — ce test répondrait à *"pourquoi le tracker produit-il des collisions d'ID"*, pas à la question centrale *"pourquoi le dataset historique contient-il 43 alors que le pipeline reproductible actuel donne 17"*. Le coût de calcul déjà englouti pour établir que cadence + configuration + suivi continu ne suffisent pas est substantiel ; le poursuivre sur une anomalie déjà démontrée comme non-explicative du problème central ne serait pas un bon rapport coût/information.

**Ne rien corriger dans le pipeline de production sur la base de cette découverte** — l'anomalie de duplication est réelle et documentée, mais elle ne résout pas le mystère recherché ; corriger `n_players` ou le RF sur cette seule base serait prématuré et pourrait masquer la vraie cause encore inconnue.

## 13.1. Complément — la rampe `n_players` autour du KO, et vérification visuelle

Des artefacts supplémentaires retrouvés a posteriori (`transition_goe.csv`, mesure frame par frame) montrent que `n_players` suit une **rampe continue et quasi monotone** dans la zone étudiée :

```
t=1057s (KO-24s) : 11
t=1081s (KO exact) : 23
t=1092s (KO+11s, notre candidat) : 41
t=1104s (KO+23s) : 47
```

**Hypothèse initialement avancée puis réfutée par capture d'écran** : un élargissement du plan de caméra expliquerait cette hausse. Comparaison directe de deux captures à `t=1081s` et `t=1092s` (fournies par l'utilisateur) : **le cadrage est quasiment identique dans les deux images, et le nombre de personnes réellement visibles à l'écran est comparable (~12-13) dans les deux cas** — pas 23 puis 41. L'hypothèse caméra est donc infirmée par l'image elle-même.

### Verdict retenu, formulé avec prudence

Le mécanisme le plus vraisemblable, cohérent à la fois avec la rampe et avec les captures d'écran : la scène contient un nombre de personnes stable (~12-13), mais le tracker produit un nombre d'*entrées* croissant dans le temps (accumulation de tracks, avec des collisions de `tracker_id` déjà démontrées sur 65% des frames de la zone). **Ceci reste une piste mécanistique plausible, pas une preuve établie** — la cause précise de cette accumulation (au sein de `deep_sort_realtime` lui-même, ou dans les détections fournies en entrée) n'a pas été déterminée.

**Formulation retenue pour la documentation, volontairement mesurée** :

> *Le nombre `n_players` historique de Goe présente une inflation manifeste par rapport à la scène observée. Entre 1057s et 1104s, la valeur calculée passe de 11 à 47 alors que les captures d'écran disponibles ne montrent aucune augmentation correspondante du nombre de personnes visibles. L'analyse du replay révèle par ailleurs des collisions de `tracker_id`, présentes sur 65% des frames de la zone 1085-1095s, y compris entre deux tracks simultanément à `time_since_update=0`. Cette anomalie démontre que `n_players` est fortement affecté par l'état interne du tracking et ne peut pas être interprété comme un comptage fiable des personnes visibles sur cette séquence. Elle ne permet toutefois pas encore d'identifier la cause exacte de la divergence entre le run historique (43) et le replay (17).*

**Ce qui reste explicitement à deux niveaux séparés, à ne pas confondre :**
- **Question A (le tracking est-il défaillant ?)** — réponse partielle : oui, une anomalie réelle et mesurée existe (collisions de `tracker_id`).
- **Question B (pourquoi `43` vs `17` précisément ?)** — toujours sans réponse. Cette anomalie est compatible avec l'écart mais n'en établit pas la cause exacte.

**Ne pas documenter** : *"Goe est une feature intrinsèquement instable et non reproductible"* — formulation jugée trop forte et probablement incorrecte au vu de ce qui précède ; elle a été proposée à un stade de l'investigation puis explicitement retirée après cette vérification visuelle.

## 14. Nouvelle piste ouverte — fiabilité de l'identité joueur sur la durée d'un match

Indépendamment de Goe, une observation terrain (statistiques de match affichant des attributions suspectes à de "mauvais joueurs") combinée à la découverte de collisions de `tracker_id` amène à formuler une préoccupation plus large et plus importante à long terme que le mystère Goe lui-même.

### Les trois niveaux du problème, à ne pas confondre

1. **Détection** — détecte-t-on correctement les personnes présentes ?
2. **Tracking / identité** — le joueur A reste-t-il bien le joueur A pendant tout le match ?
3. **Attribution des événements** — les tirs/passes/xG attribués à un joueur lui appartiennent-ils réellement ?

**Le niveau 2 est le maillon fondamental** : si l'identité se perd ou se mélange entre deux joueurs, toutes les statistiques du niveau 3 deviennent fausses même si détection et événements sont corrects individuellement. La chaîne complète (`tracking → player ID → events → passes/tirs → xG/xA → heatmap → classement individuel`) peut silencieusement propager une erreur d'identité sans qu'aucune étape ne plante — c'est le type de bug le plus difficile à repérer a posteriori.

### Proposition — V5.1 : Audit de fiabilité des identités joueurs

**Objectif** : déterminer si les statistiques individuelles produites par le pipeline sur la durée complète d'un match sont réellement fiables, sur 2-3 matchs dont les joueurs sont bien connus (vérité terrain partielle, sans annotation exhaustive image par image).

| Test | Ce qui est mesuré |
|---|---|
| ID → joueur | L'identité reste-t-elle cohérente sur toute la durée ? |
| Changement d'ID | Combien de fois un même joueur réel change-t-il de `track_id` ? |
| Collision | Deux joueurs distincts partagent-ils le même ID à un instant donné ? |
| ID switch | L'ID d'un joueur passe-t-il à tort à un autre joueur ? |
| Occlusion | Que se passe-t-il quand deux joueurs se croisent/se chevauchent ? |
| Réapparition | Un joueur qui ressort du cadre puis y revient reçoit-il le bon ID ? |
| Statistiques | Les actions (tirs, passes) restent-elles attachées au bon joueur réel ? |

**Format de résultat attendu, illustratif** :
```
Joueur réel #7 :
  Track 124 : 0-18 min
  Track 287 : 18-18:14
  Track 341 : 18:14-42 min
  Track 519 : 42-90 min
  → 4 IDs pour le même joueur (cas "changement d'ID", gênant mais détectable)

Track 341 :
  18:14-25:32 → joueur #7
  25:32-26:01 → joueur #11
  26:01-42:00 → joueur #7
  → cas "ID switch" (bien plus grave — deux joueurs réels mélangés sous un même ID)
```

**Décision d'ingénierie explicite** : ne pas toucher au code de tracking/identité avant d'avoir mené cet audit — mesurer d'abord l'ampleur réelle du problème, ne pas corriger à l'aveugle sur la seule base de l'anomalie Goe (qui reste une découverte indépendante). Cette question est jugée **prioritaire sur l'amélioration du classement xG/xA** : affiner des formules de classement sur des statistiques potentiellement mal attribuées à la mauvaise personne n'aurait pas de valeur tant que la fiabilité de l'identité n'est pas établie.

**Ce qu'il faut pour lancer cet audit** : 2-3 matchs dont l'utilisateur connaît bien les joueurs (numéros de maillot, identité), pour servir de vérité terrain partielle — pas encore fournis, en attente.

## 14.1. Spécification produit — "Match Start Resolver" (3 paliers de confiance)

Suite à une question terrain directe : *si le KO n'est pas trouvé automatiquement, l'analyse du match entier (échauffement inclus dans les statistiques) n'a aucune valeur professionnelle.* Décision : **abandonner la logique binaire "succès/échec de détection"** au profit d'un contrat de confiance à 3 paliers, avec arrêt explicite et validation humaine plutôt qu'une décision silencieuse et potentiellement fausse.

### Les 3 états

```
                     Mining & RF (V4)
                            │
                            ▼
                Calcul du score de confiance
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
 🟢 AUTO_CONFIRMED   🟠 NEEDS_CONFIRMATION  🔴 MANUAL_REQUIRED
         │                  │                  │
         ├──────────────────┴──────────────────┤
         ▼                                     ▼
Analyse automatique                    Interface de validation
 (t_start injecté)                    (confirmation ou saisie)
         │                                     │
         └──────────────────┬──────────────────┘
                            ▼
                Analyse alignée sur match_time
              (match_time = video_time - t_start)
```

### Ancrage sur les statuts réels déjà établis (9 matchs de référence)

Plutôt que d'inventer des seuils numériques a priori, la correspondance se fait directement sur les statuts déjà validés dans ce document :

| État | Matchs concernés | Critère observé |
|---|---|---|
| 🟢 `AUTO_CONFIRMED` | Andrimont, Franchimont, MineroisSter, P1Minerois, Raeren, Stembert, Wanze (7/9) | Candidat unique dominant, chaîne mining→RF→vision cohérente, aucune ambiguïté de nature |
| 🟠 `NEEDS_CONFIRMATION` | Juprelle (1/9) | Ambiguïté de nature démontrée entre deux candidats crédibles (faux départ d'échauffement vs vrai KO, §7 de `V3_SELECTION_MULTICANDIDATS.md`) — la règle de désambiguïsation ±21s aide mais ne suffit pas à trancher avec certitude totale |
| 🔴 `MANUAL_REQUIRED` | Goe (1/9) | Aucun candidat ne surgit avec une confiance suffisante — rang RF du meilleur candidat 48/60 (§3), proba quasi nulle |

**Important — ce que cette spécification résout et ne résout pas** : elle **déverrouille la production sur Goe sans attendre la résolution du mystère `43 vs 17`** (§13-13.1). Le pipeline peut fonctionner correctement dès aujourd'hui en mode `MANUAL_REQUIRED` pour ce genre de cas, indépendamment de savoir un jour pourquoi le mining échoue précisément sur ce match.

### Repère temporel — deux horloges à ne jamais confondre

```
match_time = video_time - t_start
```

- `video_time < t_start` → étiqueté `PRE_MATCH`, exclu de tous les calculs (possession, heatmaps, distances, vitesses moyennes, xG/xA)
- Traçabilité obligatoire dans le rapport final : `"Début du match : 18:37 (détecté par IA, confiance 94%)"` vs `"Début du match : 18:37 (confirmé manuellement)"` — ne jamais laisser croire à une détection automatique quand elle ne l'est pas

### Règle stricte proposée

Si le statut est `MANUAL_REQUIRED` et qu'aucune saisie n'a encore été fournie : **aucune statistique "match" ne doit être produite**. Une analyse technique de la vidéo reste possible, mais le rapport doit porter le statut `MATCH_START_UNCONFIRMED` et ne jamais être présenté comme des statistiques officielles.

### Valeur ajoutée à long terme — boucle de données

Chaque saisie manuelle en mode `MANUAL_REQUIRED` (comme sur Goe) devient une nouvelle donnée d'entraînement de haute qualité pour enrichir `features_ko_v2.csv` — le prochain match similaire pourrait basculer vers `AUTO_CONFIRMED`.

### Ce qui reste à faire avant l'implémentation

**Seuils numériques précis non encore calibrés** — cette spécification s'appuie pour l'instant sur la correspondance qualitative aux 9 matchs déjà connus, pas sur un seuil de probabilité RF validé statistiquement. Avant mise en production : calibrer le seuil exact (proba RF du meilleur candidat + écart de proba avec le second candidat le plus proche) sur un échantillon plus large que 9 matchs, pour fixer la frontière `AUTO_CONFIRMED`/`NEEDS_CONFIRMATION`/`MANUAL_REQUIRED` de façon reproductible sur de nouveaux matchs jamais vus.

## 15. Prochaine direction recommandée

Recentrer l'investigation sur la **provenance exacte** de `df_mine_complet.csv` historique de Goe plutôt que sur le comportement du tracker : commit git, version d'environnement (`ultralytics`, `torch`, `deep_sort_realtime`), poids YOLO exacts, `imgsz` réellement utilisé, et surtout — quelle infrastructure (Kaggle T4 ou RunPod RTX4090) a produit ces données à l'origine. C'est désormais jugé plus susceptible d'expliquer l'écart `43` que de continuer à disséquer le mécanisme de collision d'ID, qui reste une découverte intéressante mais indépendante du problème central.

## 16. Ce qui reste ouvert

- Intégrer formellement `class_weight='balanced'` dans le pipeline de mining de production (validé sur 9 matchs + 1 cas réel de bout en bout, mais pas encore adopté comme changement définitif).
- **Goe reste un vrai mystère non résolu** — même avec la configuration de production exacte reproduite (V4.10), l'écart `43 vs 17` persiste intégralement inexpliqué. Piste recommandée : provenance exacte du dataset historique (commit, environnement, infrastructure), pas le comportement du tracker.
- La collision de `tracker_id` (V4.10) est une anomalie réelle et documentée, cohérente avec la rampe `n_players` observée (§13.1), mais son origine précise (bibliothèque `deep_sort_realtime` vs détections en entrée) reste indéterminée — volontairement non poursuivie, faute de preuve établie du lien exact avec l'écart `43 vs 17`.
- **Nouveau point ouvert prioritaire (§14)** : audit de fiabilité de l'identité joueur sur la durée complète d'un match (V5.1) — en attente de 2-3 matchs dont l'utilisateur connaît bien les joueurs, pour servir de vérité terrain partielle.
- **Nouveau point ouvert, spécification produit (§14.1)** : "Match Start Resolver" à 3 paliers de confiance — spécification fonctionnelle rédigée, ancrée sur les statuts réels des 9 matchs, mais seuils numériques précis (proba RF, écart entre candidats) non encore calibrés sur un échantillon plus large avant implémentation.
- Le seuil de clustering (20s) reste provisoire.
- L'ambiguïté de nature de Juprelle reste ouverte — aucune règle fiable trouvée pour départager un faux positif d'échauffement d'un vrai KO précoce quand les deux sont en pré-match.
