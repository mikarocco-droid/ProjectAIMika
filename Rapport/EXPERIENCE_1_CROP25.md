# PROJET SCOUTIA — EXPÉRIENCE 1 : CROP TERRAIN 25%
## Rapport de référence — BILAN FINAL — devient la Baseline v2.0 officielle du projet

**Statut :** Expérience terminée, pipeline complet (Niveau 1 → Niveau 3 → décision) validé sur les 10 matchs. Toute expérience future se compare à ce document, pas à la Baseline v1.0.

---

## 1. OBJECTIF

Mesurer l'impact du crop terrain (suppression des ~25% supérieurs de l'image)
sur la capacité du Niveau 1 (DINOv2) à détecter des candidats coup d'envoi,
**sans modifier** :
- le prompt Vision (Niveau 3)
- les poids du modèle
- la logique métier (Niveau 4)
- les paramètres de décision

La seule variable expérimentale était la représentation DINOv2 utilisée pour
le Niveau 1.

---

## 2. PROTOCOLE — UNE SEULE VARIABLE MODIFIÉE

### Variable modifiée
```
CROP_TOP_RATIO = 0.25
```
Retrait des 25% supérieurs de chaque image avant le calcul de l'embedding
DINOv2. Appliqué uniquement au Niveau 1 — jamais au Niveau 3.

### Variables rigoureusement figées (identiques à la Baseline v1.0)
- Modèle : `facebook/dinov2-base`, embedding `pooler_output` (768-dim)
- Méthode d'échantillonnage d'entraînement : positifs KO±5s à pas 0.5s (21
  échantillons), négatifs sur grille régulière 8s depuis t=0, exclusion ±30s
  autour du KO — méthode reconstruite et validée par analyse forensique des
  `.npz` baseline (cosinus moyen 0,995–1,000 sur les négatifs communs)
- Classifieur : régression logistique, `class_weight="balanced"`,
  entraînement leave-one-match-out sur les 9 autres matchs
- Niveau 2 (filtre) : `MIN_DURATION_S=5.0`, `MIN_MAX_SCORE=0.8`,
  `MIN_MEAN_SCORE=0.3`, `MAX_CANDIDATES=5` (cap chronologique)
- Niveau 3 (IA Vision) : **non modifié** — même prompt, mêmes 9 observations,
  même schéma de sortie
- Niveau 4 (règle de décision) : **non modifié** — même score pondéré,
  même seuil 12/18

### Outils de validation utilisés
- `embedding_generator_v1.py` — générateur canonique documenté
- `valider_regeneration_vs_baseline.py` — validation à 3 critères avant tout
  lancement à grande échelle
- `experience1_analyse_recall.py` — calcul automatique du rappel Niveau 1

---

## 3. RÉSULTATS NIVEAU 1

| Métrique | Baseline v1.0 | Expérience 1 (crop 25%) |
|---|---|---|
| KO présent parmi les candidats | 5/10 | **8/10** |
| Rappel Niveau 1 | 50% | **80%** |

**Améliorations :** Stembert ❌→✅, Goe ❌→✅, Juprelle ❌→✅
**Aucune régression observée.**

---

## 4. RÉSULTATS NIVEAU 3 — REPASSAGE COMPLET SUR LES 10 MATCHS

Contrairement au bilan initial (qui n'avait retesté que 5 matchs), **7 des 10
matchs avaient en réalité des candidats changés** par rapport à la baseline
(Franchimont, Juprelle, Goe, Stembert, Raeren, Wanze, P1Minerois) et ont tous
été repassés au Niveau 3, inchangé.

| Match | Niveau 3 en Baseline v1.0 | Niveau 3 en Expérience 1 |
|---|---|---|
| Franchimont | ✅ | ✅ |
| Raeren | ✅ | ✅ |
| Wanze | ✅ | ✅ |
| matchTest | ✅ (non retesté, candidats identiques) | ✅ |
| **P1Minerois** | **❌ (seul échec Niveau 3 de la baseline)** | **✅** |
| Juprelle | — (absent du Niveau 1) | ✅ |
| Goe | — (absent du Niveau 1) | ✅ |
| Stembert | — (absent du Niveau 1) | ✅ |

**Précision Niveau 3 conditionnelle** (parmi les matchs où le Niveau 1 fournit le KO) :
- Baseline v1.0 : 4/5 = 80%
- Expérience 1 : **7/7 = 100%**

### Point de méthode important — P1Minerois

P1Minerois était le seul échec du Niveau 3 en Baseline v1.0 (le bon candidat
était présent au rang 1, mais classé "jeu ouvert" à 15%). Avec les nouveaux
candidats de l'Expérience 1, il est désormais correctement classifié.

**Formulation prudente à retenir :** on ne peut pas affirmer que *"le crop a
corrigé le Niveau 3"* — ce serait impossible à prouver, puisque le Niveau 3
lui-même n'a pas changé. La formulation correcte est : **le changement de
candidats produit par le crop 25% a supprimé l'unique cas d'échec du Niveau 3
observé en baseline — le candidat présenté à Vision est désormais
correctement classifié.** Les causes possibles (nouvelle fenêtre temporelle
plus favorable, plateau mieux centré sur l'événement, différence d'images
effectivement envoyées au modèle) ne sont pas distinguées par ce test. Le
résultat opérationnel, lui, est sans ambiguïté : le pipeline complet réussit
désormais sur ce match.

---

## 5. RÉSULTAT FINAL — PIPELINE COMPLET SUR LES 10 MATCHS

| Match | Niveau 1 | Niveau 3 | Pipeline |
|---|---|---|---|
| Franchimont | ✅ | ✅ | ✅ |
| Raeren | ✅ | ✅ | ✅ |
| Wanze | ✅ | ✅ | ✅ |
| Juprelle | ✅ | ✅ | ✅ |
| Goe | ✅ | ✅ | ✅ |
| Stembert | ✅ | ✅ | ✅ |
| P1Minerois | ✅ | ✅ | ✅ |
| matchTest | ✅ | — (inchangé) | ✅ |
| Andrimont | ❌ | — | Escalade |
| MineroisSter | ❌ | — | Escalade |

---

## 6. MÉTRIQUES FINALES — BASELINE v2.0

| Métrique | Baseline v1.0 | Crop25 v2.0 |
|---|---|---|
| Rappel Niveau 1 | 50% | **80%** |
| Précision Vision conditionnelle | 80% | **100%** |
| Décisions automatiques correctes | 4/10 | **8/10** |
| Fausse décision automatique | 0 | **0** |
| Escalades nécessaires | 6/10 | **2/10** |

---

## 7. CONCLUSION

Ce résultat est presque idéal pour l'architecture choisie : **DINO reste
volontairement un outil de rappel, pas de décision. Vision reste un outil de
compréhension, pas de recherche exhaustive. Les règles Python restent
responsables de la décision finale.** Et surtout : le crop 25% améliore
fortement la couverture sans dégrader la fiabilité.

Le pipeline passe d'un système **prudent mais trop souvent aveugle** (40% de
réussite globale, 6 escalades sur 10) à un système **beaucoup plus couvrant,
toujours sans faux positif** (80% de réussite globale, 2 escalades sur 10,
zéro fausse décision automatique dans les deux versions).

**Les deux échecs restants (Andrimont, MineroisSter) ne constituent pas une
surprise** : ils avaient déjà été identifiés, lors des premières analyses des
embeddings DINOv2 (avant même la construction du Niveau 3), comme les cas
présentant le plus fort décalage de distribution ("domain shift"). L'Expérience
1 ne les résout pas, mais elle réduit le problème de cinq cas difficiles à
deux — ce qui confirme que le crop terrain améliore significativement la
qualité de la représentation, tout en isolant plus clairement les cas qui
nécessiteront une approche différente.

---

## 8. DÉCISION — CROP 25% + PIPELINE COMPLET DEVIENT LA BASELINE v2.0 OFFICIELLE

**Baseline v2.0 = crop25 + pipeline complet (Niveau 1 → Niveau 3 → décision)
validé sur 10 matchs.**

À partir de maintenant, toute nouvelle expérience se compare à ce document
(rappel N1 80%, précision N3 100%, réussite globale 80%, 0 fausse décision),
et non plus à la Baseline v1.0.

---

## 9. DIAGNOSTIC DES 2 ÉCHECS RESTANTS (ÉTAPE 2 — DÉTAILS DANS `DIAGNOSTIC_ANDRIMONT_MINEROISSTER.md`)

Un diagnostic dédié (5 ratios de crop testés isolément sur Andrimont, inspection
visuelle du vrai KO, analyse des plus proches voisins DINO sur les deux
matchs) a établi que :
- Aucun ratio de crop (20% à 40%) ne fait apparaître le KO sur Andrimont
- Le contenu de la scène du vrai KO est visuellement normal dans les deux cas
- Les plus proches voisins DINO du vrai KO, pour Andrimont **et** MineroisSter,
  sont systématiquement d'autres frames du même match — jamais un KO d'un
  autre match
- Conclusion prudente : des caractéristiques propres à chaque vidéo (nature
  exacte non isolée — colorimétrie, compression, contraste, netteté, bruit,
  rendu de l'encodage, ou combinaison de ces facteurs) dominent actuellement
  la représentation DINOv2 pour ces deux vidéos spécifiques

---

## 10. PROCHAINE ÉTAPE — EXPÉRIENCE 2

**L'Expérience 2 ne devrait pas être un autre ratio de crop** — cette piste
est épuisée pour Andrimont et MineroisSter (5 ratios testés, aucun effet).

La vraie question scientifique pour l'Expérience 2 devient :

**"Peut-on supprimer la domination de la signature vidéo dans DINO pour
récupérer les 2 cas restants (Andrimont et MineroisSter) ?"**

C'est désormais le seul verrou identifié du projet. Pistes candidates à
tester isolément (une seule variable à la fois, comme pour l'Expérience 1) :
- Normalisation colorimétrique avant DINO
- Égalisation d'histogramme
- Correction gamma
- Réduction des écarts de luminosité entre vidéos
- Changement d'encodeur (DINOv2 → autre modèle de vision)
