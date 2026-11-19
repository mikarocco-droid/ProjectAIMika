# PROJET SCOUTIA — EXPÉRIENCE 2 : ENCODEUR SIGLIP
## Rapport de référence — Baseline v2 (DINOv2 + crop 25%) reste la référence opérationnelle

**Statut :** Expérience terminée sur les 10 matchs. SigLIP n'est PAS adopté comme remplacement de DINOv2, mais l'expérience apporte une découverte importante à documenter et à réutiliser plus tard.

---

## 1. OBJECTIF

Répondre à la question posée à l'issue du diagnostic Andrimont/MineroisSter
(voir `DIAGNOSTIC_ANDRIMONT_MINEROISSTER.md`) : le crop terrain et les
prétraitements colorimétriques ayant été épuisés sans résoudre les deux
derniers échecs, **un encodeur différent de DINOv2 peut-il les récupérer,
sans dégrader le reste du pipeline ?**

---

## 2. PROTOCOLE

### Screening préliminaire (Étape 0)
Un screening rapide à 3-4 matchs (Franchimont, Andrimont, MineroisSter, Wanze),
classifieur entraîné sur seulement 2-3 matchs, a montré un signal fort sur
Andrimont (KO retrouvé au rang 1) mais un résultat ambigu sur Franchimont —
jugé non concluant à cause de la faiblesse du set d'entraînement, pas
généralisable en l'état.

### Expérience complète (seule variable modifiée : l'encodeur)
```
ENCODER_NAME = "google/siglip-base-patch16-224"
```
remplace `facebook/dinov2-base`. **Tout le reste identique à la Baseline v2** :
crop 25%, méthode d'échantillonnage (positifs ±5s/0.5s, négatifs grille 8s,
exclusion ±30s), classifieur en régression logistique, leave-one-out complet
sur les 9 autres matchs, seuils de détection de plateaux (`PLATEAU_THRESHOLD`,
`MIN_PLATEAU_S`, `MERGE_GAP_S`), filtre généreux du Niveau 2, `MAX_CANDIDATES=5`.

### Séquence de test (décisionnelle, pour limiter le calcul en cas d'échec précoce)
1. Génération des 9 embeddings SigLIP (hors Andrimont) — prérequis incompressible
2. **Test décisif sur Andrimont seul**, avec le vrai leave-one-out à 9 matchs
3. Poursuite uniquement si succès : Franchimont et Wanze (vérification anti-régression), puis les 6 matchs restants

---

## 3. RÉSULTATS — LES 10 MATCHS

| Match | DINOv2 (Baseline v2) | SigLIP | Différence |
|---|---|---|---|
| Andrimont | Non | Oui (rang 4/5) | Amélioration |
| Franchimont | Oui (rang 1/6) | Oui (rang 1/10) | Confirmé |
| Wanze | Oui (rang 1/3) | Oui (rang 2/2) | Confirmé |
| Juprelle | Oui (rang 5/5) | Non (rang 7/25) | Régression |
| matchTest | Oui (rang 1/1) | Oui (rang 1/2) | Confirmé |
| MineroisSter | Non | Non (0 candidat) | Statu quo |
| Raeren | Oui (rang 2/5) | Non (0 candidat) | Régression |
| Stembert | Oui (rang 1/1) | Oui (rang 1/2) | Confirmé |
| Goe | Oui (rang 1/1) | Non (rang 9/9) | Régression |
| P1Minerois | Oui | Oui (rang 1/21) | Confirmé |

---

## 4. LES DEUX MÉTRIQUES — ET POURQUOI LEUR ÉCART EST LA VRAIE DÉCOUVERTE

| Métrique | DINOv2 (Baseline v2) | SigLIP |
|---|---|---|
| Rappel brut (KO présent parmi tous les candidats, avant cap) | 8/10 = 80% | 8/10 = 80% — IDENTIQUE |
| Rappel opérationnel (KO survit au cap MAX_CANDIDATES=5) | 8/10 = 80% | 6/10 = 60% |

**Le rappel brut est rigoureusement identique.** SigLIP ne voit pas moins de
coups d'envoi que DINOv2 — il les classe différemment dans le temps. C'est la
distinction la plus importante de cette expérience : ce n'est pas un problème
de capacité de représentation, c'est un problème de distribution des scores
qui interagit mal avec un cap chronologique conçu et calibré pour DINOv2.

### Diagnostic causal des 4 échecs SigLIP

| Cause | Matchs concernés | Nombre |
|---|---|---|
| KO absent (vrai échec de représentation) | MineroisSter, Raeren | 2 |
| KO présent mais rang > 5 (échec de calibrage du pipeline) | Juprelle (rang 7/25), Goe (rang 9/9) | 2 |

**La moitié des échecs de SigLIP ne sont pas imputables à l'encodeur
lui-même**, mais à un paramètre du Niveau 2 (MAX_CANDIDATES=5) qui n'a jamais
été recalibré pour la distribution de score, nettement moins tranchée, que
produit SigLIP.

---

## 5. CONCLUSION

SigLIP ne dégrade pas la capacité de retrouver les coups d'envoi (rappel brut
identique à DINOv2), mais il modifie fortement la distribution des scores de
similarité. Le pipeline, calibré autour de DINOv2 (MAX_CANDIDATES=5 et
sélection chronologique), n'est plus optimal avec SigLIP. Dans ces
conditions, le rappel opérationnel passe de 80% à 60%, principalement parce
que le vrai KO apparaît plus fréquemment au-delà du cinquième candidat
retenu.

**Andrimont reste le signal le plus important de cette expérience.** Après
l'échec de toutes les pistes de prétraitement testées sur ce match (crop 25%
à 70%, égalisation d'histogramme, CLAHE, normalisation z-score), SigLIP le
retrouve directement, avec un entraînement leave-one-out complet à 9 matchs —
confirmé, pas un artefact du screening préliminaire. C'est la première fois
depuis le tout début du projet qu'une méthode voit réellement ce coup
d'envoi.

**Verdict opérationnel : DINOv2 + crop 25% (Baseline v2) reste la
référence.** SigLIP n'est pas adopté en l'état — le coût des 3 régressions
dépasse le bénéfice de l'unique amélioration, sur le pipeline actuel.

Cette expérience est néanmoins une réussite, pas un échec : elle répond sans
ambiguïté à la question posée ("un encodeur généraliste peut-il remplacer
immédiatement DINOv2 ?" → non), tout en révélant une information précieuse
(SigLIP voit réellement Andrimont) qui pourra être réexploitée dans une
future expérimentation — par exemple un pipeline hybride, ou un recalibrage
du Niveau 2 spécifiquement pour SigLIP.

---

## 6. CE QUI N'EST PAS FAIT MAINTENANT, ET POURQUOI

MAX_CANDIDATES n'est volontairement pas modifié dans le cadre de cette
expérience, même si le diagnostic de la section 4 le désignerait comme un
levier évident. Le modifier maintenant ouvrirait une deuxième variable
(encodeur + cap) et casserait la propreté de la comparaison Baseline v2 vs
Expérience 2. Un recalibrage du cap pour SigLIP, s'il est tenté, doit être
une expérience distincte (Expérience 3, une seule variable : le cap, avec
SigLIP comme configuration figée), pas un ajustement a posteriori de
celle-ci.

---

## 7. DÉCISION

- Baseline v2 (DINOv2 + crop 25%) reste la référence opérationnelle du
  projet — rappel Niveau 1 80%, précision Niveau 3 100%, précision globale
  80%, aucune régression depuis sa validation.
- SigLIP n'est pas intégré au pipeline de production.
- Le signal Andrimont est documenté et conservé comme piste pour une future
  expérience (pipeline hybride, recalibrage du cap avec SigLIP, ou nouvel
  encodeur).
- Andrimont et MineroisSter restent, à ce stade, en dehors du pipeline
  automatique — escalade humaine, comme documenté depuis la Baseline v1.0.
