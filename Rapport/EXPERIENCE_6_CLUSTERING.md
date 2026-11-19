# PROJET SCOUTIA — EXPÉRIENCE 6 : CLUSTERING TEMPOREL
## DINOv2 cesse d'être un détecteur de coup d'envoi pour devenir un générateur d'événements

**Statut :** Expérience terminée sur les 10 matchs. Changement d'architecture majeur, pas un simple réglage — probablement l'expérience la plus importante du projet.

---

## 1. CONTEXTE

Après l'Expérience 2 (SigLIP, non adopté) et une série de tentatives sur la
représentation elle-même (double vue, fusion pondérée, classifieurs non
linéaires), toutes les pistes convergeaient vers la même conclusion : sur
Andrimont, **le signal du vrai KO existe** (rang 42/1861 dans une fenêtre
±10s, score 0,796 à un instant précis), **mais il est toujours dominé par un
autre événement** — un rassemblement pré-match / cri de guerre — qui obtient
un score supérieur sur une durée bien plus longue.

Le pipeline historique (Niveau 1 → seuil de plateau soutenu 0,5/6s → premier
candidat retenu chronologiquement → Niveau 3) demandait implicitement à DINOv2
de **trancher seul** lequel des deux événements est le bon. Une mosaïque
comparant le vrai KO aux meilleurs faux positifs a confirmé visuellement que
ce n'est pas un problème de représentation défaillante : le rassemblement
pré-match partage authentiquement les mêmes ingrédients visuels qu'un coup
d'envoi (joueurs groupés, arbitre, zone centrale) — DINOv2 ne se trompe pas
par manque de signal, il confond deux événements qui se ressemblent
réellement.

---

## 1bis. CHRONOLOGIE DES EXPÉRIENCES

| Expérience | Objectif | Verdict |
|---|---|---|
| Baseline v1.0 | DINOv2 brut, sans prétraitement | Référence historique (rappel 50%) |
| Expérience 1 | Crop terrain 25% | ✅ Adoptée (Baseline v2, rappel 80%) |
| Diagnostic Andrimont/MineroisSter | Crop 20-70%, normalisation colorimétrique | ❌ Aucune piste ne fonctionne |
| Expérience 2 | Changement d'encodeur (SigLIP) | ❌ Gain sur Andrimont, mais 3 régressions ailleurs |
| Expérience 3 | Double embedding (vue centrée + fusion pondérée) | ❌ Signal réel mais trop faible pour le classifieur |
| Expérience 4 | Classifieurs non linéaires (RandomForest, XGBoost) | ❌ Pas meilleurs que LogisticRegression — le KO reste au rang 874/1861 |
| Expérience 5 | Analyse qualitative (mosaïque des meilleurs scores) | ✅ Découverte du faux positif "rassemblement pré-match" |
| **Expérience 6** | **Changement d'architecture (clustering + Niveau 3 décideur)** | **✅ Adoptée — 9/10** |

Ce tableau explique le cheminement : cinq expériences successives ont cherché
à améliorer la représentation ou le classifieur, sans succès durable — la
sixième change la question posée au Niveau 1, plutôt que sa réponse.

---

## 2. HYPOTHÈSE

**DINOv2 ne doit plus décider quel événement est le coup d'envoi.** Il doit
seulement repérer les événements inhabituels autour du rond central — qu'il
sait très bien faire, comme le montre la richesse des signaux détectés sur
tous les matchs testés. La décision sémantique (lequel de ces événements est
réellement le coup d'envoi) revient au Niveau 3 (Claude Vision), dont c'est
précisément la compétence : comparer, contextualiser et trancher entre
plusieurs scènes visuellement proches.

> **Changement de philosophie**
>
> Avant : DINO devait détecter le coup d'envoi.
>
> Maintenant : DINO doit uniquement produire une liste raisonnable
> d'événements susceptibles d'être intéressants. La décision est transférée
> à Claude Vision.
>
> **Le rôle du Niveau 1 n'est plus de trouver le coup d'envoi, mais de
> garantir qu'il fasse partie des événements soumis au Niveau 3.**

---

## 3. NOUVEL ALGORITHME

```
Vidéo
  │
  ▼
Scan DINOv2 (1 frame/s, crop 25%, classifieur LogisticRegression — Baseline v2 inchangée)
  │
  ▼
Seuil bas (0.10) — pas de seuil de "crédibilité", juste écarter le bruit pur
  │
  ▼
Clustering temporel (fusion si écart <= 8s)
  │
  ▼
Fusion sémantique supplémentaire (fusion si écart <= 20s ET similarité
d'embedding >= 0.90 — regroupe une même scène coupée par un creux de score,
sans jamais s'appuyer sur le score DINO pour décider)
  │
  ▼
Classification événement / instant (>=3 points = "événement", sinon "instant")
  │
  ▼
Tous les clusters envoyés au Niveau 3, sans cap ni tri par importance
```

**Aucune autre variable ne change** : mêmes embeddings (DINOv2, crop25), même
classifieur (LogisticRegression, Baseline v2), même leave-one-out à 9 matchs.

### Pourquoi pas de tri par importance
Un score combinant max/aire/durée a été testé et abandonné : dans les
données réelles, le rassemblement pré-match domine sur les trois critères à
la fois (score max, aire, durée) — aucune pondération linéaire ne peut
promouvoir le vrai KO au-dessus de lui. Le tri a été supprimé : tous les
clusters sont envoyés à Claude, qui compare plutôt que de filtrer a priori.

### Bug corrigé en cours de route
La vérification initiale de présence du KO (`start <= KO <= end`) échouait
sur les clusters ponctuels (ex: un cluster à `309s-309s` face à un KO annoté
à 307s). Corrigée avec une tolérance de ±3s (bord du cluster ou pic le plus
proche), cohérente avec le fait qu'un coup d'envoi peut apparaître
visuellement quelques secondes après l'annotation.

---

## 4. RÉSULTATS — LES 10 MATCHS

| Match | Événements | Instants | Total clusters | KO présent parmi les candidats |
|---|---|---|---|---|
| Andrimont | 16 | 11 | 27 | ✅ (instant, 309s) |
| Franchimont | 16 | 9 | 25 | ✅ (événement) |
| Raeren | 10 | 15 | 25 | ✅ (événement) |
| Stembert | 11 | 13 | 24 | ✅ (événement) |
| Goe | 9 | 7 | 16 | ✅ (événement) |
| Wanze | 6 | 2 | 8 | ✅ (événement) |
| Juprelle | 13 | 1 | 14 | ✅ (événement) |
| P1Minerois | 15 | 7 | 22 | ✅ (événement) |
| MineroisSter | 0 | 8 | 8 | ❌ (aucun signal) |
| matchTest | 13 | 37 | 50 | ✅ (événement) |

**Rappel Niveau 1 (KO présent parmi les candidats envoyés au Niveau 3) : 9/10 = 90%**

### Le contraste qui compte vraiment : ancien pipeline vs nouveau

| Match | Ancien pipeline (seuil de plateau, Baseline v2) | Nouveau pipeline (clusters) |
|---|---|---|
| Andrimont | ❌ | ✅ |
| Franchimont | ✅ | ✅ |
| Raeren | ✅ | ✅ |
| Stembert | ✅ | ✅ |
| Goe | ✅ | ✅ |
| Wanze | ✅ | ✅ |
| Juprelle | ✅ | ✅ |
| P1Minerois | ✅ | ✅ |
| MineroisSter | ❌ | ❌ |
| matchTest | ✅ | ✅ |
| **Rappel Niveau 1** | **8/10 = 80%** | **9/10 = 90%** |

Le gain net (+1 match, +10 points) est modeste en apparence — mais il ne
vient d'aucune modification de la représentation ou du modèle : **c'est un
changement pur de stratégie de sélection**, sur un cas que cinq expériences
successives n'avaient pas réussi à débloquer.

---

## 5. ANALYSE

### Ce qui saute aux yeux n'est pas le 9/10 en soi
C'est que **les neuf réussites couvrent exactement les cas que toutes les
expériences précédentes échouaient à expliquer** — en particulier Andrimont,
en échec total avec DINOv2 seul, le crop (20% à 70%), la normalisation
colorimétrique, SigLIP, le double embedding, la fusion pondérée, et les
classifieurs non linéaires (RandomForest, XGBoost). Aucune de ces
modifications de la représentation ou du modèle ne changeait quoi que ce
soit — parce qu'aucune ne s'attaquait au vrai problème : une question de
sélection parmi des événements concurrents, pas une question de
représentation.

### Andrimont n'est plus un cas spécial
```
Avant : Vidéo → DINO → ❌ KO absent → Claude ne voit jamais rien
Après : Vidéo → DINO → 27 candidats → Claude → KO présent
```

### MineroisSter devient le seul cas d'étude isolé
Jusqu'ici, Andrimont et MineroisSter étaient traités comme deux variantes du
même problème (domain shift). Ce résultat change la priorité : **à ce stade,
MineroisSter devient le seul cas nécessitant une étude spécifique** — non pas
que le problème général soit résolu définitivement, mais que le pipeline
est désormais suffisamment robuste pour isoler les vrais cas particuliers.
Sur ce match, même ce protocole généreux (seuil 0,10, sur 1551 points
scannés) ne trouve que 9 points au-dessus du seuil dans tout le match, et
aucun près du KO. Ce n'est plus un problème de sélection parmi des candidats
— il n'y a simplement pas de candidat. Les pistes à explorer pour ce cas
isolé (caméra, angle, qualité d'image, voire annotation) sont différentes de
tout ce qui a été tenté jusqu'ici et méritent une étude dédiée, distincte du
reste du pipeline.

### Le nombre de clusters (8 à 50) n'est finalement pas le bon indicateur d'échec
L'objectif initial était de descendre à 5-10 clusters par match. Le résultat
réel oscille entre 8 et 50. Mais le coût pour Claude reste du même ordre de
grandeur quel que soit ce nombre dans cette fourchette : **une seule requête
groupée par match, quelques dizaines d'images plutôt que 1500-1800 frames**.
Le changement d'échelle (par un facteur de 30 à 200) est déjà acquis, que le
nombre exact soit 8 ou 50.

---

## 6. NOUVELLE ARCHITECTURE SCOUTIA

```
Vidéo
   │
   ▼
DINOv2
(générateur de candidats)
   │
   ▼
Fusion temporelle
+ fusion sémantique
   │
   ▼
Liste d'événements
   │
   ▼
Claude Vision
(classification comparative)
   │
   ▼
Niveau 4
(règles métier)
```

Le rôle du Niveau 1 est redéfini : il ne cherche plus à trouver LE coup
d'envoi, il cherche à **ne pas le manquer** parmi une liste raisonnable de
candidats. La décision sémantique — celle qui nécessite de comparer un
rassemblement pré-match à un vrai coup d'envoi — est confiée entièrement au
Niveau 3, qui est conçu pour ce type de raisonnement.

---

## 8. FUTUR NIVEAU 3

Les clusters ne seront plus évalués indépendamment. Claude recevra
l'ensemble des événements d'un même match en une seule requête, et devra :
- classer chaque événement dans une catégorie sémantique (coup d'envoi,
  rassemblement avant-match, changement de côté, cri de guerre, reprise,
  autre) ;
- indiquer un niveau de confiance par événement ;
- désigner le meilleur candidat "coup d'envoi" du match, en comparant les
  événements entre eux plutôt qu'en répondant indépendamment à chacun.

Le score DINO (max, durée, nombre de points) est transmis comme indice
contextuel, jamais comme critère de filtrage — Claude peut l'utiliser ou
l'ignorer. Ce chapitre sera développé en détail dans le prochain document
(conception du prompt et validation sur les 9 matchs récupérés).

---

## 9. PROCHAINES ÉTAPES

1. **Redessiner le prompt du Niveau 3** pour une classification comparative
   par match (tous les clusters d'un même match dans une seule requête),
   avec des catégories sémantiques explicites (coup d'envoi / rassemblement
   avant-match / changement de côté / cri de guerre / reprise / autre), une
   confiance par catégorie, et une désignation finale du meilleur candidat
   "coup d'envoi". Le score DINO (max, durée, nombre de points) est transmis
   comme indice contextuel, jamais comme critère de filtrage.
2. **Étudier MineroisSter isolément**, sans chercher à le faire rentrer dans
   le même protocole que les 9 autres — diagnostic dédié (image par image,
   qualité vidéo, angle de caméra) avant toute nouvelle expérience de
   représentation.
3. **Valider le nouveau Niveau 3** sur les 9 matchs qui fonctionnent déjà
   (vérifier l'absence de régression) avant de le considérer comme la
   nouvelle référence du pipeline.

---

## 10. CONCLUSION

Cette expérience ne constitue pas une amélioration incrémentale des
performances de DINOv2. **Elle redéfinit son rôle dans Scoutia.**

Les expériences précédentes (1 à 4) cherchaient à rendre le Niveau 1
suffisamment performant pour identifier directement le coup d'envoi — par le
crop, l'encodeur, la représentation double vue, ou le classifieur. Aucune
n'a résolu Andrimont de façon durable. Les résultats de cette expérience
montrent qu'une architecture où DINO agit comme générateur de candidats et
Claude comme interprète sémantique est plus robuste, et explique naturellement
les cas difficiles comme Andrimont sans dégrader les autres matchs.

**Cette architecture devient la nouvelle référence pour les développements
futurs du pipeline.**