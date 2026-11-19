# PROJET SCOUTIA — BASELINE v1.0
## Détection automatique du coup d'envoi — architecture à 4 niveaux

**Date de gel :** version validée sur les 10 matchs de référence, pipeline figé
**Statut :** baseline officielle de comparaison — toute évolution future se mesure contre ces chiffres

---

## 1. ARCHITECTURE FIGÉE

```
Vidéo
  ↓
Niveau 1 (DINOv2 + classifieur logistique, leave-one-match-out)
  → scan complet 1 frame/s, score de similarité au concept "arrêt de jeu organisé"
  ↓
Niveau 2 (filtre généreux + cap)
  → détection de plateaux (segments > seuil), filtre durée/score, cap chronologique
  ↓
Niveau 3 (IA Vision — Claude, observations factuelles + catégorie + probabilité)
  → 9 observations booléennes/catégorielles, jamais de décision finale
  ↓
Niveau 4 (règle métier Python, score pondéré)
  → décide seul ; la probabilité du LLM n'est qu'indicative
  ↓
Journal JSONL cumulatif (faits bruts, pour mesure future, pas intuition)
```

**Principe directeur validé par ce benchmark :** chaque niveau a un rôle unique et n'empiète pas sur les autres. DINO ne décide jamais. Le LLM Vision ne décide jamais. Seul le Niveau 4 décide, sur la base de faits observés, jamais sur la base d'une probabilité seule.

---

## 2. PARAMÈTRES GELÉS (extraits des scripts réellement exécutés)

### Niveau 1 — Détection de plateaux
| Paramètre | Valeur |
|---|---|
| `SAMPLE_EVERY_S` | 1.0 (1 frame/s) |
| `MAX_MINUTES` | 30 |
| `PLATEAU_THRESHOLD` | 0.5 |
| `MIN_PLATEAU_S` | 6.0 |
| `MERGE_GAP_S` | 4.0 |
| `CREDIBLE_SCORE_THRESHOLD` | 0.7 (indicatif, non utilisé par la décision finale) |
| Modèle d'embedding | `facebook/dinov2-base`, pooler_output (768-dim) |
| Classifieur | Régression logistique, `class_weight="balanced"`, entraînement leave-one-match-out sur les 9 autres matchs |

### Niveau 2 — Filtre généreux
| Paramètre | Valeur |
|---|---|
| `MIN_DURATION_S` | 5.0 |
| `MIN_MAX_SCORE` | 0.8 |
| `MIN_MEAN_SCORE` | 0.3 |
| `MAX_CANDIDATES` | 5 (cap chronologique, pas par score) |

### Niveau 3 — IA Vision
| Paramètre | Valeur |
|---|---|
| Modèle | Claude Sonnet 5 (`claude-sonnet-5`) |
| `N_FRAMES_PER_CANDIDATE` | 7 |
| Fenêtre d'extraction | `[start_t - 2s, end_t + 3s]` |
| `max_tokens` | 900 |
| Format de sortie | JSON structuré : 9 observations + catégorie (A-I) + probabilité + justification |
| Observations demandées | ballon visible, ballon sur rond central, équipes dans leur moitié, joueurs attendent le coup de sifflet, jeu démarre pendant la séquence, arbitre visible, ressemble à une remise en jeu organisée, ballon au centre (1ère image), ballon au centre (dernière image), durée d'attente avant départ |
| Principe de prompt | Observation factuelle d'abord, catégorie ensuite, probabilité en dernier — jamais l'inverse |

### Niveau 4 — Règle de décision (score pondéré)
| Fait | Poids |
|---|---|
| Équipes dans leur moitié | +3 |
| Ballon au centre (1ère image) | +3 |
| Jeu démarre pendant la séquence | +4 |
| Ballon quitte le centre (calculé : 1ère image=centre ET dernière image≠centre) | +4 |
| Arbitre visible | +2 |
| Joueurs immobiles au début | +2 |
| **Score max** | **18** |
| **Seuil de validité** | **12/18 (67%)**, appliqué au score max *applicable* (le fait dérivé "ballon quitte le centre" est exclu du calcul, pas compté comme faux, si la 1ère image ne montre pas le ballon au centre — évite la double pénalité en cas d'occlusion) |

Règle : premier candidat (ordre chronologique) dont le score ≥ seuil relatif → retenu. Sinon, validation humaine requise. La probabilité du LLM n'intervient jamais dans la décision, seulement comme indicateur de cohérence (écart >40 points entre score et probabilité = signal à examiner).

---

## 3. RÉSULTATS SUR LES 10 MATCHS DE RÉFÉRENCE

| Match | KO réel (s) | Candidats → Niveau 3 | KO présent (Niveau 1) | Niveau 3 reconnaît | Décision finale |
|---|---|---|---|---|---|
| Franchimont | 318 | 5 | Oui | Oui | ✅ Correcte (275-370s) |
| Andrimont | 307 | 1 | Non | — | Escalade (domain shift) |
| Raeren | 410 | 5 | Oui | Oui | ✅ Correcte (374-412s) |
| Stembert | 553 | 1 | Non (raté de 3s) | — | Escalade (calibrage temporel) |
| Goe | 1081 | 2 | Non | — | Escalade (signal trop faible/court) |
| Wanze | 986 | 1 | Oui | Oui | ✅ Correcte (940-995s) |
| Juprelle | 548 | 0 | Non | — | Escalade (0 candidat, domain shift) |
| P1Minerois | 192 | 2 | **Oui** | **Non** | Escalade (échec Niveau 3) |
| MineroisSter | 723 | 3 | Non | — | Escalade (domain shift confirmé, cas connu) |
| matchTest | 8 | 1 | Oui | Oui | ✅ Correcte (0-13s) |

### Classification des échecs en 3 familles

| Famille | Définition | Matchs | Responsable |
|---|---|---|---|
| **A** | KO présent + reconnu | Franchimont, Raeren, Wanze, matchTest (4) | Pipeline complet OK |
| **B** | KO présent, non reconnu | P1-Minerois (1) | Niveau 3 |
| **C** | KO absent dès le Niveau 1 | Andrimont, Stembert, Goe, Juprelle, MineroisSter (5) | Niveau 1 |

**5 des 6 échecs viennent du Niveau 1. Un seul du Niveau 3. Zéro du Niveau 4.**

---

## 4. LES 3 MÉTRIQUES DE RÉFÉRENCE

| Métrique | Valeur | Ce qu'elle mesure |
|---|---|---|
| **Rappel Niveau 1** | **5/10 = 50%** | Le vrai KO est-il présent parmi les candidats envoyés au Niveau 3 ? Mesure DINO seul, sans ambiguïté. |
| **Précision Niveau 3 conditionnelle** | **4/5 = 80%** | Parmi les matchs où le KO est présent, le Niveau 3 le reconnaît-il ? |
| **Charge Niveau 3** | **1,6 faux candidat / match** | 16 faux candidats sur 21 candidats totaux envoyés au Niveau 3 (5 vrais). Mesure le coût de production réel. |
| Précision des décisions automatiques | **4/4 = 100%** | Quand le pipeline décide, a-t-il raison ? Aucune fausse décision sur tout le benchmark. |
| Taux d'escalade | 6/10 = 60% | Fréquence de la validation humaine requise. |

---

## 5. CONCLUSION DE CETTE PHASE

- **Le Niveau 4 est terminé.** Comportement validé : jamais de choix aberrant, préfère l'escalade à une fausse confiance. Ne plus y toucher.
- **Le Niveau 3 est déjà solide** (80% conditionnel, charge de calcul très raisonnable). Amélioration possible mais pas prioritaire.
- **Le Niveau 1 est le goulet d'étranglement.** C'est là que se concentre la quasi-totalité des échecs (5/6). C'est là que doit porter le prochain cycle d'expérimentation.

---

## 6. PROTOCOLE POUR LA SUITE

Une seule variable modifiée à la fois, dans cet ordre de priorité suggéré :
1. Crop terrain (retirer ciel/décor de l'image avant embedding DINOv2)
2. Encodeur alternatif ou fine-tuné
3. Augmentation d'embeddings / données d'entraînement supplémentaires

Après chaque modification : relancer exactement les 10 mêmes matchs, comparer uniquement les 3 métriques de référence (section 4) contre cette baseline. Toute amélioration doit être démontrée par ces chiffres, pas par impression sur un cas isolé.

**Fichier de données associé :** `niveau3_raw_log_MASTER.jsonl` (21 lignes, faits bruts par candidat, 9 matchs représentés — Juprelle absent du log car 0 candidat produit par le Niveau 1, ce qui est en soi une donnée correcte).
