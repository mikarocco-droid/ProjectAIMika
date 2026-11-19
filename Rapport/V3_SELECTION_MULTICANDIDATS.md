# V3 — SÉLECTION MULTI-CANDIDATS ET DÉSAMBIGUÏSATION PAR CONTEXTE LOINTAIN

**Statut : architecture validée sur le benchmark de référence complet à 9 matchs — 5 succès automatiques corrects, 2 ambiguïtés honnêtement signalées, 2 échecs de mining déjà connus, 0 faux résultat confiant. Pas encore modèle de production définitif** (résolution des ambiguïtés et amélioration du mining restent à traiter, séparément). Complète `INVESTIGATION_OFFSET_S_DEUX_HORIZONS.md` (§11 : le problème d'orchestration identifié comme seule pièce manquante avant un remplacement sérieux de la baseline — maintenant traité).

---

## 1. Origine — pourquoi V2 seul ne suffit pas

Le prototype V2 (2 horizons) est validé comme moteur de vision : 5/9 succès nets, stabilité parfaite (dispersion 0,00s sur 3 runs, §10 du document précédent). Mais son **orchestration** — s'arrêter au premier candidat positif trouvé chronologiquement — a révélé deux failles distinctes en creusant le cas Juprelle :

1. **Un faux positif précoce peut verrouiller le résultat** avant que le bon candidat ne soit jamais évalué (Juprelle : `292,0` accepté à tort, `556,0` — le vrai KO — jamais atteint dans le run original).
2. **Le pool contient souvent plusieurs positifs réels**, pas un faux et un vrai, mais deux (ou plus) coups d'envoi légitimes — le premier du match, et une ou plusieurs reprises après un but. L'arrêt anticipé ne les distingue pas.

## 2. Vérification systématique sur les 9 matchs

En comparant, pour chaque match, le premier positif trouvé (là où l'arrêt anticipé a coupé) au reste du pool jamais évalué :

| Match | Candidats jamais testés au-delà du 1er positif | Signal a priori |
|---|--:|---|
| Andrimont | 20 | `1242,048` — meilleur candidat RF du mining original |
| Franchimont | 20 | `1132,0`, `1154,368` — suspects (déjà positifs sous un autre prompt) |
| Goe | 0 | pool entièrement couvert (aucun positif nulle part) |
| Juprelle | 18 | `556,0` confirmé positif (le vrai KO) |
| MineroisSter | 8 | — |
| P1Minerois | 0 | pool entièrement couvert |
| Raeren | 13 | — |
| Stembert | 13 | `1264,0` — suspect |
| Wanze | 4 | `1391,472` — le cas le plus documenté de toute l'investigation |

**Test ciblé sur les 3 candidats suspects** (`Franchimont 1132,0`, `Stembert 1264,0`, `Wanze 1391,472`), avec la logique V2 figée :

| Candidat | Décision V2 |
|---|:-:|
| Franchimont 1132,0 | False (rejeté) |
| Stembert 1264,0 | **True** |
| Wanze 1391,472 | **True** |

**Conclusion : au moins 3 matchs sur 9 (Juprelle, Stembert, Wanze) ont plusieurs positifs réels dans leur pool complet.** Ce n'est pas un cas isolé.

## 3. Deux problèmes distincts, deux réponses différentes

### 3.1 Reprise après un but (Stembert, Wanze) — résolu par le contexte lointain

Hypothèse testée : à `t-45s`/`t-60s` d'une vraie reprise après un but, la scène montre du jeu déjà actif (le but vient d'être marqué) ; à `t-45s`/`t-60s` du premier KO réel, la scène montre du pré-match/échauffement.

**Résultat — 6/6 confirmations, sans exception** :

| Candidat | Contexte -45s | Contexte -60s |
|---|:-:|:-:|
| Franchimont 1132,0 | jeu déjà en cours | jeu déjà en cours |
| Stembert 1264,0 | jeu déjà en cours | jeu déjà en cours |
| Wanze 1391,472 | jeu déjà en cours | jeu déjà en cours |

**Cette règle fonctionne parfaitement pour distinguer un premier KO d'une reprise après un but.**

### 3.2 Faux positif d'échauffement vs vrai KO précoce (Juprelle) — non résolu

`292,0` (faux, échauffement) et `556,0` (vrai KO) sont **tous les deux avant le début réel du match** — la règle de contexte lointain ne peut pas les distinguer (les deux montreraient du pré-match à -45/-60s). Deux heuristiques simples testées échouent aussi :
- Proba RF : favorise `292,0` (0,005 > 0,000) — **le mauvais candidat**.
- Rapidité de confirmation (nombre d'appels, robustesse du nombre d'offsets qui confirment) : favorise aussi `292,0` — **le mauvais candidat également**.

**Aucune règle fiable trouvée pour ce cas précis.** Documenté honnêtement comme limite connue plutôt que de forcer une règle ad hoc sur un seul exemple.

## 4. Architecture V3 proposée

```
Pool complet (jusqu'a 30 candidats, tri chronologique)
         │
         ▼
Evaluer TOUS les candidats avec V2 (2 horizons) — arret anticipe DESACTIVE
         │
         ▼
Collecter TOUS les positifs
         │
    ┌────┴────┬─────────────┬──────────────────┐
    │          │             │                   │
  0 positif  1 positif    2+ positifs        2+ positifs
    │          │        (tous mais 1 en    (2+ en contexte
    │          │      contexte "jeu deja    "pre-match" -
    │          │       en cours" -> exclus)  cas Juprelle)
    ▼          ▼             ▼                   ▼
  repli    ce candidat   le seul restant    AMBIGUITE :
 (comme            (retenu, avec note      presenter tous
 avant)             "autres positifs        les candidats,
                     identifies comme       PAS de choix
                     reprises apres but,     automatique
                     ecartes")
```

**Point cle** : le tri "contexte lointain" (`-45s`/`-60s`, jeu actif ou pas) s'applique à **chaque** positif trouvé, pas seulement au premier — y compris le candidat le plus précoce, pour ne pas simplement présumer qu'il est le bon parce qu'il est premier chronologiquement.

## 5. Coût — compromis assumé

Désactiver l'arrêt anticipé signifie évaluer l'intégralité du pool (jusqu'à 30 candidats × jusqu'à 6 appels V2 = jusqu'à 180 appels/match dans le pire cas), contre le coût actuel bien inférieur (arrêt dès le premier positif, ~20-90 appels/match observés). **C'est un vrai compromis, pas un gain gratuit** — plus cher, mais correct là où l'arrêt anticipé peut se tromper silencieusement. À mesurer sur les 9 matchs avant d'adopter définitivement, cohérent avec la discipline "mesurer avant de corriger" suivie depuis le début.

## 6. Ce qui reste non résolu

Le cas Juprelle (§3.2) — deux positifs tous deux en pré-match, sans règle de départage fiable. **V3 gère ce cas honnêtement (signale l'ambiguïté) plutôt que de le résoudre.** Trouver une vraie règle de désambiguïsation pour ce sous-cas précis nécessitera probablement d'observer plusieurs exemples similaires sur d'autres matchs avant qu'un signal fiable ne se dégage — pas quelque chose à inventer sur un seul cas.

## 7. Validation sur Juprelle (`notebook_v3_validation_juprelle.py`) — terminée

Le point manquant identifié (contexte lointain jamais testé sur `292,0` et `556,0` eux-mêmes) a été complété (4 appels) :

| Candidat | Contexte -45s | Contexte -60s | Raisonnement |
|---|:-:|:-:|---|
| `292,0` (faux) | pré-match | pré-match | *"Échauffement d'une équipe avec plusieurs ballons"* |
| `556,0` (vrai KO) | pré-match | pré-match | *"Une équipe est regroupée en cercle avant de jouer"* |

**Résultat : `DECISION V3 : AMBIGUITE — [292,0, 556,0]`** — exactement le comportement prévu et documenté en §3.2/§6, confirmé empiriquement, pas juste supposé.

**Point notable** : les deux raisonnements sont cohérents et plausibles chacun pour ce qu'ils décrivent (échauffement réel pour l'un, rassemblement d'équipe classique avant un vrai coup d'envoi pour l'autre) — la règle de contexte lointain ne se trompe pas, elle atteint simplement sa limite naturelle : deux scènes de pré-match légitimement différentes, mais toutes deux antérieures au vrai début du match, donc indiscernables par ce critère seul.

## 8. Bilan de cette phase de validation

| Sous-problème | Statut |
|---|---|
| Reprise après un but (Stembert, Wanze, et témoin Franchimont) | ✅ Résolu — 6/6 confirmations, aucun faux filtrage d'un vrai KO |
| Faux positif d'échauffement vs vrai KO précoce (Juprelle) | ⚠️ Non résolu, mais géré honnêtement (ambiguïté signalée, pas de choix silencieux erroné) |

**V3 est validé comme une amélioration comportementale nette par rapport à V2+arrêt anticipé**, même sur le seul cas qu'il ne peut pas encore trancher automatiquement : passer de *"confiance élevée sur `292,0`, faux"* à *"ambiguïté signalée entre `292,0` et `556,0`"* est un progrès réel, pas neutre — un système en aval (humain ou règle future) peut travailler avec une ambiguïté signalée, pas avec une fausse certitude.

## 9. Résultat final — test complet sur les 9 matchs (`notebook_v3_9matchs.py`)

**Ne pas résumer ce résultat par "5/9".** Le résultat réel est plus riche et plus informatif :

> Sur les 9 matchs de référence, V3 produit **5 succès automatiques corrects**, **2 cas d'ambiguïté honnêtement signalés**, et **2 échecs de mining déjà connus**, avec **0 faux résultat confiant**.

### Résultat détaillé, match par match

| Match | Résultat V3 | Interprétation |
|---|:-:|---|
| Andrimont | ✅ Correct | KO trouvé à +0,5 s |
| Franchimont | ✅ Correct | Le bon candidat survit aux filtres (après avoir été initialement mal classé comme "faux positif bloque la passe 2") |
| Goe | ❌ Mining | Pas de candidat exploitable — déjà connu |
| Juprelle | ⚠️ Ambiguïté de nature | Faux positif `292,0` (échauffement) + vrai candidat `556,0` |
| MineroisSter | ✅ Correct | KO trouvé à -0,97 s |
| P1Minerois | ❌ Mining | Pool insuffisant — déjà connu |
| Raeren | ✅ Correct | Un nouveau positif tardif (`1028,5`) découvert par ce test, correctement filtré |
| Stembert | ⚠️ Ambiguïté de précision | 3 candidats cohérents autour de 553s (`541,75`/`549,0`/`557,0`) ; le vrai faux positif tardif `1264,0` correctement filtré séparément |
| Wanze | ✅ Correct | `1391,472` — le cas le plus documenté et le plus résistant de toute l'investigation (jamais stabilisé après 5 séries de correctifs de prompt) — enfin correctement filtré |

**Répartition** : 5/9 (55,6%) résolution automatique correcte · 2/9 (22,2%) ambiguïtés correctement signalées · 2/9 (22,2%) échecs de mining, en amont de la vision · **0/9 (0%) faux résultat confiant**.

**Vue alternative, la plus importante** : en comptant "résultat exploitable sans raconter une fausse histoire au système" (bon KO OU ambiguïté honnête, jamais un mauvais candidat présenté avec confiance) : **7/9 matchs**, contre une baseline qui pouvait produire un mauvais candidat avec `confidence='élevée'`.

### Trois victoires techniques à retenir

1. **Franchimont** — le problème qui semblait initialement être un problème d'`offset_s`, puis un problème d'orchestration pure, est résolu par la combinaison des deux corrections (2 horizons + filtre de contexte).
2. **Raeren et Wanze** — le filtre de contexte lointain ne se contente pas de trouver un candidat positif : il sait dire *"cette scène ressemble à un coup d'envoi, mais elle correspond à une reprise après un événement lointain et ne doit pas être retenue"*. Sur Wanze en particulier, ceci résout enfin un cas qui a résisté à 5 séries de correctifs de prompt antérieurs.
3. **Zéro faux résultat confiant** — probablement le critère le plus important pour une chaîne de production. Un système qui répond *"je ne sais pas, deux possibilités"* est plus exploitable qu'un système qui retourne `confidence=0.95` sur un mauvais candidat.

### Juprelle — conservé comme cas adversarial permanent

On sait maintenant précisément ce qu'il contient :
- `292,0` → faux positif, scène d'avant-match/échauffement.
- `556,0` → vrai événement de coup d'envoi, à +8s de la vérité terrain.
- La proba RF favorise, à tort, le faux candidat.
- La vision seule (formation + mise en jeu + contexte lointain) ne suffit pas à les départager, puisque les deux sont légitimement en pré-match.
- **V3 a le mérite de ne pas inventer une décision définitive** sur ce cas plutôt que de deviner.

**Ce candidat doit être conservé comme test de non-régression permanent** pour toute modification future de l'architecture.

## 10. Statut — validé, mais pas encore modèle de production définitif

**V3 = architecture validée sur le benchmark de référence, mais pas encore modèle de production définitif.** Deux choses restent à faire avant de considérer le problème comme réellement réglé :

1. **Décider comment résoudre les ambiguïtés** — Juprelle (ambiguïté de *nature* : faux événement vs vrai KO) et Stembert (ambiguïté de *précision* : plusieurs candidats décrivant vraisemblablement le même KO) sont deux problèmes différents, qui appelleront probablement des solutions différentes.
2. **Améliorer le mining pour Goe/P1Minerois** — désormais clairement séparé de la vision. C'est une bonne nouvelle : plus besoin de toucher à la vision pour tenter de réparer ces deux matchs.

**Ne plus toucher au prompt de vision pour l'instant** — les expériences successives (matrice `offset_s`, formation/mise en jeu séparées, contexte lointain) ont suffisamment démontré que le problème restant n'est plus principalement là.

## 11. Conclusion du benchmark

> V3 constitue une amélioration majeure de la baseline. Sur 9 matchs, il obtient 5 résolutions correctes automatiques, identifie honnêtement 2 situations ambiguës et laisse 2 échecs attribuables au mining, sans aucun faux résultat confiant. L'architecture semble donc avoir corrigé le principal défaut de la baseline — la validation prématurée d'un candidat incorrect — tout en améliorant fortement la robustesse face aux reprises et événements voisins. Les travaux futurs doivent désormais se concentrer séparément sur la résolution des ambiguïtés et sur le mining, plutôt que sur une nouvelle réécriture générale du prompt de vision.

C'est la première fois, depuis le début de cette investigation, qu'on dispose d'une base suffisamment solide pour arrêter de tourner en rond autour de la détection du coup d'envoi et passer à l'étape suivante.
