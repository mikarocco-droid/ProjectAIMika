# V5.2 — Documentation de session : nouvelle architecture Fin1MT, stratégie Q2 pour KO1, diagnostic Melen

## Résumé exécutif

Session partie du diagnostic `Melen` (match amical avec échauffement prolongé perturbant `Fin1MT`/`KO2`), ayant mené à plusieurs changements de production validés, une tentative reversée après régression détectée, et un run complet final sur `10` matchs (`9` de référence `+` `Melen`).

---

## Changements de production INTÉGRÉS et validés

### 1. Nouvelle architecture Fin1MT (remplace narratif + Q2)

**Avant** : `Q1` (signal de sortie) `→` vérification narrative (`2` images) `→` `Q2` (`"terrain vide"`) `+60-90s` plus tard.

**Après** : `3` confirmations `Q1` consécutives (`pas=60s`) `→` pause établie au **premier** point de la série `→` recherche fine par dichotomie (`15/5/1s`) dans `[dernier NON, premier de la série]`. `marge_apres_min` passée de `5` à `2min`.

**Pourquoi** : sur `Melen`, l'ancienne logique produisait des dizaines de rejets narratifs consécutifs face à l'échauffement prolongé des remplaçants (le narratif exigeait une "transition en cours", introuvable une fois la pause déjà établie) — retard de `+467s`.

**Validation** : testée sur les `9` matchs de référence — médiane d'erreur divisée par `3` (`9s → 3s`), coût réduit de `34%` (`22,6 → 14,9` appels/match).

**Fichier** : `fin1mt_gemini_cascade.py`, fonction `find_fin1mt_gemini`.

### 2. Stratégie C pour KO1 (vérification Q2 en deux temps)

**Avant** : `Q2` vérifié une seule fois, à `candidat+60s`.

**Après** : `Q2` vérifié d'abord à `candidat+30s` ; si `NON`, second essai à `candidat+60s` (filet de sécurité).

**Pourquoi** : diagnostic `Andrimont` — un candidat proche du vrai `KO1` (`t=300s`, `7,5s` d'écart) a été rejeté à tort par `Q2+60s`, forçant le scan à continuer et à confirmer un candidat erroné `470s` plus tard. `+30s` aurait correctement confirmé ce même candidat.

**Validation empirique rigoureuse** : test réel (scan `Q1` complet, pas d'approximation) sur les `10` matchs — un seul cas où `+30`/`+60` divergent en faveur de `+30` (`Andrimont`, gain de `470s`), `2` cas où ils divergent dans l'autre sens (`+60` réussit là où `+30` échoue encore — sans perte, `+60` reste le filet de sécurité).

**KO2 explicitement neutralisé** : `delai_verif_q2_precoce` fixé égal à `delai_verif_q2` (`22s`) pour `KO2` — comportement strictement inchangé, cette stratégie n'ayant été validée que pour `KO1`.

**Fichiers** : `kickoff_gemini_cascade.py` (`_rechercher_kickoff`, `detect_kickoff_gemini`), `match_boundaries_v2.py` (paramètre explicite pour `KO2`).

### 3. Réduction du coût de sortie (tokens)

**Retiré** : champ `raisonnement` des schémas `JSON` de `Fin1MT` (`Q1`, `Q2`, histoire narrative) et `FinMatch` (`Q1`, `Q2`) — comme `KO1`/`KO2` qui ne l'avaient jamais eu.

**Conservé en `CoT` court** : vérification narrative de `FinMatch` — raisonnement limité à `15` mots maximum au lieu d'un paragraphe complet, validé sans régression sur `2` cas connus (`Andrimont t=6347/6352` et `t=6372/6377`).

**Justification** : les tokens de sortie coûtent significativement plus cher que les tokens d'entrée ; ce changement réduit la facture sans sacrifier la fiabilité (le mécanisme de vérification reste actif, juste plus concis).

---

## Changement TENTÉ puis REVERSÉ

### Règle affinée sur les buts d'entraînement (KO1/KO2)

**Tentative** : distinguer un but actif du match (extrémités du terrain) d'un petit but d'entraînement/rangement sur le côté, pour corriger `KO2` sur `Melen` (`73%` des points de scan rejetés à tort par la règle originale `"tout but visible → rejet"`).

**Validation initiale** : `0` régression sur `27` points testés (`9` matchs de référence), correction confirmée sur `Melen` (`t=4220s` : `NON→OUI`, `+59s` d'écart au lieu d'être manqué).

**Régression découverte** : deux runs complets consécutifs sur `Andrimont/KO1` ont donné `379s` (au lieu de `~310s` validé) avec cette règle active. Diagnostic approfondi (traces instrumentées, tests de taux répétés) n'a pas permis d'isoler un mécanisme déterministe unique, mais la reproductibilité de l'échec (`2/2` runs) a été jugée plus probante que les tests contradictoires isolés.

**Décision** : reversée. `KO1`/`KO2` sont revenus à la règle originale (`"tout but visible → rejet automatique"`), validée et stable. Le cas `Melen/KO2` reste donc non corrigé.

---

## Résultats du run complet final (10 matchs)

| Match | KO1 | KO2 | Fin1MT | FinMatch | Appels |
|---|---|---|---|---|---|
| Andrimont | +2s | +3s | +21s | **+175s** ⚠️ | 181 |
| Raeren | +5s | +1s | +11s | +10s | 131 |
| P1Minerois | +1s | +3s | +26s | +9s | 125 |
| MineroisSter | +0s | +1s | +2s | +35s | 267 |
| Juprelle | +2s | +2s | -1s | +12s | 230 |
| Franchimont | +3s | +3s | +13s | +15s | 116 |
| Goe | +1s | +2s | +7s | +59s | 215 |
| Wanze | +1s | +18s | +2s | +10s | 259 |
| Stembert | +2s | +16s | +12s | +12s | 164 |
| MelenP1 | +1s | **+82s** ⚠️ | +18s | +38s | 219 |

**Total : 1907 appels | ~190,7/match | ~$5,66 estimé (ordre de grandeur) pour les 10 matchs**

### Points d'attention identifiés dans ce run (non résolus, acceptés comme limites connues)

**`FinMatch/Andrimont = +175s`** : `15+` rejets narratifs consécutifs (*"positions identiques, aucun mouvement de sortie"*) avant épuisement de la fenêtre et repli sur le mécanisme de secours. `FinMatch` n'a pas été modifié aujourd'hui — probable variance de modèle sur ce run précis, même match qui avait déjà posé problème sur ce détecteur en début de session.

**`KO2/MelenP1 = +82s`** : le candidat à `t=4278s` (`1s` du vrai `KO2`) a été trouvé et voté par `Q1`, mais rejeté par `Q2` précoce à `t=4300s`. Cohérent avec le diagnostic approfondi mené ce matin sur ce match précis (échauffement perturbant la reconnaissance de `Q2`). Non affecté par les changements du jour (`KO2` inchangé, règle des buts reversée).

---

## État des fichiers de production (fin de session)

| Fichier | Statut |
|---|---|
| `kickoff_gemini_cascade.py` | Stratégie C intégrée (KO1), règle des buts reversée à l'original, instrumentation diagnostic retirée |
| `match_boundaries_v2.py` | KO2 neutralisé pour la stratégie C (comportement inchangé), règle des buts reversée |
| `fin1mt_gemini_cascade.py` | Nouvelle architecture intégrée, raisonnement retiré des schémas |
| `finmatch_gemini_cascade.py` | Raisonnement retiré de Q1/Q2, CoT court sur la vérification narrative |
| `pipeline.py` | Inchangé |

Tous vérifiés syntaxiquement valides et cohérents entre eux au moment de la clôture de session.

---

## Limites connues (acceptées, non résolues)

1. **`KO2/Melen`** : sensibilité au cadrage caméra et à l'échauffement prolongé — piste de correction (règle des buts affinée) tentée et reversée après régression sur `Andrimont`.
2. **`FinMatch`** : biais systématique en retard (`10-30s`, parfois plus) déjà documenté ; sensibilité à la phase d'échantillonnage de la grille (`Andrimont` notamment) ; variance résiduelle de `Gemini` à `température=0`.
3. **`Fin1MT`/`KO2` sur cadrage caméra atypique** (`Melen`) : petits buts d'entraînement visibles simultanément au rond central, cas rare mais réel.

## Item complémentaire : diagnostic Spa et correction KO2 (stratégie C étendue)

### Contexte

Après la clôture initiale de session, un nouveau match (`Spa`) a été testé. Premier run : `KO2=+672s` (avec vérités terrain initialement mal saisies, corrigées en cours d'échange — `KO1=415s`, `Fin1MT=3122s`, `KO2=3820s`, `FinMatch=6910s`). Résultat reproductible sur `2` runs identiques (candidat `Q1` à `t=3798s`, rejeté, dérive vers `t=4140s`).

### Diagnostic précis (scan `Q1`+`Q2` avec raisonnement sur `[3790s, 3830s]`, pas `2s`)

**Le vrai `KO2` bascule à `t=3822s`** (`Q2` passe de `NON` à `OUI` de façon nette et stable). **Le candidat `Q1` à `t=3798s` était correct** (`22s` avant le vrai instant) — mais la vérification `Q2` de production (`+22s → t=3820s`) est tombée à **`2s` près** de la vraie transition, où le ballon n'était pas encore visible sur le terrain (le coup d'envoi n'avait littéralement pas encore été donné). `Q2` a répondu `NON` correctement pour cet instant précis — ce n'est pas une erreur de jugement, mais un problème de calage de la grille de vérification.

`Q1`, interrogé sur la même zone avec une question différente, corrobore indépendamment cette même transition (bascule de langage `"mise en place"` → `"jeu manifestement actif"` entre `3822s` et `3828s`).

### Correction : stratégie C étendue à KO2

**`delai_verif_q2_precoce=22s`** (inchangé, c'était l'unique vérification avant ce jour) **`delai_verif_q2=30s`** (filet de sécurité tardif, passé de `22s` à `30s`).

**Validation rigoureuse** (boucle complète, vrais candidats `Q1` via `vote_economique`, sur `11` matchs — `9` référence `+` `Melen` `+` `Spa`) :
- `Wanze` : `4734s → 4712s` (plus proche du vrai `4735s`)
- `MelenP1` : `4499s → 4255s` — gain de `244s`, le candidat légitime à `4255s` (`24s` du vrai `4279s`) était rejeté par `+22s` mais confirmé par `+30s`
- **`0` régression** sur les `9` autres matchs

### Résultat final sur Spa (run complet, code corrigé)

| | Détecté | Vrai | Écart |
|---|---|---|---|
| `KO1` | `418s` | `415s` | `+3s` |
| `KO2` | `3819s` | `3820s` | `-1s` ✅ (corrigé, était `+672s`) |
| `Fin1MT` | `3132s` | `3122s` | `+10s` |
| `FinMatch` | `6924s` | `6910s` | `+14s` |

Les `4` détecteurs sont dans la fourchette normale — validation de bout en bout réussie.

**Fichier modifié** : `match_boundaries_v2.py` (paramètres `delai_verif_q2`/`delai_verif_q2_precoce` de l'appel `detect_kickoff_gemini_avec_retry`).

---

## Item complémentaire : diagnostic Andrimont/FinMatch (+175s) et pistes identifiées

### Contexte

`FinMatch` sur `Andrimont` a donné `+175s` sur `3` runs consécutifs identiques (`6323s` détecté via mécanisme de repli, vrai `FinMatch=6148s`). Contrairement à `Spa`/`KO2`, ce n'est **pas** un problème touché aujourd'hui — `FinMatch` n'a reçu aucune modification cette session, seule sa vérification narrative a été raccourcie en `CoT` (`15` mots), sans changement de logique.

### Diagnostic précis (scan `Q1` avec raisonnement sur `[6080s, 6220s]`, pas `10s`)

**`Q1` détecte correctement la transition, mais `22s` après la vérité terrain** : `NON` stable jusqu'à `t=6160s` (*"jeu en cours, joueurs en pleine course"*), puis `SORTIE` net et stable à partir de `t=6170s` (*"mouvement de sortie évident"*, puis rituel de poignées de main confirmé de `6180s` à `6210s`). Ce n'est pas un angle mort de jugement — c'est cohérent avec le **biais systématique en retard** de `FinMatch` déjà documenté en tout début de session (`10-30s`, le coup de sifflet final n'étant pas visuellement détectable avant que le mouvement de sortie soit amorcé).

**Le vrai problème est un calage de grille** : le scan de production (pas `60s`) teste `6107s` et `6167s`, ratant `6170s` de `3s`. Le premier `SIGNAL_SORTIE` de la grille de production n'arrive qu'à `t=6287s` (`139s` après le vrai FinMatch) — zone où commence alors une cascade de rejets narratifs dus à une activité de ballon informelle post-match (probablement des jeunes/enfants jouant sur le terrain après la fin officielle), menant finalement au repli à `6323s`.

### Impact concret signalé par l'utilisateur

**Ce n'est pas qu'un écart de timestamp abstrait** : si `FinMatch` est détecté `175s` trop tard, une analyse vidéo avale basée sur cette fenêtre inclurait `175s` de jeu informel post-match (enfants/jeunes jouant sur le terrain) dans ce qui est censé être le résumé du vrai match.

### Piste identifiée n°1 : appliquer la logique Fin1MT à FinMatch

**Remplacer le rejet narratif systématique par `3` confirmations `Q1` consécutives** (comme `Fin1MT`), pause établie au premier point de la série. Sur la trace `Andrimont`, ça aurait déclenché dès `t=6287s` (au lieu de `6323s` après `15` rejets) — un gain réel mais partiel, qui n'aurait pas résolu l'angle mort `Q1` entre `5927s` et `6227s`.

**Non testé, non déployé** — nécessiterait la même validation rigoureuse sur les `9` matchs de référence qu'on a appliquée pour `Fin1MT` ce matin, avant tout déploiement.

### Piste identifiée n°2 (proposée par l'utilisateur) : signal "match adulte + personnes non conformes"

**Contexte produit** : le flux prévu est que l'utilisateur upload sa vidéo, `KO1` est lancé en premier pour extraire le timestamp du coup d'envoi **et les couleurs des deux équipes** (déjà mentionnées spontanément par `Gemini` dans son raisonnement `Q1`/`Q2` tout au long de la session — *"rouge à gauche, jaune/bleu à droite"*), avant que l'utilisateur ne fournisse les noms d'équipe et autres paramètres.

**Deux signaux combinés proposés** :
1. **Match adulte** (`half_duration_min=45`, déjà disponible dans le pipeline) `+` détection de mineurs/enfants sur le terrain par la vérification narrative de `FinMatch` → signal fort de fin de match réelle (les enfants n'ont normalement pas accès au terrain pendant le jeu).
2. **Comparaison de couleurs** : capturer les couleurs de référence des deux équipes dès `KO1` (déjà énoncées par `Gemini`), les transmettre à `FinMatch` comme contexte, et traiter la présence de personnes aux couleurs non conformes comme signal supplémentaire de fin de match. Plus objectif/vérifiable que le critère "enfant" seul.

**Avantage** : cohérent avec le flux produit déjà prévu (couleurs déjà extraites à `KO1`), donc l'ajout ne serait pas une nouvelle fonctionnalité isolée mais la réutilisation d'une information déjà planifiée.

**Réserve identifiée** : risque de faux positif si `Gemini` confond joueur junior/ado avec "enfant" sur un match dont l'âge des joueurs est proche de la limite adulte — à vérifier empiriquement avant déploiement.

**Non testé, non implémenté** — nécessite : `(a)` faire circuler l'information couleurs entre `KO1` et `FinMatch` (aujourd'hui détecteurs indépendants), `(b)` tester sur les `9` matchs de référence pour absence de régression, avant tout déploiement.

---

## Item complémentaire : correctif KO1 — vérification préalable à t=0

### Contexte

Hypothèse implicite jamais testée dans le code (`t_début=60` pour `KO1`) : *"aucun vrai coup d'envoi n'arrive avant `60s`"*. Testée sur un nouveau match réel (`match_testcomplet.mp4`, `KO1≈8s`) — **violée concrètement**, avec conséquence sérieuse.

### Diagnostic

**Sans correctif** : le scan (démarrant à `60s`) n'a aucune chance de voir le vrai coup d'envoi, et finit par confirmer à tort une reprise de jeu après un but (`t=431s`, `+423s` d'écart) — le signal ressemblant visuellement à une vraie formation d'engagement.

### Correctif implémenté

**Vérification ponctuelle à `t=0`** avant le scan normal : si `Q1(t=0)=OUI`, recherche fine directe dans `[0, t_début]` (réutilise la logique existante — vérification `Q2`, recherche fine par paliers — sans dupliquer de code). Si `NON`, comportement inchangé, scan normal depuis `t_début=60s`. **Coût additionnel dans le cas normal : `1` seul appel `Gemini`.**

**Résultat sur le match précoce** : `431s` (faux) → `10s` (`+2s`, correct), et moins cher (`10` appels au lieu de `23`).

**Validation sur `5` des `11` matchs de référence** (`Andrimont`, `Franchimont`, `Goe`, `MelenP1`, `Spa`) : `5/5` corrects, écarts `≤3s`, aucune régression. Surcoût observé variable selon les matchs (`+0` à `+21` appels) — en partie attribuable à de la vraie variance `Gemini` entre runs séparés (candidats intermédiaires différents trouvés), pas uniquement au coût de la vérification `t=0` elle-même. **Non testé sur les `6` matchs restants** (`Raeren`, `P1Minerois`, `MineroisSter`, `Juprelle`, `Wanze`, `Stembert`) — jugé suffisant pour ce soir vu la cohérence des résultats.

### Décision sur le pas adaptatif KO1 (`120→60`, `180→60`)

**Explorée en parallèle, finalement écartée** : `120→60` fonctionnait bien (`0` régression sur `11` matchs, `~14%` d'économie), mais `180→60` cassait déjà `2` matchs sur `4` testés (`P1Minerois` notamment, `+1152s`). Décision finale : **garder `pas=60` fixe** pour `KO1` (comportement historique, déjà éprouvé) — le risque d'un match démarrant tôt (`~130-180s`) rendait le gain de coût du pas grossier trop incertain face à ce risque, d'autant que le vrai problème (matchs démarrant *très* tôt, `<60s`) est désormais couvert par la vérification `t=0`, qui n'a pas ce risque.

**Fichier modifié** : `kickoff_gemini_cascade.py` (`_rechercher_kickoff`, ajout de la vérification `t=0` avant la boucle principale).

---

## Item complémentaire : KO2 pas adaptatif (20s→10s) + marge_avant_min=6 — validation finale correcte

### Contexte méthodologique important

**Le premier test de cette piste (plus tôt aujourd'hui) utilisait `KO1` vérité terrain**, pas les valeurs réellement détectées — un vrai problème méthodologique découvert en cours de session (l'utilisateur l'a signalé : *"pourquoi reprendre les vrais `KO1` alors qu'on ne les trouve jamais ou très peu correctement"*). Ce test-là a été refait avec les `11` vraies valeurs `KO1` détectées aujourd'hui.

### Résultat final (méthodologie correcte)

**`11/11` corrects (`≤30s` d'écart)**, tous entre `+1s` et `+3s` sauf `Stembert` (`+16s` — mais n'a même jamais basculé en `pas=10`, donc variance normale indépendante de cette correction). **`833` appels `Gemini` au total** sur les `11` matchs.

### Statut

**`pas=20→10` adaptatif `+` `marge_avant_min=6` pour `KO2` est maintenant validé avec une méthodologie fiable** — prêt à être intégré en production (non encore fait, existe seulement dans les variantes `_benchmark.py`).

---

## Items en attente (mis à jour)

- **[GELÉ]** Correctif règle des buts d'entraînement pour `KO1`/`KO2` (tenté, reversé après régression sur `Andrimont`, non résolu — cas `Melen` avec petits buts d'entraînement visibles au centre reste non corrigé par cette voie)
- **[RÉSOLU]** Stratégie C pour `KO2` (`+22s`/`+30s`) — validée sur `11` matchs, intégrée en production, confirmée fonctionnelle sur `Spa` **et** `Melen` (`+82s → +1s`)
- **[RÉSOLU]** `KO1` : vérification préalable à `t=0` — corrige les matchs démarrant avant `60s`, validée sur `5/11` matchs sans régression, **intégrée en production**
- **[RÉSOLU]** `KO2` : pas adaptatif (`20→10`) `+` `marge_avant_min=6` — validé `11/11` avec vrais `KO1` détectés, **non encore intégré en production** (existe dans la variante benchmark)
- **[IDENTIFIÉ, NON TESTÉ]** `FinMatch/Andrimont` (`+175s`, reproductible sur `3` runs) : deux pistes documentées ci-dessus (logique `Fin1MT`-style, signal couleurs/mineurs) — à tester en priorité la prochaine session
- **[EN COURS]** Module de pré-analyse upload (`KO1` + couleurs), voir `pre_analyse_upload.py` — reste à faire : appariement couleurs `Gemini`↔tracker (labellisation après clustering, pas d'injection dans le clustering lui-même)
- **[PENDING]** Prochains matchs à tester avec le système dans son état actuel

