# JALON — INVESTIGATION SUR LA REPRODUCTIBILITÉ DU MINING KO1
## (suite du JALON_10_MATCHS_ET_CONTRAT_FONCTIONNEL.md, à partir de la section 26)

**Statut** : investigation clôturée sur plusieurs branches, avec des
conclusions fermes obtenues par falsification, pas par abandon. Document
à utiliser comme contexte de départ pour une nouvelle conversation.

**Origine** : après le test d'intégration complet (section 26 du jalon
précédent), un test de recall en mode aveugle sur les 9 matchs a révélé
un écart brutal entre le "9/9" historique et un nouveau "3/9" — cette
investigation cherche à comprendre pourquoi, sans toucher au code tant
que la cause n'est pas confirmée.

---

> ⚠️ **AVERTISSEMENT CRITIQUE À NE JAMAIS PERDRE — LIRE AVANT TOUTE
> MODIFICATION DE `PROMPT_KO_VISION` (section 19) :** le prompt Gemini
> "checklist" (vérification stricte image par image) A DÉJÀ ÉTÉ
> reformulé une fois en version "histoire globale" (jugement holistique)
> — testé sur `gemini-2.5-flash`, ça a **doublé le taux de faux
> positifs** (voir tableau section 19). Cette leçon a été **oubliée une
> fois pendant cette même investigation** (la version "histoire
> globale" s'est retrouvée réintroduite sans qu'on s'en souvienne,
> découverte seulement en refouillant les transcripts). Ne JAMAIS
> reformuler ce prompt vers un jugement "holistique"/"histoire globale"
> sans re-tester explicitement sur `gemini-2.5-flash` (pas seulement un
> modèle plus puissant) sur les 9 matchs avant adoption.

---

## 1. DÉCOUVERTE FONDATRICE — LE "9/9" NE MESURAIT JAMAIS LE MINING

`features_ko_v2.csv` contient deux sources distinctes, marquées dans la
colonne `source_dataset` :
- **`v1_manuel`** (25 lignes) : candidats choisis à la main à l'origine
  du projet, **avant** que `candidates_mining.py` existe.
- **`v2_mining`** (508 lignes) : candidats produits automatiquement par
  `mine_candidates()`.

**Audit local (sans Kaggle) sur les 9 matchs** : les 10 vrais KO du
dataset viennent à **100% de `v1_manuel`**, **0% de `v2_mining`**. Le
vrai recall du mining historique (en excluant les KO injectés
manuellement) : **0/9 à ±3s, 3/9 à ±5s, 7/9 à ±10s**.

**Conséquence** : tous les benchmarks "9/9" précédents (jalon sections
20-23) mesuraient uniquement la capacité de Gemini à choisir dans un
pool où le KO était déjà garanti présent — jamais la capacité du mining
à le proposer. Ce n'est pas une régression récente, c'est une confusion
méthodologique présente depuis le début du projet.

---

## 2. LE MINING ACTUEL EST NETTEMENT MEILLEUR QUE LE DATASET HISTORIQUE

Diagnostic à 4 étages (grille brute → pics sans coupure → candidats
finaux `mine_candidates()` en mode aveugle → sifflet), testé en direct
sur 7 matchs (Colab GPU T4 et Kaggle CPU, `process_video()` fonctionne
sans GPU, juste plus lent) :

| Match | Écart historique (`v2_mining`) | Écart mining actuel |
|---|---|---|
| Andrimont | -11,0s | +5,0s |
| Goe | +11,0s | +3,0s |
| Stembert | 4,0s | -1,0s |
| Franchimont | 4,75s | -4,75s (stable — cas de contrôle) |
| P1Minerois | 9,75s | -7,75s |
| Wanze | ? | -6,0s (puis -0,1s via sifflet, cf. section 3) |
| MineroisSter | 4,98s | -0,97s |

**6/7 montrent une amélioration nette**, le seul stable (Franchimont)
était déjà bon — élimine l'hypothèse d'une amélioration générique du
pipeline, confirme que le dataset historique contenait de vraies
erreurs de mining spécifiques à certains matchs.

---

## 3. DÉCOUVERTE `top_n_whistle` — PARTIELLEMENT UTILE

Sur P1Minerois et Wanze, l'étage 4 (sifflet) a révélé des pics
extrêmement précis (0,1 à 0,5s du KO) **absents des candidats finaux**
— coupés par `top_n_whistle=10` (défaut), un match complet contenant
naturellement plus de 10 vrais coups de sifflet.

**Vérifié comme sûr avant de tester** : `_whistle_peaks()` utilise
`find_peaks()` pour détecter les vrais pics AVANT d'appliquer `top_n` —
augmenter ce paramètre ne peut jamais inventer de faux positifs.

**Résultat du test `top_n_whistle=40`** :
- MineroisSter : aucun changement (-0,97s déjà optimal) — confirme la
  non-régression sur les cas déjà bons.
- Wanze/P1Minerois : correction lancée, résultat non recueilli avant que
  l'investigation bifurque vers la piste audio (section 6).

**Limite découverte plus tard (section 6)** : même `top_n_whistle=150`
ne suffit pas toujours — sur Wanze, le pic à 985,904s n'était classé
que 121e/208 par amplitude parmi TOUS les pics du match. Un simple
paramètre ne résout donc pas tous les cas.

---

## 4. BUG DE MÉTHODOLOGIE DANS LE SCRIPT DE DIAGNOSTIC (PAS LE PIPELINE)

`_peaks()` (dans `candidates_mining.py`) trie bien par magnitude en
interne (`np.argsort(-values[idx])`) pour sélectionner les `top_n`
meilleurs — **mais la ligne finale `return sorted(...)` re-trie le
retour par TEMPS**, perdant l'information de force. Le script de
diagnostic utilisait `.index()` sur ce retour pour calculer un "rang" —
donc un **rang chronologique**, pas un rang de force.

**Confirmé dans le code source** (ligne 119-120 de `_peaks()`), et
reproduit sur un cas simulé contrôlé (pic faible mais précoce sortait
"rang 1" avec l'ancienne méthode).

**Important** : ce bug n'affecte QUE l'étage 2 du diagnostic (analyse
informative), **pas** le pipeline de production — `_peaks()` fonctionne
correctement en interne pour sélectionner ses `top_n` candidats, peu
importe l'ordre du retour final.

**Corrigé** : nouvelle fonction `rang_par_magnitude()` ajoutée aux deux
scripts de diagnostic (Colab et Kaggle), testée et validée.

**Conséquence sur les résultats déjà obtenus** : à revérifier au cas
par cas. Confirmé après re-test : Wanze (rang 13 par magnitude, dans le
top-15 — l'écart initial ×25 était une erreur de calcul, pas un vrai
rang), Goe (rang 3, déjà correct), P1Minerois (rang 30-45, confirmé
loin). Andrimont : jamais formellement reconfirmé après le fix avant
que l'investigation bifurque vers la piste audio.

---

## 5. SIX HYPOTHÈSES TESTÉES ET FALSIFIÉES (PAS ABANDONNÉES PAR LASSITUDE)

Chacune avec une preuve concrète, pas une supposition écartée :

| # | Hypothèse | Test effectué | Résultat |
|---|---|---|---|
| 1 | `whistle_score` seul (amplitude) suffit | Comparaison train/test | Insuffisant — mismatch de méthode confirmé (`_whistle_score_at` grille vs `whistle_score` précis), mais recall RF non amélioré par un fix partiel |
| 2 | `top_n_whistle` plus large résout tout | Testé à 40/80/150 | Seul 150 récupère le pic Wanze — pas une solution générale, dépend du match |
| 3 | Croisement mouvement + sifflet | Testé | Pas de gain démontré |
| 4 | Forme/isolement du pic audio (prominence, largeur, isolement, densité) | 4 mesures sur 4 matchs, critère de décision fixé À L'AVANCE | Aucune des 4 mesures légitimes ne passe le seuil <20 sur les 4 matchs (pire cas : amplitude=113, prominence=161, largeur=311, isolement=329) |
| 5 | Densité locale de pics (`densite_1s`/`densite_2s`) | Repéré comme "parfait" (rang 1/N partout) | **Artefact mathématique** : `find_peaks(distance=3s)` garantit structurellement qu'aucun pic n'est à moins de 3s d'un autre — la densité sur fenêtre <3s vaut 0 pour TOUS les pics, pas seulement le KO. Confirmé par preuve mathématique directe. |
| 6 | Rampe de `n_players` comme signature discriminante | Comparée sur 6 matchs (2 difficiles, 4 récupérés/stables) | **Universelle** (6/6, y compris les cas déjà bons) — ne discrimine pas les cas difficiles des cas faciles |

**Point méthodologique central retenu** : *"Une observation intéressante
n'entre dans le code que si elle démontre une capacité de discrimination
sur un benchmark défini à l'avance."* Règle explicite pour éviter le
cycle "regarder les données → trouver un motif → inventer une feature →
tester → découvrir un artefact" déjà vécu plusieurs fois.

---

## 6. CE QUI RESTE VALIDE ET UTILE (PAS À REJETER)

- La rampe de `n_players` après un KO est un phénomène physique réel et
  généralisable (dispersion progressive des joueurs) — utile à connaître,
  mais pas discriminant.
- Les matchs peuvent être résolus par des **canaux différents** selon le
  cas (mouvement pour certains, sifflet pour d'autres) — ne jamais
  comparer naïvement la forme d'un signal entre un match "facile" et un
  match "difficile" sans vérifier par quel canal chacun a réellement été
  résolu.
- `CHANGEPOINT_KO` (détecteur de rupture de régime plutôt que de pic) —
  **idée V2 conditionnelle**, fondement réel (la rampe existe) mais
  aucune valeur discriminante démontrée. Ne pas coder maintenant.

---

## 7. ÉTAT ACTUEL DU PIPELINE

```
KO1
 → Mining : encore imparfait sur certains matchs (Andrimont, P1Minerois
   restent difficiles ; aucun signal suffisamment discriminant trouvé
   dans les modalités testées à ce jour)
 → Scoring RF : à évaluer une fois le pool de candidats correctement
   établi pour les 9 matchs — PAS ENCORE FAIT SYSTÉMATIQUEMENT
 → Gemini : ne pas toucher, fonctionne déjà bien sur des candidats
   corrects (ex. Stembert)
```

**Ne pas modifier `candidates_mining.py`, le RF, ni Gemini sur la base
des découvertes actuelles** — aucune n'a démontré un gain suffisant pour
justifier un changement de production.

---

## 8. CE QUI RESTE À FAIRE (PROCHAINE PRIORITÉ)

**✅ TERMINÉ** : diagnostic à 4 étages sur les 9 matchs, avec
environnement figé (`ultralytics==8.4.120`) — voir tableau final
section 10. Résultat : 8/9 sous 10s, 5/9 sous 5s, seul Goe résiste
(11,0s).

**✅ TERMINÉ** : scoring RF systématique sur les 9 `df_mine_complet.csv`
— voir résultat complet et décisif en section 11. 5/9 cas de
"RF_MAUVAIS" confirmés : le mining trouve un bon candidat, le RF
(entraîné sur `features_ko_v2.csv` non corrigé) le classe mal.

**✅ TERMINÉ ET CLOS** : investigation complète de la correction
`whistle_score` (tests contrefactuels, stabilité multi-seed, diagnostic
de distribution) — voir conclusion définitive en section 13. **Décision
actée : ne pas corriger `whistle_score` uniformément.** L'effet est
réel mais mécaniquement hétérogène (biais de la méthode grille tantôt
défavorable, tantôt favorable au candidat cible selon le match).

**✅ TERMINÉ ET VALIDÉ** : stratégie adaptative top-30 (section 14) —
`detecter_ko_par_vision_adaptatif()` prêt, testé par mocks, fiabilité
confirmée sur la vraie durée de production (1800s). Prêt pour un test
en conditions réelles sur Andrimont, Juprelle, Wanze.

**✅ TERMINÉ ET ABANDONNÉ** : pré-filtre visuel à 1 image (section 15)
— testé en conditions réelles, 2 faux rejets de sécurité sur 3 matchs
malgré un prompt permissif. Ne pas retenter sans nouvelle preuve
préalable.

**✅ TERMINÉ ET ABANDONNÉ** : motif C / score de formation visuelle
(section 16) — testé sur les 9 matchs avec vrai passage YOLO, 3/8 en
top-5 (37,5%), aucune composante individuelle ne dépasse ce seuil.
Aucune corrélation exploitable avec la performance du RF. Ne pas
retenter de nouvelle variante sans preuve préalable.

**✅ TERMINÉ ET FERMÉ** : fusion multi-signaux (section 17) — 4
stratégies testées (moyenne, top-2, consensus, poids appris en
leave-one-match-out), aucune ne dépasse le RF seul (38% top-10 pour
les deux meilleures). Nuance : démontre que cette fusion à 4 signaux
n'égale pas le RF — ne démontre PAS que le RF (11 features) est sans
défaut.

**PROCHAINE ACTION UNIQUE ET PRIORITAIRE** : retest effectué (section
19) — **les deux causes se cumulent, aucune ne suffit seule** :
1. **Revoir la logique de décision finale** pour ne plus s'arrêter au
   premier positif trouvé (Andrimont et Wanze ont encore un faux positif
   chronologiquement avant le vrai KO, même avec le prompt corrigé) —
   collecter tous les positifs du pool, choisir selon un critère plus
   robuste que "le premier".
2. **Investiguer séparément le cas Juprelle** : le vrai KO lui-même est
   rejeté par Gemini (pas un problème de prompt ni d'arrêt anticipé) —
   possible désalignement temporel (candidat à 8s du vrai KO).

**Test du pré-filtre (section 15) toujours à refaire** — sa décision
d'abandon reste suspendue.

**Prochaine étape réelle** : tester `detecter_ko_par_vision_adaptatif()`
en conditions réelles (vrais appels Gemini) sur Andrimont, Juprelle et
Wanze — vérifier si l'élargissement top-30 retrouve effectivement le
bon KO sur ces 3 matchs, et mesurer le coût réel engagé.

Pistes complémentaires, **seulement si de nouvelles données le
justifient** (ne pas relancer une nouvelle feature sans preuve
préalable, cf. principe méthodologique section 13) :
- **Figer `scikit-learn`** dans tous les scripts de scoring RF
  (jamais fait jusqu'ici, contrairement à `ultralytics`/`torch` en
  section 10) — nécessaire avant de retester P1Minerois et Stembert
  (cas frontière, section 11) de façon reproductible.
- Vérifier si `stabilite_avant_s` présente le même type de biais
  mécanique que `whistle_score` (même diagnostic de distribution,
  transposé à cette feature).
- Envisager un ré-entraînement du RF avec plus de données (au-delà des
  9 matchs actuels) plutôt qu'une correction de feature isolée.
- Sur Goe (seul cas `MINING_MAUVAIS`) : le problème reste en amont
  (génération/sélection du candidat sifflet, cf. section 3), aucune
  piste de scoring RF n'est pertinente pour ce cas.
- Sur Raeren (taux top-30 = 37%, le plus faible) : accepter comme échec
  probable pour l'instant, ne pas chercher une solution dédiée sans
  nouvelle piste.

---

## 9. FICHIERS ET NOTEBOOKS DE RÉFÉRENCE

- `candidates_mining.py`, `ko_features.py`, `detect_ko_vision.py` —
  versions actuelles, dans `codeprojectai` (dataset compagnon Kaggle).
- `colab_diagnostic_4_etages_9matchs.py` / `notebook_diagnostic_4_etages_kaggle.py`
  — diagnostic à 4 étages, **corrigés** (rang par magnitude), configurés
  avec `top_n_whistle=40` et `MATCHS_A_TRAITER` à ajuster match par match.
- `notebook_diagnostic_forme_isolement_sifflet.py` — diagnostic audio
  seul (prominence, largeur, isolement, densité), **conclusion négative
  actée** (section 5, hypothèse 4-5), gardé pour référence mais piste
  fermée.
- `notebook_correction_whistle_v2_mining.py` — script de correction du
  dataset (`refine_whistle_scores()` sur les 508 lignes `v2_mining`),
  **jamais lancé** — en attente de la priorité 8.2/8.3 ci-dessus avant
  de décider si nécessaire.

---

## 10. DÉCOUVERTE — SUSPICION DE NON-REPRODUCTIBILITÉ CPU vs GPU

**Contexte** : lors du re-test des 4 matchs manquants (Andrimont,
Juprelle, Raeren, P1Minerois) sur Kaggle **sans GPU** (CPU, DeepSort en
FP32), Andrimont a donné un résultat radicalement différent de son tout
premier test réussi (Colab, GPU T4) :

| | GPU (1er test) | CPU (ce test) |
|---|---|---|
| Grille à t=312s | geo=1.00, speed=42.2 | geo=0.89, speed=39.2 |
| Candidat t=312 dans le pool final | ✅ présent | ❌ absent, remplacé par t=296 |
| Écart au KO connu | +5,0s | **-11,0s** |

**Point crucial** : ce n'est pas juste "le RF classe différemment le
même candidat" — **le pool de candidats en entrée du RF a lui-même
changé**. La chaîne complète (détections YOLO → features → grille →
pics → fusion) diverge selon le matériel, pas seulement le scoring final.

**Ce qui n'est PAS encore établi** : une corrélation temporelle forte
(même vidéo, même code supposé, matériel différent, résultat différent)
n'est pas encore une preuve causale. Une seule observation ne suffit
pas à conclure — il faut un test contrôlé dédié.

### Pourquoi c'est plus prioritaire que le scoring RF

Si le pool de candidats change selon CPU/GPU, alors comparer "mining
actuel vs historique" ou scorer les 9 fichiers avec le RF risque
d'attribuer à tort au RF (ou au dataset `features_ko_v2.csv`) des
écarts qui viennent en réalité de l'environnement d'exécution. Sans
clarifier ce point, un même code pourrait donner GPU→8/9, CPU→5/9, et
faire croire à tort à une amélioration ou régression du pipeline.

### Plan de vérification (avant tout scoring RF)

1. **Andrimont seul, en GPU**, avec un environnement strictement
   identique au test CPU qui vient d'être fait (même
   `candidates_mining.py`, mêmes paramètres — `top_n_whistle=40`,
   même `frame_skip`, même résolution, même `imgsz`) :
   - Si le GPU reproduit **t=312, +5,0s** → preuve forte que le
     matériel influence le mining, passer à l'étape 2.
   - Si le GPU donne encore **-11,0s** (comme le CPU) → l'hypothèse
     CPU/GPU tombe, chercher ailleurs (version du code, du modèle
     YOLO/Ultralytics, ou la vidéo source elle-même).
2. **Si confirmé sur Andrimont** : reproduire le même test sur 1-2
   autres matchs pour vérifier la généralité du phénomène.
3. **Si le phénomène se généralise** : figer un environnement de
   référence explicite pour tout futur benchmark (GPU vs CPU, versions
   exactes des dépendances), et le documenter systématiquement à côté
   de chaque résultat.
4. **Seulement après** : scorer les 9 pools de candidats avec le RF
   actuel.
5. **Seulement après ça** : décider si `whistle_score`/
   `stabilite_avant_s` doivent être corrigés dans `features_ko_v2.csv`.

**Décision actée** : ne pas adopter `top_n_whistle=40` comme nouveau
défaut de production tant que ce point n'est pas clarifié — impossible
de savoir actuellement quelle part des écarts observés vient de ce
paramètre et quelle part vient de la divergence CPU/GPU.

### Résultat du test contrôlé (résolution du mystère)

Andrimont testé 2 fois de suite sur GPU, environnement non figé
(`pip install ultralytics` sans version) : les deux runs ont donné un
résultat **identique** entre eux (t=307,5, +0,5s), mais **différent**
du tout premier test fait plusieurs jours plus tôt (t=312, +5,0s).

**Conclusion** : ce n'était ni un problème CPU/GPU, ni un aléa interne
du pipeline (le pipeline est déterministe pour une version donnée) —
c'était une **dérive de version des dépendances** (`ultralytics`
installée sans version figée, mise à jour entre les sessions). Version
figée à `ultralytics==8.4.120`, `torch 2.10.0+cu128`, confirmée stable
sur GPU T4.

### Tableau final des 9 matchs, environnement figé et documenté

| Match | Historique | Mining (env. non figé) | Env. figé `v8.4.120` |
|---|---|---|---|
| Andrimont | -11,0s | +5,0s | **+0,5s** |
| Franchimont | 4,75s | -4,75s | **+0,6s** |
| Goe | 11,0s | +3,0s | 11,0s ⚠️ |
| Juprelle | ? | +4,5s | +8,0s |
| MineroisSter | 4,98s | -0,97s | **+0,97s** |
| P1Minerois | 9,75s | -7,75s | +3,9s |
| Raeren | 5,17s | ? | +5,2s |
| Stembert | 4,0s | -1,0s | +4,0s |
| Wanze | ? | -6,0s | +6,0s |

**Bilan global (env. figé)** : 8/9 sous 10s, 5/9 sous 5s, 3/9 sous 1s —
écart moyen absolu de 4,46s. **Le seul cas qui résiste est Goe**
(11,0s) — cohérent avec la découverte antérieure : son meilleur pic
sifflet (1082,384s, écart -1,4s) reste hors du `top_n_whistle=40` en
amplitude, malgré son élargissement.

**Ce que ça établit fermement** : figer explicitement les versions de
dépendances (`ultralytics`, `torch`) est indispensable pour tout futur
benchmark — sans ça, un même code peut légitimement donner des résultats
substantiellement différents d'une session à l'autre, rendant toute
comparaison "avant/après" non fiable.

---

## 11. SCORING RF SYSTÉMATIQUE — RÉSULTAT DÉCISIF

**⚠️ CORRECTION APPLIQUÉE** (après vérification indépendante) : le
tableau original de cette section reposait sur un **seul
`random_state=0`**, exactement l'erreur méthodologique que la section
13 a explicitement identifiée et corrigée pour `whistle_score` — mais
jamais rétro-appliquée ici. Un test à 30 seeds a révélé que **2 des 8
catégorisations étaient des cas frontière**, pas des verdicts fermes.
Tableau corrigé ci-dessous. Point technique additionnel identifié à
cette occasion : la version de `scikit-learn` n'a jamais été figée
dans ce jalon (contrairement à `ultralytics`/`torch`, section 10) — à
corriger pour toute future comparaison RF.

**Méthode** : pour chacun des 9 `df_mine_complet.csv` (environnement
figé, section 10), deux mesures strictement séparées :
1. **Recall du mining** : un candidat existe-t-il à ≤3s/≤5s/≤10s du
   vrai KO, indépendamment de tout scoring ?
2. **Rang RF** : le RF (entraîné sur les 8 AUTRES matchs de
   `features_ko_v2.csv` non corrigé, aucune fuite) classe où le
   meilleur candidat proche du KO, parmi TOUS les candidats du pool —
   **mesuré sur 30 `random_state`, pas un seul** (correction appliquée).

**Règle appliquée** : si le recall à ≤10s est faux, le RF n'est jamais
appelé — la catégorie est directement `MINING_MAUVAIS`, pour ne jamais
confondre un échec de génération avec un échec de scoring (cf. section
8, point sur Goe).

### Tableau final (30 seeds, robustesse de la catégorie)

| Match | Écart meilleur candidat | Rang RF moyen (30 seeds) | % seeds top-10 | Catégorie robuste |
|---|---|---|---|---|
| Andrimont | 0,50s | 29,6 | 0% | 🟠 RF_MAUVAIS (robuste) |
| Franchimont | 0,61s | 1,0 | 100% | 🟢 OK (robuste) |
| Goe | 11,0s | — | — | 🔴 MINING_MAUVAIS |
| Juprelle | 8,0s | 43,5 | 0% | 🟠 RF_MAUVAIS (robuste) |
| MineroisSter | 0,97s | 3,3 | 100% | 🟢 OK (robuste) |
| **P1Minerois** | 3,89s | **7,6** | **83%** | ⚠️ **CAS FRONTIÈRE — surtout OK** |
| Raeren | 5,17s | 33,1 | 3% | 🟠 RF_MAUVAIS (robuste) |
| **Stembert** | 4,00s | **11,1** | **33%** | ⚠️ **CAS FRONTIÈRE — surtout RF_MAUVAIS** |
| Wanze | 6,00s | 36,1 | 0% | 🟠 RF_MAUVAIS (robuste) |

### Bilan corrigé

- **🟢 2/9 robustement OK** (Franchimont, MineroisSter) — rang quasi
  parfait dans 100% des seeds.
- **🟠 4/9 robustement RF_MAUVAIS** (Andrimont, Juprelle, Raeren, Wanze)
  — hors top-10 dans 97-100% des seeds, catégorisation fiable.
- **⚠️ 2/9 cas frontière** (P1Minerois, Stembert) — le rang oscille
  suffisamment (P1Minerois : 5-13 selon la seed ; Stembert : 5-16) pour
  qu'un seul tirage (ou une version différente de scikit-learn, jamais
  figée ici) fasse basculer la catégorie dans un sens ou l'autre. **Ne
  pas chercher d'explication spécifique à "pourquoi le RF classe mal
  P1Minerois"** (bon dans 83% des cas) **ni ignorer Stembert comme
  "déjà résolu"** (échoue dans 67% des seeds).
- **🔴 1/9 MINING_MAUVAIS** (Goe) — déjà expliqué (section 3) : signal
  sifflet précis mais hors `top_n_whistle=40` en amplitude.

**Conclusion inchangée sur le fond** : la majorité des échecs restants
ne sont pas des échecs de mining — ce sont des échecs de **scoring
RF**, mais avec une nuance importante : **4 cas sont fermement établis
(Andrimont, Juprelle, Raeren, Wanze), 2 cas restent incertains
(P1Minerois, Stembert)** et mériteraient d'être retestés avec une
version de scikit-learn figée avant toute décision les concernant
spécifiquement.

### Cas le plus flagrant, toujours valide : Andrimont

Écart de seulement **0,50 seconde** — le mining a quasiment trouvé le
KO exact — mais rang RF moyen **29,6/92 sur 30 seeds (0% top-10)**,
robustement hors du top-10 exploité par Gemini. C'est le cas choisi
pour le test ciblé de la section 12 (le signal est le plus "propre"
pour valider ou invalider l'hypothèse whistle_score, avant de toucher
aux 508 lignes du dataset) — ce choix reste pleinement justifié après
cette correction.

---

## 12. TEST CIBLÉ EN COURS — CORRECTION `whistle_score` SUR ANDRIMONT SEUL

**Objectif** : avant de corriger les 508 lignes `v2_mining` de
`features_ko_v2.csv` (coût important, irréversible sans sauvegarde),
valider l'hypothèse sur un seul candidat — celui d'Andrimont (écart
0,50s, rang RF 26/92).

**Protocole** (discipline anti-sur-ajustement, cf. section 5) :
1. Recalculer `whistle_score` du candidat concerné avec la méthode
   précise (`ko_features.whistle_score()`), sans toucher aux autres
   candidats du pool ni à aucune autre feature.
2. Réutiliser exactement le même RF (entraîné sur les 8 autres matchs,
   inchangé) pour re-scorer.
3. Comparer explicitement :
   - Rang avant (26) vs après
   - Probabilité RF avant vs après
   - **Nombre de candidats devant lui** avant/après
   - **Variation moyenne des AUTRES candidats** (contrôle indispensable
     — si tout le classement bouge uniformément, ce n'est pas une
     preuve spécifique à `whistle_score`)
   - Top-10 RF complet avant/après

**Trois issues possibles, définies à l'avance** (évite de réinterpréter
le résultat après coup) :
- **A. 26 → 1-10** : preuve forte que `whistle_score` est une cause
  majeure — justifie la correction complète.
- **B. 26 → 15-25** : contribue mais insuffisant seul — tester
  `stabilite_avant_s` séparément avant de conclure.
- **C. 26 → ~26** : la piste `whistle_score` est probablement mauvaise
  — ne pas lancer la correction complète, chercher ailleurs.

**Statut** : test réalisé sur Andrimont sur un seul `random_state=0`.
**Résultat confirmé robuste par la suite** (test de stabilité à 30
seeds, section 13) — voir conclusion complète en section 13.

### Résultat — Andrimont

| Mesure | Avant | Après |
|---|---|---|
| `whistle_score` | 0,360 | **4,700** (×13) |
| Rang RF | 26/92 | **10/92** |
| Proba RF | 0,0150 | 0,0850 |
| Candidats devant lui | 25 | 9 |

**Contrôle validé parfaitement** : variation moyenne des autres
candidats = 0,000000, aucun autre rang modifié — la preuve la plus
solide possible que l'effet est spécifique à ce candidat, pas un
artefact de recalcul global.

**Nuance importante** : le candidat entre **exactement** à la limite
du top-10 (position 10, pas une position confortable comme 3 ou 5).
Résultat réel et spécifique (issue A au sens strict), mais **marginal**
— pas une victoire nette de type "26 → 1-3". Ne pas généraliser sur ce
seul cas avant un 2e test indépendant (Wanze, section 12bis).

---

## 13. CONCLUSION FINALE DU CHANTIER `whistle_score` — NE PAS CORRIGER UNIFORMÉMENT

### Chronologie complète des tests (après le test Andrimont isolé, section 12)

1. **Test contrefactuel n°2 (Wanze)** — candidat testé à 6,0s du KO (pas
   comparable à Andrimont à 0,50s). Amélioration 30→17, mais analyse
   jugée moins propre que souhaité à cause de l'écart temporel plus
   grand. Gardé en test secondaire.
2. **Test contrefactuel P1Minerois** (candidat à 3,89s, comparable à
   Andrimont) — sur un seul `random_state=0` : 11→7, entrée dans le
   top-10. Semblait confirmer la réplication.
3. **Généralisation à 9 matchs, un seul `random_state=0`** — révèle un
   tableau très hétérogène (4 améliorés, 2 neutres, 3 dégradés), avec
   Raeren en chute catastrophique (17→54).
4. **Test de stabilité à 30 `random_state` (Andrimont seul)** — 30/30
   améliore, écart-type qui s'effondre (6,3→1,0) : preuve solide et
   robuste sur ce match.
5. **Test de stabilité à 30 `random_state` (Raeren seul)** — chevauchement
   total entre les deux distributions (non corrigé 12-67, corrigé
   23-67) : l'effet initial "17→54" était un artefact d'un seul tirage,
   pas un effet causal reproductible.
6. **Généralisation du test à 30 seeds sur les 9 matchs** — résultat
   définitif : **P1Minerois s'avère lui aussi négatif en moyenne**
   (7,9→9,3 sur 30 seeds, 20 seeds dégradées contre 3 améliorées) — la
   "réplication" du point 2 était elle-même trompeuse, exactement la
   même leçon que Raeren.

### Tableau définitif (30 seeds, moyenne du rang RF)

| Match | Catégorie | Moyenne rang (non corrigé → corrigé) | Seeds améliorées/dégradées |
|---|---|---|---|
| Andrimont | 🟢 Positif robuste | 27,5 → 9,4 | 30/0 |
| Stembert | 🟢 Positif robuste | 7,0 → 1,6 | 30/0 |
| Wanze | 🟢 Positif robuste | 31,9 → 15,4 | 30/0 |
| MineroisSter | 🟢 Positif modéré | 7,4 → 6,5 | 14/1 |
| Franchimont | ⚪ Neutre | 1,0 → 1,0 | 0/0 (déjà rang 1) |
| Goe | ⚪ Neutre | 58,2 → 55,6 | 2/0 (déjà `MINING_MAUVAIS`) |
| Juprelle | 🔴 Négatif | 26,2 → 27,4 | 8/13 |
| P1Minerois | 🔴 Négatif | 7,9 → 9,3 | 3/20 |
| Raeren | 🔴 Négatif | 37,4 → 52,9 | 2/21 |

### Diagnostic explicatif final (section 13bis) — mécanisme identifié

Un diagnostic complémentaire (rang du candidat cible **par
`whistle_score` seul**, parmi les autres candidats non corrigés) a
révélé le mécanisme exact :

| Match | Whistle avant | Rang whistle avant | Rang whistle après |
|---|---|---|---|
| Andrimont (🟢) | 0,36 (très bas) | 52 | **7** |
| Stembert (🟢) | -0,04 (très bas) | 56 | **4** |
| Wanze (🟢) | -0,80 (très bas) | 79 | **39** |
| P1Minerois (🔴) | 3,38 (déjà favorable) | 14 | 30 |
| Raeren (🔴) | 3,51 (déjà favorable) | 8 | 36 |

**Explication mécanique confirmée** : la méthode "grille" (non corrigée,
`_whistle_score_at()`) n'introduit pas un bruit aléatoire — elle
introduit un **biais systématique différent** de la méthode précise.
Sur les cas positifs, ce biais pénalisait à tort le candidat cible
(valeur anormalement basse) ; la correction le remet à sa vraie place,
plus favorable. Sur les cas négatifs, ce même biais **favorisait par
accident** le candidat cible (valeur déjà dans la plage des vrais
positifs) ; la correction le ramène à sa vraie valeur, moins favorable.

**Ce n'est donc pas que "whistle_score corrigé est parfois bon, parfois
mauvais" de façon imprévisible** — c'est qu'une correction uniforme ne
peut, par construction, qu'aider certains matchs et en pénaliser
d'autres, selon le signe et l'amplitude du biais initial pour chaque
candidat spécifique.

### DÉCISION FINALE

**NE PAS lancer la correction complète des 508 lignes `v2_mining` de
`features_ko_v2.csv`.** Une correction uniforme améliorerait 3-4 matchs
et en dégraderait 3, de façon difficile à prédire à l'avance sans
recalculer individuellement chaque ligne — ce qui, de toute façon,
n'est plus une "correction de feature" mais une reconstruction complète
du dataset, hors du périmètre de ce chantier initial.

**Ce chantier est clos.** `notebook_correction_whistle_v2_mining.py`
reste disponible mais **ne doit pas être lancé** sur la base des
résultats actuels.

### Principe méthodologique retenu pour la suite du projet

> Une modification n'entre en production que si elle démontre une
> amélioration robuste sur **plusieurs seeds** (pas un seul tirage) et
> **sans dégradation systématique** d'autres matchs. Un résultat
> spectaculaire sur un seul `random_state` (ex. Raeren 17→54, P1Minerois
> 11→7) peut être complètement trompeur — toujours tester la stabilité
> avant de conclure à un effet causal.

---

## 14. STRATÉGIE ADAPTATIVE TOP-30 — VALIDÉE, PRÊTE À L'USAGE

**Contexte** : suite à la clôture du chantier `whistle_score` (section 13),
la piste retenue pour les 5 cas `RF_MAUVAIS` restants est d'élargir le
pool de candidats envoyés à Gemini (top-10 → top-30), en s'appuyant sur
le tri chronologique + arrêt anticipé déjà existant dans
`detecter_ko_par_vision()` pour ne pas payer un surcoût inutile.

### Mécanisme construit et testé (`detect_ko_vision.py`)

- **`candidats_deja_evalues`** ajouté à `detecter_ko_par_vision()` : permet
  de réutiliser des résultats déjà obtenus, sans jamais rappeler Gemini
  dessus.
- **`detecter_ko_par_vision_adaptatif()`** : nouvelle fonction à 2 passes —
  top-10 d'abord (coût inchangé si ça suffit), élargissement à top-30
  **seulement si la passe 1 échoue** (`confidence='faible'`,
  `ambiguity=True`), en réutilisant les 10 premiers résultats sans
  aucun appel redondant.
- **Testé par mocks** : aucun surcoût quand le top-10 suffit ; aucun
  appel dupliqué lors de l'élargissement (vérifié précisément,
  candidat par candidat).

### Fiabilité du top-30 mesurée sur les 5 cas `RF_MAUVAIS` (30 seeds, sans Gemini)

| Match | Taux top-30 | Décision |
|---|---|---|
| P1Minerois | 100% | Déjà résolu par le top-10 lui-même — n'était pas vraiment `RF_MAUVAIS` (son classement initial venait d'un seul tirage `random_state=0`) |
| Andrimont | 73% | ✅ Élargissement utile |
| Juprelle | 70% | ✅ Élargissement utile |
| Wanze | 50% | ⚠️ Gain incertain, à tester quand même |
| Raeren | 37% | 🔴 Probablement non résolu même en top-30 |

### Vérification de la vraie borne de production (1800s, pas 1400s)

Andrimont étant une vidéo de match complet, un test direct à
`DURATION_S=1800` (au lieu de la découpe artificielle à 1400s utilisée
dans tous les tests précédents) a confirmé que la stratégie se
généralise bien : 105 candidats générés (vs 92 à 1400s, +14% modeste),
**taux top-30 strictement identique (73%)**.

### Statut

**Prêt pour un test en conditions réelles** (vrais appels Gemini) sur
Andrimont, Juprelle et Wanze — Raeren restera probablement non résolu
quelle que soit la taille du pool, ne pas y consacrer d'appels
supplémentaires en espérant un miracle.

---

## 15. PRÉ-FILTRE VISUEL À 1 IMAGE — TESTÉ ET ABANDONNÉ

### Idée testée

Ajouter un filtre bon marché (1 seul appel Gemini, sur l'image
`juste_avant` seule) **avant** le vote complet à 3 images, pour rejeter
à bas coût les candidats clairement hors sujet (tir, remise en jeu
localisée, etc.) sans payer les 2-3 appels du vote complet dessus.

### Vérification visuelle préalable (avant tout code)

Inspection de vraies images déjà extraites dans ce projet :
- **Wanze** (vrai KO, t=984s) et **Juprelle** (vrai KO, t=546s) :
  motif net — deux équipes séparées de part et d'autre de la ligne
  médiane, arbitre visible en jaune fluo, joueurs immobiles.
- **Wanze** (faux candidat, échauffement) : motif clairement absent —
  une seule équipe visible, pas d'arbitre, cadrage excentré.
- **Point de vigilance identifié** : sur **Raeren** (candidat t=408s,
  correspondant au vrai KO mais décalé de quelques secondes), **le
  ballon est invisible** malgré une formation par ailleurs nette. Le
  ballon ne doit donc **jamais** être un critère bloquant du filtre —
  leçon intégrée dans le prompt avant tout test.

### Implémentation (`detect_ko_vision.py`)

- **`PROMPT_PREFILTRE_RAPIDE`** — prompt à 1 image, délibérément
  permissif ("en cas de doute, réponds oui"), ballon explicitement non
  requis.
- **`demander_gemini_prefiltre()`** — 1 appel, ne bloque jamais en cas
  d'échec technique (retourne `None`, traité comme "laisser passer").
- **Intégré dans `_evaluer()`** via `utiliser_prefiltre` (défaut
  `False` — aucune régression du comportement existant).
- Testé par mocks : économie de coût confirmée, sécurité en cas
  d'échec confirmée, non-régression du défaut confirmée.

### Résultat du benchmark en conditions réelles (3 matchs, vrais appels Gemini)

| Match | Résultat sans filtre | Résultat avec filtre | Identique ? | Faux rejet de sécurité |
|---|---|---|---|---|
| Wanze | 323,984 | 323,984 | ✅ | 0 |
| Juprelle | 640,112 | 1315,504 | ❌ | 1 (t=640,112) |
| Andrimont | 109,232 | 113,360 | ❌ | 1 (t=118,528) |

**2 faux rejets de sécurité sur 3 matchs**, malgré un prompt conçu pour
être permissif. Sur 2 des 3 matchs, le résultat final change — le
pré-filtre a activement fait perdre le bon candidat, pas juste raté un
gain d'économie.

### Interprétation

Même avec des instructions explicites de permissivité, **le modèle
applique un jugement plus strict que demandé sur une image isolée** —
cohérent avec un principe déjà observé plusieurs fois dans cette
investigation (RF instable sur 1 seed vs 30, whistle_score isolé
trompeur vs test complet) : **un jugement basé sur un seul point de
donnée est structurellement moins fiable qu'un jugement basé sur
plusieurs**. Le vote à 3 images fonctionne précisément parce qu'il
combine plusieurs vues ; en retirer 2 pour économiser du coût casse
cette robustesse.

### DÉCISION

**Pré-filtre abandonné, non intégré au pipeline de production.** Ne
pas retenter de variante corrigée du prompt — le taux d'échec (2/3) est
trop élevé pour espérer une correction fiable sans un nouveau cycle de
test coûteux, et ça reproduirait le pattern déjà évité avec
`whistle_score` (chaque nouvelle variante semble prometteuse jusqu'à
preuve du contraire).

> ⚠️ **DÉCISION SUSPENDUE — VOIR SECTION 19.** Ce test comparait le
> pré-filtre à une référence ("timestamp sans pré-filtre") produite par
> le vote complet utilisant le prompt "histoire globale" — prompt
> depuis identifié comme défaillant sur `gemini-2.5-flash` (section 19).
> Vérification faite : les deux "faux rejets de sécurité" (Andrimont
> t=109,232 ; Juprelle t=640,112) étaient eux-mêmes des candidats loin
> du vrai KO (307,0 et 548,0 respectivement) — probablement de **vrais
> faux positifs produits par la référence elle-même**, pas de vrais
> rejets à tort du pré-filtre. **Cette conclusion n'est donc plus
> fiable dans un sens ou dans l'autre** et doit être retestée avec le
> prompt checklist restauré avant toute décision définitive sur le
> pré-filtre.

**Ce qui reste valide indépendamment** : le mode adaptatif top-30
(section 14) n'a jamais dépendu du pré-filtre pour fonctionner — il
reste utilisable seul, avec son propre mécanisme d'économie (arrêt
anticipé chronologique), sans cette optimisation supplémentaire.

---

## 16. MOTIF C (FORMATION VISUELLE) — TESTÉ ET ABANDONNÉ

### Idée testée

Suite à une observation visuelle qualitative (un petit groupe de 2-6
personnes hors-équipe — arbitre/gardien —, séparé de deux groupes de
joueurs, proche du centre, est visible sur la plupart des vrais KO),
tester si un score calculé **directement sur les positions et labels
d'équipe trackés** (`frames_data`, sans RF, sans Gemini, sans nouvelle
feature audio) pouvait classer le vrai KO parmi les meilleurs candidats
de façon indépendante.

**Clarification technique actée avant le test** : `frames_data` n'a
jamais été sauvegardé dans ce projet — ce test nécessite un vrai
passage `process_video()` (YOLO) par match, pas un simple
post-traitement de CSV. `torch==2.10.0+cu128` ajouté au pinning
(oubli initial corrigé), cohérent avec la leçon de la section 10.

### 6 composantes calculées par candidat

`separation_equipes` (labels d'équipe réels, pas juste `x<mid_x`),
`groupe_hors_equipe_plausible` (taille 2-6), `proximite_centre`,
`compacite_centrale`, `immobilite`, `symetrie` — score final = moyenne
des 6, normalisées 0-1. Logique validée par tests unitaires sur des
cas simulés contrôlés avant tout lancement réel.

### Résultat sur les 9 matchs (vrai passage YOLO complet)

| Match | Écart candidat | Rang motif C | Rang RF (rappel) |
|---|---|---|---|
| Andrimont | 0,50s | 5/92 ✅ | 29,6 (RF_MAUVAIS) |
| Franchimont | 0,61s | 3/90 ✅ | 1,0 (excellent) |
| Stembert | 4,00s | 3/90 ✅ | 11,1 (cas frontière) |
| Juprelle | 8,00s | 12/81 | 43,5 (RF_MAUVAIS) |
| Wanze | 6,00s | 15/83 | 36,1 (RF_MAUVAIS) |
| MineroisSter | 0,97s | 32/86 ❌ | 3,3 (excellent) |
| P1Minerois | 3,89s | 77/93 ❌❌ | 7,6 (cas frontière) |
| Raeren | 5,17s | 69/86 ❌❌ | 33,1 (RF_MAUVAIS) |
| Goe | 11,0s | 26/81 (non interprétable, `MINING_MAUVAIS`) | — |

**Bilan (8 matchs interprétables, Goe exclu)** : 3/8 en top-5 (37,5%),
loin du seuil de 7/9 fixé à l'avance. **Aucune corrélation exploitable
avec la performance du RF** sur le même match — bon sur des cas où le
RF est mauvais (Andrimont) ET sur des cas où le RF est déjà excellent
(Franchimont), mauvais sur d'autres cas où le RF est mauvais (Raeren)
ET sur des cas où le RF est déjà excellent (MineroisSter). Pas de
motif de compensation sélective identifiable.

### Vérification complémentaire (post-traitement, sans coût YOLO) : rang par composante individuelle

Aucune des 6 composantes prise isolément ne dépasse le score combiné :
`separation_equipes`/`groupe_hors_equipe_plausible`/`compacite_centrale`
plafonnent à 3/9 en top-5, `symetrie` à 1/9, `proximite_centre` et
`immobilite` à 0/9. **Confirme que le problème est le signal
sous-jacent, pas la pondération** — une meilleure combinaison des 6
composantes n'aurait pas sauvé le résultat.

### DÉCISION

**Motif C abandonné**, avec la même rigueur que `whistle_score`
(section 13) : hypothèse testée honnêtement sur les 9 matchs, échec
confirmé à deux niveaux (score combiné ET composantes individuelles),
aucune nouvelle variante de calcul à chercher sans preuve nouvelle.
Observation humaine qualitativement séduisante, mais qui ne se traduit
pas en signal quantitatif exploitable avec cette implémentation.

---

## 17. FUSION MULTI-SIGNAUX (GEO/SPEED/WHISTLE/MOTIF_C) — TESTÉE ET FERMÉE

### Idée testée

Suite à l'observation qu'un signal peut être excellent sur un match et
mauvais sur un autre (`whistle_score` : très bon sur Andrimont, très
mauvais sur Raeren ; motif C : très bon sur Andrimont/Stembert, très
mauvais sur MineroisSter/P1Minerois), tester si la **fusion** de
plusieurs signaux indépendants peut faire émerger le vrai KO même
quand aucun signal seul n'est fiable sur un match donné — sans
modifier le RF, sans nouveau traitement vidéo, sans Gemini.

### 4 stratégies testées, sur les 4 signaux déjà produits (geo, speed, whistle, motif_c)

- **A. Moyenne simple** des percentiles de rang.
- **B. Moyenne des 2 meilleurs signaux** (ne pénalise pas un candidat à
  cause d'un seul signal mauvais).
- **C. Consensus** (nombre de signaux plaçant le candidat dans le
  top-5/10/20).
- **D. Poids appris** par régression logistique, en **leave-one-match-out
  strict** (même discipline que le RF tout au long du jalon).

### Résultat sur 8 matchs interprétables (Goe exclu, `MINING_MAUVAIS`)

| Stratégie | Taux top-10 |
|---|---|
| RF (déjà connu) | **38% (3/8)** |
| C-consensus5 | 38% (3/8) — égalité, pas mieux |
| A-moyenne, B-top2, C-consensus10, C-consensus20 | 25% (2/8) |
| D-poids appris | **12% (1/8) — pire que tout le reste** |

**Aucune stratégie de fusion ne dépasse le RF.** La meilleure
(consensus-5) l'égale sans le battre. La plus sophistiquée (poids
appris, leave-one-match-out) est la pire de toutes.

### Détail intéressant, sans changer la conclusion

Les poids appris par régression sont étonnamment stables entre les 8
folds (`speed` toujours négatif, `whistle`/`motif_c` toujours positifs,
amplitude 2-4) — une vraie régularité, pas du bruit. Mais cette
régularité ne se traduit pas en meilleure performance de classement :
le signal disponible dans ces 4 features est probablement trop faible
ou trop bruité pour être exploité, même avec des poids "corrects".

### DÉCISION, AVEC NUANCE IMPORTANTE

**Fusion fermée** — pas de nouvelle formule à chercher sans preuve
nouvelle, même discipline que les pistes précédentes.

**Nuance à ne pas perdre** : ce résultat démontre uniquement que *"une
fusion alternative basée sur ces 4 signaux ne fait pas mieux que le
RF"* — **pas** que *"le RF n'a aucun problème"*. Le RF utilise 11
features complètes (dont `pct_immobile`, `synchronisation_std_s`,
`ball_score`, `stabilite_avant_s`, `activite_apres`, absentes de cette
expérience de fusion à 4 signaux). Ne pas sur-interpréter ce résultat
comme un blanc-seing donné au RF actuel.

### Ce que ça confirme pour la suite

Le vrai levier restant n'est pas d'améliorer le classement fin du RF
(6 pistes déjà fermées : `whistle_score`, `stabilite_avant_s` via
vérification directe, pré-filtre, motif C, fusion) — c'est d'accepter
que **le RF n'a pas besoin d'être parfait**, seulement suffisamment bon
pour ne pas jeter le bon candidat hors d'un pool que Gemini peut
ensuite départager. D'où la priorité complète sur le **mode adaptatif
top-30** (section 14), seule piste restante jamais encore testée en
conditions réelles.

---

## 18. DÉCOUVERTE MAJEURE — LE PROBLÈME N'EST PAS LE POOL, C'EST LA STRATÉGIE DE PARCOURS

### Contexte : le premier test top-30 semblait être un échec total

Test réel (vrais appels Gemini) sur Andrimont, Juprelle, Wanze — top-30
forcé, résultat catastrophique : **0/3 récupérés, 2/3 aggravés**
(Juprelle 92s→356s, Wanze 662s→897s d'écart).

### Diagnostic précis avant toute conclusion hâtive

Plutôt que d'accepter ce résultat comme "le top-30 ne marche pas",
diagnostic détaillé candidat par candidat (chronologique), avec 3
catégories définies à l'avance :
- **Cas A** : le vrai KO n'est jamais dans le pool → problème mining/RF
- **Cas B** : le vrai KO est dans le pool mais jamais évalué (arrêt
  anticipé avant de l'atteindre) → problème de stratégie de parcours
- **Cas C** : le vrai KO est évalué mais rejeté → problème de vision

### Résultat : 3/3 en Cas B — le vrai KO n'a JAMAIS été examiné

| Match | Rang RF vrai KO | Candidat ayant déclenché l'arrêt | Timestamp final erroné |
|---|---|---|---|
| Andrimont | 26/30 | t=113,36s | 109,23s |
| Juprelle | 18/30 | t=116,0s | 116,0s |
| Wanze | 30/30 | t=89,25s | 89,25s |

**Motif frappant** : les 3 faux positifs qui ont déclenché l'arrêt
tombent tous dans une fenêtre étroite (**89-118 secondes**, début de
match) sur 3 vidéos différentes — probablement des **formations
d'avant-match** (présentation d'équipes, échauffement organisé) que
Gemini confond avec un vrai coup d'envoi, exactement le type de
confusion visuelle qu'on avait anticipé avec le motif C (section 16),
mais cette fois commise par Gemini lui-même, pas par une heuristique
simplifiée.

### Ce que ça change radicalement

**Le problème n'est ni le RF** (le vrai KO était bien dans le pool,
rang 18-30) **ni la qualité du jugement de Gemini sur le vrai KO**
(jamais testé — impossible de savoir s'il aurait été bien classé,
puisqu'il n'a jamais été vu). Le seul coupable identifié est la
**politique "premier positif trouvé = arrêt de la recherche"**, qui
laisse le système s'arrêter sur un faux positif précoce avant même
d'atteindre le bon candidat, plus tardif dans le match.

### Nouveau paramètre diagnostique ajouté (`detect_ko_vision.py`)

`desactiver_arret_anticipe` (défaut `False`, comportement de
production totalement inchangé) — permet de forcer l'évaluation de
**tous** les candidats du pool, sans s'arrêter au premier positif.
Testé par mocks : non-régression du défaut confirmée, nouveau mode
confirmé fonctionnel.

### Test en cours — le vrai KO est-il accepté quand on lui donne sa chance ?

Script préparé (`notebook_test_sans_arret_anticipe.py`) pour répondre à
la question qui décidera de la suite : si on désactive l'arrêt
anticipé, Gemini accepte-t-il correctement le vrai KO une fois qu'il
est enfin examiné ? Si oui sur au moins 2/3 matchs → confirmation
totale que le problème est uniquement la stratégie de parcours, pas la
vision — la solution devient : collecter tous les positifs du pool
plutôt que de s'arrêter au premier, puis choisir parmi eux (par
exemple le plus tardif, ou celui avec le rang RF le plus élevé parmi
les positifs). Si non → le problème est plus profond, la vision
elle-même confond aussi le vrai KO avec d'autres moments.

**Statut : résultat à venir.**

---

## 19. ⚠️ PROMPT GEMINI RESTAURÉ EN VERSION "CHECKLIST" — LEÇON DÉJÀ OUBLIÉE UNE FOIS, À NE JAMAIS RÉOUBLIER

### Ce qui s'est passé

En creusant les faux positifs découverts en section 18 (t=89-118s,
probablement des formations d'avant-match), l'utilisateur a eu
l'intuition suivante : *"je pense qu'il ne raconte pas l'histoire d'un
KO mais qu'il regarde individuellement chaque image... on en avait déjà
parlé dans des tests précédents de ça, voir jalon peut-être"*.

**Vérification faite dans les transcripts d'une session antérieure**
(`2026-08-15-12-58-38-scoutia-ko-detection-vision-pipeline.txt`) :
cette intuition était juste, et le sujet avait déjà été testé
empiriquement, avec un résultat clair et une décision prise —
**décision qui avait ensuite été perdue**, le prompt "histoire globale"
s'étant retrouvé dans `detect_ko_vision.py` sans qu'aucune trace de cet
historique ne subsiste dans le code ni dans ce jalon.

### Ce que le test antérieur avait établi

Deux versions du prompt Gemini ont été comparées sur `gemini-2.5-flash`,
même modèle, mêmes 3 matchs (Andrimont, Stembert, MineroisSter), seule
variable changée :

- **Version "checklist"** (originale) : vérifie 3 critères stricts
  image par image (image1 = statique, image2 = frappe, image3 =
  dispersion).
- **Version "histoire globale"** (reformulée ensuite) : demande de
  juger les 3 images comme un seul récit cohérent, sans vérification
  stricte critère par critère.

| Match | Checklist (original) | Histoire globale (testé) |
|---|---|---|
| Andrimont | ~2-3 faux positifs/10 | **5 faux positifs/10** |
| Stembert | ~2 faux positifs/10 | **4 faux positifs/10, + vrai KO lui-même raté** |
| MineroisSter | ~3 faux positifs/10 | **7 faux positifs/10** |

**Le taux de faux positifs a quasiment doublé** avec la version
"histoire globale". Cause identifiée à l'époque : assouplir le critère
donne au modèle la permission implicite de rationaliser presque
n'importe quelle action de jeu (tir, contre-attaque) comme "cohérente
avec un début de match", puisque la plupart des actions ont un moment
où plusieurs joueurs avancent ensemble. Un signal encore plus
préoccupant avait été noté : sur Stembert, le vrai KO était rejeté avec
un raisonnement **hallucinant une incohérence temporelle qui n'existait
pas** dans les images — pas juste une erreur de jugement, une
confabulation complète.

**Testé aussi sur `gemini-3.1-pro-preview`** : effet inverse (modèle
devenu trop conservateur, rejetant 2/3 des vrais KO) — **le même prompt
produit des comportements opposés selon le modèle**, confirmant que
l'effet n'est jamais neutre et doit être re-testé spécifiquement par
modèle.

**Décision prise à l'époque** : revenir au prompt "checklist" comme
référence stable, et envisager `gemini-3.1-pro-preview` comme piste
séparée si besoin de plus de capacité — **pas** assouplir le prompt.

### Portée de cette découverte — ce qui est affecté, et ce qui ne l'est pas

**Non affecté** (aucun appel Gemini impliqué) : `whistle_score`
(sections 12-13, purement RF), motif C (section 16, calculé directement
sur `frames_data`), fusion multi-signaux (section 17, combine des
signaux RF/audio/positionnels). Ces conclusions restent valides.

**Affecté, à des degrés différents** :
- **Section 18 (top-30, early stop)** : directement et entièrement
  concerné — c'est ce test qui a déclenché cette découverte.
- **Section 15 (pré-filtre) — à retester, décision suspendue.** Les 2
  "faux rejets de sécurité" qui avaient motivé l'abandon du pré-filtre
  étaient des candidats (Andrimont t=109,232 ; Juprelle t=640,112) très
  éloignés du vrai KO (307,0 et 548,0) — ce sont donc probablement des
  **faux positifs produits par la référence elle-même** (le vote
  complet avec le prompt défaillant), pas de vrais rejets à tort du
  pré-filtre. Cette conclusion doit être retestée avec le prompt
  checklist restauré avant toute décision définitive.

### Ce qui explique probablement les faux positifs de la section 18

Le pipeline testé dans cette session-ci (`detect_ko_vision.py` avant
correction) utilisait `gemini-2.5-flash` **avec** le prompt "histoire
globale" — exactement la combinaison invalidée par le test antérieur.
Les faux positifs à t=89-118s (probablement des formations
d'avant-match) sont cohérents avec ce mécanisme : le prompt permissif
laisse Gemini rationaliser une scène ambiguë comme le début d'un récit
de coup d'envoi.

**Conséquence importante** : le diagnostic de la section 18 ("le
problème est la stratégie de parcours / early stop") n'est peut-être
pas la cause profonde — ou seulement une partie. La vraie cause
probable est plus simple : **le prompt lui-même est trop permissif sur
Flash**, et n'importe quelle stratégie de parcours aurait fini par
accepter un faux positif quelque part.

### Action prise

`detect_ko_vision.py` restauré avec le prompt "checklist" original
(texte exact retrouvé et vérifié caractère par caractère dans les
transcripts). **Un avertissement explicite et détaillé a été ajouté
directement dans le code source**, juste au-dessus de la définition du
prompt, pour qu'un futur oubli soit structurellement impossible — même
sans relire ce jalon. Le reste du fichier (logique de sélection
chronologique + arrêt anticipé, pré-filtre, mode adaptatif top-30,
`desactiver_arret_anticipe`) reste inchangé — ces évolutions sont
indépendantes du choix checklist/histoire globale.

### Prochaine étape

Refaire le test de la section 18 (early stop, top-30 forcé sur
Andrimont/Juprelle/Wanze) **avec le prompt checklist restauré**, pour
voir si les faux positifs précoces disparaissent. Si oui, la
"découverte" de la section 18 doit être requalifiée : ce n'était pas
(uniquement) un problème de stratégie de parcours, c'était le prompt.
Si les faux positifs persistent même avec le prompt checklist, alors le
diagnostic de la section 18 (stratégie de parcours) reste valide en
plus de ce problème de prompt.

### RÉSULTAT DU RETEST — les deux problèmes sont réels et se cumulent

Test relancé (`notebook_test_sans_arret_anticipe.py`, arrêt anticipé
désactivé, prompt checklist restauré) sur les 3 mêmes matchs.

**Ce que le verdict automatique du script dit** ("2/3 vrai KO accepté")
est incomplet — il répond à *"le vrai KO est-il accepté quand on
l'évalue"*, pas à *"un faux positif existe-t-il encore AVANT le vrai KO
chronologiquement"* (la vraie question pour l'arrêt anticipé en
production).

| Match | Faux positif AVANT le vrai KO ? | Vrai KO accepté ? | Diagnostic |
|---|---|---|---|
| Andrimont | ✅ Oui, nouveau faux positif à t=194,9s (les anciens t=109/113 sont maintenant correctement rejetés) | ✅ Oui (307,5s) | Prompt améliore, mais arrêt anticipé échouerait ENCORE (faux positif différent, toujours avant) |
| Wanze | ✅ Oui, faux positif à t=89,25s persiste (2/3, inchangé) | ✅ Oui (980,0s) | Idem — le prompt n'a pas éliminé ce faux positif précis |
| Juprelle | ❌ Non (aucun faux positif avant 556s) | ❌ **Non — rejeté (0/2)** | Problème DIFFÉRENT : vrai rejet du bon candidat (cas C), pas un problème de prompt ni d'arrêt anticipé |

### Conclusion révisée — les deux causes se cumulent, aucune n'est seule suffisante

**Corriger le prompt seul ne suffit pas.** Sur Andrimont et Wanze,
même avec le prompt checklist restauré, un faux positif chronologique
antérieur au vrai KO subsiste — la politique "premier positif = stop"
(section 18) reste un vrai problème indépendant à corriger. Sur
Juprelle, un problème différent apparaît : Gemini rejette le bon
candidat lui-même, peut-être à cause de son décalage temporel de 8s par
rapport au vrai KO (l'image "juste_avant" pourrait déjà montrer le jeu
en cours plutôt que la formation statique attendue par le prompt).

**Action combinée nécessaire** :
1. Revoir la logique de décision finale pour ne plus s'arrêter au
   premier positif (collecter tous les positifs du pool, choisir selon
   un critère plus robuste — cf. section 18).
2. Investiguer séparément le cas Juprelle (candidat à 8s du vrai KO,
   possible désalignement image/prompt) — pas résolu par la correction
   du prompt.

---

## 20. CE QUE CE DOCUMENT NE COUVRE PAS

- La feuille de route des marqueurs temporels complets du match (fin de
  mi-temps, KO2, prolongations, tirs au but) — voir
  `ANALYSE_MARQUEURS_TEMPORELS_MATCH.md`, document séparé, non affecté
  par cette investigation (dépend de KO1 stabilisé en amont).
- Toute décision définitive sur Andrimont/P1Minerois comme "échecs
  acceptés" — dépend du résultat de la section 8, pas encore tranché.
