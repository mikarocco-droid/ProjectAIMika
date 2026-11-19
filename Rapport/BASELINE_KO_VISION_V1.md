# BASELINE KO VISION V1 — VALIDÉE

**Statut : FIGÉE.** Ne pas modifier le prompt, le code de `detect_ko_vision.py`, ni les paramètres ci-dessous sans repasser par un benchmark complet. Cf. section "Règle de non-régression" en fin de document.

---

## 1. Configuration de référence

| Paramètre | Valeur |
|---|---|
| Modèle | `gemini-3.1-pro-preview` |
| Température | `0.0` |
| Vote | jusqu'à 5 appels indépendants, majorité stricte, arrêt anticipé généralisé (dès que la majorité est mathématiquement acquise compte tenu des appels restants — pas simplement "2 votes d'accord") |
| Coût observé sur benchmark | **3 appels/candidat dans 100% des cas** (le plafond à 5 n'a jamais été sollicité sur ce jeu de test) |
| Pré-filtre 1 image | désactivé par défaut (`utiliser_prefiltre=False`) — permissif par construction si activé (`None` ou `True` laissent toujours passer) |
| Offset images | ±2 s (`offset_s=2.0`) |
| Pool de candidats | top-10 RF (mode standard) ou top-10→30 adaptatif à 2 passes |
| Arrêt anticipé pipeline | activé — traitement chronologique, verrouillage sur le premier positif trouvé, fenêtre d'ambiguïté `seuil_ambiguite_secondes=10.0` |
| Décision par vote | `avant_compatible AND mise_en_jeu_visible AND transition_apres`, recalculée en code — jamais le `transition_visible` auto-rapporté par Gemini seul |

## 2. Contenu du prompt (résumé — voir `detect_ko_vision.py` pour le texte complet)

Le prompt actuel résulte de 6 corrections successives, chacune motivée par un faux positif concret observé :

1. **Checklist stricte** (base historique) — jamais reformuler en "histoire globale" sans re-validation complète (cf. avertissement déjà présent dans le fichier — cette erreur a été commise une fois pendant l'investigation).
2. **Anti-hallucination géométrique** — un but visible ne doit jamais être confondu avec le rond central.
3. **Vocabulaire "regroupée ≠ stabilisée"** — une équipe en cercle/cri de guerre reste rejetée même si elle semble immobile.
4. **Séparation stricte par moitié de terrain** — chaque équipe doit être de son côté, pas seulement "distincte par couleur".
5. **Cas "les deux équipes du même côté"** — rejet même si un rond central et un ballon central sont visibles.
6. **Vérifications préalables** (but visible sur image 1 OU 2 ; maillots mélangés sur image 1) — placées en tout début de prompt, avec rejet automatique sans compensation possible par le reste de la scène.

Complété par 2 corrections de code (indépendantes du texte du prompt) :
- Recalcul strict de la décision en Python (AND des 3 critères), au lieu de faire confiance au jugement auto-rapporté de Gemini.
- `temperature=0.0` — a éliminé la variance inter-runs mesurée avant son ajout (certains candidats oscillaient entre 25% et 75% de "True" selon le tirage, à prompt et images strictement identiques).

## 3. Résultat du benchmark de validation

**25 candidats de référence × 3 runs identiques = 75 évaluations.**

| Métrique | Résultat | Seuil de réussite | Statut |
|---|---|---|---|
| A. Vrais KO détectés | **9/9**, unanimité 3/3 sur chaque match, chaque run | 9/9 obligatoire | ✅ |
| B. Faux positifs **avant** le vrai KO | **0** (sur 4 candidats surveillés) | 0 obligatoire | ✅ |
| C. Faux positifs **après** le vrai KO | **0** (sur 12 candidats surveillés, y compris `Wanze 1391,47` — 100% d'échec sur les 4 runs précédant `temperature=0.0`) | toléré | ✅ (mieux qu'attendu) |
| D. Stabilité inter-runs | **25/25** décisions finales identiques sur les 3 runs | 100% visé | ✅ |
| E. Coût réel (appels/candidat) | **3,00 / 5 max** — jamais de 4ᵉ ou 5ᵉ appel sollicité | mesure informative | — |

### 3.1 Jeu de candidats de référence (test de non-régression)

| Match | t (s) | Type | Position / KO | Écart au KO (s) | Décision | Appels |
|---|--:|---|---|--:|:-:|:-:|
| Andrimont | 194,928 | piège | avant_KO | −112,07 | False | 3 |
| **Andrimont** | **307,000** | **vrai_ko** | **KO** | 0,00 | **True** | 3 |
| Andrimont | 383,776 | piège | après_KO | 76,78 | False | 3 |
| Andrimont | 614,704 | piège | après_KO | 307,70 | False | 3 |
| Andrimont | 937,408 | piège | après_KO | 630,41 | False | 3 |
| Andrimont | 980,000 | piège | après_KO | 673,00 | False | 3 |
| Andrimont | 1262,944 | piège | après_KO | 955,94 | False | 3 |
| Andrimont | 1273,504 | piège | après_KO | 966,50 | False | 3 |
| Andrimont | 1322,064 | piège | après_KO | 1015,06 | False | 3 |
| **Franchimont** | **318,000** | **vrai_ko** | **KO** | 0,00 | **True** | 3 |
| **Goe** | **1081,000** | **vrai_ko** | **KO** | 0,00 | **True** | 3 |
| **Juprelle** | **548,000** | **vrai_ko** | **KO** | 0,00 | **True** | 3 |
| Juprelle | 640,112 | piège | après_KO | 92,11 | False | 3 |
| Juprelle | 920,750 | piège | après_KO | 372,75 | False | 3 |
| Juprelle | 955,392 | piège | après_KO | 407,39 | False | 3 |
| **MineroisSter** | **723,000** | **vrai_ko** | **KO** | 0,00 | **True** | 3 |
| **P1Minerois** | **192,000** | **vrai_ko** | **KO** | 0,00 | **True** | 3 |
| **Raeren** | **410,000** | **vrai_ko** | **KO** | 0,00 | **True** | 3 |
| **Stembert** | **553,000** | **vrai_ko** | **KO** | 0,00 | **True** | 3 |
| Wanze | 89,250 | piège | avant_KO | −896,75 | False | 3 |
| Wanze | 218,608 | piège | avant_KO | −767,39 | False | 3 |
| Wanze | 818,640 | piège | avant_KO | −167,36 | False | 3 |
| **Wanze** | **986,000** | **vrai_ko** | **KO** | 0,00 | **True** | 3 |
| Wanze | 1216,000 | piège | après_KO | 230,00 | False | 3 |
| Wanze | 1391,472 | piège | après_KO | 405,47 | False | 3 |

*(Source complète, run par run : `detail_3runs.csv` — à conserver comme pièce justificative.)*

## 4. Historique — comment cette baseline a été atteinte

Point de départ (prompt "checklist" simple, avant durcissement) : **13 faux positifs sur 18 candidats testés (72%)**. Progression au fil des corrections ciblées :

| Étape | Faux positifs restants |
|---|---|
| Prompt initial | 13/18 |
| + anti-hallucination géométrique | 8/18 |
| + séparation par moitié de terrain | 5/18 |
| + cas "même côté" | 3/18 |
| + vérifications préalables (but/mélange en tête de prompt) | 1/18 |
| + `temperature=0.0` | **0/25** (jeu élargi à 9 vrais KO + 16 pièges) |

Chaque correction a été **motivée par un faux positif réel et vérifié visuellement** (jamais une hypothèse théorique), et chaque itération a été re-testée sur l'ensemble du jeu de candidats — pas seulement le cas qui l'a motivée — pour détecter les régressions (au moins 2 régressions réelles ont été détectées et corrigées en cours de route : `Andrimont 194,928` et `Juprelle 920,75`).

## 5. Règle de non-régression

**Ne pas modifier `PROMPT_KO_VISION`, `demander_gemini_ko()`, `voter_transition()`, ni les paramètres par défaut de `detecter_ko_par_vision()` / `detecter_ko_par_vision_adaptatif()` sans repasser l'intégralité du benchmark à 25 candidats × 3 runs.**

Toute modification future doit être justifiée par un échec concret observé sur un **nouveau** cas (pas une reformulation "pour améliorer" sans preuve d'échec), et validée par comparaison avant/après sur ce même jeu de référence — c'est exactement la discipline qui a permis de détecter 2 régressions invisibles autrement pendant cette investigation.

## 6. Prochaine étape — benchmark de généralisation (PAS une modification)

Le benchmark ci-dessus répond à *"les corrections n'ont-elles rien cassé de connu ?"*. Il ne répond pas à *"le système fonctionne-t-il sur des vidéos jamais vues ?"*. Prochaine phase, sur un jeu de matchs **totalement hors de ce benchmark**, en gardant strictement la configuration figée en section 1 :

1. **Vrais KO inconnus** → mesure de recall.
2. **Faux positifs avant KO** → métrique critique n°1 en production.
3. **Faux positifs après KO** → tolérés, mesurés séparément.
4. **Nombre moyen d'appels Gemini** — comparer au 3,00 mesuré ici.
5. **% de candidats nécessitant réellement 4 ou 5 votes** — jamais observé sur ce jeu ; reste à voir si ça se produit sur des cas plus ambigus.
6. **Stabilité** sur les nouveaux candidats (répéter le principe multi-runs).
7. **Écart temporel** entre le candidat détecté et le vrai KO, sur les cas où la détection réussit.
8. **Veo vs caméra manuelle** — une différence de confiance avait déjà été observée par le passé, jamais quantifiée séparément.

**Important** : ne pas toucher aux paramètres pendant ce nouveau benchmark. Si un problème apparaît sur les nouveaux matchs, il faudra d'abord établir clairement que c'est un problème de généralisation (le prompt actuel ne couvre pas un cas de figure nouveau) et non une régression due à un changement de configuration entre-temps.
