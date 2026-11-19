# Protocole de comparaison — Gemini vs autres IA sur l'image P1Minerois (t=3652s)

## Contexte

Sur cette image précise, `Gemini` (température=0, prompt identique) a donné **deux verdicts opposés** à deux moments différents : `OUI` (accepté comme candidat KO2) puis `NON` (avec un raisonnement correct rejetant la scène). Objectif : voir si un autre modèle est plus cohérent sur ce même jugement, avec exactement le même prompt et la même image.

## Ce qu'il faut utiliser

**Image** : `1788759400167_image.png` (déjà dans tes fichiers uploadés) — l'image à `t=3652s` du match P1Minerois.

**Prompt exact** : fichier `prompt_q1_ko2_exact.txt` ci-joint — copie-colle-le tel quel, sans le modifier, pour que la comparaison soit honnête.

## Protocole recommandé

1. **Envoie l'image + le prompt à l'autre IA, 5 à 10 fois**, dans des conversations **séparées** (pas dans la même conversation à la suite, pour éviter que le modèle se souvienne de sa réponse précédente et se contente de la répéter).
2. **Note à chaque fois** : les 5 critères (`true`/`false`) et la décision finale recalculée (`≥3/4` sur les critères `1-4` **ET** `deux_equipes_visibles=true`).
3. **Compte le taux d'accord** : sur `10` essais, combien disent `OUI` vs `NON` ? Si c'est `10/10` identique → le modèle est plus cohérent que `Gemini` sur ce cas. Si c'est mitigé (par exemple `6/10` vs `4/10`) → l'ambiguïté est probablement dans l'image elle-même, pas spécifique à `Gemini`.

## Tableau à remplir

| Essai | zone_centrale | avant_match | amorce_separation | pas_autre_remise | deux_equipes | Décision finale |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

**Taux de OUI : __/10**

## Repère — ce que Gemini a donné sur ce point précis

| Essai | Décision | Raisonnement résumé |
|---|---|---|
| 1 (dans la cascade réelle) | OUI | (pas de raisonnement capturé, décision brute) |
| 2 (diagnostic dédié) | NON | "joueurs regroupés autour de l'arbitre... préparation d'un coup franc... aucune amorce de séparation" |

## Si tu testes avec l'API (pas juste l'interface web)

Si l'autre fournisseur a une API avec un paramètre de température, mets-le aussi à `0` pour une comparaison honnête (`Gemini` était à `0.0` dans nos tests) — sinon la comparaison ne serait pas équitable (un modèle à température non nulle aura presque toujours plus de variance, ce ne serait pas un vrai test de fiabilité du modèle lui-même).

## Ce qu'on fera du résultat

- **Si l'autre modèle est nettement plus stable** (proche de `10/10` dans un sens) → ça vaudra la peine d'évaluer un changement de modèle pour cette tâche précise, au moins pour `KO2`.
- **Si l'autre modèle est aussi partagé que Gemini** → ça confirme que c'est une vraie ambiguïté de la scène, pas une faiblesse spécifique à `Gemini`, et on documente la limite plutôt que de chercher encore un modèle "parfait".
