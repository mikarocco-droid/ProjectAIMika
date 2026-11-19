Voici un résumé complet à copier-coller dans la nouvelle conversation :

Résumé — Suite d'investigation sur le cas Raeren (après le jalon section 25)

Contexte : le jalon documente jusqu'à la section 25 (test d'intégration complet, Andrimont réussi, Raeren en échec). Voici tout ce qui s'est passé depuis, en résumé chronologique.

1. Élimination de fausses pistes
Chronologie vérifiée : les corrections de bugs sur overlay.py/tracker.py/player_reid.py précèdent la construction de features_ko_v2.csv — donc pas la cause de l'écart observé.
Config ffmpeg : testé fps natif vs -vf fps=4 (config originale) — la config originale a été restaurée comme référence, sans que ça résolve le problème.
2. Découverte du vrai problème : le mining en mode aveugle échoue sur Raeren
Relancé mine_candidates(true_kickoff_t=None, ...) (mode aveugle, comme en vraie prod) sur Raeren.
Le candidat le plus proche du KO connu (référentiel = 410,0s) atterrit en rang 10/10 à t=404,8s, avec une proba RF de seulement 0,015 — extrêmement faible.
3. Découverte d'une erreur de référentiel (~1s)
En observant les images, le vrai instant du KO est plutôt ~409s, pas 410s comme indiqué dans le référentiel historique du projet.
4. Autopsie de mean_speed_change_px_s (feature clé du RF)
Reproduit exactement _transition_synchro() de ko_features.py avec instrumentation complète (par joueur suivi).
Pas de problème de filtre de couverture (15-16 joueurs sur ~22 passent le seuil).
Test d'agrégations (médiane/moyenne/P75/P90/P95/max) sur 390-420s : aucune métrique ne montre de pic isolé net — plutôt un plateau élevé de 406 à 410s (cohérent avec le fait que le KO réel est ~409, pas 410).
5. Test contrôlé ±2 vs ±4 (fenêtre d'images)
⚠️ Un premier tableau comparatif a été mal lu par moi puis corrigé : ±4 améliore bien Stembert (retrouve le vrai KO avec ambiguïté honnête, alors que ±2 se trompait avec fausse confiance) sur le top-10 avec injection.
Mais le vrai test décisif (mining aveugle + ±2 vs ±4 dans la même exécution) a montré que ±4 ne change rien sur Raeren : le candidat à 404,8s reste False dans les deux cas, contrat final = 352,0s (repli, confidence faible, ambiguity=True).
6. Découverte de la cause racine probable
Deux métriques différentes dans le pipeline : génération de candidats (_window_features, médiane brute) vs scoring RF (_transition_synchro, delta par joueur).
Sur Raeren 390-420s, _window_features ne montre aucun pic isolé — juste une rampe continue croissante (26,4 à 390s → 59,9 à 420s). Le vrai pic retenu par _peaks() dans cette zone est t=400,0s, pas 404,8 (le candidat final vient probablement d'une fusion avec une autre source — sifflet ou événement).
Conclusion actuelle : sur ce match précis, le mining n'a jamais eu de bon signal ponctuel à proposer — le jeu s'intensifie progressivement sans rupture nette détectable par ces métriques (similaire au cas Franchimont, déjà documenté comme limite connue).
7. Piste en cours, résultat en attente
Hypothèse : si le signal est une rampe continue plutôt qu'un pic, ajouter une image à contexte étendu (t+10, +15, +20) pourrait permettre à Gemini de voir que le jeu est "pleinement lancé" plus tard, confirmant indirectement qu'une transition a bien eu lieu vers cette zone.
Notebook envoyé pour extraire 6 images (t-2/0/+2/+10/+15/+20 autour de 404,8s) — résultat pas encore reçu/analysé, c'est la prochaine étape.

Fichiers utiles à ramener dans la nouvelle conversation : detect_ko_vision.py (version actuelle, tri chronologique pur restauré + paramètre offset_s configurable), candidates_mining.py (avec la borne DUREE_MAX_RECHERCHE_KO_S), et le jalon complet pour le contexte global.


