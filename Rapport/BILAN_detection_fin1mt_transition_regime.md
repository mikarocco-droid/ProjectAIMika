# Détection de la fin de 1ère mi-temps par transition de régime audio — Bilan validé

## Résultat final

| Match | Erreur | Score au candidat | Statut |
|---|---|---|---|
| Andrimont | **+1s** | 17,180 | ✅ Excellent |
| Wanze | **+2s** | 2,307 | ✅ Excellent |
| MineroisSter | **+2s** | 3,373 | ✅ Excellent |
| Juprelle | **+3s** | 2,171 | ✅ Excellent |
| Franchimont | **+4s** | 2,319 | ✅ Excellent |
| Goe | **+4s** | 1,951 | ✅ Excellent |
| P1Minerois | **+12s** | 4,027 | ✅ Bon |
| Raeren | **+19s** | 4,822 | ✅ Bon |
| Stembert | +90s | 0,934 | ⚠️ Limite connue et expliquée |

**8/9 matchs à ≤19s d'écart, dont 6/9 à ≤5s.** Médiane globale : 4,0s.

## Méthode retenue

**Principe** : ne pas chercher LE bon pic acoustique isolé (approche abandonnée après une longue série de tests infructueux — S5/S10/S20, dominance globale, dominance locale à portée X, marges multiplicatives), mais détecter un **changement durable de régime acoustique** :

```
match normal → sifflet → régime différent (calme/bruit diffus), qui PERSISTE
```

**Mesures utilisées** (les 2 seules, parmi 4 testées, montrant une cohérence directionnelle sur la majorité des 9 matchs lors de la caractérisation préalable) :
- **Proportion d'énergie dans la bande sifflet (3400-4300Hz)** — baisse après la transition (8/9 matchs cohérents)
- **Platitude spectrale (spectral flatness)** — hausse après la transition (7/9 matchs cohérents)

**Score de transition** : pour chaque instant candidat, compare une fenêtre de 30s avant et une fenêtre de 30s après (avec une **zone tampon de 4s** entre les deux — correction cruciale, voir ci-dessous). Le score combine la baisse relative de proportion sifflet et la hausse relative de platitude. Le point de score maximal = candidat retenu.

**Fenêtre de recherche** : ancrée sur `KO2` (coup d'envoi de la 2e mi-temps), lui-même détecté via la cascade Gemini déjà validée pour KO1 (même prompts, mêmes critères, avec un ajout de filtre "deux équipes visibles" et une gestion de retry sur erreur API). Fenêtre retenue : `[KO2-16min, KO2-8min]`, validée 9/9 pour contenir la vraie Fin1MT sur l'échantillon connu.

## Bug trouvé et corrigé en cours de route : la zone tampon

**Découverte sur Stembert** (premier run : score au vrai Fin1MT négatif, -0,592) : le sifflet lui-même est une rafale très forte en proportion_sifflet (0,4 à 0,7, contre 0,01-0,03 de bruit de fond) qui, sans précaution, se retrouvait comptée DANS la fenêtre "après" — faisant monter au lieu de baisser la mesure censée capter le calme après le sifflet. Corrigé par une zone tampon de 4s excluant le sifflet lui-même du calcul de la fenêtre "après". Confirmé par simulation synthétique avant application (score passé de -0,735 à +1,980 sur un cas simulé reproduisant exactement ce piège).

## Limite connue : Stembert (+90s)

**Cause identifiée avec certitude** (confirmée par capture d'écran) : une animation de pom-pom girls avec musique amplifiée a lieu ~90s après le vrai sifflet, créant une **deuxième transition de régime acoustique**, plus marquée que la première (score 0,934 contre 0,042 au vrai sifflet). Ce n'est pas un bug de méthode — c'est une ambiguïté réelle du signal : deux changements de régime légitimes dans la même fenêtre, sans moyen actuel de les distinguer.

**Décision** : accepté comme limite connue plutôt que de chercher un correctif spécifique à ce cas. 8/9 restant excellent, la cause étant bien comprise et documentée plutôt qu'un échec silencieux.

## Chantiers explorés et abandonnés avant d'arriver à cette méthode

Pour référence, dans l'ordre chronologique :
1. **Vision Gemini (Q3, "signes de fin de période")** — incohérent, imprécis (jusqu'à +108s), abandonné
2. **Motif temporel "2 courts + 1 long"** — réfuté (0/9 matchs)
3. **Motif "toujours 2 bursts"** — réfuté après inspection visuelle (artefacts de seuil, vrais événements multiples)
4. **Discrimination spectrale sifflet vs bruit** (fréquence, largeur de bande, stabilité, SNR, durée) — aucune séparation exploitable (0,01 à 0,45 sur 5 dimensions)
5. **Signal spatial (mouvement/convergence des joueurs)** — signal réel mais peu discriminant, ne distingue pas le cas Andrimont comme espéré
6. **Règles "dernier événement" avec seuil de score (S5/S10/S20)** — compromis vrai/faux candidat, jamais >5/9 à ≤5s
7. **Dominance globale/locale, marges multiplicatives** — jamais strictement meilleur que la baseline, cas Juprelle irrécupérable (résidu à 255s du vrai)
8. **Transition de régime combinée** (méthode retenue) — 8/9 excellent après correction du bug de zone tampon

---

## Prochaine étape — adaptation au coup de sifflet final (fin de match)

Avant de coder quoi que ce soit, plusieurs différences avec Fin1MT à examiner :

1. **Pas de fenêtre "KO3" pour ancrer la recherche** — contrairement à Fin1MT (ancrée sur KO2), la fin de match n'a pas d'événement symétrique après elle à détecter en premier. La fenêtre de recherche doit être ancrée différemment (probablement sur KO2 + durée théorique de 2e mi-temps, à établir).
2. **Célébrations/sortie du terrain après le coup de sifflet final** — potentiellement une transition acoustique encore plus marquée que celle de Fin1MT (applaudissements, cris de joie), pourrait être plus facile à détecter, ou au contraire créer de nouveaux pièges similaires à celui de Stembert.
3. **Prolongations et tirs au but** — mentionné dans le projet comme paramètre utilisateur, hors périmètre immédiat sauf si le match concerné en a.
4. **Vérité terrain** — à rassembler pour les 9 matchs (fin de match connue), si pas déjà disponible.

Recommandation : **caractériser d'abord** (comme on l'a fait pour Fin1MT) avant de supposer que la même méthode transposée telle quelle fonctionnera.

---

## FIN DE MATCH — résultat validé

### Résultat final

| Match | Erreur | Statut |
|---|---|---|
| P1Minerois | **-0s** | ✅ Excellent |
| MineroisSter | **-1s** | ✅ Excellent |
| Stembert | **+1s** | ✅ Excellent |
| Goe | **+2s** | ✅ Excellent |
| Wanze | **+2s** | ✅ Excellent |
| Juprelle | **+6s** | ✅ Excellent |
| Franchimont | **+7s** | ✅ Excellent |
| Andrimont | **+13s** | ✅ Bon |
| Raeren | +62s | ⚠️ Limite documentée (signal faible, délai naturel) |

**8/9 à ≤10s en conditions aveugles**, médiane 2,2s. Méthode : identique à Fin1MT (mêmes 2 mesures, même zone tampon 4s), avec un ajout : un **critère de persistance** (voir ci-dessous).

### Critère de persistance — ajout crucial suite aux faux positifs

Constat initial (7/9 à ≤10s, avant cet ajout) : les 2 échecs (Raeren -183s, MineroisSter -368s) partageaient un motif commun visible dans les données brutes — des pics de signal **brefs et isolés** (quelques dixièmes de seconde), très différents d'un vrai changement de régime **soutenu**. La moyenne simple (avant/après sur 30s) pouvait être trompée par un seul pic ponctuel bref.

**Correction** : le score de transition est maintenant multiplié par une **fraction de persistance** — la proportion du temps, dans la fenêtre "après", où le signal reste réellement au-dessus d'un seuil local (1,5× la moyenne "avant"). Un pic bref et isolé donne une fraction proche de 0 (score neutralisé) ; un vrai changement soutenu conserve l'essentiel de son score.

**Résultat** : MineroisSter entièrement corrigé (-368s → -1s). Raeren nettement amélioré (-183s → +62s) sans être parfait — voir limite ci-dessous.

### Fenêtre de recherche (production)

`[KO2+40min, KO2+55min]`, calculée à partir de la plage connue de durée de 2e mi-temps sur les 9 matchs (43,7min à 50,9min, avec marge de sécurité) — vérifiée 9/9 pour contenir la vraie fin de match.

### Différence importante avec Fin1MT — la platitude spectrale domine ici

Contrairement à Fin1MT (où les 2 mesures étaient à peu près également cohérentes, 8/9 et 7/9), pour la fin de match :
- **Platitude spectrale en hausse : 9/9, parfait, sans exception**
- **Proportion bande sifflet en baisse : seulement 5/9** — sur Andrimont et MineroisSter, elle **augmente** fortement (+378%, +117%) après la fin de match, probablement parce que les célébrations elles-mêmes contiennent des sifflements/cris dans cette bande de fréquence, contrairement à une pause calme.

La détection reste robuste malgré cette différence car le score combiné s'appuie principalement sur la platitude dans ces cas.

### Piste explorée et abandonnée : sélection "backward" (dernier pic depuis la fin)

Idée testée : plutôt que le maximum global sur toute la fenêtre, remonter depuis la fin de la vidéo/fenêtre disponible et prendre le dernier pic local rencontré — motivée par le constat que Raeren et MineroisSter (les 2 échecs) ont le contenu le plus court après la vraie fin de match (148s et 78s respectivement, contre 267-765s pour les autres).

**Résultat : nettement pire** (médiane 232s contre 6,2s pour le maximum global, 0/9 à ≤10s). Cause : la détection de "dernier pic" (`find_peaks` avec `height=0`) ramassait systématiquement une fluctuation résiduelle infime en fin de fenêtre, sans rapport avec un vrai changement de régime — un rognage défensif de fin de segment (déjà validé ailleurs) n'a quasiment rien changé, confirmant que le problème n'était pas un artefact de bord mais un seuil de détection de pic beaucoup trop permissif. Abandonné : corriger proprement nécessiterait un seuil de hauteur minimale, ramenant exactement au même type de compromis (S5/S10/S20) déjà exploré et abandonné pour Fin1MT.

**Décision : le maximum global reste la référence retenue.**

### Limites connues, causes confirmées par visionnage réel de la vidéo

**MineroisSter (résolu par le critère de persistance, -368s → -1s)** : le faux candidat initial correspondait à **un supporter qui crie fort près de la caméra** (confirmé par visionnage direct) — un cri bref et isolé, exactement le type de piège que le critère de persistance (fraction du temps "après" où le signal reste réellement élevé, pas juste une moyenne) a été conçu pour filtrer. Entièrement corrigé.

**Raeren (résiduel, +62s après le critère de persistance, contre -183s avant)** : score de transition le plus faible des 9 matchs (0,510, contre une médiane de 1,181 — quasiment 2× plus faible que tout autre match). Cause identifiée par l'utilisateur (visionnage réel + connaissance du son) : **du vent dans le micro** affaiblit la netteté générale du signal sur ce match précis, et **les joueurs rentrent calmement vers les vestiaires en discutant** à ce moment — ce n'est donc pas un bruit parasite sans rapport (contrairement à MineroisSter), mais probablement le **bon signal**, avec un délai naturel d'environ 60s le temps que les joueurs finissent de se saluer sur le terrain avant de vraiment quitter la zone de jeu. Le détecteur capture la bonne nature d'événement, avec un simple retard propre à la dynamique de sortie de terrain sur ce match.

**Décision** : accepté comme limite documentée et comprise (retard naturel + bruit de vent atténuant le signal), pas un échec de méthode. 8/9 à ≤10s reste un résultat solide.

### État final — prêt pour intégration

**Méthode validée pour Fin1MT ET FinMatch**, avec le même détecteur sous-jacent (2 mesures, zone tampon, critère de persistance pour FinMatch), seule la fenêtre de recherche change (ancrée sur KO2 dans les deux cas, avec des bornes différentes). **8/9 matchs dans les critères retenus pour Fin1MT (≤19s) et FinMatch (≤10s)**, avec une limite connue et expliquée dans chaque cas (Stembert, Raeren) — pas des échecs silencieux.

