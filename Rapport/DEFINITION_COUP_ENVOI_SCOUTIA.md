# DÉFINITION DU COUP D'ENVOI SCOUTIA

**Statut :** Document de référence, pas de code. Objectif : définir l'événement
avant de chercher le signal qui le détecte. Toutes les affirmations ci-dessous
s'appuient sur les expériences déjà menées dans ce projet (DINO/Claude,
géométrie, ballon, vitesse, rupture d'état) — aucune n'est spéculative.

---

## 1. CE QUE CHAQUE EXPÉRIENCE A DÉMONTRÉ (conditions NÉCESSAIRES, jamais SUFFISANTES)

| Condition testée | Ce qu'elle capture | Pourquoi elle échoue seule |
|---|---|---|
| **Séparation géométrique des équipes** | Les deux équipes occupent chacune leur moitié | Vraie aussi pendant un rassemblement organisé, un jeu ouvert équilibré, ou une pause — 0/10 en décision unique (audit A0) |
| **Ballon immobile / stable** | Le jeu est arrêté | Signal peu fiable sur ces vidéos (détection instable confirmée visuellement) ; de plus, vrai aussi pour un coup franc, une remise en jeu, un arrêt sur blessure |
| **Accélération collective (vitesse moyenne)** | Beaucoup de joueurs démarrent en même temps | Une contre-attaque produit une accélération collective PLUS FORTE qu'un KO (démontré sur Andrimont, Raeren, Stembert) |
| **Rupture de régime (changement d'état)** | Le comportement collectif change durablement | Un pressing, une récupération haute, un coup franc joué vite produisent aussi des ruptures durables |

**Conclusion empirique directe** : chacune de ces quatre conditions est
nécessaire (un vrai KO les présente presque toujours), mais aucune n'est
suffisante (chacune se retrouve aussi ailleurs dans le match). C'est
exactement le motif qui explique pourquoi chaque expérience a fonctionné
sur certains matchs et échoué sur d'autres.

---

## 2. CE QUI DISTINGUE UN VRAI KO DES AUTRES ÉVÉNEMENTS SIMILAIRES

### KO vs Contre-attaque
- **Contre-attaque** : l'accélération collective démarre depuis un état
  **déjà actif** (le match est en cours juste avant). Quelques joueurs
  accélèrent, pas la totalité des deux équipes.
- **KO** : l'accélération démarre depuis un état **immobile ou quasi
  immobile depuis plusieurs secondes** (pas juste l'instant d'avant — une
  fenêtre soutenue de faible activité). C'est l'ÉTAT AVANT qui différencie,
  pas l'ampleur du changement.

### KO vs Coup franc / Remise en jeu
- **Coup franc / remise** : arrêt localisé — les joueurs se regroupent
  autour d'un point précis du terrain, pas une séparation des deux équipes
  sur toute la largeur. Reprise généralement rapide (quelques secondes),
  et un seul camp initie l'action.
- **KO** : séparation des deux équipes sur **toute la largeur du terrain**,
  chaque camp dans sa moitié, pendant une durée notable (plusieurs
  secondes à quelques dizaines de secondes selon les habitudes pré-match).

### KO vs Sortie de but / Dégagement gardien
- Implique un seul joueur (le gardien) à l'initiative, pas une configuration
  symétrique à deux équipes. Le reste des joueurs n'est pas nécessairement
  organisé en formation stable.

### KO vs Reprise de mi-temps
- Géométriquement quasi identique à un KO (c'est un KO, au sens propre).
  Le signal recherché doit être **le même**. La difficulté observée
  (Juprelle notamment) n'est pas un problème de définition mais de moment
  où la vidéo commence par rapport à cet événement.

---

## 3. LA DÉFINITION PROPOSÉE — LA PREMIÈRE TRANSITION IRRÉVERSIBLE, PAS UNE SÉQUENCE D'IMAGES

**Reformulation plus juste que la version précédente** : un coup d'envoi
n'est pas trois états à reconnaître. C'est **la première transition
irréversible entre le mode "avant-match" et le mode "match"**. La nuance
compte : on ne demande plus "est-ce que cette fenêtre ressemble à un KO ?"
mais "est-ce que le système vient de changer de mode ?".

C'est ce qui distingue structurellement un KO d'une contre-attaque : une
contre-attaque est un pic **à l'intérieur** d'un régime déjà établi (le jeu
était déjà lancé avant et après). Un KO est le passage **entre deux
régimes** qui ne redevient jamais l'ancien.

```
État A — STABLE (nécessaire, ET durable, ET stable — pas juste "faible")
  - deux équipes séparées sur la largeur du terrain (geo_separation élevé)
  - activité FAIBLE ET peu VARIABLE sur plusieurs secondes consécutives
    (voir note sur la stabilité ci-dessous — un état oscillant entre faible
    et moyenne n'est PAS un état stable, même si sa moyenne est basse)

        ↓  [transition A→B, doit être SOUDAINE et COLLECTIVE]

État B — DÉPART COLLECTIF
  - une proportion importante des joueurs des DEUX équipes démarre
    quasi simultanément (pas 2-3 joueurs isolés comme une contre-attaque)

        ↓  [transition B→C, doit être IRRÉVERSIBLE]

État C — JEU ÉTABLI
  - l'activité collective reste élevée et NE RETOMBE PAS au niveau de
    l'état A dans les secondes qui suivent — c'est cette irréversibilité,
    pas l'ampleur du changement, qui exclut une contre-attaque
```

### Note sur la stabilité de l'état A (pas seulement la moyenne)

Un état "faible en moyenne mais oscillant" n'est pas un état stable :

```
Stable (valide)          Oscillant (PAS valide, même moyenne faible)
0.12  0.11  0.13  0.10    0.15  0.27  0.11  0.30  0.09  0.28
```

La condition sur l'état A porte donc sur **la moyenne ET la variance**
de l'activité, mesurées sur une fenêtre de plusieurs secondes — pas
uniquement sur un seuil de moyenne.

### Le ballon n'est PAS une condition — c'est un indice

Le ballon a été retiré des conditions ci-dessus, volontairement. Il ne
fait pas partie de la définition du KO — il est un **indice de confiance
additionnel**, à consulter seulement quand il est fiable :

| Indice bonus (jamais requis) | Renforce la confiance si présent |
|---|---|
| Ballon détecté de façon stable pendant l'état A | ✓ |
| Ballon détecté en mouvement pendant/après la transition B | ✓ |
| Ballon proche du centre du terrain pendant l'état A | ✓ (si calibration terrain disponible) |

Cette séparation est nécessaire précisément parce que `BallTracker` s'est
révélé peu fiable sur les vidéos de ce projet (faux positifs récurrents
confirmés visuellement sur Andrimont) — la définition du KO doit rester
valable même quand cet indice est absent ou erroné.

**Condition suffisante = la séquence A→B→C dans cet ordre, chacune validée
indépendamment.** C'est la raison probable pour laquelle chaque expérience
isolée de ce projet a produit des faux positifs : chacune détectait un seul
état ou une seule transition, jamais la séquence complète.

---

## 4. LA QUESTION QUI RESTE OUVERTE — ET LA PLUS IMPORTANTE

Ce document définit bien le KO lui-même (section 3). **Il ne définit pas
encore la frontière de l'état A** — c'est-à-dire comment distinguer, dans
la période qui précède le KO :

- la présentation des équipes,
- l'échauffement,
- les joueurs qui discutent entre eux,
- l'attente du signal de l'arbitre,
- **de la véritable formation de coup d'envoi**.

### Réponse empirique partielle, confirmée sur Andrimont (diagnostic visuel)

Sur Andrimont, trois "faux départs" ont été observés avant le vrai KO : à
chaque fois, les joueurs se placent dans une configuration déjà proche
d'un KO (arbitre au centre, ballon au centre, `geo_separation` élevé) —
**mais une ou les deux équipes interrompent la mise en place pour un rituel
de groupe (cri de guerre)**, et la séparation géométrique retombe.

| | Durée avant reprise | Retombe ensuite ? |
|---|---|---|
| Faux départ 1 (185-191s) | ~6s | Oui |
| Faux départ 2 (219-225s) | ~6s | Oui |
| Faux départ 3 (271-275s) | ~4s | Oui |
| **Vraie formation (275-307s)** | **~32s** | **Non — mène directement au KO** |

**La différence n'est donc pas la présence d'une bonne géométrie — les
faux départs l'ont déjà. C'est la durée de stabilité SANS interruption.**

**⚠️ Mise en garde explicite — n'a jamais été validé au-delà d'un seul
match.** Les 32 secondes observées sur Andrimont ne sont PAS un seuil à
retenir. Il est tout à fait possible qu'un autre match montre un vrai KO
précédé de 10 secondes de stabilité, ou de 60. **La durée de stabilité de
l'état avant le KO est une VARIABLE À MESURER sur plusieurs matchs, pas
une constante à fixer.** Toute règle de type "durée > X secondes ⇒ vrai
KO" serait prématurée et reproduirait exactement l'erreur déjà commise
avec `kickoff_detector.py` (seuils choisis avant d'avoir mesuré leur
distribution réelle).

matchTest, à l'inverse, ne présente pas ce genre de faux départ dans les
quelques secondes précédant son KO (t=8s) — cohérent avec le fait que ce
match ait toujours été plus simple à traiter par les approches précédentes
une fois le bon instant identifié.

**Ce qui reste à vérifier** : ce motif (faux départs courts qui retombent,
suivis d'un état stable plus long) se retrouve-t-il sur d'autres matchs
(Franchimont, Juprelle, Goe, Wanze, etc.), et avec quelle distribution de
durées ? Voir section 7 pour l'observateur construit à cet effet.

---

## 5. CE QUI RESTE À VÉRIFIER (avant tout nouveau code)

1. **La frontière de l'état A** (section 4) — priorité avant tout le reste.
2. **Durée minimale de l'état A** — combien de secondes de stabilité
   faut-il exiger avant de considérer la transition comme valide ?
3. **Proportion minimale de joueurs concernés par la transition A→B** —
   combien de joueurs doivent démarrer ensemble pour exclure une
   contre-attaque à 3-4 joueurs ?
4. **Durée minimale de persistance de l'état C** — combien de temps le jeu
   doit-il rester actif après la transition pour confirmer qu'elle est
   irréversible, et non un pic isolé ?
5. Ces durées/proportions ne doivent pas être choisies arbitrairement —
   elles devraient elles-mêmes être **mesurées** sur les cas déjà connus
   (vrai KO, contre-attaque, coup franc, échauffement) avant d'être
   fixées, exactement comme l'observatoire l'a fait pour
   `mean_speed_change_px_s`.

---

## 6. CE QUE CE DOCUMENT CHANGE CONCRÈTEMENT

- On ne cherche plus "quel signal détecte le KO" — on construit une
  **machine à états explicite**, où chaque transition est un test booléen
  indépendant, jamais une somme pondérée :

```
A0 — Formation en cours (transitoire)
     ↓
Formation séparée détectée (geo_separation élevé) ?  — non → rester en A0
     ↓ oui
        ═══> entrée en A1
A1 — Formation stable
     │
     ├── L'activité remonte / la séparation retombe (rituel d'équipe,
     │   replacement, etc.) AVANT tout départ collectif ?
     │        oui → RETOUR EN A0 (faux départ — c'est le cas observé
     │              3 fois sur Andrimont avant le vrai KO)
     │
     └── Départ collectif détecté (proportion de joueurs) pendant A1 ?
              ↓ oui
              ═══> transition vers B
B — Départ collectif
     ↓
Activité maintenue après la transition (pas une contre-attaque) ?
     ↓ oui
===> C — JEU ÉTABLI = KO CONFIRMÉ (à l'instant de la transition A1→B)
```

**Le point essentiel de cette version** : A0 et A1 forment une boucle —
le système peut entrer et sortir de A1 plusieurs fois (chaque faux départ
observé sur Andrimont est exactement un cycle A0→A1→A0) avant qu'un
épisode A1 se prolonge suffisamment pour déclencher la transition vers B.
Ce n'est qu'en observant plusieurs matchs (section 7) qu'on pourra dire à
partir de quelle durée un épisode A1 devient statistiquement différent
d'un faux départ.

- **Avantage direct** : chaque échec devient localisable. Si le détecteur
  se trompe, on saura dire "il a échoué sur la condition de stabilité de
  l'état A" ou "sur la persistance de l'état C" — plus jamais "le score
  global était insuffisant", qui n'indique rien d'actionnable.
- Les signaux déjà validés dans ce projet deviennent des **tests de
  conditions**, pas des scores à combiner :
  - État A (stabilité) : `geo_separation` + variance de l'activité sur
    fenêtre glissante (jamais mesurée comme condition de DURÉE jusqu'ici)
  - Transition A→B : proportion de joueurs en transition simultanée
    (la piste "synchronisation" déjà évoquée, jamais codée comme test)
  - État C (persistance) : le critère "durable" déjà présent dans
    `detecteur_rupture_etat.py`, à isoler comme test propre
- **Prochaine étape concrète** (pas encore lancée) : répondre à la question
  de la section 4 (frontière avant-match) sur les cas déjà connus, avant
  d'écrire la machine à états elle-même.

---

## 7. OBSERVATEUR DE MACHINE À ÉTATS (en cours)

Pas un détecteur — un journal d'événements. Pour chaque match, on enregistre
chaque entrée/sortie de l'état A1 (formation stable), avec sa durée et ses
caractéristiques (proportion de joueurs déjà immobiles, synchronisation),
sans jamais décider lequel de ces épisodes "est" le KO. La décision viendra
seulement après avoir vu la distribution de ces durées sur plusieurs matchs
— pas avant.

Exemple de journal attendu (format, pas données réelles) :
```
t=185s  entrée A1
t=191s  sortie A1   (durée=6s)   -- faux départ
t=219s  entrée A1
t=225s  sortie A1   (durée=6s)   -- faux départ
t=271s  entrée A1
t=275s  sortie A1   (durée=4s)   -- faux départ
t=275s  entrée A1
t=307s  transition A1→B          -- durée de l'épisode=32s, mène au KO
```

Voir `observateur_machine_etats.py` / `observateur_machine_etats_kaggle.py`
pour l'implémentation, et le tableau croisé multi-matchs qui en résulte.
