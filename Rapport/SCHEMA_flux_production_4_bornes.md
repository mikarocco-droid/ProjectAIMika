# Flux de production complet — détection des 4 bornes du match

## Ordre de CALCUL (différent de l'ordre chronologique du match)

```
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1 — KO1 (coup d'envoi 1ère mi-temps)                       │
│  Méthode : cascade Gemini Q1/Q2 (déjà validée 9/9)                │
│  Fenêtre de recherche : début de la vidéo                         │
│  Dépend de : rien (point de départ)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2 — KO2 (coup d'envoi 2e mi-temps)                         │
│  Méthode : même cascade Gemini Q1/Q2 (validée 9/9 aujourd'hui,    │
│            filtre "deux équipes visibles" + gestion retry)        │
│  Fenêtre de recherche : [KO1+49min, KO1+68min]                    │
│  Dépend de : KO1 uniquement                                       │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    ↓                           ↓
┌───────────────────────────────┐ ┌───────────────────────────────┐
│  ÉTAPE 3a — Fin1MT             │ │  ÉTAPE 3b — FinMatch           │
│  Méthode : transition audio    │ │  Méthode : transition audio    │
│  (proportion sifflet +         │ │  + critère de persistance      │
│  platitude spectrale,          │ │  (proportion sifflet +         │
│  zone tampon 4s)               │ │  platitude, tampon 4s)         │
│  Fenêtre : [KO2-16min,KO2-8min]│ │  Fenêtre : [KO2+40min,KO2+55min]│
│  Dépend de : KO2 uniquement    │ │  Dépend de : KO2 uniquement    │
│  (PAS de Fin1MT)               │ │  (indépendant de Fin1MT aussi) │
│  Résultat : 8/9 à ≤19s         │ │  Résultat : 8/9 à ≤10s         │
└───────────────────────────────┘ └───────────────────────────────┘
```

**Point clé** : Fin1MT et FinMatch dépendent tous les deux de KO2, mais AUCUN des deux ne dépend de l'autre — ils peuvent être calculés en parallèle une fois KO2 connu. Aucune dépendance circulaire : c'est un graphe simple KO1 → KO2 → {Fin1MT, FinMatch}, même si l'ordre de calcul (KO2 avant Fin1MT) diffère de l'ordre chronologique réel du match (Fin1MT avant KO2).

---

## Ordre CHRONOLOGIQUE du match (pour référence, différent de l'ordre de calcul ci-dessus)

```
Vidéo
  │
  ├── KO1 (t≈300s)
  │     │
  │     │   ◄── 1ère mi-temps ──►
  │     │
  ├── Fin1MT (t≈KO1+46min)         ← calculé APRÈS KO2 (étape 3a)
  │     │
  │     │   ◄── pause (8-15min) ──►
  │     │
  ├── KO2 (t≈KO1+55min)            ← calculé en 2e (étape 2)
  │     │
  │     │   ◄── 2e mi-temps ──►
  │     │
  └── FinMatch (t≈KO2+48min)       ← calculé APRÈS KO2 (étape 3b)
```

---

## Découpage du tracking (process_video) une fois les 4 bornes connues

```python
# Une fois KO1, Fin1MT, KO2, FinMatch tous calculés (étapes 1 à 3b ci-dessus) :

process_video(
    video_path,
    start_time_s=KO1 - 10,
    end_time_s=Fin1MT
)
# → ne tracke que la 1ère mi-temps utile, pas toute la durée théorique

process_video(
    video_path,
    start_time_s=KO2 - 10,
    end_time_s=FinMatch
)
# → ne tracke que la 2e mi-temps utile

# Le fichier vidéo reste le même (video_path inchangé) — seuls
# start_time_s/end_time_s changent, donc les numéros de frames restent
# ceux de la vidéo originale (pas de recalcul, pas de fichiers coupés
# à créer, cohérent avec le fix §12.15+§12.19 déjà en place).
```

**Ce qui NE change PAS** : le tracking (YOLO/DeepSort) ne détecte jamais les bornes lui-même en direct — il reçoit des fenêtres déjà calculées par les détecteurs dédiés (Gemini pour KO1/KO2, audio pour Fin1MT/FinMatch). Pas de dépendance circulaire, pas de "je track pour découvrir où m'arrêter".

---

## Prochaine étape (avant de coder)

Confirmer ce schéma correspond bien à l'intention avant d'intégrer dans `pipeline.py`/`main.py` — notamment vérifier l'état actuel de ces deux fichiers pour brancher proprement sans casser ce qui fonctionne déjà pour KO1.
