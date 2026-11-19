# Montage d'un joueur — mise en production (19/11/2026)

## Ce qui tourne
- `analysis/montage_joueur.py` : fonction `monter_joueur(...)` appelée par le pipeline (mode joueur), qui lance
  `analysis/montage_joueur_script.py` dans un processus séparé (multiprocessing interdit dans un worker Celery).
- Méthode validée sur Kaggle (SRU Verviers : n° 6, gardien n° 1) : lecture des numéros PARSeq affiné ; gardien reconnu
  par la couleur de son maillot + vérification Gemini ; arrêts et tirs adverses repris de l'analyse du match ;
  6 mètres retirés ; TOUS les buts du match dans le résumé ; fiche « Bilan du joueur » jointe au rapport du match.

## Interrupteur et variables d'environnement
| Variable | Rôle |
| --- | --- |
| `SCOUTIA_PROFIL_KO=2026-11` | détection des buts validée (11 matchs) : vraies détections, épisodes, début de match, redatage, reprise |
| `SCOUTIA_JOUEUR_V30=1` | active la nouvelle méthode (sinon : ancienne méthode du mode joueur) |
| `SCOUTIA_PARSEQ_POIDS` | chemin de `parseq_scoutia_maillots_v2.pt` (hors dépôt) |
| `SCOUTIA_DOSSIER_JOUEUR` | dossier où chercher des fichiers réutilisables (lectures `lectures_v30_<match>_v2.json`) |
| `GEMINI_API_KEY` | clé Gemini (déjà utilisée par le pipeline) |
| `SCOUTIA_CACHE_JOUEUR` | (facultatif) dossier du cache de suivi, si ce n'est pas `<sortie>/cache` |

## Application
- Formulaire : position « Gardien » → champ « Couleur du maillot du gardien » (obligatoire).
- Base : colonnes `position_joueur`, `couleur_maillot_gardien` (ajoutées automatiquement au démarrage) ; reprises à
  la relance.

## Serveur
- `ffmpeg` (assemblage du montage), `requirements.txt` (dont timm, pytorch-lightning, nltk, pypdf).
- Code PARSeq téléchargé par torch.hub (préchargé dans `Dockerfile.runpod`) ; GPU conseillé (lecture des numéros
  ≈ 1 h 30 sur processeur pour un match, quelques minutes sur GPU d'après les runs Kaggle).

## Vérification avant activation
`test_module_montage_joueur.py` (Kaggle) : le module doit redonner les résultats des cellules sur SRU (n° 6, gardien).
