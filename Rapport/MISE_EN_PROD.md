# ScoutIA — notice de mise en production (25/11/2026)

Remplace `MISE_EN_PROD_JOUEUR.md` (19-20/11), qui ne couvrait que le mode joueur.

## 1. Ce qui tourne
- **Analyse du match** (`pipeline.py`) : détection des buts (profil validé « 2026-11 » : 11 matchs de référence, puis
  SRU, Warsage, Baelen : 18 vrais buts sur 18, 1 faux but « confirmé » sur 3 matchs), bornes, équipes, rapport PDF.
- **Résumé vidéo** (`export/resume_match.py`) : tous les buts + occasions (classification et journal Gemini), 12 min au
  plus ; découpé SANS réencodage (qualité d'origine) ; chaque séquence est gardée (`sequences/`) avec les « autres
  moments à voir ». L'utilisateur recompose son résumé sur la page de résultats.
- **Mode joueur** (`analysis/montage_joueur.py` → `analysis/montage_joueur_script.py`, processus séparé) : lecture des
  numéros PARSeq, gardien par la couleur + vérification Gemini, montage avec repère, fiche « Bilan du joueur » (page 1 :
  présence, buts marqués et passes décisives indiqués par l'utilisateur ; page 2 : chronologie). Séquences gardées pour
  « Composer le montage » (ses séquences, ses ballons écartés, les autres faits de jeu du match).
- **Page de résultats** (`app.py`, `templates/results.html`) : validation des buts (vidéo, équipe, pas un but, but
  oublié), score, rôle du joueur, compositions ; rapport et fiche refaits à chaque réponse ; montage du joueur refait
  sans les buts écartés. Les réponses sont partagées entre utilisateurs de la même vidéo (empreinte du fichier).
- **Réutilisation** (`analysis/cache_analyses.py`) : même fichier vidéo déjà analysé (30 jours, même version) → suivi et
  réponses Gemini repris : pas de GPU, presque 0 $ de Gemini.
- **Jetons** (remplacent les formules) : 1 jeton = 3 € ; match 48 h = 2, match rapide = 3, joueur 48 h = 3, joueur
  rapide = 4 ; paquets 10 / 20+1 / 30+2 / 50+5 / 100+12 ; 2 jetons d'essai (match 48 h seulement) ; parrainage 2 jetons
  pour 3 filleuls ayant analysé un match ; validité 24 mois ; jetons rendus si l'analyse échoue.
  **Prix à confirmer après mesure du coût réel en production.**
- **Mode « sous 48 h »** = Gemini Flex (−50 %), bascule au tarif standard 36 h après la commande. Le GPU n'est pas encore
  économisé (même file que les analyses rapides).

## 2. Variables d'environnement
| Variable | Rôle | Valeur conseillée |
| --- | --- | --- |
| `SCOUTIA_PROFIL_KO` | réglages validés de la détection des buts | `2026-11` |
| `SCOUTIA_JOUEUR_V30` | méthode actuelle du mode joueur | `1` |
| `SCOUTIA_CAMERA_TYPE` | type de caméra transmis au pipeline (toute la validation : `high_side`) | `high_side` |
| `SCOUTIA_PARSEQ_POIDS` | chemin de `parseq_scoutia_maillots_v2.pt` (hors dépôt) | à déposer sur le serveur |
| `SCOUTIA_DOSSIER_JOUEUR` | dossier des fichiers réutilisables du joueur (lectures) | dossier du serveur |
| `SCOUTIA_CACHE_JOUEUR` | (facultatif) cache de suivi, si ce n'est pas `<sortie>/cache` | — |
| `SCOUTIA_CACHE_ANALYSES_DIR` | dossier du cache de réutilisation (≈ 100 à 200 Mo par match, estimation) | disque du serveur |
| `SCOUTIA_CACHE_ANALYSES_JOURS` | durée de conservation de ce cache | `30` |
| `SCOUTIA_RESUME_MAX_MIN` | durée maximale du résumé de base (0 = sans limite) | `12` |
| `SCOUTIA_RESUME_COPIE` | découpe sans réencodage (qualité d'origine) | `1` |
| `JETON_PRIX_EUR`, `JETONS_MATCH_48H`, `JETONS_MATCH_RAPIDE`, `JETONS_JOUEUR_48H`, `JETONS_JOUEUR_RAPIDE` | prix et coûts | 3 / 2 / 3 / 3 / 4 |
| `JETONS_ESSAI`, `JETONS_PARRAINAGE`, `JETONS_VALIDITE_MOIS` | essai, parrainage, validité | 2 / 2 / 24 |
| `RETOURS_MOIS_APRES_SAISON`, `SAISON_DEBUT_MOIS` | conservation des confirmations de buts | 12 / 7 |
| `GEMINI_API_KEY` | clé Gemini (compte PAYANT, condition de Flex et de la confidentialité annoncée) | — |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` | paiement | — |

Les durées de conservation des résultats par formule restent dans `config.FILE_RETENTION_DAYS`.

## 3. Serveur
- `ffmpeg` installé (ou, à défaut, le paquet Python `imageio-ffmpeg`, utilisé automatiquement par le résumé).
- `requirements.txt` (dont timm, pytorch-lightning, nltk, pypdf, fpdf2) ; GPU pour l'analyse (≈ 2 h par match sur Kaggle
  T4 ×2 ; durée sur la carte de production À MESURER).
- Code PARSeq téléchargé par torch.hub (préchargé dans `Dockerfile.runpod`).
- Nettoyage automatique (`cleanup_expired_outputs`) : résultats et fichiers du joueur à l'expiration ; jetons expirés ;
  confirmations de buts 12 mois après la fin de la saison. Les analyses de démonstration ne sont jamais supprimées.

## 4. Stripe
- Paquets en paiement unique, prix fixé par l'application (aucun produit à créer).
- Activer **Bancontact** dans le tableau de bord Stripe ; activer le webhook `checkout.session.completed` vers
  `/webhook/stripe`. Un paiement n'est crédité qu'une fois (page de retour ou webhook).
- **Tester en mode test Stripe avant la mise en ligne.**

## 5. Vérifications avant l'ouverture
1. Une vraie analyse de bout en bout (envoi → résultats → validation des buts → résumé et rapport à jour).
2. Le même fichier envoyé une 2e fois : le journal doit afficher « ♻️ Match déjà analysé », durée et coût très réduits.
3. Une analyse « sous 48 h » : journal Gemini en Flex ; coût mesuré.
4. Un achat de jetons en mode test Stripe (page de retour ET webhook).
5. Mesurer le coût réel d'un match (SRU par exemple) : GPU + Gemini, puis ajuster les prix.
6. Compléter les mentions en jaune des CGU et de la politique de confidentialité (adresse, contact, hébergeur).

## 6. Limites connues (à garder en tête)
- Faux buts : environ 3 par match, rarement « confirmés » (1 sur 3 matchs) ; l'utilisateur les écarte sur la page.
- Coup d'envoi faussé si les équipes sont en place longtemps avant (Baelen : 63 s).
- Occasions : environ la moitié seulement sont retrouvées ; types d'action peu fiables (non affichés).
- Joueur : actions manquées quand son numéro n'est pas lisible ; couleurs testées : rouge, bleu clair, noir, jaune.
- Possession : non fiable de nuit (masquée).
- Seul le même FICHIER vidéo est reconnu pour la réutilisation et le partage des corrections.
- Caméras autres que Veo, matchs de jour : non testés.
