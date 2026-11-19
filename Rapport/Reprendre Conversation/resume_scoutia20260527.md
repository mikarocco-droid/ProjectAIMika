Voici le résumé complet et à jour de la dernière session :
🎯 Scoutia — Résumé session du 27 mai 2026
Ce qui a été fait
1. Détection de fin de match (kickoff_detector.py + pipeline.py)

Problème : la vidéo Andrimont continue après le match (public, gamins qui jouent)
Solution implémentée : find_match_end_offset qui scanne les frames_data du tracking
Critère "vrai jeu" = ≥4 joueurs avec couleur d'équipe assignée (les gamins sans vareuse = ignorés)
Seuil : 25 min sans vrai jeu → fin du match (couvre mi-temps ~15 min + prolongations + tirs au but)
On garde toujours le last_real_game_time et non le premier silence
apply_match_end coupe events/frames_data avec +60s de marge (célébrations, coup de sifflet)
Minimum 45 min de jeu avant de commencer à surveiller
Script de découpe vidéo (cut_video.py)

Script ffmpeg généré pour couper les 15 premières minutes et les 30 dernières
-c copy = pas de réencodage, rapide
Output dans /kaggle/working/clips/
Run en cours — vidéo Andrimont (107 min)

Démarré à 07:52, à ~50% vers 14:55 (7h d'écoulé)
Tracking terminé à 13:57, phase SHOT→GOAL en cours (412 tirs, ~18% fait à la dernière mise à jour)
Estimation de fin : ~18h00 (heure Kaggle)
GK throw bloquage 3s = comportement normal et intentionnel
Timestamps corrects ✅ (fix ball["frame"] = analyzed confirmé)
Fichiers modifiés dans cette session

kickoff_detector.py — ajout find_match_end_offset
pipeline.py — intégration apply_match_end après kickoff offset
kaggle_notebook.py — checks mis à jour
cut_video.py — nouveau script
Pending

Résultats du run Andrimont (buts détectés ? rapport PDF ?)
Rapport PDF toujours en suspens (numéros maillots, commentaires — session précédente)
V23 à tester (fix -1 poteau → but 02:20)
Crédits Anthropic API (erreur 400)
Objectif 25 matchs (actuellement 9)