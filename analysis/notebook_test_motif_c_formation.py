"""
notebook_test_motif_c_formation.py
=======================================
NOUVEAU DIAGNOSTIC (pas de RF, pas de Gemini, pas de nouvelle feature
audio) : teste si le "motif C" (petit groupe hors-equipe - arbitre/
gardien - separe de deux groupes de joueurs, pres du centre) est un
signal INDEPENDANT capable de classer le vrai KO parmi les meilleurs
candidats, sans passer par le RF actuel (dont on sait qu'il est
instable sur plusieurs matchs, cf. jalon section 11).

IMPORTANT (clarification avant execution) : ce test necessite un vrai
passage process_video() (YOLO) par match, car frames_data (positions
individuelles des joueurs) n'a JAMAIS ete sauvegarde dans ce projet -
il est toujours genere en memoire puis jete. Ce n'est donc pas "gratuit"
au sens strict, mais ca REUTILISE le meme passage deja necessaire pour
mine_candidates() - donc pas de cout YOLO SUPPLEMENTAIRE par rapport a
un diagnostic normal.

METHODE : pour chaque candidat deja produit par mine_candidates(),
calculer un score de "formation KO" base UNIQUEMENT sur les positions
et labels d'equipe des joueurs trackes (frames_data), independamment
de geo_separation/speed/whistle_score deja utilises par le RF :

  1. separation_equipes : les deux equipes (labels reels "team0"/
     "team1", pas juste x<mid_x) sont-elles bien separees de part et
     d'autre de la ligne mediane ?
  2. groupe_hors_equipe : existe-t-il un petit groupe (2-6 personnes)
     sans label d'equipe classe (arbitre/gardien), distinct des deux
     groupes principaux ?
  3. proximite_centre : le barycentre de tous les joueurs est-il proche
     du centre horizontal du terrain (frame_w/2) ?
  4. compacite_centrale : le groupe hors-equipe (ou a defaut l'ensemble)
     est-il compact (faible dispersion spatiale) ?
  5. immobilite : la position moyenne des joueurs change-t-elle peu sur
     une fenetre de quelques secondes autour du candidat ?
  6. symetrie : les deux equipes sont-elles en miroir l'une de l'autre
     par rapport a l'axe central ?

Score final = moyenne des 6 composantes (chacune normalisee 0-1).
Classement de TOUS les candidats par ce score, sans aucun RF.

QUESTION POSEE : le vrai KO ressort-il dans le top 3/5 de ce
classement sur 7-8 matchs / 9 ? Si oui -> vraie brique independante.
Si non -> observation humaine seduisante mais pas un signal
exploitable, a abandonner proprement.
"""

import subprocess, os, sys

# --- CELLULE 1 : setup ---

subprocess.run(["git", "clone", "--depth", "1",
                 "https://github.com/mikarocco-droid/ProjectAIMika.git",
                 "/kaggle/working/ProjectAIMika"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "ultralytics==8.4.120", "torch==2.10.0+cu128",
                 "opencv-python-headless", "anthropic",
                 "python-dotenv", "werkzeug", "scikit-learn", "joblib",
                 "numpy", "scipy", "psutil", "deep-sort-realtime", "pandas"], check=True)
subprocess.run(["apt-get", "install", "-y", "-q", "tesseract-ocr", "ffmpeg"], check=True)
os.chdir("/kaggle/working/ProjectAIMika")
sys.path.insert(0, "/kaggle/working/ProjectAIMika")
for pkg_dir in ["vision", "analysis", "rendering"]:
    init_path = os.path.join(pkg_dir, "__init__.py")
    if os.path.isdir(pkg_dir) and not os.path.exists(init_path):
        open(init_path, "a").close()

COMPANION_DATASET = "/kaggle/input/datasets/michaelziant/codeprojectai"
import shutil
for fname in ["ko_features.py", "candidates_mining.py"]:
    src = os.path.join(COMPANION_DATASET, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join("analysis", fname))
        print(f"  Copie : {fname}")

with open("param.env", "w") as f:
    f.write("SPORT=football\n")

print("Setup termine")
try:
    import ultralytics
    print(f"ultralytics version : {ultralytics.__version__}")
except Exception:
    pass
try:
    import torch
    print(f"torch version : {torch.__version__} (cuda dispo: {torch.cuda.is_available()})")
except Exception:
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

DURATION_S = 1400.0

MATCHS = [
    {"name": "Andrimont",    "ko_connu": 307.0,  "video": "/kaggle/input/datasets/michaelziant/stertest/match-p4-andrimont-2026-04-26.mp4"},
    {"name": "Franchimont",  "ko_connu": 318.0,  "video": "/kaggle/input/datasets/michaelziant/p4franchimont/ster-b-franchimont-b-2026-04-09_0.mp4"},
    {"name": "Goe",          "ko_connu": 1081.0, "video": "/kaggle/input/datasets/michaelziant/stergoe/ster-b-goe-2026-03-20_0.mp4"},
    {"name": "Juprelle",     "ko_connu": 548.0,  "video": "/kaggle/input/datasets/michaelziant/juprelle/ster-juprelle-2026-01-25_0.mp4"},
    {"name": "MineroisSter", "ko_connu": 723.0,  "video": "/kaggle/input/datasets/michaelziant/mineroisster/minerois-ster-2025-11-09_0.mp4"},
    {"name": "P1Minerois",   "ko_connu": 192.0,  "video": "/kaggle/input/datasets/michaelziant/p1minerois/p1-minerois-2026-03-28_0.mp4"},
    {"name": "Raeren",       "ko_connu": 410.0,  "video": "/kaggle/input/datasets/michaelziant/p4raeren/ster-b-raeren-b-2026-04-03_0.mp4"},
    {"name": "Stembert",     "ko_connu": 553.0,  "video": "/kaggle/input/datasets/michaelziant/stembert/ster-b-stembert-2026-03-07_0.mp4"},
    {"name": "Wanze",        "ko_connu": 986.0,  "video": "/kaggle/input/datasets/michaelziant/wanzebas/ster-wanze-bas-oha-2026-02-28_0.mp4"},
]

RESULTS_DIR = "/kaggle/working/motif_c"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================


import numpy as np
import pandas as pd
from main import process_video
from analysis.candidates_mining import mine_candidates
from analysis.ko_features import _players_at


def _score_composantes(frames_data, fps, frame_w, t, window_s=4.0):
    """
    Calcule les 6 composantes du score de formation KO a l'instant t.
    Retourne un dict de sous-scores (chacun 0.0-1.0), ou None si pas
    assez de joueurs detectes pour juger.
    """
    joueurs = _players_at(frames_data, fps, t, tolerance_s=1.5)
    if len(joueurs) < 6:
        return None

    mid_x = frame_w / 2.0
    positions = np.array([j["center"] for j in joueurs], dtype=float)
    teams = [j.get("team") for j in joueurs]

    team_labels = [tl for tl in set(teams) if tl is not None]
    groupes_equipes = {tl: positions[[i for i, tt in enumerate(teams) if tt == tl]] for tl in team_labels}
    idx_hors_equipe = [i for i, tt in enumerate(teams) if tt is None]
    groupe_hors_equipe = positions[idx_hors_equipe] if idx_hors_equipe else np.empty((0, 2))

    # --- 1. Separation des equipes (vrais labels, pas juste x<mid_x) ---
    if len(groupes_equipes) >= 2:
        moyennes_x = [g[:, 0].mean() for g in groupes_equipes.values() if len(g) > 0]
        if len(moyennes_x) >= 2:
            cote_gauche = sum(1 for mx in moyennes_x if mx < mid_x)
            separation_equipes = 1.0 if (cote_gauche == 1) else 0.3
        else:
            separation_equipes = 0.0
    else:
        separation_equipes = 0.0

    # --- 2. Presence d'un groupe hors-equipe de taille plausible (2-6) ---
    n_hors_equipe = len(groupe_hors_equipe)
    groupe_hors_equipe_plausible = 1.0 if 2 <= n_hors_equipe <= 6 else (0.3 if n_hors_equipe == 1 else 0.0)

    # --- 3. Proximite du centre (barycentre global proche de mid_x) ---
    barycentre_x = positions[:, 0].mean()
    ecart_centre_norm = abs(barycentre_x - mid_x) / (frame_w / 2.0)
    proximite_centre = max(0.0, 1.0 - ecart_centre_norm)

    # --- 4. Compacite du groupe hors-equipe (faible dispersion) ---
    if n_hors_equipe >= 2:
        dispersion = np.std(groupe_hors_equipe, axis=0).mean()
        compacite_centrale = max(0.0, 1.0 - dispersion / (frame_w * 0.1))
    else:
        compacite_centrale = 0.5

    # --- 5. Immobilite (position moyenne stable sur quelques secondes) ---
    joueurs_avant = _players_at(frames_data, fps, t - window_s / 2, tolerance_s=1.0)
    joueurs_apres = _players_at(frames_data, fps, t + window_s / 2, tolerance_s=1.0)
    if len(joueurs_avant) >= 6 and len(joueurs_apres) >= 6:
        bary_avant = np.array([j["center"] for j in joueurs_avant]).mean(axis=0)
        bary_apres = np.array([j["center"] for j in joueurs_apres]).mean(axis=0)
        deplacement = np.linalg.norm(bary_apres - bary_avant)
        immobilite = max(0.0, 1.0 - deplacement / (frame_w * 0.05))
    else:
        immobilite = 0.5

    # --- 6. Symetrie des deux equipes autour de l'axe central ---
    if len(groupes_equipes) >= 2 and all(len(g) > 0 for g in groupes_equipes.values()):
        moyennes = [g.mean(axis=0) for g in groupes_equipes.values()]
        dists = [abs(mv[0] - mid_x) for mv in moyennes]
        if max(dists) > 0:
            symetrie = 1.0 - abs(dists[0] - dists[1]) / max(dists)
        else:
            symetrie = 1.0
    else:
        symetrie = 0.0

    return {
        "separation_equipes": separation_equipes,
        "groupe_hors_equipe_plausible": groupe_hors_equipe_plausible,
        "proximite_centre": proximite_centre,
        "compacite_centrale": compacite_centrale,
        "immobilite": immobilite,
        "symetrie": symetrie,
        "n_hors_equipe": n_hors_equipe,
    }


resultats_globaux = []

for m in MATCHS:
    match_name, t_ko, video_source = m["name"], m["ko_connu"], m["video"]
    resultat_path = os.path.join(RESULTS_DIR, f"{match_name}_motif_c.csv")

    print(f"\n{'='*70}\n{match_name} (KO connu = {t_ko}s)\n{'='*70}")

    if os.path.exists(resultat_path):
        print(f"  deja traite - saute.")
        continue

    trimmed_video = f"/kaggle/working/{match_name}_motifc.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-ss", "0", "-i", video_source, "-t", str(DURATION_S),
        "-vf", "fps=4", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "64k", trimmed_video,
    ], check=True, capture_output=True)

    events, jersey_map, fps, total_frames, frames_data = process_video(
        trimmed_video, return_frames=True
    )
    frame_w = frames_data[0]["frame_w"]
    video_duration_reelle = total_frames / fps
    print(f"  process_video termine : {len(frames_data)} frames_data")

    df_mine = mine_candidates(
        frames_data=frames_data, fps=fps, frame_w=frame_w,
        video_path=trimmed_video, match_name=match_name,
        video_duration_s=video_duration_reelle,
        true_kickoff_t=None, terminal_events=events,
        top_n_whistle=40,
    )

    scores = []
    for _, row in df_mine.iterrows():
        composantes = _score_composantes(frames_data, fps, frame_w, row["t"])
        if composantes is None:
            score_final = 0.0
            composantes = {}
        else:
            score_final = np.mean([
                composantes["separation_equipes"], composantes["groupe_hors_equipe_plausible"],
                composantes["proximite_centre"], composantes["compacite_centrale"],
                composantes["immobilite"], composantes["symetrie"],
            ])
        scores.append({"t": row["t"], "score_formation_c": score_final, **composantes})

    df_scores = pd.DataFrame(scores)
    df_scores["ecart_ko"] = (df_scores["t"] - t_ko).abs()
    df_scores = df_scores.sort_values("score_formation_c", ascending=False).reset_index(drop=True)
    df_scores["rang_formation_c"] = df_scores.index + 1

    idx_meilleur_candidat = df_scores["ecart_ko"].idxmin()
    ligne_ko = df_scores.loc[idx_meilleur_candidat]

    print(f"  Meilleur candidat (ecart={ligne_ko['ecart_ko']:.2f}s) : "
          f"rang par motif C = {ligne_ko['rang_formation_c']}/{len(df_scores)}, "
          f"score={ligne_ko['score_formation_c']:.2f}")

    df_scores.to_csv(resultat_path, index=False)

    resultats_globaux.append({
        "match": match_name, "ecart_candidat": round(ligne_ko["ecart_ko"], 2),
        "rang_formation_c": int(ligne_ko["rang_formation_c"]),
        "score_formation_c": round(ligne_ko["score_formation_c"], 3),
        "n_candidats": len(df_scores),
        "top3": ligne_ko["rang_formation_c"] <= 3,
        "top5": ligne_ko["rang_formation_c"] <= 5,
        "top10": ligne_ko["rang_formation_c"] <= 10,
    })

    if os.path.exists(trimmed_video):
        os.remove(trimmed_video)


# --- TABLEAU FINAL ---

df_final = pd.DataFrame(resultats_globaux)
print(f"\n{'='*100}\nTABLEAU FINAL - RANG DU VRAI KO SELON LE SCORE MOTIF C\n{'='*100}")
print(df_final.to_string(index=False))

df_final.to_csv(os.path.join(RESULTS_DIR, "resume_motif_c_9matchs.csv"), index=False)

print(f"\n{'='*100}\nVERDICT\n{'='*100}")
n_top3 = df_final["top3"].sum()
n_top5 = df_final["top5"].sum()
n_top10 = df_final["top10"].sum()
n_total = len(df_final)
print(f"Top-3  : {n_top3}/{n_total}")
print(f"Top-5  : {n_top5}/{n_total}")
print(f"Top-10 : {n_top10}/{n_total}")

if n_top5 >= 7:
    print(f"\n>>> Signal solide : le motif C place le vrai KO dans le top-5 sur au moins")
    print(f">>> 7/{n_total} matchs - vraie brique independante a considerer, potentiellement")
    print(f">>> comme feature RF ou comme filtre complementaire (PAS un remplacement du RF).")
elif n_top10 >= 7:
    print(f"\n>>> Signal modere : utile en top-10 mais pas assez discriminant pour un")
    print(f">>> classement fin. A combiner avec d'autres signaux plutot qu'a utiliser seul.")
else:
    print(f"\n>>> Signal insuffisant - le motif C, tel que calcule ici, ne discrimine pas")
    print(f">>> mieux que le hasard sur cet echantillon. Observation humaine seduisante,")
    print(f">>> mais pas un signal exploitable en l'etat. A abandonner proprement, sans")
    print(f">>> chercher une nouvelle variante de calcul sans preuve supplementaire.")
