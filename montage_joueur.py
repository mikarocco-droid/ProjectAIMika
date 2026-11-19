# analysis/montage_joueur.py
# 19/11/2026 — POINT D'ENTRÉE DE PRODUCTION du montage et du bilan d'un joueur (méthode validée sur Kaggle :
# n° 6 et gardien n° 1 de SRU Verviers). Le travail est fait par analysis/montage_joueur_script.py, lancé dans un
# PROCESSUS SÉPARÉ : la lecture des numéros utilise plusieurs processus (multiprocessing), interdits dans un worker
# Celery « daemon » ; un sous-processus isole aussi la mémoire (PARSeq, vidéo).
#
# Utilisation :
#   from analysis.montage_joueur import monter_joueur
#   res = monter_joueur(video=..., cache_dir=..., sortie=..., match="sruverviers", numero=6, couleur="rouge",
#                       bornes=[[ko1, fin1], [ko2, fin2]], buts=[{"time_abs": 715.0, "statut": "confirmé",
#                       "equipe": "maillot red / short white"}, ...])
#   res["trouve"], res["montage"], res["fiche"], res["rapport_et_fiche"], res["message"], …
#
# Ligne de commande (tests) :  python -m analysis.montage_joueur parametres.json
import json
import os
import runpy
import subprocess
import sys

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "montage_joueur_script.py")
COULEURS_VALIDES = ["rouge", "bleu", "bleu clair", "blanc", "noir", "jaune", "vert", "orange", "violet", "gris"]


def monter_joueur(video, cache_dir, sortie, match, numero, couleur, bornes=None, buts=None, gardien=False,
                  couleur_maillot=None, entrees=None, gemini_key=None, parseq_poids=None, rapport_pdf=None,
                  duree_analyse_min=0.0, attribution=None, delai_max_s=6 * 3600):
    """Monte le résumé d'un joueur et produit sa fiche. Renvoie un dict (voir resultat_montage_joueur.json) ;
    en cas d'échec technique : {"trouve": False, "erreur": "...", "message": None}."""
    couleur = str(couleur or "").strip().lower()
    if couleur not in COULEURS_VALIDES:
        return {"trouve": False, "erreur": f"couleur inconnue : {couleur!r} (attendu : {', '.join(COULEURS_VALIDES)})",
                "message": None}
    if gardien and str(couleur_maillot or "").strip().lower() not in COULEURS_VALIDES:
        return {"trouve": False, "erreur": "gardien : couleur du maillot du gardien manquante ou inconnue", "message": None}
    os.makedirs(sortie, exist_ok=True)
    params = {
        "SORTIE": sortie,
        "ENTREES": [d for d in (entrees or []) if d and os.path.isdir(d)] or [sortie],
        "GEMINI_KEY": gemini_key or os.environ.get("GEMINI_API_KEY", ""),
        "PARSEQ_POIDS": parseq_poids or os.environ.get("SCOUTIA_PARSEQ_POIDS", ""),
        "RAPPORT_PDF": rapport_pdf,
        "RELAIS": {"match": match, "video": video, "cache": cache_dir, "actions_abs": [], "bornes": bornes,
                   "buts": buts or [], "t0": None, "duree_analyse_min": duree_analyse_min, "joueur_seul": True},
        "MATCH": match, "VIDEO": video, "CACHE": cache_dir, "NUMERO": int(numero), "COULEUR": couleur,
        "GARDIEN": bool(gardien), "COULEUR_MAILLOT": str(couleur_maillot or "").strip().lower(),
        "FENETRES": [], "MODE": "complet", "DUREE_ANALYSE_MIN": duree_analyse_min, "ATTRIBUTION": attribution,
    }
    fichier_p = os.path.join(sortie, "parametres_montage_joueur.json")
    json.dump(params, open(fichier_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    resultat_f = os.path.join(sortie, "resultat_montage_joueur.json")
    if os.path.exists(resultat_f):
        os.remove(resultat_f)
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=racine + os.pathsep + os.environ.get("PYTHONPATH", ""))
    journal = os.path.join(sortie, "journal_montage_joueur.txt")
    try:
        with open(journal, "w", encoding="utf-8") as j:
            p = subprocess.run([sys.executable, "-m", "analysis.montage_joueur", fichier_p], cwd=racine, env=env,
                               stdout=j, stderr=subprocess.STDOUT, timeout=delai_max_s)
    except subprocess.TimeoutExpired:
        return {"trouve": False, "erreur": f"montage du joueur trop long (> {delai_max_s / 3600:.0f} h)", "message": None,
                "journal": journal}
    if os.path.exists(resultat_f):
        res = json.load(open(resultat_f, encoding="utf-8"))
        res["journal"] = journal
        return res
    return {"trouve": False, "erreur": f"montage du joueur interrompu (code {p.returncode}) — voir {journal}",
            "message": None, "journal": journal}


def _executer(fichier_parametres):
    """Exécution dans le sous-processus : le script tourne comme « __main__ » (nécessaire au multiprocessing)."""
    params = json.load(open(fichier_parametres, encoding="utf-8"))
    try:
        runpy.run_path(SCRIPT, init_globals={"PARAMS_MONTAGE": params}, run_name="__main__")
    except SystemExit as e:          # « joueur introuvable » : le résultat a déjà été écrit par le script
        print(f"   arrêt du script : {e}")


if __name__ == "__main__":
    _executer(sys.argv[1])
