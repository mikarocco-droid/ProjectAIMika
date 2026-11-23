# outils/demo_resultat.py — 22/11/2026
# Crée une analyse « de démonstration » dans la base de l'application À PARTIR D'UN RUN DÉJÀ FAIT (Kaggle), pour tester la
# page de résultats (résumé en haut, « Composer mon résumé », séquences) SANS refaire d'analyse (ni GPU, ni Gemini).
#
# Préparer un dossier avec la sortie du run, par exemple pour Warsage :
#     demo_warsage/
#         resume_warsage.mp4            (résumé de base)
#         sequences/                    (dézippé de sequences_warsage.zip : s000.mp4… + sequences.json)
#         bout_en_bout_warsage.json     (buts, bornes)
#         rapport_warsage.pdf           (facultatif)
# Puis, dans le dossier du projet (là où se trouve app.py) :
#     python outils/demo_resultat.py --dossier demo_warsage --email votre@adresse.be --titre "P1 vs Warsage"
# et ouvrir la page indiquée (/results/<id>). Il faut ffmpeg sur la machine pour « Refaire mon résumé ».
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, db, Analysis, User  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True)
    ap.add_argument("--email", required=True, help="compte de l'application qui verra l'analyse")
    ap.add_argument("--titre", default="Match de démonstration")
    a = ap.parse_args()
    dos = os.path.abspath(a.dossier)
    reel = next(iter(sorted(glob.glob(os.path.join(dos, "resume_*.mp4")))), None)
    seq = os.path.join(dos, "sequences")
    if not reel or not os.path.exists(os.path.join(seq, "sequences.json")):
        raise SystemExit("❌ il faut resume_<match>.mp4 et sequences/sequences.json dans le dossier")
    be = next(iter(glob.glob(os.path.join(dos, "bout_en_bout_*.json"))), None)
    relais = json.load(open(be, encoding="utf-8")) if be else {}
    ko1 = float((relais.get("bornes") or [[0, 0]])[0][0])
    buts = [{"time_abs": b["time_abs"], "time": float(b["time_abs"]) - ko1, "status": b.get("statut"),
             "equipe": b.get("equipe")} for b in relais.get("buts", [])]
    pdf = next(iter(glob.glob(os.path.join(dos, "rapport_*.pdf"))), None)
    result = {"reel": reel, "sequences_resume": seq, "pdf": pdf, "demo": True,
              "summary": {"goals": len(buts), "goals_confirmes": sum(1 for b in buts if str(b["status"]).startswith("confirm")),
                          "shots": 0},
              "buts": buts}
    with app.app_context():
        u = User.query.filter_by(email=a.email).first()
        if not u:
            raise SystemExit(f"❌ aucun compte {a.email} dans la base")
        an = Analysis(user_id=u.id, filename=f"{a.titre}.mp4", sport="football", mode="match", status="done",
                      progress=100, progress_msg="Démonstration", result_json=json.dumps(result), output_dir=dos)
        db.session.add(an)
        db.session.commit()
        print(f"✅ analyse de démonstration n°{an.id} créée → ouvrir /results/{an.id}")


if __name__ == "__main__":
    main()
