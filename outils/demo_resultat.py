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
    ap.add_argument("--bornes", default=None, help="(facultatif) bornes connues si l'analyse ne les a pas trouvées : "
                                                   "\"coup d'envoi,fin 1re MT,reprise,fin\" en heure vidéo, ex. \"00:10,47:40,47:43,84:46\"")
    ap.add_argument("--joueur", default=None, help="(facultatif) dossier joueur_* d'un montage : montage, fiche, "
                                                   "fiche_donnees_*.json, resultat_montage_joueur.json")
    a = ap.parse_args()
    dos = os.path.abspath(a.dossier)
    reel = next(iter(sorted(glob.glob(os.path.join(dos, "resume_*.mp4")))), None)
    seq = os.path.join(dos, "sequences")
    # 25/11/2026 : résumé absent → refait ICI à partir des séquences (celles du résumé de base), sans Kaggle
    if not reel and os.path.exists(os.path.join(seq, "sequences.json")):
        from export.resume_match import recomposer_resume
        ids = [x["id"] for x in json.load(open(os.path.join(seq, "sequences.json"), encoding="utf-8")) if x.get("dans_resume")]
        print(f"   résumé absent : reconstruction à partir de {len(ids)} séquence(s)…")
        reel = recomposer_resume(seq, ids, os.path.join(dos, "resume_reconstruit.mp4"))
        print(f"   résumé reconstruit : {reel}" if reel else "   ❌ reconstruction impossible (ffmpeg ?)")
    if not reel or not os.path.exists(os.path.join(seq, "sequences.json")):
        raise SystemExit("❌ il faut resume_<match>.mp4 et sequences/sequences.json dans le dossier")
    be = next(iter(glob.glob(os.path.join(dos, "bout_en_bout_*.json"))), None)
    relais = json.load(open(be, encoding="utf-8")) if be else {}
    bn = relais.get("bornes") or [[0, 0]]
    if a.bornes:                                  # 25/11 : bornes données à la main (analyse sans mi-temps trouvée)
        _sx = lambda h: sum(int(x) * 60 ** i for i, x in enumerate(reversed(h.strip().split(":"))))
        _b4 = [float(_sx(h)) for h in a.bornes.split(",")]
        if len(_b4) == 4:
            bn = [[_b4[0], _b4[1]], [_b4[2], _b4[3]]]
            print(f"   bornes données : {a.bornes}")
    ko1 = float(bn[0][0])
    # 23/11 : équipes (A/B) reconstituées à partir des tenues du relais, comme le rapport
    tenues_l = sorted({b.get("equipe") for b in relais.get("buts", []) if b.get("equipe")})
    tenues = {"A": tenues_l[0], "B": tenues_l[1]} if len(tenues_l) == 2 else {}
    inv = {v: k for k, v in tenues.items()}
    buts = [{"time_abs": b["time_abs"], "time": float(b["time_abs"]) - ko1, "status": b.get("statut"),
             "equipe": inv.get(b.get("equipe")), "tenue_equipe": b.get("equipe"), "niveau_equipe": b.get("niveau_equipe")}
            for b in relais.get("buts", [])]
    pdf = next(iter(glob.glob(os.path.join(dos, "rapport_*.pdf"))), None)
    result = {"reel": reel, "sequences_resume": seq, "pdf": pdf, "demo": True,
              "attribution_equipes": {"tenues": tenues} if tenues else None,
              "bornes_mi_temps": ({"ko1": bn[0][0], "fin1mt": bn[0][1], "ko2": bn[1][0], "finmatch": bn[1][1]}
                                  if len(bn) >= 2 else None),
              "summary": {"goals": len(buts), "goals_confirmes": sum(1 for b in buts if str(b["status"]).startswith("confirm")),
                          "shots": 0},
              "buts": buts}
    if a.joueur:      # 24/11/2026 : section « joueur » de la page (montage, fiche, rôle dans les buts)
        dj = os.path.abspath(a.joueur)
        # 24/11 : le fichier peut être dans un sous-dossier (archive dézippée dans un dossier du même nom)
        _rj = sorted(glob.glob(os.path.join(dj, "**", "resultat_montage_joueur.json"), recursive=True), key=len)
        if not _rj:
            raise SystemExit(f"❌ resultat_montage_joueur.json introuvable dans {dj} : le montage du joueur est-il terminé, "
                             f"et l'archive bien dézippée ?")
        dj = os.path.dirname(_rj[0])
        print(f"   dossier du joueur : {dj}")
        rj = json.load(open(_rj[0], encoding="utf-8"))
        loc = lambda p: os.path.join(dj, os.path.basename(p)) if p and os.path.exists(os.path.join(dj, os.path.basename(p))) else None
        result["player_reel"] = loc(rj.get("montage"))
        result["joueur"] = {"demande": True, "trouve": bool(rj.get("trouve")), "numero": str(rj.get("numero") or ""),
                            "couleur": None, "gardien": bool(rj.get("gardien")), "fiche": loc(rj.get("fiche")),
                            "fiche_donnees": loc(rj.get("fiche_donnees")), "buts_impliques": rj.get("buts_impliques") or [],
                            "rapport_et_fiche": loc(rj.get("rapport_et_fiche")),
                            # séquences : dossier clips_* copié dans le dossier joueur (avec sequences.json)
                            "sequences": next((os.path.dirname(f) for f in glob.glob(os.path.join(dj, "**", "sequences.json"),
                                                                                   recursive=True)), None),
                            "message": rj.get("message")}
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
