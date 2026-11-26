# worker_gpu.py — 26/11/2026
# PROGRAMME DE LA MACHINE GPU (RunPod) en production.
#   1. demande une analyse au site (HTTPS, jeton partagé) ; 2. télécharge la vidéo depuis R2 ; 3. fait l'analyse
#   (analyse_execution.executer_analyse, le même calcul que partout ailleurs) ; 4. dépose les résultats sur R2 ;
#   5. prévient le site. Sans travail pendant SCOUTIA_GPU_INACTIF_MIN minutes : le signale au site, qui supprime la
#   machine (la machine n'a PAS la clé RunPod).
# Variables : SCOUTIA_SITE_URL, SCOUTIA_WORKER_TOKEN, R2_*, GEMINI_API_KEY, réglages SCOUTIA_* (transmis par le site).
import json
import os
import shutil
import threading
import time
import traceback
from datetime import datetime

import boto3
import requests

SITE = os.environ.get("SCOUTIA_SITE_URL", "").rstrip("/")
JETON = os.environ.get("SCOUTIA_WORKER_TOKEN", "")
POD_ID = os.environ.get("RUNPOD_POD_ID", "local")
INACTIF_MIN = float(os.environ.get("SCOUTIA_GPU_INACTIF_MIN", 15))
TRAVAIL = os.environ.get("SCOUTIA_TRAVAIL", "/app/travail")
BUCKET = os.environ.get("R2_BUCKET_NAME", "scoutia-videos")


def _r2():
    return boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")


def _site(chemin, donnees, essais=5):
    """Appel au site, avec quelques nouvelles tentatives (coupure réseau passagère)."""
    for k in range(essais):
        try:
            r = requests.post(f"{SITE}{chemin}", json=dict(donnees, pod_id=POD_ID),
                              headers={"Authorization": f"Bearer {JETON}"}, timeout=120)
            if r.status_code == 403:
                raise SystemExit("❌ jeton refusé par le site (SCOUTIA_WORKER_TOKEN)")
            r.raise_for_status()
            return r.json()
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [SITE] {chemin} : {e} (essai {k + 1}/{essais})", flush=True)
            time.sleep(10 * (k + 1))
    return {}


def _preparer_parseq():
    """Poids PARSeq affinés (lecture des numéros) : téléchargés depuis R2 au démarrage s'ils ne sont pas là."""
    dest = os.environ.get("SCOUTIA_PARSEQ_POIDS") or "/app/models/parseq_scoutia_maillots_v2.pt"
    if not os.path.exists(dest):
        cle = os.environ.get("SCOUTIA_PARSEQ_R2_KEY", "modeles/parseq_scoutia_maillots_v2.pt")
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _r2().download_file(BUCKET, cle, dest)
            print(f"  PARSeq : poids téléchargés depuis R2 ({cle})", flush=True)
        except Exception as e:
            print(f"  ⚠️  PARSeq : poids introuvables sur R2 ({cle}) : {e} — le mode joueur ne pourra pas lire les numéros")
    os.environ["SCOUTIA_PARSEQ_POIDS"] = dest


def _envoyer_resultats(dossier, prefixe):
    client = _r2()
    n = 0
    for racine, _, fichiers in os.walk(dossier):
        for f in fichiers:
            local = os.path.join(racine, f)
            cle = prefixe + os.path.relpath(local, dossier).replace("\\", "/")
            client.upload_file(local, BUCKET, cle)
            n += 1
    return n


def traiter(job):
    aid = int(job["id"])
    dossier = os.path.join(TRAVAIL, "sortie", str(aid))
    video = os.path.join(TRAVAIL, "videos", f"{aid}_{os.path.basename(job['r2_key'])}")
    os.makedirs(os.path.dirname(video), exist_ok=True)
    shutil.rmtree(dossier, ignore_errors=True)
    debut = time.time()
    fini = threading.Event()

    def _battement():            # nouvelles au site toutes les minutes (sinon il croit la machine perdue)
        while not fini.wait(60):
            mn = (time.time() - debut) / 60
            _site("/api/worker/progression", {"id": aid, "progress": min(95, 5 + mn / 130 * 90),
                                              "message": f"Analyse en cours ({mn:.0f} min)…"}, essais=1)
    threading.Thread(target=_battement, daemon=True).start()
    try:
        print(f"\n══ ANALYSE #{aid} ({job.get('vitesse')}, {job.get('mode')}) — {job.get('filename')}", flush=True)
        _r2().download_file(BUCKET, job["r2_key"], video)
        _site("/api/worker/progression", {"id": aid, "progress": 5, "message": "Vidéo reçue, analyse en cours…"}, essais=1)
        from analyse_execution import executer_analyse
        cree = datetime.fromisoformat(job["cree_le"]) if job.get("cree_le") else None
        result, empreinte = executer_analyse(aid, video, job.get("sport", "football"), job.get("plan", "free"), dossier,
                                             vitesse=job.get("vitesse", "rapide"), cree_le=cree, **{
                                                 k: job.get(k) for k in ("mode", "player_id", "team_names",
                                                                         "player_position", "kickoff_s_precalcule",
                                                                         "team_colors_gemini_precalcule",
                                                                         "couleur_maillot_gardien") if k in job},
                                             use_match_boundary_segments=job.get("use_match_boundary_segments", True))
        fini.set()
        _site("/api/worker/progression", {"id": aid, "progress": 97, "message": "Envoi des résultats…"}, essais=1)
        prefixe = f"resultats/{aid}/"
        n = _envoyer_resultats(dossier, prefixe)
        print(f"  résultats déposés sur R2 : {n} fichier(s) | {(time.time() - debut) / 60:.0f} min", flush=True)
        _site("/api/worker/termine", {"id": aid, "ok": True, "result": json.loads(json.dumps(result, default=str)),
                                      "sortie_distante": dossier, "prefixe_r2": prefixe, "empreinte": empreinte})
    except Exception as e:
        fini.set()
        traceback.print_exc()
        _site("/api/worker/termine", {"id": aid, "ok": False, "erreur": f"{type(e).__name__}: {e}"[:500]})
    finally:
        fini.set()
        for p in (video, dossier):
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))


def boucle():
    if not (SITE and JETON):
        raise SystemExit("❌ SCOUTIA_SITE_URL et SCOUTIA_WORKER_TOKEN sont nécessaires")
    print(f"Machine GPU ScoutIA prête ({POD_ID}) → {SITE}", flush=True)
    _preparer_parseq()
    derniere_activite = time.time()
    while True:
        rep = _site("/api/worker/prochaine", {})
        job = rep.get("analyse")
        if job:
            traiter(job)
            derniere_activite = time.time()
            continue
        if (time.time() - derniere_activite) / 60 >= INACTIF_MIN:
            if _site("/api/worker/inactif", {}).get("arret"):
                print("Plus de travail : la machine va être supprimée par le site.", flush=True)
                while True:
                    time.sleep(3600)
            derniere_activite = time.time()
        time.sleep(20)


if __name__ == "__main__":
    boucle()
