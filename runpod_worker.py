#!/usr/bin/env python3
# runpod_worker.py
# ─────────────────────────────────────────────────────────────────
# ScoutIA — Worker RunPod V9.6
# Ce script tourne dans le container Docker sur RunPod.
# Il reçoit un job, télécharge la vidéo depuis R2,
# lance le pipeline, uploade les résultats sur R2,
# et notifie l'app Flask via webhook.
# ─────────────────────────────────────────────────────────────────

import os
import sys
import json
import time
import shutil
import logging
import subprocess
import traceback
import requests

import runpod

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s %(message)s",
    stream = sys.stdout,
)
log = logging.getLogger()


# ─────────────────────────────────────────
# CONFIG (depuis variables d'environnement RunPod)
# ─────────────────────────────────────────
GITHUB_REPO      = os.getenv("GITHUB_REPO",      "https://github.com/mikarocco-droid/ProjectAIMika.git")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY",   "")
CLAUDE_API_KEY   = os.getenv("CLAUDE_API_KEY",   "")
R2_ACCOUNT_ID    = os.getenv("R2_ACCOUNT_ID",    "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY    = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET        = os.getenv("R2_BUCKET_NAME",   "scoutia-videos")
R2_PUBLIC_URL    = os.getenv("R2_PUBLIC_URL",     "")
WEBHOOK_SECRET   = os.getenv("WEBHOOK_SECRET",    "")
APP_WEBHOOK_URL  = os.getenv("APP_WEBHOOK_URL",   "")  # ex: https://scoutia.fr/webhook/runpod

WORK_DIR    = "/tmp/scoutia"
PROJECT_DIR = f"{WORK_DIR}/ProjectAIMika"


# ─────────────────────────────────────────
# SETUP — clone + environnement
# ─────────────────────────────────────────
def setup_project():
    """Clone le projet GitHub et configure l'environnement."""
    os.makedirs(WORK_DIR, exist_ok=True)

    if os.path.exists(PROJECT_DIR):
        shutil.rmtree(PROJECT_DIR)

    log.info(f"Clone {GITHUB_REPO}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", GITHUB_REPO, PROJECT_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Clone failed : {result.stderr}")

    log.info("Clone OK")

    # Créer les dossiers nécessaires
    for d in ["uploads", "outputs", "instance", "models", ".env", "outputs/learning"]:
        os.makedirs(f"{PROJECT_DIR}/{d}", exist_ok=True)

    for pkg in ["analysis", "analytics", "ai", "vision", "video",
                "rendering", "export", "payments", "sports", "tracking"]:
        os.makedirs(f"{PROJECT_DIR}/{pkg}", exist_ok=True)
        init = f"{PROJECT_DIR}/{pkg}/__init__.py"
        if not os.path.exists(init):
            open(init, "w").close()

    # Écrire le .env
    env_content = f"""SECRET_KEY=runpod-worker
DATABASE_URI=sqlite:///db.sqlite3
DEBUG=false
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
FPS=25
FRAME_WIDTH=1920
FRAME_HEIGHT=1080
YOLO_MODEL=yolo11m.pt
YOLO_CONFIDENCE=0.4
BALL_METHOD=hybrid
YOLO_BATCH_SIZE=16
FRAME_SKIP_EVERY=4
SHOT_ZONE_FOOTBALL=900
HIGHLIGHT_MAX=15
HIGHLIGHT_MIN_SCORE=4
MONTAGE_RESOLUTION=1280x720
CLAUDE_API_KEY={CLAUDE_API_KEY}
GEMINI_API_KEY={GEMINI_API_KEY}
STRIPE_SECRET_KEY=
PLAN_FREE_MAX=1
PLAN_STARTER_MAX=20
PLAN_PRO_MAX=100
PLAN_UNIQUE_MAX=1

R2_ACCOUNT_ID={R2_ACCOUNT_ID}
R2_ACCESS_KEY_ID={R2_ACCESS_KEY_ID}
R2_SECRET_ACCESS_KEY={R2_SECRET_KEY}
R2_BUCKET_NAME={R2_BUCKET}
R2_PUBLIC_URL={R2_PUBLIC_URL}

DELETE_RAW_VIDEO=true
"""
    with open(f"{PROJECT_DIR}/.env/param.env", "w") as f:
        f.write(env_content)

    log.info("Environnement configuré")


# ─────────────────────────────────────────
# R2 — Download / Upload
# ─────────────────────────────────────────
def get_r2_client():
    import boto3
    endpoint = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url          = endpoint,
        aws_access_key_id     = R2_ACCESS_KEY_ID,
        aws_secret_access_key = R2_SECRET_KEY,
        region_name           = "auto",
    )


def download_from_r2(r2_key, local_path):
    """Télécharge un fichier depuis R2."""
    log.info(f"Download R2 : {r2_key} → {local_path}")
    client = get_r2_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client.download_file(R2_BUCKET, r2_key, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    log.info(f"Download OK : {size_mb:.1f} MB")


def upload_to_r2(local_path, r2_key):
    """Upload un fichier vers R2. Retourne l'URL publique."""
    if not os.path.exists(local_path):
        log.warning(f"Upload R2 : fichier manquant {local_path}")
        return None
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    log.info(f"Upload R2 : {local_path} → {r2_key} ({size_mb:.1f} MB)")
    client = get_r2_client()
    client.upload_file(local_path, R2_BUCKET, r2_key)
    url = f"{R2_PUBLIC_URL}/{r2_key}" if R2_PUBLIC_URL else None
    log.info(f"Upload OK : {url}")
    return url


# ─────────────────────────────────────────
# WEBHOOK — notifier l'app Flask
# ─────────────────────────────────────────
def notify_app(analysis_id, status, data=None):
    """Notifie l'app Flask que l'analyse est terminée."""
    if not APP_WEBHOOK_URL:
        log.warning("APP_WEBHOOK_URL non configuré — notification ignorée")
        return

    payload = {
        "analysis_id": analysis_id,
        "status":      status,
        "secret":      WEBHOOK_SECRET,
        "data":        data or {},
    }

    try:
        resp = requests.post(
            APP_WEBHOOK_URL,
            json    = payload,
            timeout = 10,
            headers = {"Content-Type": "application/json"},
        )
        log.info(f"Webhook → {resp.status_code}")
    except Exception as e:
        log.error(f"Webhook erreur : {e}")


# ─────────────────────────────────────────
# PROGRESS CALLBACK
# ─────────────────────────────────────────
def make_progress_callback(analysis_id):
    """Retourne un callback qui notifie la progression en temps réel."""
    last_pct = [0]

    def callback(pct):
        pct = int(pct)
        if pct - last_pct[0] >= 5:  # notifier tous les 5%
            last_pct[0] = pct
            notify_app(analysis_id, "processing", {
                "progress":     pct,
                "progress_msg": f"Analyse en cours... {pct}%",
            })

    return callback


# ─────────────────────────────────────────
# HANDLER PRINCIPAL
# ─────────────────────────────────────────
def handler(job):
    """
    Point d'entrée RunPod.
    job["input"] contient :
        - analysis_id  : ID de l'analyse en DB
        - video_r2_key : clé R2 de la vidéo
        - sport        : "football", "basketball", etc.
        - plan         : "free", "starter", "pro", "unique"
        - mode         : "match" ou "joueur"
        - player_id    : ID joueur si mode="joueur"
    """
    job_input   = job.get("input", {})
    analysis_id = job_input.get("analysis_id", "unknown")
    video_r2_key = job_input.get("video_r2_key")
    sport       = job_input.get("sport",     "football")
    plan        = job_input.get("plan",      "pro")
    mode        = job_input.get("mode",      "match")
    player_id   = job_input.get("player_id", None)

    log.info(f"Job reçu : analysis_id={analysis_id} sport={sport} mode={mode}")

    # Notifier démarrage
    notify_app(analysis_id, "processing", {
        "progress":     5,
        "progress_msg": "Démarrage de l'analyse...",
    })

    try:
        # ── 1. Setup projet ──────────────────────────
        setup_project()
        sys.path.insert(0, PROJECT_DIR)
        os.chdir(PROJECT_DIR)

        # ── 2. Download vidéo depuis R2 ──────────────
        local_video = f"{PROJECT_DIR}/uploads/{analysis_id}.mp4"
        download_from_r2(video_r2_key, local_video)

        notify_app(analysis_id, "processing", {
            "progress":     10,
            "progress_msg": "Vidéo reçue — lancement YOLO...",
        })

        # ── 3. Lancer le pipeline ────────────────────
        output_dir = f"{PROJECT_DIR}/outputs/{analysis_id}"
        os.makedirs(output_dir, exist_ok=True)

        # Vider le cache modules
        for m in list(sys.modules.keys()):
            if any(m.startswith(p) for p in
                   ['config','pipeline','main','vision','analysis',
                    'analytics','ai','video','rendering','export',
                    'payments','sports','tracking']):
                del sys.modules[m]

        from pipeline import run_pipeline

        start  = time.time()
        result = run_pipeline(
            video_path        = local_video,
            sport             = sport,
            output_dir        = output_dir,
            plan              = plan,
            mode              = mode,
            player_id         = player_id,
            progress_callback = make_progress_callback(analysis_id),
        )
        elapsed = time.time() - start
        log.info(f"Pipeline terminé en {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # ── 4. Upload résultats sur R2 ───────────────
        outputs = {}

        # PDF
        if result.get("pdf") and os.path.exists(result["pdf"]):
            key = f"outputs/{analysis_id}/rapport.pdf"
            url = upload_to_r2(result["pdf"], key)
            outputs["pdf_url"]  = url
            outputs["pdf_key"]  = key

        # Reel vidéo
        if result.get("reel") and os.path.exists(result["reel"]):
            key = f"outputs/{analysis_id}/reel.mp4"
            url = upload_to_r2(result["reel"], key)
            outputs["reel_url"] = url
            outputs["reel_key"] = key

        # Heatmaps
        heatmap_urls = {}
        for name, path in result.get("heatmaps", {}).items():
            if path and os.path.exists(path):
                key = f"outputs/{analysis_id}/heatmaps/{name}.png"
                url = upload_to_r2(path, key)
                if url:
                    heatmap_urls[name] = url
        if heatmap_urls:
            outputs["heatmap_urls"] = heatmap_urls

        # ── 5. Préparer le résultat ──────────────────
        summary = result.get("summary", {})
        final_data = {
            "outputs":    outputs,
            "summary":    summary,
            "stats":      result.get("stats",          {}),
            "highlights": result.get("highlights",     [])[:15],
            "jersey_map": result.get("jersey_map",     {}),
            "mvp":        result.get("mvp"),
            "formation":  result.get("formation"),
            "possession": result.get("possession",     {}),
            "match_story": result.get("match_story",   ""),
            "elapsed_s":  round(elapsed, 1),
        }

        # ── 6. Notifier succès ───────────────────────
        notify_app(analysis_id, "done", final_data)

        log.info(f"Job {analysis_id} terminé — "
                 f"buts={summary.get('goals',0)} "
                 f"tirs={summary.get('shots',0)} "
                 f"highlights={len(result.get('highlights',[]))}")

        return {"status": "done", "analysis_id": analysis_id, **final_data}

    except Exception as e:
        error_msg = traceback.format_exc()
        log.error(f"Job {analysis_id} erreur :\n{error_msg}")
        notify_app(analysis_id, "error", {
            "progress_msg": f"Erreur : {str(e)[:200]}",
        })
        return {"status": "error", "analysis_id": analysis_id, "error": str(e)}

    finally:
        # Nettoyer les fichiers locaux pour libérer le disque
        try:
            if os.path.exists(local_video):
                os.remove(local_video)
            log.info("Nettoyage local OK")
        except Exception:
            pass


# ─────────────────────────────────────────
# DÉMARRAGE
# ─────────────────────────────────────────
if __name__ == "__main__":
    log.info("ScoutIA RunPod Worker V9.6 — démarrage")
    runpod.serverless.start({"handler": handler})
