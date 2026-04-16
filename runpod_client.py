# runpod_client.py
# ─────────────────────────────────────────────────────────────────
# ScoutIA — Client RunPod
# Utilisé par app.py pour soumettre les jobs d'analyse
# ─────────────────────────────────────────────────────────────────

import os
import json
import logging
import requests

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY",   "")
RUNPOD_ENDPOINT  = os.getenv("RUNPOD_ENDPOINT",  "")  # ex: https://api.runpod.ai/v2/ENDPOINT_ID/run
RUNPOD_ENABLED   = bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT)


def submit_analysis_job(analysis_id, video_r2_key, sport, plan, mode="match", player_id=None):
    """
    Soumet un job d'analyse à RunPod.
    Retourne True si le job est bien soumis, False sinon.

    Si RunPod n'est pas configuré → fallback sur pipeline local (thread).
    """
    if not RUNPOD_ENABLED:
        log.warning("RunPod non configuré — fallback pipeline local")
        return False

    payload = {
        "input": {
            "analysis_id":  analysis_id,
            "video_r2_key": video_r2_key,
            "sport":        sport,
            "plan":         plan,
            "mode":         mode,
            "player_id":    player_id,
        }
    }

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    try:
        resp = requests.post(
            RUNPOD_ENDPOINT,
            json    = payload,
            headers = headers,
            timeout = 15,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id")
        log.info(f"RunPod job soumis : {job_id} pour analyse {analysis_id}")
        return True

    except Exception as e:
        log.error(f"RunPod submit erreur : {e}")
        return False


def upload_video_to_r2(local_path, analysis_id):
    """
    Upload la vidéo sur R2 avant de soumettre le job RunPod.
    Retourne la clé R2 du fichier uploadé.
    """
    import boto3

    r2_account_id = os.getenv("R2_ACCOUNT_ID", "")
    r2_access_key = os.getenv("R2_ACCESS_KEY_ID", "")
    r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "")
    r2_bucket     = os.getenv("R2_BUCKET_NAME", "scoutia-videos")

    if not all([r2_account_id, r2_access_key, r2_secret_key]):
        log.warning("R2 non configuré — impossible d'uploader")
        return None

    r2_key = f"uploads/{analysis_id}/video.mp4"
    endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"

    client = boto3.client(
        "s3",
        endpoint_url          = endpoint,
        aws_access_key_id     = r2_access_key,
        aws_secret_access_key = r2_secret_key,
        region_name           = "auto",
    )

    log.info(f"Upload R2 : {local_path} → {r2_key}")
    client.upload_file(local_path, r2_bucket, r2_key)
    log.info(f"Upload R2 OK : {r2_key}")
    return r2_key
