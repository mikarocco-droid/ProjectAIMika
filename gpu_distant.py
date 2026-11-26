# gpu_distant.py — 26/11/2026
# Pilotage des machines GPU RunPod depuis le SITE : création à la demande, suppression, état.
# API REST RunPod (https://rest.runpod.io/v1/pods, jeton « Bearer »). Aucune clé RunPod n'est envoyée à la machine GPU :
# c'est le SITE qui la crée et la supprime (la machine signale seulement qu'elle n'a plus de travail).
import os

import requests

import config

_API = "https://rest.runpod.io/v1"


def _entetes():
    return {"Authorization": f"Bearer {config.RUNPOD_API_KEY}", "Content-Type": "application/json"}


def variables_pour_machine():
    """Variables transmises à la machine GPU : adresse du site, jeton partagé, réglages de l'analyse, clés utiles."""
    env = {"SCOUTIA_SITE_URL": config.SITE_URL, "SCOUTIA_WORKER_TOKEN": config.WORKER_TOKEN,
           "SCOUTIA_GPU_INACTIF_MIN": str(config.GPU_INACTIF_MIN), "MODE": "gpu_distant"}
    for k in config.VARIABLES_POUR_GPU:
        if os.environ.get(k):
            env[k] = os.environ[k]
    # jeton R2 PROPRE aux machines GPU (révocable sans toucher au site), s'il est défini
    if os.environ.get("R2_ACCESS_KEY_ID_GPU") and os.environ.get("R2_SECRET_ACCESS_KEY_GPU"):
        env["R2_ACCESS_KEY_ID"] = os.environ["R2_ACCESS_KEY_ID_GPU"]
        env["R2_SECRET_ACCESS_KEY"] = os.environ["R2_SECRET_ACCESS_KEY_GPU"]
    return env


def creer_machine():
    """Crée une machine GPU. Renvoie son identifiant, ou None (avec la raison dans le journal)."""
    if not (config.RUNPOD_API_KEY and config.RUNPOD_IMAGE and config.SITE_URL and config.WORKER_TOKEN):
        print("  [GPU] création impossible : RUNPOD_API_KEY, RUNPOD_IMAGE, SCOUTIA_SITE_URL ou SCOUTIA_WORKER_TOKEN manquant")
        return None
    corps = {"name": "scoutia-gpu", "imageName": config.RUNPOD_IMAGE, "gpuTypeIds": config.RUNPOD_GPU_TYPES,
             "gpuTypePriority": "custom", "gpuCount": 1, "cloudType": config.RUNPOD_CLOUD_TYPE,
             "containerDiskInGb": config.RUNPOD_DISQUE_GO, "volumeInGb": 0, "env": variables_pour_machine(),
             "interruptible": False}
    try:
        r = requests.post(f"{_API}/pods", json=corps, headers=_entetes(), timeout=60)
        if r.status_code >= 300:
            print(f"  [GPU] RunPod a refusé la création ({r.status_code}) : {r.text[:300]}")
            return None
        pod_id = (r.json() or {}).get("id")
        print(f"  [GPU] machine créée : {pod_id} ({config.RUNPOD_GPU_TYPES[0]}…, {config.RUNPOD_CLOUD_TYPE})")
        return pod_id
    except Exception as e:
        print(f"  [GPU] création impossible : {e}")
        return None


def supprimer_machine(pod_id):
    """Supprime la machine (plus rien n'est facturé, ni calcul ni disque)."""
    if not (pod_id and config.RUNPOD_API_KEY):
        return False
    try:
        r = requests.delete(f"{_API}/pods/{pod_id}", headers=_entetes(), timeout=60)
        ok = r.status_code < 300 or r.status_code == 404
        print(f"  [GPU] machine {pod_id} supprimée" if ok else f"  [GPU] suppression refusée ({r.status_code}) : {r.text[:200]}")
        return ok
    except Exception as e:
        print(f"  [GPU] suppression impossible : {e}")
        return False


def etat_machine(pod_id):
    """« RUNNING », « EXITED »… ; None si la machine n'existe plus ; « ? » si RunPod ne répond pas."""
    try:
        r = requests.get(f"{_API}/pods/{pod_id}", headers=_entetes(), timeout=30)
        if r.status_code == 404:
            return None
        return (r.json() or {}).get("desiredStatus") or "?"
    except Exception:
        return "?"
