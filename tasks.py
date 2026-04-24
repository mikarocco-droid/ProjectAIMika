# tasks.py — Tâches Celery ScoutIA
# Wrape run_pipeline() en tâche asynchrone

import os
import time
import traceback
from celery import Celery
from celery.utils.log import get_task_logger

# ─────────────────────────────────────────
# INIT CELERY
# ─────────────────────────────────────────
BROKER  = os.getenv("CELERY_BROKER_URL",     "redis://localhost:6379/0")
BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "scoutia",
    broker  = BROKER,
    backend = BACKEND,
)

celery_app.conf.update(
    task_serializer        = "json",
    result_serializer      = "json",
    accept_content         = ["json"],
    task_track_started     = True,
    task_acks_late         = True,          # requeue si worker crash
    worker_prefetch_multiplier = 1,         # 1 tâche à la fois par worker
    task_routes            = {
        "tasks.run_analysis": {"queue": "pipeline"},
    },
    result_expires         = 86400,         # résultats conservés 24h
)

logger = get_task_logger(__name__)


# ─────────────────────────────────────────
# HELPERS PROGRESSION
# ─────────────────────────────────────────
def _update_state(task, step, total, message):
    """Met à jour le statut visible depuis l'API."""
    task.update_state(
        state   = "PROGRESS",
        meta    = {
            "step":    step,
            "total":   total,
            "message": message,
            "pct":     int(step / total * 100),
        }
    )


# ─────────────────────────────────────────
# TÂCHE PRINCIPALE
# ─────────────────────────────────────────
@celery_app.task(
    bind             = True,
    name             = "tasks.run_analysis",
    max_retries      = 2,
    soft_time_limit  = 7200,   # 2h soft limit
    time_limit       = 7800,   # 2h10 hard limit
)
def run_analysis(self, video_path, sport="football", output_dir=None,
                 mode="match", plan="free", analysis_id=None,
                 use_coarse_scan=True):
    """
    Tâche Celery : lance run_pipeline() en arrière-plan.

    Returns:
        dict avec summary, highlights, pdf_path, reel_path
    """
    task_id = self.request.id
    logger.info(f"[{task_id}] START video={video_path} sport={sport}")

    if output_dir is None:
        output_dir = os.path.join("outputs", task_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # ── Étape 1 : import pipeline ──────────────────────────────
        _update_state(self, 1, 10, "Initialisation pipeline...")
        from pipeline import run_pipeline

        # ── Étape 2 : lancement ────────────────────────────────────
        _update_state(self, 2, 10, "Analyse en cours...")
        t0 = time.time()

        result = run_pipeline(
            video_path      = video_path,
            sport           = sport,
            output_dir      = output_dir,
            mode            = mode,
            plan            = plan,
            analysis_id     = analysis_id,
            use_coarse_scan = use_coarse_scan,
        )

        elapsed = time.time() - t0

        # ── Étape 3 : préparer résultat sérialisable ───────────────
        _update_state(self, 9, 10, "Finalisation...")

        summary    = result.get("summary", {})
        highlights = result.get("highlights", [])

        # Nettoyer pour JSON (pas d'objets numpy etc.)
        clean_result = {
            "task_id":    task_id,
            "status":     "SUCCESS",
            "elapsed_s":  round(elapsed, 1),
            "video_path": video_path,
            "output_dir": output_dir,
            "summary": {
                "goals":     summary.get("goals", 0),
                "shots":     summary.get("shots", 0),
                "total_xg":  summary.get("total_xg", 0),
                "players":   summary.get("players", 0),
                "passes":    summary.get("passes", 0),
                "possession": summary.get("possession", {}),
                "mvp":       summary.get("mvp_jersey", "?"),
                "formation": summary.get("formation", "?"),
            },
            "highlights_count": len(highlights),
            "pdf_path":  result.get("pdf"),
            "reel_path": result.get("reel"),
            "coarse_stats": result.get("coarse_stats", {}),
        }

        logger.info(f"[{task_id}] DONE in {elapsed:.1f}s | "
                    f"goals={clean_result['summary']['goals']} "
                    f"shots={clean_result['summary']['shots']}")

        return clean_result

    except Exception as exc:
        logger.error(f"[{task_id}] ERROR : {exc}\n{traceback.format_exc()}")
        # Retry automatique (max 2 fois)
        raise self.retry(exc=exc, countdown=30)


# ─────────────────────────────────────────
# TÂCHE LÉGÈRE — coarse scan seul (pre-check)
# ─────────────────────────────────────────
@celery_app.task(
    bind            = True,
    name            = "tasks.run_coarse_only",
    soft_time_limit = 600,
)
def run_coarse_only(self, video_path, sport="football"):
    """
    Lance uniquement le coarse scan pour pré-analyser une vidéo.
    Utile pour afficher rapidement 'X segments détectés' à l'utilisateur.
    """
    try:
        from coarse_scan import run_coarse_scan
        segments, stats = run_coarse_scan(video_path, sport=sport)
        return {
            "status":   "SUCCESS",
            "segments": segments,
            "stats":    stats,
        }
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10, max_retries=1)
