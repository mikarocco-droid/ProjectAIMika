# tasks.py — Tâches Celery ScoutIA
#
# 24/10/2026 : CHEMIN D'ANALYSE UNIQUE. La tâche délègue à app.run_analysis(), la fonction complète qui :
#   - passe l'analyse en « processing », lance run_pipeline() avec TOUS les paramètres (dont le joueur demandé),
#   - refuse une analyse incomplète (service d'IA indisponible, crédit ou quota) au lieu de la livrer,
#   - enregistre le statut « done » et le résultat (result_json) dans la base,
#   - supprime la vidéo brute et envoie les fichiers vers R2.
# Avant : la tâche lançait run_pipeline() elle-même et n'écrivait RIEN dans la base (analyse « pending » pour
# toujours, page de résultats vide, vidéo jamais supprimée) et ne transmettait pas le joueur demandé.
# Pas de relance automatique : une erreur ne coûte plus jusqu'à trois analyses complètes (GPU et Gemini).

import os
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
    task_acks_late         = True,          # requeue si le worker meurt en cours de tâche
    worker_prefetch_multiplier = 1,         # 1 tâche à la fois par worker
    task_routes            = {
        "tasks.run_analysis": {"queue": "pipeline"},
    },
    result_expires         = 86400,         # résultats Celery conservés 24h (le résultat utile est en base)
)

logger = get_task_logger(__name__)


def _update_state(task, step, total, message):
    """Progression visible depuis /api/status."""
    task.update_state(state="PROGRESS",
                      meta={"step": step, "total": total, "message": message, "pct": int(step / total * 100)})


# ─────────────────────────────────────────
# TÂCHE PRINCIPALE
# ─────────────────────────────────────────
@celery_app.task(
    bind             = True,
    name             = "tasks.run_analysis",
    max_retries      = 0,      # pas de relance automatique (voir l'en-tête)
    soft_time_limit  = 7200,   # 2 h : l'exception est rattrapée par app.run_analysis -> statut « error »
    time_limit       = 7800,   # 2 h 10 : arrêt forcé
)
def run_analysis(self, analysis_id, video_path=None, sport="football", plan="free",
                 mode="match", player_id=None, r2_key=None, team_names=None,
                 player_position=None, kickoff_s_precalcule=None,
                 team_colors_gemini_precalcule=None, use_match_boundary_segments=True):
    """Lance l'analyse complète d'app.py et renvoie son statut final (lu en base)."""
    task_id = self.request.id
    logger.info(f"[{task_id}] START analyse #{analysis_id} mode={mode} player_id={player_id}")
    _update_state(self, 1, 10, "Analyse en cours...")

    # Import différé : app.py importe ce module au chargement (import circulaire évité)
    from app import run_analysis as _run_analysis_complete, app as _flask_app, db as _db, Analysis as _Analysis

    _run_analysis_complete(
        analysis_id, video_path, sport, plan,
        mode                          = mode,
        player_id                     = player_id,
        r2_key                        = r2_key,
        team_names                    = team_names,
        player_position               = player_position,
        kickoff_s_precalcule          = kickoff_s_precalcule,
        team_colors_gemini_precalcule = team_colors_gemini_precalcule,
        use_match_boundary_segments   = use_match_boundary_segments,
    )

    with _flask_app.app_context():
        a = _db.session.get(_Analysis, analysis_id)
        statut, message = (a.status, a.progress_msg) if a else ("introuvable", None)
    logger.info(f"[{task_id}] FIN analyse #{analysis_id} : {statut} — {message}")
    return {"analysis_id": analysis_id, "status": statut, "message": message}


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
