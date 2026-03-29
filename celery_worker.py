# ─────────────────────────────────────────
# CELERY WORKER V6 — SCOUT IA
# ─────────────────────────────────────────

import os
import json
import traceback
import logging

from celery import Celery

# ─────────────────────────────────────────
# CONFIG LOGS
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CELERY INIT
# ─────────────────────────────────────────
celery = Celery(
    "scout_tasks",
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
)

# Optionnel mais recommandé
celery.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Brussels",
    enable_utc=True
)

# ─────────────────────────────────────────
# TASK PRINCIPALE
# ─────────────────────────────────────────
@celery.task(bind=True)
def process_video(
    self,
    analysis_id,
    video_path,
    mode="match",
    sport="football",
    numero_joueur=None,
    couleur_maillot=None,
    position_joueur=None,
    couleur_gardien_domicile=None,
    couleur_gardien_visiteur=None
):
    """
    Task principale Celery pour lancer pipeline V6
    """

    # Import ici pour éviter conflits Flask/Celery
    from app import app, db, Analysis
    import scout  # ton module pipeline

    with app.app_context():

        a = db.session.get(Analysis, analysis_id)

        if not a:
            logger.error(f"Analysis {analysis_id} introuvable")
            return

        # ─────────────────────────────────────────
        # START
        # ─────────────────────────────────────────
        try:
            a.status = "processing"
            a.progress = 5
            a.progress_msg = "Initialisation..."
            db.session.commit()

            # ─────────────────────────────────────────
            # 1. SETUP
            # ─────────────────────────────────────────
            self.update_state(state='PROGRESS', meta={'progress': 10})

            output_dir, frames_dir = scout.setup_dirs(video_path)

            a.progress = 10
            a.progress_msg = "Préparation des fichiers..."
            db.session.commit()

            # ─────────────────────────────────────────
            # 2. EXTRACTION FRAMES
            # ─────────────────────────────────────────
            frames, duration, interval = scout.extract_frames(video_path, frames_dir)

            a.progress = 25
            a.progress_msg = f"{len(frames)} frames extraites"
            db.session.commit()

            # ─────────────────────────────────────────
            # 3. ANALYSE
            # ─────────────────────────────────────────
            if mode == "joueur":

                a.progress = 40
                a.progress_msg = f"Analyse joueur #{numero_joueur}"
                db.session.commit()

                analysis_result = scout.analyze_player(
                    frames,
                    duration,
                    interval,
                    numero=int(numero_joueur),
                    couleur=couleur_maillot,
                    position=position_joueur,
                    sport=sport,
                    couleur_gardien_domicile=couleur_gardien_domicile,
                    couleur_gardien_visiteur=couleur_gardien_visiteur
                )

            else:
                # MODE MATCH

                a.progress = 40
                a.progress_msg = "Analyse IA du match..."
                db.session.commit()

                analysis_result = scout.analyze_frames(
                    frames,
                    duration,
                    interval,
                    sport=sport
                )

                # ─────────────────────────────────────────
                # 4. AFFINAGE
                # ─────────────────────────────────────────
                a.progress = 70
                a.progress_msg = "Affinage des highlights..."
                db.session.commit()

                analysis_result = scout.refine_all_highlights(
                    video_path,
                    output_dir,
                    analysis_result
                )

                # ─────────────────────────────────────────
                # 5. CUT CLIPS
                # ─────────────────────────────────────────
                a.progress = 85
                a.progress_msg = "Découpe des clips..."
                db.session.commit()

                clips = scout.cut_highlights(
                    video_path,
                    output_dir,
                    analysis_result.get("highlights", [])
                )

                # ─────────────────────────────────────────
                # 6. REEL
                # ─────────────────────────────────────────
                a.progress = 95
                a.progress_msg = "Création du highlight reel..."
                db.session.commit()

                scout.create_highlight_reel(output_dir, clips)

            # ─────────────────────────────────────────
            # FIN
            # ─────────────────────────────────────────
            analysis_result["_mode"] = mode

            a.status = "done"
            a.progress = 100
            a.progress_msg = "Analyse terminée"
            a.result_json = json.dumps(analysis_result, ensure_ascii=False)
            a.output_dir = output_dir

            db.session.commit()

            logger.info(f"✅ Analyse {analysis_id} terminée")

        # ─────────────────────────────────────────
        # ERREUR
        # ─────────────────────────────────────────
        except Exception as e:

            error_trace = traceback.format_exc()

            print("\n❌ CELERY ERROR:")
            print(error_trace)

            logger.error(f"❌ CELERY ERROR: {str(e)}")
            logger.error(error_trace)

            a.status = "error"
            a.progress = 0
            a.progress_msg = str(e)

            db.session.commit()