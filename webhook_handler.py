# webhook_handler.py
# ─────────────────────────────────────────────────────────────────
# ScoutIA — Handler webhook RunPod
# À intégrer dans app.py :
#
#   from webhook_handler import handle_runpod_webhook
#
#   @app.route("/webhook/runpod", methods=["POST"])
#   def runpod_webhook():
#       return handle_runpod_webhook(request, db, Analysis)
#
# ─────────────────────────────────────────────────────────────────

import os
import json
import logging
from flask import jsonify

log = logging.getLogger(__name__)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")


def handle_runpod_webhook(request, db, Analysis):
    """
    Reçoit la notification RunPod quand une analyse est terminée.
    Met à jour la DB et stocke les résultats.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "payload vide"}), 400

        # Vérifier le secret
        if WEBHOOK_SECRET and data.get("secret") != WEBHOOK_SECRET:
            log.warning("Webhook secret invalide")
            return jsonify({"error": "unauthorized"}), 401

        analysis_id = data.get("analysis_id")
        status      = data.get("status")
        payload     = data.get("data", {})

        if not analysis_id:
            return jsonify({"error": "analysis_id manquant"}), 400

        # Trouver l'analyse en DB
        analysis = Analysis.query.get(analysis_id)
        if not analysis:
            log.error(f"Webhook : analyse {analysis_id} introuvable en DB")
            return jsonify({"error": "analyse introuvable"}), 404

        log.info(f"Webhook RunPod : analysis={analysis_id} status={status}")

        if status == "processing":
            # Mise à jour progression
            progress     = payload.get("progress", analysis.progress)
            progress_msg = payload.get("progress_msg", "")
            analysis.progress     = progress
            analysis.progress_msg = progress_msg
            db.session.commit()

        elif status == "done":
            # Stocker les résultats
            outputs   = payload.get("outputs", {})
            summary   = payload.get("summary", {})
            highlights = payload.get("highlights", [])
            stats     = payload.get("stats", {})
            jersey_map = payload.get("jersey_map", {})

            result_data = {
                "summary":    summary,
                "stats":      stats,
                "highlights": highlights,
                "jersey_map": jersey_map,
                "mvp":        payload.get("mvp"),
                "formation":  payload.get("formation"),
                "possession": payload.get("possession", {}),
                "match_story": payload.get("match_story", ""),
                "pdf":        outputs.get("pdf_url"),
                "reel":       outputs.get("reel_url"),
                "heatmaps":   outputs.get("heatmap_urls", {}),
                "elapsed_s":  payload.get("elapsed_s"),
            }

            analysis.status       = "done"
            analysis.progress     = 100
            analysis.progress_msg = "Analyse terminée !"
            analysis.result_json  = json.dumps(result_data)
            db.session.commit()

            log.info(f"Analyse {analysis_id} terminée — "
                     f"buts={summary.get('goals',0)} "
                     f"tirs={summary.get('shots',0)}")

        elif status == "error":
            error_msg = payload.get("progress_msg", "Erreur inconnue")
            analysis.status       = "error"
            analysis.progress_msg = error_msg
            db.session.commit()
            log.error(f"Analyse {analysis_id} erreur : {error_msg}")

        return jsonify({"ok": True}), 200

    except Exception as e:
        log.error(f"Webhook erreur : {e}")
        return jsonify({"error": str(e)}), 500
