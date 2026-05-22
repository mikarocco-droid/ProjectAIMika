# app.py
# -*- coding: utf-8 -*-

import os
import uuid
import json
import threading
import shutil
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(".env/param.env")

from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, send_from_directory, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import config
from config import allowed_file
# Celery — import conditionnel (fallback threading si Celery indispo)
try:
    from tasks import run_analysis as celery_run_analysis
    from tasks import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    print("  ⚠️  Celery non disponible — fallback threading")
from payments.stripe_handler import (
    create_checkout_session,
    create_portal_session,
    verify_checkout_session,
    handle_webhook
)

# ─────────────────────────────────────────
# SPORTS VALIDES
# ─────────────────────────────────────────
VALID_SPORTS = {
    "football", "mini-foot", "basketball", "handball",
    "rugby", "hockey sur glace", "hockey sur gazon",
    "tennis", "tennis de table", "padel"
}

# ─────────────────────────────────────────
# INIT APP
# ─────────────────────────────────────────
app = Flask(__name__)

config.validate()

app.secret_key                        = config.SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_URI
app.config["UPLOAD_FOLDER"]           = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"]      = config.MAX_UPLOAD_SIZE
app.config["DEBUG"]                   = config.DEBUG

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


# ─────────────────────────────────────────
# R2 / S3 CLIENT (optionnel)
# ─────────────────────────────────────────
def get_r2_client():
    """Retourne un client boto3 R2, ou None si R2 non configuré."""
    if not config.R2_ENABLED:
        return None
    try:
        import boto3
        return boto3.client(
            "s3",
            endpoint_url          = config.R2_ENDPOINT_URL,
            aws_access_key_id     = config.R2_ACCESS_KEY_ID,
            aws_secret_access_key = config.R2_SECRET_ACCESS_KEY,
            region_name           = "auto",
        )
    except ImportError:
        print("⚠️  boto3 non installé — pip install boto3")
        return None


def r2_generate_presigned_upload(filename, content_type="video/mp4", expires=3600):
    """
    Génère un presigned URL pour upload direct depuis le navigateur vers R2.
    Retourne (presigned_url, r2_key) ou (None, None) si R2 désactivé.
    """
    client = get_r2_client()
    if not client:
        return None, None

    r2_key = f"uploads/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
    try:
        url = client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket":      config.R2_BUCKET_NAME,
                "Key":         r2_key,
                "ContentType": content_type,
            },
            ExpiresIn=expires,
        )
        return url, r2_key
    except Exception as e:
        print(f"R2 presigned URL error: {e}")
        return None, None


def r2_download_to_local(r2_key, local_path):
    """Télécharge une vidéo depuis R2 vers le disque local du worker."""
    client = get_r2_client()
    if not client:
        return False
    try:
        client.download_file(config.R2_BUCKET_NAME, r2_key, local_path)
        return True
    except Exception as e:
        print(f"R2 download error: {e}")
        return False


def r2_delete(r2_key):
    """Supprime un fichier sur R2."""
    client = get_r2_client()
    if not client or not r2_key:
        return
    try:
        client.delete_object(Bucket=config.R2_BUCKET_NAME, Key=r2_key)
        print(f"  R2 supprimé : {r2_key}")
    except Exception as e:
        print(f"R2 delete error: {e}")


def r2_upload_outputs(local_dir, analysis_id):
    """
    Upload les outputs (highlights, PDF…) vers R2 après analyse.
    Retourne le préfixe R2 utilisé, ou None si R2 désactivé.
    """
    client = get_r2_client()
    if not client or not os.path.isdir(local_dir):
        return None

    prefix = f"outputs/{analysis_id}/"
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_file = os.path.join(root, fname)
            rel_path   = os.path.relpath(local_file, local_dir)
            r2_key     = prefix + rel_path.replace("\\", "/")
            try:
                client.upload_file(local_file, config.R2_BUCKET_NAME, r2_key)
            except Exception as e:
                print(f"R2 upload output error ({fname}): {e}")
    return prefix


# ─────────────────────────────────────────
# NETTOYAGE AUTOMATIQUE
# ─────────────────────────────────────────
def delete_raw_video(video_path, r2_key=None):
    """Supprime la vidéo brute (local + R2) après analyse."""
    if not config.DELETE_RAW_VIDEO_AFTER_ANALYSIS:
        return

    # Local
    if video_path and os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"  Vidéo brute supprimée : {video_path}")
        except Exception as e:
            print(f"  Erreur suppression locale : {e}")

    # R2
    if r2_key:
        r2_delete(r2_key)


def cleanup_expired_outputs():
    """
    Supprime les dossiers outputs des analyses expirées selon le plan de l'utilisateur.
    À appeler via un cron ou au démarrage.
    """
    with app.app_context():
        analyses = Analysis.query.filter_by(status="done").all()
        now      = datetime.utcnow()
        deleted  = 0

        for a in analyses:
            user = db.session.get(User, a.user_id)
            if not user:
                continue

            plan            = user.plan or "free"
            retention_days  = config.FILE_RETENTION_DAYS.get(plan, 7)
            expires_at      = a.created_at + timedelta(days=retention_days)

            if now > expires_at:
                # Supprime le dossier output local
                out_dir = a.output_dir or os.path.join(
                    config.OUTPUT_FOLDER, str(a.id)
                )
                if out_dir and os.path.isdir(out_dir):
                    try:
                        shutil.rmtree(out_dir)
                        print(f"  Outputs expirés supprimés : analyse #{a.id}")
                    except Exception as e:
                        print(f"  Erreur suppression outputs #{a.id}: {e}")

                # Supprime les outputs R2 si configuré
                if config.R2_ENABLED and a.r2_output_prefix:
                    _r2_delete_prefix(a.r2_output_prefix)

                # Met à jour le statut
                a.output_dir       = None
                a.r2_output_prefix = None
                db.session.commit()
                deleted += 1

        if deleted:
            print(f"  Nettoyage : {deleted} analyse(s) expirée(s) purgées")


def _r2_delete_prefix(prefix):
    """Supprime tous les fichiers R2 sous un préfixe donné."""
    client = get_r2_client()
    if not client:
        return
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config.R2_BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                client.delete_object(Bucket=config.R2_BUCKET_NAME, Key=obj["Key"])
        print(f"  R2 préfixe supprimé : {prefix}")
    except Exception as e:
        print(f"  R2 delete prefix error: {e}")


def start_cleanup_scheduler():
    """Lance le nettoyage automatique toutes les 24h en background."""
    def _loop():
        import time
        while True:
            try:
                cleanup_expired_outputs()
            except Exception as e:
                print(f"Cleanup scheduler error: {e}")
            time.sleep(86400)  # 24h

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print("✅  Scheduler nettoyage démarré (toutes les 24h)")


# ─────────────────────────────────────────
# HELPER PLAN
# ─────────────────────────────────────────
class PlanObj:
    def __init__(self, name, data):
        self.name    = name.capitalize()
        self.videos  = data.get("max_analyses",  3)
        self.pdf     = data.get("pdf",           False)
        self.montage = data.get("montage",       False)


def get_plan_obj(plan_name):
    data = config.PLANS.get(plan_name, config.PLANS["free"])
    return PlanObj(plan_name, data)


# ─────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────
class User(UserMixin, db.Model):
    id              = db.Column(db.Integer,     primary_key=True)
    email           = db.Column(db.String(150), unique=True, nullable=False)
    password        = db.Column(db.String(200), nullable=False)
    plan            = db.Column(db.String(50),  default="free")
    is_admin        = db.Column(db.Boolean,     default=False)
    stripe_customer = db.Column(db.String(100))
    stripe_sub      = db.Column(db.String(100))

    def can(self, feature):
        return config.PLANS.get(self.plan, {}).get(feature, False)

    def analyses_count(self):
        return Analysis.query.filter_by(user_id=self.id).count()

    def analyses_left(self):
        max_a = config.PLANS.get(self.plan, {}).get("max_analyses", 0)
        return max(0, max_a - self.analyses_count())


class Analysis(db.Model):
    id                 = db.Column(db.Integer,     primary_key=True)
    user_id            = db.Column(db.Integer,     nullable=False)
    filename           = db.Column(db.String(200))
    sport              = db.Column(db.String(50),  default="football")
    mode               = db.Column(db.String(20),  default="match")
    player_id          = db.Column(db.String(50),  nullable=True)
    status             = db.Column(db.String(20),  default="pending")
    progress           = db.Column(db.Integer,     default=0)
    progress_msg       = db.Column(db.String(200), default="En attente...")
    result_json        = db.Column(db.Text)
    output_dir         = db.Column(db.String(300))
    # Champs R2 (null si stockage local)
    r2_video_key       = db.Column(db.String(500), nullable=True)
    r2_output_prefix   = db.Column(db.String(500), nullable=True)
    created_at         = db.Column(db.DateTime,    default=datetime.utcnow)
    # Celery — tracking tâche asynchrone
    celery_task_id     = db.Column(db.String(200), nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")

        if not email or not pwd:
            flash("Email et mot de passe requis")
            return redirect(url_for("register"))

        if len(pwd) < 8:
            flash("Mot de passe trop court (8 caracteres minimum)")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email deja utilise")
            return redirect(url_for("register"))

        user = User(
            email    = email,
            password = generate_password_hash(pwd),
            plan     = "free"
        )
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash("Compte cree avec succes !")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")
        user  = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, pwd):
            login_user(user)
            return redirect(url_for("dashboard"))

        flash("Email ou mot de passe incorrect")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# ─────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pricing")
def pricing():
    return render_template("pricing.html", plans=config.PLANS)


@app.route("/dashboard")
@login_required
def dashboard():
    analyses = (
        Analysis.query
        .filter_by(user_id=current_user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )
    plan = get_plan_obj(current_user.plan)
    used = current_user.analyses_count()

    return render_template(
        "dashboard.html",
        analyses      = analyses,
        used          = used,
        plan          = plan,
        analyses_left = current_user.analyses_left()
    )


@app.route("/results/<int:id>")
@login_required
def results(id):
    a = db.session.get(Analysis, id)

    if not a or a.user_id != current_user.id:
        flash("Analyse introuvable")
        return redirect(url_for("dashboard"))

    result = json.loads(a.result_json) if a.result_json else {}

    return render_template(
        "results.html",
        analysis    = a,
        result      = result,
        can_pdf     = current_user.can("pdf"),
        can_montage = current_user.can("montage")
    )


# ─────────────────────────────────────────
# API — PRESIGNED URL (mode R2)
# ─────────────────────────────────────────
@app.route("/api/upload-url", methods=["POST"])
@login_required
def get_upload_url():
    """
    Génère un presigned URL pour upload direct vers R2.
    Utilisé par le frontend JS quand R2 est activé.
    Retourne : { presigned_url, r2_key, analysis_id }
    """
    if not config.R2_ENABLED:
        return jsonify({"error": "R2 non configuré"}), 400

    if current_user.analyses_left() <= 0:
        return jsonify({"error": "quota_exceeded"}), 403

    filename     = request.json.get("filename", "video.mp4")
    content_type = request.json.get("content_type", "video/mp4")
    sport        = request.json.get("sport", "football")
    mode         = request.json.get("mode", "match")
    player_id    = request.json.get("player_id") or None

    if not allowed_file(filename):
        return jsonify({"error": "format non supporté"}), 400

    safe_filename = secure_filename(filename)
    presigned_url, r2_key = r2_generate_presigned_upload(safe_filename, content_type)

    if not presigned_url:
        return jsonify({"error": "impossible de générer l'URL"}), 500

    # Crée l'analyse en BDD dès maintenant (status=pending)
    analysis = Analysis(
        user_id      = current_user.id,
        filename     = safe_filename,
        sport        = sport,
        mode         = mode,
        player_id    = player_id,
        status       = "pending",
        r2_video_key = r2_key,
    )
    db.session.add(analysis)
    db.session.commit()

    return jsonify({
        "presigned_url": presigned_url,
        "r2_key":        r2_key,
        "analysis_id":   analysis.id,
    })


# ─────────────────────────────────────────
# API — CONFIRME UPLOAD R2 TERMINÉ
# ─────────────────────────────────────────
@app.route("/api/upload-complete/<int:id>", methods=["POST"])
@login_required
def upload_complete(id):
    """
    Appelé par le frontend après que l'upload R2 est terminé.
    Lance l'analyse en background.
    """
    a = db.session.get(Analysis, id)
    if not a or a.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403
    if a.status != "pending":
        return jsonify({"error": "déjà lancé"}), 400

    thread = threading.Thread(
        target = run_analysis,
        args   = (a.id, None, a.sport, current_user.plan, a.mode, a.player_id),
        kwargs = {"r2_key": a.r2_video_key},
        daemon = True,
    )
    thread.start()

    return jsonify({"ok": True, "analysis_id": a.id})


# ─────────────────────────────────────────
# UPLOAD (mode local — fallback si R2 absent)
# ─────────────────────────────────────────
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if current_user.analyses_left() <= 0:
        flash(f"Quota atteint pour votre plan {current_user.plan} — passez a Pro")
        return redirect(url_for("pricing"))

    f         = request.files.get("video")
    sport     = request.form.get("sport",     "football")
    mode      = request.form.get("mode",      "match")
    player_id        = request.form.get("player_id", "").strip() or None
    player_position  = request.form.get("position", "").strip() or None

    # Noms équipes saisis par l'utilisateur
    team_name_0       = request.form.get("team_name_0", "").strip() or None
    team_name_1       = request.form.get("team_name_1", "").strip() or None
    preview_upload_id = request.form.get("preview_upload_id", "").strip() or None
    user_team_id_raw  = request.form.get("user_team_id", "").strip()

    # user_team_id : 0 = équipe A est la mienne, 1 = équipe B est la mienne
    # Si l'utilisateur a cliqué "C'est mon équipe" sur l'équipe B (tid=1),
    # on inverse les noms pour que son équipe soit toujours team_0
    user_team_id = None
    try:
        user_team_id = int(user_team_id_raw) if user_team_id_raw else None
    except ValueError:
        pass

    team_names = {}
    if user_team_id == 1:
        # L'utilisateur a sélectionné l'équipe B comme la sienne → inverser
        if team_name_0: team_names["1"] = team_name_0
        if team_name_1: team_names["0"] = team_name_1
        user_team_id = 0   # après inversion, son équipe est toujours 0
    else:
        if team_name_0: team_names["0"] = team_name_0
        if team_name_1: team_names["1"] = team_name_1

    if not f or f.filename == "":
        flash("Aucune video selectionnee")
        return redirect(url_for("dashboard"))

    if not allowed_file(f.filename):
        flash(f"Format non supporte — formats acceptes : {', '.join(config.ALLOWED_EXTENSIONS)}")
        return redirect(url_for("dashboard"))

    if sport not in VALID_SPORTS:
        flash(f"Sport non reconnu : {sport}")
        return redirect(url_for("dashboard"))

    if mode not in ["match", "player"]:
        mode = "match"

    filename = secure_filename(f.filename)
    path     = os.path.join(config.UPLOAD_FOLDER, filename)
    f.save(path)

    analysis = Analysis(
        user_id   = current_user.id,
        filename  = filename,
        sport     = sport,
        mode      = mode,
        player_id = player_id,
        status    = "pending",
        team_name_0 = team_name_0,
        team_name_1 = team_name_1,
    )
    db.session.add(analysis)
    db.session.commit()

    if CELERY_AVAILABLE:
        celery_run_analysis.apply_async(
            kwargs = {
                "video_path":      path,
                "sport":           sport,
                "output_dir":      os.path.join(config.OUTPUT_FOLDER, str(analysis.id)),
                "mode":            mode,
                "plan":            current_user.plan,
                "analysis_id":     analysis.id,
                "use_coarse_scan": True,
                "team_names":      team_names or None,
            },
            task_id = f"analysis_{analysis.id}",
            queue   = "pipeline",
        )
        analysis.celery_task_id = f"analysis_{analysis.id}"
        db.session.commit()
    else:
        # Fallback : threading (mode dev sans Redis)
        thread = threading.Thread(
            target = run_analysis,
            args   = (analysis.id, path, sport, current_user.plan,
                      mode, player_id, None, team_names or None),
            kwargs = {"player_position": player_position},
            daemon = True
        )
        thread.start()

    flash("Video uploadee — analyse en cours...")
    return redirect(url_for("dashboard"))


# ─────────────────────────────────────────
# BACKGROUND PROCESS
# ─────────────────────────────────────────
def run_analysis(
    analysis_id, video_path, sport, plan,
    mode="match", player_id=None, r2_key=None, team_names=None,
    player_position=None
):
    with app.app_context():
        a              = db.session.get(Analysis, analysis_id)
        a.status       = "processing"
        a.progress     = 0
        a.progress_msg = "Analyse en cours..."
        db.session.commit()

    local_tmp = None  # chemin temporaire si téléchargé depuis R2

    try:
        # ── Mode R2 : télécharge la vidéo localement pour le pipeline ──
        if r2_key and config.R2_ENABLED:
            local_tmp  = os.path.join(
                config.UPLOAD_FOLDER,
                f"tmp_{analysis_id}_{os.path.basename(r2_key)}"
            )
            ok = r2_download_to_local(r2_key, local_tmp)
            if not ok:
                raise RuntimeError("Impossible de télécharger la vidéo depuis R2")
            video_path = local_tmp

        output_dir = os.path.join(config.OUTPUT_FOLDER, str(analysis_id))

        result = run_pipeline(
            video_path     = video_path,
            sport          = sport,
            output_dir     = output_dir,
            analysis_id    = analysis_id,
            save_annotated = config.PLANS.get(plan, {}).get("montage", False),
            plan           = plan,
            mode           = mode,
            player_id       = player_id,
            player_position = player_position,
            team_names      = team_names or None,
        )

        # ── Supprime la vidéo brute dès que l'analyse est terminée ──
        delete_raw_video(video_path, r2_key)

        # ── Upload les outputs vers R2 si activé ──
        r2_prefix = None
        if config.R2_ENABLED:
            r2_prefix = r2_upload_outputs(output_dir, analysis_id)

        with app.app_context():
            a                  = db.session.get(Analysis, analysis_id)
            a.status           = "done"
            a.progress         = 100
            a.progress_msg     = "Analyse terminee"
            a.result_json      = json.dumps(result)
            a.output_dir       = output_dir
            a.r2_output_prefix = r2_prefix
            db.session.commit()

    except Exception as e:
        # Nettoyage en cas d'erreur
        if local_tmp and os.path.exists(local_tmp):
            os.remove(local_tmp)

        with app.app_context():
            a              = db.session.get(Analysis, analysis_id)
            a.status       = "error"
            a.progress_msg = str(e)[:200]
            db.session.commit()
        print(f"ERROR analyse {analysis_id} : {e}")


# ─────────────────────────────────────────
# API
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# API — UPLOAD PREVIEW + DÉTECTION ÉQUIPES
# ─────────────────────────────────────────
import glob as _glob

@app.route("/api/upload-preview", methods=["POST"])
@login_required
def api_upload_preview():
    """Upload préliminaire XHR — reçoit la vidéo, retourne un upload_id."""
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "no video"}), 400

    tmp_dir = os.path.join(config.UPLOAD_FOLDER, "previews")
    os.makedirs(tmp_dir, exist_ok=True)

    uid      = str(uuid.uuid4())[:8]
    ext      = os.path.splitext(secure_filename(video.filename))[1] or ".mp4"
    tmp_path = os.path.join(tmp_dir, f"prev_{uid}{ext}")
    video.save(tmp_path)

    return jsonify({"upload_id": uid, "tmp_path": tmp_path})


@app.route("/api/detect-teams/<upload_id>")
@login_required
def api_detect_teams(upload_id):
    """Détecte les 2 équipes depuis l'upload préliminaire."""
    tmp_dir = os.path.join(config.UPLOAD_FOLDER, "previews")
    matches = _glob.glob(os.path.join(tmp_dir, f"prev_{upload_id}.*"))
    if not matches:
        return jsonify({"success": False, "error": "upload not found"}), 404

    video_path  = matches[0]
    preview_dir = os.path.join(tmp_dir, f"preview_{upload_id}")

    try:
        from analysis.detect_teams_preview import detect_teams_preview
        # sport depuis la session ou formulaire si disponible
        _prev_sport = request.args.get("sport", "football")
        result = detect_teams_preview(
            video_path          = video_path,
            output_dir          = preview_dir,
            bootstrap_duration  = 30.0,
            sport               = _prev_sport,
        )

        if result.get("success"):
            for tid in ["team_0", "team_1"]:
                t = result.get(tid, {})
                # Couleur maillot
                bgr = t.get("color_bgr")
                if bgr:
                    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
                    t["color_hex"] = f"#{r:02x}{g:02x}{b:02x}"
                # Couleur short
                short_bgr = t.get("short_bgr")
                if short_bgr:
                    try:
                        b, g, r = int(short_bgr[0]), int(short_bgr[1]), int(short_bgr[2])
                        t["short_hex"] = f"#{r:02x}{g:02x}{b:02x}"
                        t["short_bgr"] = [b, g, r]
                    except Exception:
                        t.pop("short_bgr", None)
                if t.get("preview_frame") and os.path.exists(t["preview_frame"]):
                    fname = os.path.basename(t["preview_frame"])
                    t["preview_url"] = f"/api/preview-image/{upload_id}/{fname}"
                # Rendre color_bgr sérialisable (peut être tuple ou numpy array)
                if t.get("color_bgr"):
                    t["color_bgr"] = [int(x) for x in t["color_bgr"]]
                # Purger les valeurs None non sérialisables
                t = {k: v for k, v in t.items() if v is not None}
                result[tid] = t

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/preview-image/<upload_id>/<filename>")
@login_required
def api_preview_image(upload_id, filename):
    """Sert les images de preview des équipes."""
    from flask import send_from_directory
    preview_dir = os.path.join(
        config.UPLOAD_FOLDER, "previews", f"preview_{upload_id}"
    )
    return send_from_directory(preview_dir, filename)


@app.route("/api/status/<int:id>")
@login_required
def status(id):
    a = db.session.get(Analysis, id)

    if not a or a.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    # Progression temps réel depuis Celery
    celery_info = {}
    if CELERY_AVAILABLE and getattr(a, "celery_task_id", None):
        try:
            task = celery_app.AsyncResult(a.celery_task_id)
            if task.state == "PROGRESS":
                celery_info = {
                    "celery_state": task.state,
                    "pct":          task.info.get("pct", 0),
                    "message":      task.info.get("message", ""),
                }
            elif task.state == "SUCCESS":
                celery_info = {"celery_state": "SUCCESS", "pct": 100}
            elif task.state == "FAILURE":
                celery_info = {"celery_state": "FAILURE", "pct": 0}
        except Exception:
            pass

    return jsonify({
        "status":       a.status,
        **celery_info,
        "progress":     a.progress,
        "progress_msg": a.progress_msg
    })


@app.route("/api/delete/<int:id>", methods=["DELETE"])
@login_required
def delete_analysis(id):
    a = db.session.get(Analysis, id)

    if not a or a.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    if a.status == "processing":
        return jsonify({"error": "analyse en cours"}), 400

    # Supprime les fichiers associés
    out_dir = a.output_dir or os.path.join(config.OUTPUT_FOLDER, str(a.id))
    if out_dir and os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    if config.R2_ENABLED and a.r2_output_prefix:
        _r2_delete_prefix(a.r2_output_prefix)

    db.session.delete(a)
    db.session.commit()
    return jsonify({"success": True})


# ─────────────────────────────────────────
# FICHIERS OUTPUT
# ─────────────────────────────────────────
@app.route("/files/<int:analysis_id>/<path:filename>")
@login_required
def files(analysis_id, filename):
    a = db.session.get(Analysis, analysis_id)

    if not a or a.user_id != current_user.id:
        return "Forbidden", 403

    directory = a.output_dir if a.output_dir else \
                os.path.join(config.OUTPUT_FOLDER, str(analysis_id))

    return send_from_directory(directory, filename)


# ─────────────────────────────────────────
# TÉLÉCHARGEMENTS — reel, pdf, stream, heatmap
# ─────────────────────────────────────────
@app.route("/download/<int:id>/reel")
@login_required
def download_reel(id):
    a = db.session.get(Analysis, id)
    if not a or a.user_id != current_user.id:
        return "Forbidden", 403
    result   = json.loads(a.result_json) if a.result_json else {}
    reel_path = result.get("reel")
    if reel_path and reel_path.startswith("http"):
        return redirect(reel_path)
    if reel_path and os.path.exists(reel_path):
        dl_name = f"scoutia_highlights_{a.filename.rsplit('.', 1)[0]}.mp4"
        return send_from_directory(os.path.dirname(reel_path),
                                   os.path.basename(reel_path),
                                   as_attachment=True,
                                   download_name=dl_name)
    flash("Reel non disponible")
    return redirect(url_for("results", id=id))


@app.route("/download/<int:id>/pdf")
@login_required
def download_pdf(id):
    a = db.session.get(Analysis, id)
    if not a or a.user_id != current_user.id:
        return "Forbidden", 403
    result   = json.loads(a.result_json) if a.result_json else {}
    pdf_path = result.get("pdf")
    if pdf_path and pdf_path.startswith("http"):
        return redirect(pdf_path)
    if pdf_path and os.path.exists(pdf_path):
        dl_name = f"scoutia_rapport_{a.filename.rsplit('.', 1)[0]}.pdf"
        return send_from_directory(os.path.dirname(pdf_path),
                                   os.path.basename(pdf_path),
                                   as_attachment=True,
                                   download_name=dl_name)
    flash("PDF non disponible")
    return redirect(url_for("results", id=id))


@app.route("/stream/<int:id>/reel")
@login_required
def stream_reel(id):
    """Lecture inline du reel dans le navigateur."""
    a = db.session.get(Analysis, id)
    if not a or a.user_id != current_user.id:
        return "Forbidden", 403
    result    = json.loads(a.result_json) if a.result_json else {}
    reel_path = result.get("reel")
    if reel_path and reel_path.startswith("http"):
        return redirect(reel_path)
    if reel_path and os.path.exists(reel_path):
        return send_from_directory(os.path.dirname(reel_path),
                                   os.path.basename(reel_path))
    return "Reel non disponible", 404


@app.route("/heatmap/<int:id>/<name>")
@login_required
def heatmap(id, name):
    """Servir une heatmap depuis les outputs."""
    a = db.session.get(Analysis, id)
    if not a or a.user_id != current_user.id:
        return "Forbidden", 403
    result   = json.loads(a.result_json) if a.result_json else {}
    heatmaps = result.get("heatmaps", {})
    path     = heatmaps.get(name)
    if path and path.startswith("http"):
        return redirect(path)
    if path and os.path.exists(path):
        return send_from_directory(os.path.dirname(path),
                                   os.path.basename(path))
    return "Heatmap non disponible", 404


# ─────────────────────────────────────────
# STRIPE — CHECKOUT
# ─────────────────────────────────────────
@app.route("/create-checkout", methods=["POST"])
@app.route("/checkout/<plan>",  methods=["GET"])
@login_required
def checkout(plan=None):
    if request.method == "POST":
        plan = request.form.get("plan")

    if plan not in ["starter", "pro", "unique"]:
        flash("Plan invalide")
        return redirect(url_for("pricing"))

    url = create_checkout_session(
        user_email  = current_user.email,
        plan        = plan,
        success_url = request.host_url + "payment/success",
        cancel_url  = request.host_url + "pricing"
    )

    if not url:
        flash("Erreur lors de la creation du paiement")
        return redirect(url_for("pricing"))

    return redirect(url)


# ─────────────────────────────────────────
# STRIPE — SUCCES PAIEMENT
# ─────────────────────────────────────────
@app.route("/payment/success")
@login_required
def payment_success():
    session_id = request.args.get("session_id")

    if not session_id:
        flash("Session de paiement introuvable")
        return redirect(url_for("dashboard"))

    data = verify_checkout_session(session_id)

    if not data:
        flash("Paiement non confirme — contactez le support")
        return redirect(url_for("dashboard"))

    current_user.plan            = data["plan"]
    current_user.stripe_customer = data["customer_id"]
    current_user.stripe_sub      = data["subscription_id"]
    db.session.commit()

    flash(f"Bienvenue sur le plan {data['plan'].capitalize()} !")
    return redirect(url_for("dashboard"))


# ─────────────────────────────────────────
# STRIPE — PORTAIL CLIENT
# ─────────────────────────────────────────
@app.route("/billing")
@login_required
def billing():
    if not current_user.stripe_customer:
        flash("Aucun abonnement actif")
        return redirect(url_for("pricing"))

    url = create_portal_session(
        stripe_customer_id = current_user.stripe_customer,
        return_url         = request.host_url + "dashboard"
    )

    if not url:
        flash("Erreur portail Stripe")
        return redirect(url_for("dashboard"))

    return redirect(url)


# ─────────────────────────────────────────
# STRIPE — WEBHOOK
# ─────────────────────────────────────────
@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    payload    = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")

    data = handle_webhook(payload, sig_header)

    if not data:
        return jsonify({"error": "invalid"}), 400

    event_type = data["event_type"]

    if event_type in ["payment_success", "renewal_success"]:
        user = User.query.filter_by(stripe_customer=data["customer_id"]).first()
        if user:
            user.plan       = data.get("plan", user.plan)
            user.stripe_sub = data.get("subscription_id", user.stripe_sub)
            db.session.commit()

    elif event_type == "subscription_canceled":
        user = User.query.filter_by(stripe_customer=data["customer_id"]).first()
        if user:
            user.plan       = "free"
            user.stripe_sub = None
            db.session.commit()

    return jsonify({"ok": True}), 200


# ─────────────────────────────────────────
# ADMIN — DÉCORATEUR
# ─────────────────────────────────────────
def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────
# ADMIN — ROUTES
# ─────────────────────────────────────────
@app.route("/admin")
@login_required
@admin_required
def admin():
    users    = User.query.order_by(User.id.desc()).all()
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).all()

    stats = {
        "total_users":    User.query.count(),
        "total_analyses": Analysis.query.count(),
        "done":           Analysis.query.filter_by(status="done").count(),
        "processing":     Analysis.query.filter_by(status="processing").count(),
        "errors":         Analysis.query.filter_by(status="error").count(),
        "plan_free":      User.query.filter_by(plan="free").count(),
        "plan_starter":   User.query.filter_by(plan="starter").count(),
        "plan_pro":       User.query.filter_by(plan="pro").count(),
        "plan_unique":    User.query.filter_by(plan="unique").count(),
    }

    return render_template("admin.html", users=users, analyses=analyses, stats=stats)


@app.route("/admin/user/<int:id>/plan", methods=["POST"])
@login_required
@admin_required
def admin_change_plan(id):
    user     = db.session.get(User, id)
    new_plan = request.form.get("plan")

    if not user:
        flash("Utilisateur introuvable")
        return redirect(url_for("admin"))

    if new_plan not in config.PLANS:
        flash("Plan invalide")
        return redirect(url_for("admin"))

    user.plan = new_plan
    db.session.commit()
    flash(f"Plan de {user.email} mis a jour -> {new_plan}")
    return redirect(url_for("admin"))


@app.route("/admin/user/<int:id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_user(id):
    user = db.session.get(User, id)

    if not user:
        flash("Utilisateur introuvable")
        return redirect(url_for("admin"))

    if user.id == current_user.id:
        flash("Impossible de supprimer votre propre compte")
        return redirect(url_for("admin"))

    Analysis.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f"Utilisateur {user.email} supprime")
    return redirect(url_for("admin"))


@app.route("/admin/analysis/<int:id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_analysis(id):
    a = db.session.get(Analysis, id)

    if not a:
        flash("Analyse introuvable")
        return redirect(url_for("admin"))

    db.session.delete(a)
    db.session.commit()
    flash(f"Analyse {id} supprimee")
    return redirect(url_for("admin"))


# ─────────────────────────────────────────
# ERREURS
# ─────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404


@app.errorhandler(413)
def too_large(e):
    flash("Fichier trop volumineux (max 10 GB)")
    return redirect(url_for("dashboard"))


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    # Lance le nettoyage automatique en background
    start_cleanup_scheduler()

    app.run(
        debug    = config.DEBUG,
        port     = int(os.getenv("PORT", 5000)),
        threaded = True   # requêtes longues (detect-teams) sans bloquer
    )