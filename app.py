# app.py
# -*- coding: utf-8 -*-

import os
import json
import threading
from datetime import datetime
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
from pipeline import run_pipeline
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
    id           = db.Column(db.Integer,     primary_key=True)
    user_id      = db.Column(db.Integer,     nullable=False)
    filename     = db.Column(db.String(200))
    sport        = db.Column(db.String(50),  default="football")
    mode         = db.Column(db.String(20),  default="match")   # FIX
    player_id    = db.Column(db.String(50),  nullable=True)      # FIX
    status       = db.Column(db.String(20),  default="pending")
    progress     = db.Column(db.Integer,     default=0)
    progress_msg = db.Column(db.String(200), default="En attente...")
    result_json  = db.Column(db.Text)
    output_dir   = db.Column(db.String(300))
    created_at   = db.Column(db.DateTime,    default=datetime.utcnow)


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
# UPLOAD
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
    player_id = request.form.get("player_id", "").strip() or None

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
        status    = "pending"
    )
    db.session.add(analysis)
    db.session.commit()

    thread = threading.Thread(
        target = run_analysis,
        args   = (analysis.id, path, sport, current_user.plan, mode, player_id),
        daemon = True
    )
    thread.start()

    flash("Video uploadee — analyse en cours...")
    return redirect(url_for("dashboard"))


# ─────────────────────────────────────────
# BACKGROUND PROCESS
# ─────────────────────────────────────────
def run_analysis(analysis_id, video_path, sport, plan, mode="match", player_id=None):
    with app.app_context():
        a              = db.session.get(Analysis, analysis_id)
        a.status       = "processing"
        a.progress     = 0
        a.progress_msg = "Analyse en cours..."
        db.session.commit()

    try:
        output_dir = os.path.join(config.OUTPUT_FOLDER, str(analysis_id))

        result = run_pipeline(
            video_path     = video_path,
            sport          = sport,
            output_dir     = output_dir,
            analysis_id    = analysis_id,
            save_annotated = config.PLANS.get(plan, {}).get("montage", False),
            plan           = plan,
            mode           = mode,
            player_id      = player_id
        )

        with app.app_context():
            a              = db.session.get(Analysis, analysis_id)
            a.status       = "done"
            a.progress     = 100
            a.progress_msg = "Analyse terminee"
            a.result_json  = json.dumps(result)
            a.output_dir   = output_dir
            db.session.commit()

    except Exception as e:
        with app.app_context():
            a              = db.session.get(Analysis, analysis_id)
            a.status       = "error"
            a.progress_msg = str(e)[:200]
            db.session.commit()
        print(f"ERROR analyse {analysis_id} : {e}")


# ─────────────────────────────────────────
# API
# ─────────────────────────────────────────
@app.route("/api/status/<int:id>")
@login_required
def status(id):
    a = db.session.get(Analysis, id)

    if not a or a.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    return jsonify({
        "status":       a.status,
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

    app.run(
        debug = config.DEBUG,
        port  = int(os.getenv("PORT", 5000))
    )