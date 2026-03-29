import os, json, threading, io
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(dotenv_path=r"D:\ProjetAIMika\.env\param.env")

import stripe
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, flash, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
app = Flask(__name__)
app.secret_key        = "CHANGE_MOI_EN_PRODUCTION"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["UPLOAD_FOLDER"]           = "uploads"
app.config["MAX_CONTENT_LENGTH"]      = 10 * 1024 * 1024 * 1024  # 10 GB

STRIPE_PUBLIC_KEY     = os.environ.get("STRIPE_PUBLIC_KEY",     "")
STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY",     "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICES = {
    "starter" : os.environ.get("STRIPE_PRICE_STARTER", ""),
    "pro"     : os.environ.get("STRIPE_PRICE_PRO",     ""),
    "unique"  : os.environ.get("STRIPE_PRICE_UNIQUE",  ""),
}
stripe.api_key = STRIPE_SECRET_KEY
print("STRIPE KEY   :", STRIPE_SECRET_KEY[:20] + "..." if STRIPE_SECRET_KEY else "NON TROUVÉE")
print("PRICE UNIQUE :", STRIPE_PRICES["unique"] or "NON TROUVÉE")

PLANS = {
    "free"    : {"name": "Gratuit",      "videos": 2,   "price": 0,   "type": "free"},
    "unique"  : {"name": "Vidéo unique", "videos": 1,   "price": 2.5, "type": "one_time"},
    "starter" : {"name": "Starter",      "videos": 20,  "price": 9,   "type": "subscription"},
    "pro"     : {"name": "Pro",          "videos": 100, "price": 29,  "type": "subscription"},
}

ADMIN_TOKEN = "MON_TOKEN_SECRET"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ─────────────────────────────────────────
# MODÈLES
# ─────────────────────────────────────────
class User(UserMixin, db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    email          = db.Column(db.String(150), unique=True, nullable=False)
    password       = db.Column(db.String(256), nullable=False)
    plan           = db.Column(db.String(20), default="free")
    stripe_cust_id = db.Column(db.String(100))
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    analyses       = db.relationship("Analysis", backref="user", lazy=True)

class Analysis(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    filename     = db.Column(db.String(256))
    status       = db.Column(db.String(20), default="pending")
    progress     = db.Column(db.Integer, default=0)
    progress_msg = db.Column(db.String(256), default="En attente...")
    result_json  = db.Column(db.Text)
    output_dir   = db.Column(db.String(512))
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))

# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        pwd   = request.form["password"]
        if User.query.filter_by(email=email).first():
            flash("Email déjà utilisé.", "error")
            return redirect(url_for("register"))
        user = User(email=email, password=generate_password_hash(pwd))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("dashboard"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        pwd   = request.form["password"]
        user  = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, pwd):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Email ou mot de passe incorrect.", "error")
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
    return render_template("pricing.html", plans=PLANS, pub_key=STRIPE_PUBLIC_KEY)

@app.route("/dashboard")
@login_required
def dashboard():
    analyses = Analysis.query.filter_by(user_id=current_user.id)\
                             .order_by(Analysis.created_at.desc()).all()
    plan = PLANS.get(current_user.plan, PLANS["free"])
    used = len(analyses)
    return render_template("dashboard.html", analyses=analyses, plan=plan, used=used)

@app.route("/results/<int:analysis_id>")
@login_required
def results(analysis_id):
    a = db.session.get(Analysis, analysis_id)
    if a is None:
        return redirect(url_for("dashboard"))
    if a.user_id != current_user.id:
        return redirect(url_for("dashboard"))
    result = json.loads(a.result_json) if a.result_json else {}
    return render_template("results.html", analysis=a, result=result)

# ─────────────────────────────────────────
# UPLOAD & ANALYSE
# ─────────────────────────────────────────
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    plan = PLANS.get(current_user.plan, PLANS["free"])
    used = Analysis.query.filter_by(user_id=current_user.id).count()
    if used >= plan["videos"]:
        flash(f"Quota atteint ({plan['videos']} vidéos). Upgradez votre plan.", "error")
        return redirect(url_for("dashboard"))

    f = request.files.get("video")
    if not f:
        flash("Aucune vidéo sélectionnée.", "error")
        return redirect(url_for("dashboard"))

    # Récupérer le mode et les options joueur
    mode                     = request.form.get("mode", "match")
    sport                    = request.form.get("sport", "football")
    numero_joueur            = request.form.get("numero_joueur", "")
    couleur_maillot          = request.form.get("couleur_maillot", "")
    position_joueur          = request.form.get("position_joueur", "")
    couleur_gardien_domicile = request.form.get("couleur_gardien_domicile", "")
    couleur_gardien_visiteur = request.form.get("couleur_gardien_visiteur", "")

    if mode == "joueur" and (not numero_joueur or not couleur_maillot or not position_joueur):
        flash("Veuillez renseigner le numéro, la couleur et la position du joueur.", "error")
        return redirect(url_for("dashboard"))

    filename  = secure_filename(f.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)

    # Stocker les options dans le nom pour les retrouver
    analysis = Analysis(
        user_id      = current_user.id,
        filename     = filename,
        status       = "pending",
        progress_msg = f"Mode: {mode}" + (
            f" | #{numero_joueur} {couleur_maillot} — {position_joueur}"
            if mode == "joueur" else ""
        )
    )
    db.session.add(analysis)
    db.session.commit()

    thread = threading.Thread(target=run_analysis, args=(
        analysis.id, save_path, mode, sport,
        numero_joueur, couleur_maillot, position_joueur,
        couleur_gardien_domicile, couleur_gardien_visiteur
    ))
    thread.daemon = True
    thread.start()

    return redirect(url_for("dashboard"))

def update_progress(analysis_id: int, pct: int, msg: str):
    with app.app_context():
        a = db.session.get(Analysis, analysis_id)
        if a:
            a.progress     = pct
            a.progress_msg = msg
            db.session.commit()

def run_analysis(analysis_id: int, video_path: str,
                 mode: str = "match", sport: str = "football",
                 numero_joueur: str = "", couleur_maillot: str = "",
                 position_joueur: str = "",
                 couleur_gardien_domicile: str = "",
                 couleur_gardien_visiteur: str = ""):
    with app.app_context():
        a        = db.session.get(Analysis, analysis_id)
        a.status = "processing"
        db.session.commit()
    try:
        import scout
        update_progress(analysis_id, 5,  "Préparation des dossiers...")
        output_dir, frames_dir = scout.setup_dirs(video_path)

        update_progress(analysis_id, 10, "Extraction des frames...")
        frames, duration, interval = scout.extract_frames(video_path, frames_dir)

        if mode == "joueur":
            update_progress(analysis_id, 30, f"Analyse #{numero_joueur} {couleur_maillot} ({position_joueur})...")
            analysis_result = scout.analyze_player(
                frames, duration, interval,
                numero           = int(numero_joueur),
                couleur          = couleur_maillot,
                position         = position_joueur,
                couleur_gardien_domicile = couleur_gardien_domicile,
                couleur_gardien_visiteur = couleur_gardien_visiteur,
            )
            # Pas de découpe de clips pour le mode joueur
            update_progress(analysis_id, 95, "Génération du rapport joueur...")
        else:
            update_progress(analysis_id, 30, f"Analyse IA ({len(frames)} frames)...")
            analysis_result = scout.analyze_frames(frames, duration, interval)

            update_progress(analysis_id, 75, "Affinage des highlights...")
            analysis_result = scout.refine_all_highlights(video_path, output_dir, analysis_result)

            update_progress(analysis_id, 85, "Découpe des clips...")
            clips = scout.cut_highlights(video_path, output_dir, analysis_result.get("highlights", []))

            update_progress(analysis_id, 95, "Création du highlight reel...")
            scout.create_highlight_reel(output_dir, clips)

        # Ajouter le mode dans le résultat
        analysis_result["_mode"] = mode

        with app.app_context():
            a              = db.session.get(Analysis, analysis_id)
            a.status       = "done"
            a.progress     = 100
            a.progress_msg = "Analyse terminée !"
            a.result_json  = json.dumps(analysis_result, ensure_ascii=False)
            a.output_dir   = output_dir
            db.session.commit()

    except Exception as e:
        with app.app_context():
            a              = db.session.get(Analysis, analysis_id)
            a.status       = "error"
            a.progress_msg = f"Erreur : {str(e)}"
            db.session.commit()

# ─────────────────────────────────────────
# API PROGRESSION
# ─────────────────────────────────────────
@app.route("/api/status/<int:analysis_id>")
@login_required
def api_status(analysis_id):
    a = db.session.get(Analysis, analysis_id)
    if not a or a.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403
    return jsonify({"status": a.status, "progress": a.progress, "msg": a.progress_msg})

# ─────────────────────────────────────────
# FICHIERS VIDÉO
# ─────────────────────────────────────────
@app.route("/files/<int:analysis_id>/<path:filename>")
@login_required
def serve_file(analysis_id, filename):
    a = Analysis.query.get_or_404(analysis_id)
    if a.user_id != current_user.id:
        return redirect(url_for("dashboard"))
    return send_from_directory(a.output_dir, filename)

# ─────────────────────────────────────────
# STRIPE PAIEMENT
# ─────────────────────────────────────────
@app.route("/create-checkout", methods=["POST"])
@login_required
def create_checkout():
    plan_key = request.form.get("plan")
    if plan_key not in STRIPE_PRICES or plan_key not in PLANS:
        return redirect(url_for("pricing"))

    if not current_user.stripe_cust_id:
        customer = stripe.Customer.create(email=current_user.email)
        current_user.stripe_cust_id = customer.id
        db.session.commit()

    plan_type = PLANS[plan_key]["type"]
    mode      = "payment" if plan_type == "one_time" else "subscription"

    session = stripe.checkout.Session.create(
        customer             = current_user.stripe_cust_id,
        payment_method_types = ["card"],
        mode                 = mode,
        line_items           = [{"price": STRIPE_PRICES[plan_key], "quantity": 1}],
        success_url          = url_for("payment_success", _external=True) + "?plan=" + plan_key,
        cancel_url           = url_for("pricing", _external=True),
        metadata             = {"user_id": current_user.id, "plan": plan_key}
    )
    return redirect(session.url)

@app.route("/payment/success")
@login_required
def payment_success():
    plan = request.args.get("plan", "starter")
    current_user.plan = plan
    db.session.commit()
    flash(f"✅ Plan {PLANS[plan]['name']} activé ! Merci pour votre confiance.", "success")
    return redirect(url_for("dashboard"))

@app.route("/cancel-subscription", methods=["POST"])
@login_required
def cancel_subscription():
    if current_user.stripe_cust_id:
        subs = stripe.Subscription.list(customer=current_user.stripe_cust_id, status="active")
        for sub in subs.data:
            stripe.Subscription.modify(sub.id, cancel_at_period_end=True)
        flash("Abonnement annulé. Votre plan reste actif jusqu'à la fin de la période.", "success")
    return redirect(url_for("dashboard"))

@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    if app.debug:
        return "", 200
    payload    = request.get_data()
    sig_header = request.headers.get("Stripe-Signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        return "", 400
    if event["type"] == "checkout.session.completed":
        s       = event["data"]["object"]
        user_id = s.get("metadata", {}).get("user_id")
        plan    = s.get("metadata", {}).get("plan")
        if user_id and plan:
            with app.app_context():
                u =db.session.get(User, int(user_id))
                if u:
                    u.plan = plan
                    db.session.commit()
    elif event["type"] == "customer.subscription.deleted":
        customer_id = event["data"]["object"]["customer"]
        u = User.query.filter_by(stripe_cust_id=customer_id).first()
        if u:
            u.plan = "free"
            db.session.commit()
    return "", 200

# ─────────────────────────────────────────
# PDF
# ─────────────────────────────────────────
@app.route("/pdf/<int:analysis_id>")
@login_required
def generate_pdf(analysis_id):
    a = Analysis.query.get_or_404(analysis_id)
    if a.user_id != current_user.id:
        return redirect(url_for("dashboard"))

    result = json.loads(a.result_json) if a.result_json else {}
    buf    = io.BytesIO()
    BLUE   = HexColor("#2563eb")
    GRAY   = HexColor("#f8f9ff")

    doc      = SimpleDocTemplate(buf, pagesize=A4,
                                  leftMargin=2*cm, rightMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
    s_title  = ParagraphStyle("title",  fontSize=18, textColor=BLUE, fontName="Helvetica-Bold", spaceAfter=12)
    s_sub    = ParagraphStyle("sub",    fontSize=10, textColor=HexColor("#666666"), spaceAfter=6)
    s_h2     = ParagraphStyle("h2",     fontSize=12, textColor=BLUE, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6)
    s_body   = ParagraphStyle("body",   fontSize=11, leading=16, spaceAfter=6)
    s_small  = ParagraphStyle("small",  fontSize=9,  textColor=HexColor("#555555"), leading=13)
    s_center = ParagraphStyle("center", fontSize=11, alignment=TA_CENTER)

    elems = []

    # Header
    elems.append(Paragraph("ScoutIA — Rapport d'analyse", s_title))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(a.filename, s_sub))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(f"Généré le {datetime.utcnow().strftime('%d/%m/%Y à %H:%M')}", s_sub))
    elems.append(Spacer(1, 8))
    elems.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=12))

    # Score
    if result.get("score"):
        t = Table([[Paragraph(f"<b>Score : {result['score']}</b>",
                   ParagraphStyle("sc", fontSize=16, textColor=white, alignment=TA_CENTER, fontName="Helvetica-Bold"))]],
                  colWidths=[17*cm])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),BLUE),
                                ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10)]))
        elems += [t, Spacer(1, 12)]

    # Résumé
    elems.append(Paragraph("RÉSUMÉ DU MATCH", s_h2))
    elems.append(Paragraph(result.get("resume", ""), s_body))
    elems.append(Spacer(1, 8))

    # Highlights
    if result.get("highlights"):
        elems.append(Paragraph(f"HIGHLIGHTS ({len(result['highlights'])})", s_h2))
        rows = [["Temps", "Type", "Description", "Importance"]]
        for h in result["highlights"]:
            stars = "★" * h.get("importance", 1) + "☆" * (5 - h.get("importance", 1))
            rows.append([
                Paragraph(f"<b>{h['timestamp_debut']}</b><br/><font size=8>→ {h['timestamp_fin']}</font>", s_small),
                Paragraph(f"<b>{h['type'].upper()}</b>", s_small),
                Paragraph(h["description"], s_small),
                Paragraph(stars, s_small)
            ])
        t = Table(rows, colWidths=[2.2*cm, 2.5*cm, 10*cm, 2.3*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),BLUE), ("TEXTCOLOR",(0,0),(-1,0),white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,0),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,GRAY]),
            ("GRID",(0,0),(-1,-1),0.3,HexColor("#e8e8e8")),
            ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),6), ("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
        elems += [t, Spacer(1, 12)]

    # Top joueurs
    if result.get("top_joueurs"):
        elems.append(Paragraph("TOP JOUEURS", s_h2))
        rows = [["Note", "Joueur", "Points forts"]]
        for j in result["top_joueurs"]:
            rows.append([
                Paragraph(f"<b><font size=14>{j['note']}</font>/10</b>", s_center),
                Paragraph(f"<b>{j['joueur']}</b>", s_small),
                Paragraph(j["points_forts"], s_small)
            ])
        t = Table(rows, colWidths=[2*cm, 5*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),BLUE), ("TEXTCOLOR",(0,0),(-1,0),white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,0),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,GRAY]),
            ("GRID",(0,0),(-1,-1),0.3,HexColor("#e8e8e8")),
            ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),6), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        elems += [t, Spacer(1, 12)]

    # Tactique
    if result.get("observations_tactiques"):
        elems.append(Paragraph("OBSERVATIONS TACTIQUES", s_h2))
        elems.append(Paragraph(result["observations_tactiques"], s_body))

    # Footer
    elems += [Spacer(1,20), HRFlowable(width="100%", thickness=0.5, color=HexColor("#e8e8e8")),
              Spacer(1,6), Paragraph("ScoutIA — Rapport généré automatiquement par IA", s_sub)]

    doc.build(elems)
    buf.seek(0)
    nom = f"rapport_{a.filename.replace('.mp4','').replace(' ','_')}.pdf"
    return Response(buf, mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={nom}"})

# ─────────────────────────────────────────
# ADMIN
# ─────────────────────────────────────────
@app.route("/admin/reset/<email>")
def admin_reset(email):
    if request.args.get("token") != ADMIN_TOKEN:
        return "Accès refusé", 403
    user = User.query.filter_by(email=email).first()
    if not user:
        return "Utilisateur non trouvé", 404
    Analysis.query.filter_by(user_id=user.id).delete()
    db.session.commit()
    return f"✅ {email} remis à 0 — plan: {user.plan} — analyses supprimées"

@app.route("/admin/users")
def admin_users():
    if request.args.get("token") != ADMIN_TOKEN:
        return "Accès refusé", 403
    users  = User.query.all()
    result = ""
    for u in users:
        nb = Analysis.query.filter_by(user_id=u.id).count()
        result += f"{u.email} | plan: {u.plan} | analyses: {nb}<br>"
    return result

@app.route("/admin/setplan/<email>/<plan>")
def admin_setplan(email, plan):
    if request.args.get("token") != ADMIN_TOKEN:
        return "Accès refusé", 403
    user = User.query.filter_by(email=email).first()
    if not user:
        return "Utilisateur non trouvé", 404
    if plan not in PLANS:
        return f"Plan invalide. Choix : {list(PLANS.keys())}", 400
    user.plan = plan
    db.session.commit()
    return f"✅ {email} → plan '{plan}' activé"

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)