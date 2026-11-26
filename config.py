# config.py

import os
from dotenv import load_dotenv

load_dotenv(".env/param.env")

# ─────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────
SECRET_KEY   = os.getenv("SECRET_KEY",   "dev-only-change-me")
DATABASE_URI = os.getenv("DATABASE_URI", "") or os.getenv("DATABASE_URL", "") or "sqlite:///db.sqlite3"
if DATABASE_URI.startswith("postgres://"):          # 26/11/2026 : PostgreSQL en production (forme courte acceptée)
    DATABASE_URI = "postgresql://" + DATABASE_URI[len("postgres://"):]
DEBUG        = os.getenv("DEBUG",        "false").lower() == "true"

# ─────────────────────────────────────────
# UPLOADS
# ─────────────────────────────────────────
UPLOAD_FOLDER      = os.getenv("UPLOAD_FOLDER",  "uploads")
OUTPUT_FOLDER      = os.getenv("OUTPUT_FOLDER",  "outputs")
MAX_UPLOAD_SIZE    = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024 * 1024))
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "m4v"}

# ─────────────────────────────────────────
# CLOUDFLARE R2  (optionnel — stockage cloud)
# Si R2_ACCOUNT_ID est absent → stockage local
# ─────────────────────────────────────────
R2_ACCOUNT_ID        = os.getenv("R2_ACCOUNT_ID",        "")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID",     "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME       = os.getenv("R2_BUCKET_NAME",       "scoutia-videos")
R2_PUBLIC_URL        = os.getenv("R2_PUBLIC_URL",        "")  # ex: https://pub-xxx.r2.dev

# True si R2 est configuré
R2_ENABLED = bool(R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY)

# URL endpoint S3-compatible pour R2
R2_ENDPOINT_URL = (
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    if R2_ACCOUNT_ID else ""
)

# ─────────────────────────────────────────
# RÉTENTION DES FICHIERS (en jours)
# ─────────────────────────────────────────
# 26/11/2026 : avec les JETONS, plus de formules → une seule durée de conservation des résultats pour tous
RETENTION_JOURS = int(os.getenv("RETENTION_JOURS", 30))
# 26/11/2026 : erreurs signalées par les utilisateurs → e-mail à l'administrateur (résumé, au plus une fois par heure)
ADMIN_EMAIL  = os.getenv("ADMIN_EMAIL", "")
SMTP_HOST    = os.getenv("SMTP_HOST", "")
SMTP_PORT    = int(os.getenv("SMTP_PORT", 587))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")
SMTP_FROM    = os.getenv("SMTP_FROM", "") or SMTP_USER

FILE_RETENTION_DAYS = {
    "free":    int(os.getenv("RETENTION_FREE",    7)),
    "starter": int(os.getenv("RETENTION_STARTER", 30)),
    "pro":     int(os.getenv("RETENTION_PRO",     90)),
    "unique":  int(os.getenv("RETENTION_UNIQUE",  30)),
}

# Suppression de la vidéo brute après analyse (recommandé : True)
DELETE_RAW_VIDEO_AFTER_ANALYSIS = (
    os.getenv("DELETE_RAW_VIDEO", "true").lower() == "true"
)

# ─────────────────────────────────────────
# VIDÉO
# ─────────────────────────────────────────
VIDEO_PATH   = os.getenv("VIDEO_PATH",   "match.mp4")
FPS          = int(os.getenv("FPS",          30))
FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  1280))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 720))

# ─────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────
YOLO_MODEL      = os.getenv("YOLO_MODEL",      "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", 0.4))
BALL_METHOD     = os.getenv("BALL_METHOD",     "hybrid")

# Défauts conservateurs (test local / Kaggle T4)
# En prod RunPod RTX 4090 → YOLO_BATCH_SIZE=4  FRAME_SKIP_EVERY=4  YOLO_IMGSZ=960
YOLO_BATCH_SIZE  = int(os.getenv("YOLO_BATCH_SIZE",  4))
FRAME_SKIP_EVERY = int(os.getenv("FRAME_SKIP_EVERY", 4))
YOLO_IMGSZ       = int(os.getenv("YOLO_IMGSZ",       640))  # défaut 640 (Kaggle T4) ; RunPod RTX4090 → env YOLO_IMGSZ=960

# ─────────────────────────────────────────
# TRACKING
# ─────────────────────────────────────────
TRACKER_MAX_AGE   = int(os.getenv("TRACKER_MAX_AGE",    30))
POSSESSION_RADIUS = int(os.getenv("POSSESSION_RADIUS",  50))
DRIBBLE_MOVE_MIN  = int(os.getenv("DRIBBLE_MOVE_MIN",   30))
LONG_PASS_MIN     = int(os.getenv("LONG_PASS_MIN",      120))

# ─────────────────────────────────────────
# ZONES DE TIR PAR SPORT
# ─────────────────────────────────────────
SHOT_ZONES = {
    "football":   {"axis": "x", "threshold": int(os.getenv("SHOT_ZONE_FOOTBALL",   900))},
    "mini-foot":  {"axis": "x", "threshold": int(os.getenv("SHOT_ZONE_MINIFOOT",   900))},
    "handball":   {"axis": "x", "threshold": int(os.getenv("SHOT_ZONE_HANDBALL",   850))},
    "basketball": {"axis": "y", "threshold": int(os.getenv("SHOT_ZONE_BASKETBALL", 200))},
}

# ─────────────────────────────────────────
# HIGHLIGHTS
# ─────────────────────────────────────────
HIGHLIGHT_MAX           = int(os.getenv("HIGHLIGHT_MAX",           25))  # 25 pour match 90min
HIGHLIGHT_MIN_SCORE     = int(os.getenv("HIGHLIGHT_MIN_SCORE",      4))
HIGHLIGHT_WINDOW_FRAMES = int(os.getenv("HIGHLIGHT_WINDOW_FRAMES", 90))
HIGHLIGHT_BEFORE_SEC    = int(os.getenv("HIGHLIGHT_BEFORE_SEC",     5))
HIGHLIGHT_AFTER_SEC     = int(os.getenv("HIGHLIGHT_AFTER_SEC",      4))

# ─────────────────────────────────────────
# MONTAGE
# ─────────────────────────────────────────
MONTAGE_RESOLUTION = os.getenv("MONTAGE_RESOLUTION", "1280x720")

# ─────────────────────────────────────────
# IA — CLAUDE
# ─────────────────────────────────────────
CLAUDE_API_KEY    = os.getenv("CLAUDE_API_KEY",    "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL",      "claude-sonnet-4-20250514")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", 500))

# ─────────────────────────────────────────
# STRIPE
# ─────────────────────────────────────────
STRIPE_PUBLIC_KEY     = os.getenv("STRIPE_PUBLIC_KEY",     "")
STRIPE_SECRET_KEY     = os.getenv("STRIPE_SECRET_KEY",     "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

STRIPE_PRICE_IDS = {
    "starter": os.getenv("STRIPE_PRICE_STARTER", ""),
    "pro":     os.getenv("STRIPE_PRICE_PRO",     ""),
    "unique":  os.getenv("STRIPE_PRICE_UNIQUE",  ""),
}

# ─────────────────────────────────────────
# PLANS
# ─────────────────────────────────────────
# 24/10/2026 : alignées sur le site — gratuit = 1 analyse AVEC PDF (le PDF ne coûte rien : produit sur le serveur,
# sans Gemini). Durées maximales portées à 3 h (un match complet filmé dure ~2 h) : elles ne sont appliquées nulle
# part aujourd'hui, mais ne doivent pas bloquer un match le jour où elles le seront.
# Rappel : les variables d'environnement (.env/param.env) l'emportent sur ces valeurs par défaut.
PLANS = {
    "free": {
        "max_analyses": int(os.getenv("PLAN_FREE_MAX",      1)),
        "max_duration": int(os.getenv("PLAN_FREE_DURATION", 10800)),
        "highlights":   True,
        "pdf":          True,
        "montage":      False,
    },
    "starter": {
        "max_analyses": int(os.getenv("PLAN_STARTER_MAX",      20)),
        "max_duration": int(os.getenv("PLAN_STARTER_DURATION", 10800)),
        "highlights":   True,
        "pdf":          True,
        "montage":      False,
    },
    "pro": {
        "max_analyses": int(os.getenv("PLAN_PRO_MAX",      100)),
        "max_duration": int(os.getenv("PLAN_PRO_DURATION", 10800)),
        "highlights":   True,
        "pdf":          True,
        "montage":      True,
    },
    "unique": {
        "max_analyses": int(os.getenv("PLAN_UNIQUE_MAX",      1)),
        "max_duration": int(os.getenv("PLAN_UNIQUE_DURATION", 10800)),
        "highlights":   True,
        "pdf":          True,
        "montage":      True,
    },
}

# ─────────────────────────────────────────
# 26/11/2026 — PRODUCTION : ANALYSES SUR UNE MACHINE GPU DISTANTE (RunPod), à la demande
#   SCOUTIA_GPU_DISTANT=1 sur le SERVEUR DU SITE : les analyses sont mises en file d'attente ; une machine RunPod est
#   démarrée quand il y en a, elle demande les analyses au site (HTTPS + jeton), les fait, dépose les résultats sur R2,
#   et le site les récupère. Sans cette variable : comportement d'avant (analyse sur la machine du site).
# ─────────────────────────────────────────
GPU_DISTANT        = os.getenv("SCOUTIA_GPU_DISTANT", "0").strip() == "1"
WORKER_TOKEN       = os.getenv("SCOUTIA_WORKER_TOKEN", "")          # secret partagé site ↔ machine GPU (long, aléatoire)
SITE_URL           = os.getenv("SCOUTIA_SITE_URL", "").rstrip("/")  # ex. https://matchscoutia.com
RUNPOD_API_KEY     = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_IMAGE       = os.getenv("RUNPOD_IMAGE", "")                   # ex. utilisateur/scoutia-gpu:2026-11-26
RUNPOD_GPU_TYPES   = [g.strip() for g in os.getenv("RUNPOD_GPU_TYPES", "NVIDIA L4,NVIDIA RTX A5000,NVIDIA GeForce RTX 4090").split(",") if g.strip()]
RUNPOD_CLOUD_TYPE  = os.getenv("RUNPOD_CLOUD_TYPE", "SECURE")       # SECURE ou COMMUNITY (moins cher, moins garanti)
RUNPOD_DISQUE_GO   = int(os.getenv("RUNPOD_DISQUE_GO", 80))          # disque de la machine (vidéo + résultats)
GPU_INACTIF_MIN    = int(os.getenv("SCOUTIA_GPU_INACTIF_MIN", 15))   # machine supprimée après N min sans analyse
GPU_SILENCE_MIN    = int(os.getenv("SCOUTIA_GPU_SILENCE_MIN", 20))   # analyse remise en file si plus de nouvelles
# 26/11/2026 : analyses RAPIDES → une machine démarre TOUT DE SUITE pour chacune (pas d'attente), dans la limite de
# sécurité SCOUTIA_GPU_MAX (protège contre un emballement : bogue, abus, quota Gemini, crédit RunPod) ; analyses
# « SOUS 48 H » → au plus SCOUTIA_GPU_MAX_48H machines pour elles (elles peuvent attendre : c'est le principe de l'offre)
GPU_MAX            = int(os.getenv("SCOUTIA_GPU_MAX", 10))
GPU_MAX_48H        = int(os.getenv("SCOUTIA_GPU_MAX_48H", 1))
# variables transmises telles quelles à la machine GPU (réglages de l'analyse et clés nécessaires)
VARIABLES_POUR_GPU = ["GEMINI_API_KEY", "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME",
                      "SCOUTIA_PROFIL_KO", "SCOUTIA_JOUEUR_V30", "SCOUTIA_CAMERA_TYPE", "SCOUTIA_RESUME_MAX_MIN",
                      "SCOUTIA_RESUME_COPIE", "SCOUTIA_PARSEQ_POIDS", "SCOUTIA_PARSEQ_R2_KEY", "SCOUTIA_KO2_SANS_PAUSE",
                      "SCOUTIA_KO_REDATER_GARDE"]

# ─────────────────────────────────────────
# 25/11/2026 — JETONS (remplacent les formules Starter / Pro / Unique). Prix À VÉRIFIER après mesure en production.
# ─────────────────────────────────────────
JETON_PRIX_EUR = float(os.getenv("JETON_PRIX_EUR", 3))
COUT_JETONS = {                                  # (mode, vitesse) → jetons
    ("match", "48h"):     int(os.getenv("JETONS_MATCH_48H", 2)),
    ("match", "rapide"):  int(os.getenv("JETONS_MATCH_RAPIDE", 3)),
    ("player", "48h"):    int(os.getenv("JETONS_JOUEUR_48H", 3)),
    ("player", "rapide"): int(os.getenv("JETONS_JOUEUR_RAPIDE", 4)),
}
PAQUETS_JETONS = {                               # code → jetons achetés, offerts, prix
    "p10":  {"jetons": 10,  "bonus": 0,  "prix_eur": 30},
    "p20":  {"jetons": 20,  "bonus": 1,  "prix_eur": 60},
    "p30":  {"jetons": 30,  "bonus": 2,  "prix_eur": 90},
    "p50":  {"jetons": 50,  "bonus": 5,  "prix_eur": 150},
    "p100": {"jetons": 100, "bonus": 12, "prix_eur": 300},
}
JETONS_ESSAI          = int(os.getenv("JETONS_ESSAI", 2))          # à l'inscription, « match sous 48 h » uniquement
JETONS_PARRAINAGE     = int(os.getenv("JETONS_PARRAINAGE", 2))     # par tranche de 3 filleuls ayant analysé un match
JETONS_VALIDITE_MOIS  = int(os.getenv("JETONS_VALIDITE_MOIS", 24))

# ─────────────────────────────────────────
# EXPORT PDF
# ─────────────────────────────────────────
PDF_LOGO_PATH     = os.getenv("PDF_LOGO_PATH",     "static/logo.png")
PDF_PRIMARY_COLOR = os.getenv("PDF_PRIMARY_COLOR", "#1a73e8")

# ─────────────────────────────────────────
# MODE DEBUG
# True  = logs détaillés (Kaggle / dev local)
# False = logs minimaux (prod Hetzner)
# Peut être surchargé via .env : DEBUG=true
# ─────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ─────────────────────────────────────────
# VALIDATION AU DÉMARRAGE
# ─────────────────────────────────────────
def validate():
    warnings = []
    if SECRET_KEY == "dev-only-change-me":
        warnings.append("⚠️  SECRET_KEY non définie")
    if not CLAUDE_API_KEY:
        warnings.append("⚠️  CLAUDE_API_KEY manquante — résumé IA désactivé")
    if not STRIPE_SECRET_KEY:
        warnings.append("⚠️  STRIPE_SECRET_KEY manquante — paiement désactivé")
    if not STRIPE_WEBHOOK_SECRET:
        warnings.append("⚠️  STRIPE_WEBHOOK_SECRET manquante — webhook désactivé")
    if R2_ENABLED:
        print("✅  Cloudflare R2 activé — stockage cloud")
    else:
        print("ℹ️   R2 non configuré — stockage local (mode test)")
    print(f"ℹ️   YOLO batch={YOLO_BATCH_SIZE}  frame_skip={FRAME_SKIP_EVERY}")
    for w in warnings:
        print(w)
    return len(warnings) == 0


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )