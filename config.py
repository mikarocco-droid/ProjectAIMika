# config.py

import os
from dotenv import load_dotenv

load_dotenv(".env/param.env")

# ─────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────
SECRET_KEY   = os.getenv("SECRET_KEY",    "dev-only-change-me")
DATABASE_URI = os.getenv("DATABASE_URI",  "sqlite:///db.sqlite3")
DEBUG        = os.getenv("DEBUG",         "false").lower() == "true"

# ─────────────────────────────────────────
# UPLOADS
# ─────────────────────────────────────────
UPLOAD_FOLDER      = os.getenv("UPLOAD_FOLDER",  "uploads")
OUTPUT_FOLDER      = os.getenv("OUTPUT_FOLDER",  "outputs")
MAX_UPLOAD_SIZE    = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024 * 1024))
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "m4v"}

# ─────────────────────────────────────────
# VIDÉO
# ─────────────────────────────────────────
VIDEO_PATH   = os.getenv("VIDEO_PATH",  "match.mp4")
FPS          = int(os.getenv("FPS",          30))
FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  1280))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 720))

# ─────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────
YOLO_MODEL      = os.getenv("YOLO_MODEL",      "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", 0.4))
BALL_METHOD     = os.getenv("BALL_METHOD",     "hybrid")

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
HIGHLIGHT_MAX           = int(os.getenv("HIGHLIGHT_MAX",           15))
HIGHLIGHT_MIN_SCORE     = int(os.getenv("HIGHLIGHT_MIN_SCORE",      4))
HIGHLIGHT_WINDOW_FRAMES = int(os.getenv("HIGHLIGHT_WINDOW_FRAMES", 90))
HIGHLIGHT_BEFORE_SEC    = int(os.getenv("HIGHLIGHT_BEFORE_SEC",     5))
HIGHLIGHT_AFTER_SEC     = int(os.getenv("HIGHLIGHT_AFTER_SEC",      4))

# ─────────────────────────────────────────
# MONTAGE
# ─────────────────────────────────────────
MONTAGE_WITH_INTRO  = os.getenv("MONTAGE_WITH_INTRO",  "true").lower() == "true"
MONTAGE_WITH_LABELS = os.getenv("MONTAGE_WITH_LABELS", "true").lower() == "true"
MONTAGE_WITH_FADES  = os.getenv("MONTAGE_WITH_FADES",  "true").lower() == "true"
MONTAGE_RESOLUTION  = os.getenv("MONTAGE_RESOLUTION",  "1280x720")

# ─────────────────────────────────────────
# IA — CLAUDE
# ─────────────────────────────────────────
CLAUDE_API_KEY    = os.getenv("CLAUDE_API_KEY",    "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL",      "claude-sonnet-4-20250514")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", 500))

# ─────────────────────────────────────────
# STRIPE
# ─────────────────────────────────────────
STRIPE_PUBLIC_KEY      = os.getenv("STRIPE_PUBLIC_KEY",      "")
STRIPE_SECRET_KEY      = os.getenv("STRIPE_SECRET_KEY",      "")
STRIPE_WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET",  "")

STRIPE_PRICE_IDS = {
    "starter": os.getenv("STRIPE_PRICE_STARTER", ""),
    "pro":     os.getenv("STRIPE_PRICE_PRO",     ""),
    "unique":  os.getenv("STRIPE_PRICE_UNIQUE",  ""),
}

# ─────────────────────────────────────────
# PLANS
# ─────────────────────────────────────────
PLANS = {
    "free": {
        "max_analyses": int(os.getenv("PLAN_FREE_MAX",      3)),
        "max_duration": int(os.getenv("PLAN_FREE_DURATION", 300)),
        "highlights":   True,
        "pdf":          False,
        "montage":      False,
    },
    "starter": {
        "max_analyses": int(os.getenv("PLAN_STARTER_MAX",      20)),
        "max_duration": int(os.getenv("PLAN_STARTER_DURATION", 3600)),
        "highlights":   True,
        "pdf":          True,
        "montage":      False,
    },
    "pro": {
        "max_analyses": int(os.getenv("PLAN_PRO_MAX",      100)),
        "max_duration": int(os.getenv("PLAN_PRO_DURATION", 7200)),
        "highlights":   True,
        "pdf":          True,
        "montage":      True,
    },
    "unique": {
        "max_analyses": int(os.getenv("PLAN_UNIQUE_MAX",      1)),
        "max_duration": int(os.getenv("PLAN_UNIQUE_DURATION", 7200)),
        "highlights":   True,
        "pdf":          True,
        "montage":      True,
    },
}

# ─────────────────────────────────────────
# EXPORT PDF
# ─────────────────────────────────────────
PDF_LOGO_PATH     = os.getenv("PDF_LOGO_PATH",     "static/logo.png")
PDF_PRIMARY_COLOR = os.getenv("PDF_PRIMARY_COLOR", "#1a73e8")

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

    for w in warnings:
        print(w)

    return len(warnings) == 0


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )