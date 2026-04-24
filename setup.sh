#!/bin/bash
# setup.sh — Installation automatique ScoutIA dans Codespaces
set -e

echo "=============================="
echo " ScoutIA — Setup Codespaces"
echo "=============================="

# ── Dépendances système ────────────────
echo "→ Installation dépendances système..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg tesseract-ocr

# ── Fichier .env ───────────────────────
echo "→ Création .env/param.env..."
mkdir -p .env
if [ ! -f .env/param.env ]; then
cat > .env/param.env << 'EOF'
FLASK_SECRET_KEY=codespaces_dev_key_change_in_prod
DATABASE_URI=sqlite:///db.sqlite3
DEBUG=true
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
R2_ENABLED=false
YOLO_BATCH_SIZE=4
FRAME_SKIP_EVERY=4
DELETE_RAW_VIDEO=true
EOF
echo "  → .env/param.env créé — remplis GEMINI_API_KEY et ANTHROPIC_API_KEY"
else
echo "  → .env/param.env déjà présent"
fi

# ── Dossiers ───────────────────────────
mkdir -p uploads outputs models instance outputs/learning
for pkg in analysis analytics ai vision video rendering export payments sports tracking; do
    mkdir -p $pkg
    touch $pkg/__init__.py
done

# ── Docker Compose ─────────────────────
echo "→ Lancement Docker Compose..."
docker-compose up -d --build

echo ""
echo "=============================="
echo " Setup terminé !"
echo "=============================="
echo ""
echo " Services disponibles :"
echo "   Flask API  → http://localhost:5000"
echo "   Flower     → http://localhost:5555"
echo "   Redis      → localhost:6379"
echo ""
echo " ⚠️  Remplis tes clés API dans .env/param.env"
echo "   GEMINI_API_KEY=..."
echo "   ANTHROPIC_API_KEY=..."
echo ""
echo " Puis recharge Docker :"
echo "   docker-compose up -d"
echo ""
