#!/bin/bash
# entrypoint.sh — ScoutIA
# Usage :
#   MODE=app     → Flask/Gunicorn (défaut)
#   MODE=worker  → Celery worker pipeline
#   MODE=flower  → Celery monitoring

set -e

echo "=============================="
echo " ScoutIA — Démarrage ($MODE)"
echo "=============================="

# ── GPU check ──────────────────────────
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU detecte : {name} ({mem:.1f} GB VRAM)')
else:
    print('Mode CPU — pas de GPU detecte')
"

# ── Dossiers ───────────────────────────
mkdir -p uploads outputs models instance outputs/learning

# ── Base de données ────────────────────
python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Base de donnees OK (create_all)')
"

# ── Modèle YOLO11m ─────────────────────
python -c "
import os
from ultralytics import YOLO
model_path = 'models/yolo11m.pt'
if not os.path.exists(model_path):
    print('Telechargement yolo11m.pt...')
    model = YOLO('yolo11m.pt')
    os.makedirs('models', exist_ok=True)
    print(f'OK -> {model_path}')
else:
    print(f'yolo11m.pt deja present')
"

echo "=============================="

# ── Lancement selon MODE ───────────────
MODE=${MODE:-app}

if [ "$MODE" = "worker" ]; then
    echo " Celery Worker — queue pipeline"
    echo "=============================="
    exec celery -A tasks worker \
        --loglevel=info \
        --concurrency=1 \
        -Q pipeline \
        --max-tasks-per-child=1

elif [ "$MODE" = "flower" ]; then
    echo " Flower — monitoring Celery"
    echo "=============================="
    exec celery -A tasks flower \
        --port=5555 \
        --basic_auth=admin:scoutia2024

else
    echo " Flask — mode $( [ "$DEBUG" = "true" ] && echo "dev" || echo "prod/gunicorn" )"
    echo "=============================="
    if [ "$DEBUG" = "true" ]; then
        exec python app.py
    else
        exec gunicorn \
            --bind 0.0.0.0:5000 \
            --workers 2 \
            --timeout 3600 \
            --keep-alive 5 \
            --access-logfile - \
            --error-logfile - \
            app:app
    fi
fi
