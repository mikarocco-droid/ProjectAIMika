#!/bin/bash
# entrypoint.sh

set -e

echo "=============================="
echo " ScoutIA — Démarrage"
echo "=============================="

# Vérifier GPU
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU detecte : {name} ({mem:.1f} GB VRAM)')
else:
    print('Mode CPU — pas de GPU detecte')
"

# Créer les dossiers
mkdir -p uploads outputs models instance

# Initialiser la base de données
python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Base de donnees initialisee')
"

# Télécharger YOLO si absent
python -c "
import os
from ultralytics import YOLO

models = ['yolov8n.pt', 'yolov8x.pt']
for m in models:
    if not os.path.exists(f'models/{m}'):
        print(f'Telechargement {m}...')
        model = YOLO(m)
        os.makedirs('models', exist_ok=True)
        print(f'OK -> models/{m}')
"

echo "=============================="
echo " Lancement Flask"
echo "=============================="

# Lancer selon l'environnement
if [ "$DEBUG" = "true" ]; then
    python app.py
else
    gunicorn \
        --bind 0.0.0.0:5000 \
        --workers 2 \
        --timeout 3600 \
        --keep-alive 5 \
        app:app
fi
```

---

### 4. `.dockerignore`
```
# .dockerignore
__pycache__/
*.pyc
*.pyo
.venv/
venv/
env/
.git/
.gitignore
uploads/
outputs/
instance/
*.db
*.sqlite3
app -old.py
scout.py
*.pt
models/
.env/
node_modules/