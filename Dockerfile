# ═══════════════════════════════════════════════════════
# ScoutIA — Dockerfile RunPod V9.6
# Base : NVIDIA CUDA + Python 3.11
# GPU cible : RTX 4090 (RunPod)
# ═══════════════════════════════════════════════════════

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── Variables d'environnement ────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# ── Dossier de travail ───────────────────────────────
WORKDIR /app

# ── Dépendances système ──────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 par défaut
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python python3.11 1 \
    && python3 -m pip install --upgrade pip

# ── Dépendances Python ───────────────────────────────
RUN pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install \
    ultralytics \
    opencv-python-headless \
    anthropic \
    fpdf2 \
    python-dotenv \
    scikit-learn \
    joblib \
    numpy \
    google-genai \
    pytesseract \
    boto3 \
    deep-sort-realtime \
    boxmot \
    runpod \
    flask \
    flask-sqlalchemy \
    werkzeug

# ── Cloner le projet ─────────────────────────────────
# Le clone se fait au démarrage du container pour toujours
# avoir la dernière version du code GitHub
# (voir runpod_worker.py)

# ── Pré-télécharger le modèle YOLO ──────────────────
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" || true

# ── Copier le worker ─────────────────────────────────
COPY runpod_worker.py /app/runpod_worker.py

# ── Point d'entrée ───────────────────────────────────
CMD ["python3", "/app/runpod_worker.py"]
