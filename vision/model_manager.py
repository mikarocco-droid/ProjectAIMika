# vision/model_manager.py
# -*- coding: utf-8 -*-

import os
from ultralytics import YOLO


COCO_CLASSES = {
    "person": 0,
    "ball":   32
}

MODEL_CLASSES = {
    "football":   COCO_CLASSES,
    "basketball": COCO_CLASSES,
    "handball":   COCO_CLASSES,
    "base":       COCO_CLASSES
}


class ModelManager:

    def __init__(self):
        self._cache = {}
        os.makedirs("models", exist_ok=True)

    def get_model(self, sport):
        if "base" in self._cache:
            return self._cache["base"], "base"

        # Essayer yolo11x d'abord puis fallback
        for model_name in ["yolo11x.pt", "yolov8x.pt", "yolov8n.pt"]:
            try:
                print(f"  Chargement {model_name}...")
                model = YOLO(model_name)
                self._cache["base"] = model
                print(f"  Modele charge : {model_name}")
                return model, model_name
            except Exception as e:
                print(f"  {model_name} non disponible : {e}")
                continue

        model = YOLO("yolov8n.pt")
        self._cache["base"] = model
        return model, "yolov8n.pt"

    def is_specialized(self, sport):
        return False

    def list_available(self):
        print("Modele : yolov8n.pt (base)")


manager = ModelManager()