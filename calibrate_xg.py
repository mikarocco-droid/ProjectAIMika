# calibrate_xg.py
# -*- coding: utf-8 -*-
#
# Script de calibration xG à lancer MANUELLEMENT
# quand tu as 200+ tirs enregistrés dans xg_training_data.json
#
# Usage Kaggle :
#   python calibrate_xg.py --sport football --dir outputs/learning
#
# Ce script est indépendant du pipeline — zéro impact sur la durée d'analyse

import os
import json
import math
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve


def load_training_data(base_dir, sport):
    """Charge les données xG collectées par le learning model."""
    path = os.path.join(base_dir, sport, "xg_training_data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Lance d'abord le pipeline sur plusieurs matchs pour collecter des données."
        )
    with open(path) as f:
        data = json.load(f)
    print(f"✅ {len(data)} tirs chargés depuis {path}")
    return data


def build_features(data, frame_w=1920, frame_h=1080, pitch_l_m=105.0, goal_w_m=7.32):
    """
    Transforme les données brutes en features pour la régression logistique.
    Features : distance_effect, angle_effect
    """
    X, y = [], []

    for d in data:
        x = float(d.get("x", 0))
        yy = float(d.get("y", frame_h / 2))
        is_goal = int(d.get("is_goal", 0))

        # Centre but le plus proche
        goal_y_px = frame_h / 2.0
        goal_left  = (0.0,     goal_y_px)
        goal_right = (frame_w, goal_y_px)

        dist_left  = math.hypot(x - goal_left[0],  yy - goal_left[1])
        dist_right = math.hypot(x - goal_right[0], yy - goal_right[1])
        goal_cx, goal_cy = goal_left if dist_left <= dist_right else goal_right

        # Distance pixels → mètres
        dist_px  = math.hypot(x - goal_cx, yy - goal_cy)
        px_per_m = frame_w / pitch_l_m
        dist_m   = dist_px / max(px_per_m, 1e-6)

        # Angle entre les poteaux
        half_goal_px = (goal_w_m / pitch_l_m) * frame_w * 0.5
        post1 = (goal_cx, goal_cy - half_goal_px)
        post2 = (goal_cx, goal_cy + half_goal_px)

        v1x, v1y = post1[0] - x, post1[1] - yy
        v2x, v2y = post2[0] - x, post2[1] - yy

        dot   = v1x * v2x + v1y * v2y
        norm1 = math.hypot(v1x, v1y) + 1e-6
        norm2 = math.hypot(v2x, v2y) + 1e-6
        cos_a = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        angle = math.acos(cos_a)

        # Normalisation + non-linéarités
        distance_norm   = min(dist_m / pitch_l_m, 1.0)
        angle_norm      = angle / math.pi
        distance_effect = (1.0 - distance_norm) ** 1.3
        angle_effect    = angle_norm ** 1.7

        X.append([distance_effect, angle_effect])
        y.append(is_goal)

    return np.array(X), np.array(y)


def train_model(X, y):
    """Entraîne la régression logistique et retourne le modèle + coefficients."""
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, y)

    a = float(model.coef_[0][0])   # poids distance
    b = float(model.coef_[0][1])   # poids angle
    c = float(model.intercept_[0]) # biais

    return model, a, b, c


def evaluate_model(model, X, y):
    """Évalue la calibration et la performance du modèle."""
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"\n📊 Performance (AUC-ROC) :")
    print(f"   Moyenne : {scores.mean():.3f} ± {scores.std():.3f}")

    # Calibration — vérification que xG=0.2 → ~20% de buts
    y_proba = model.predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted = calibration_curve(y, y_proba, n_bins=5)

    print(f"\n📈 Calibration (xG prédit vs taux de buts réel) :")
    for pred, real in zip(mean_predicted, fraction_of_positives):
        diff  = abs(pred - real)
        status = "✅" if diff < 0.05 else ("⚠️" if diff < 0.10 else "❌")
        print(f"   xG prédit {pred:.2f} → buts réels {real:.2f} {status}")


def save_calibrated_model(base_dir, sport, a, b, c, n_samples):
    """Sauvegarde les coefficients calibrés dans xg_model.json."""
    path = os.path.join(base_dir, sport, "xg_model.json")

    model_data = {
        "w0":       round(c, 4),   # biais
        "w1":       round(a, 4),   # poids distance
        "w2":       round(b, 4),   # poids angle
        "n_samples": n_samples,
        "calibrated": True,
        "method":   "logistic_regression",
    }

    with open(path, "w") as f:
        json.dump(model_data, f, indent=2)

    print(f"\n✅ Modèle sauvegardé → {path}")
    print(f"   w0 (biais)    : {c:.4f}")
    print(f"   w1 (distance) : {a:.4f}")
    print(f"   w2 (angle)    : {b:.4f}")

    # Vérification sur quelques cas typiques
    print(f"\n🎯 Vérification sur cas typiques :")
    test_cases = [
        ("Tir à 30m axe",         0.05, 0.40),
        ("Tir à 20m axe",         0.12, 0.50),
        ("Surface excentré",      0.25, 0.30),
        ("Surface axe",           0.35, 0.65),
        ("Face au but 10m",       0.55, 0.80),
        ("1v1 gardien",           0.65, 0.90),
    ]
    for label, d_effect, a_effect in test_cases:
        score = c + (a * d_effect) + (b * a_effect)
        xg    = round(1 / (1 + math.exp(-score)), 3)
        print(f"   {label:25} → xG = {xg:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Calibration xG ScoutAI")
    parser.add_argument("--sport", default="football")
    parser.add_argument("--dir",   default="outputs/learning")
    parser.add_argument("--min_samples", type=int, default=100,
                        help="Nombre minimum de tirs requis (défaut: 100)")
    args = parser.parse_args()

    print(f"🔧 Calibration xG — sport={args.sport}")
    print(f"   Répertoire : {args.dir}\n")

    # Chargement
    data = load_training_data(args.dir, args.sport)

    n_goals = sum(1 for d in data if d.get("is_goal"))
    n_shots = len(data) - n_goals
    print(f"   Buts   : {n_goals}")
    print(f"   Non-buts : {n_shots}")
    print(f"   Taux de conversion : {n_goals/len(data):.1%}")

    if len(data) < args.min_samples:
        print(f"\n⚠️  Seulement {len(data)} tirs — minimum recommandé : {args.min_samples}")
        print(f"   Continue quand même avec les données disponibles...")

    # Features
    X, y = build_features(data)
    print(f"\n✅ {len(X)} features construites")

    # Entraînement
    model, a, b, c = train_model(X, y)
    print(f"\n🧠 Modèle entraîné :")
    print(f"   score = {c:.3f} + ({a:.3f} × distance) + ({b:.3f} × angle)")

    # Évaluation
    evaluate_model(model, X, y)

    # Sauvegarde
    save_calibrated_model(args.dir, args.sport, a, b, c, len(data))

    print(f"\n🚀 Relance le pipeline pour utiliser le modèle calibré !")


if __name__ == "__main__":
    main()