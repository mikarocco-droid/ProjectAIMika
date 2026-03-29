# analytics/heatmap.py
# -*- coding: utf-8 -*-

import os
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV non installe — heatmap desactivee")

import config


# ─────────────────────────────────────────
# PALETTE DE COULEURS
# ─────────────────────────────────────────
if CV2_AVAILABLE:
    COLORMAPS = {
        "jet":     cv2.COLORMAP_JET,
        "hot":     cv2.COLORMAP_HOT,
        "plasma":  cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
else:
    COLORMAPS = {}

DEFAULT_COLORMAP = "jet"


# ─────────────────────────────────────────
# EXTRACTION DES POSITIONS
# ─────────────────────────────────────────
def extract_positions(events, target="all", team=None):
    """
    Extrait les positions (x, y) depuis les events.

    target :
        "all"    -> tous les events avec coordonnées
        "ball"   -> positions du ballon
        "shot"   -> positions des tirs
        "goal"   -> positions des buts
        "player" -> positions des joueurs (possession)

    team : 0 | 1 | None (toutes équipes)
    """
    positions = []

    for e in events:

        # Filtre équipe
        if team is not None and e.get("team") != team:
            continue

        if target == "ball":
            if "ball_x" in e and "ball_y" in e:
                positions.append((float(e["ball_x"]), float(e["ball_y"])))

        elif target == "shot":
            if e["type"] == "shot" and "x" in e and "y" in e:
                positions.append((float(e["x"]), float(e["y"])))

        elif target == "goal":
            if e["type"] in ["goal", "score"] and "x" in e and "y" in e:
                positions.append((float(e["x"]), float(e["y"])))

        elif target == "player":
            if e["type"] == "possession" and "x" in e and "y" in e:
                positions.append((float(e["x"]), float(e["y"])))

        elif target == "all":
            if "x" in e and "y" in e:
                positions.append((float(e["x"]), float(e["y"])))
            elif "ball_x" in e and "ball_y" in e:
                positions.append((float(e["ball_x"]), float(e["ball_y"])))

    return positions


# ─────────────────────────────────────────
# GÉNÉRATION DE LA DENSITÉ
# ─────────────────────────────────────────
def build_density_map(positions, width, height, sigma=25):
    """
    Construit une carte de densité gaussienne
    depuis une liste de positions (x, y).
    Retourne toujours un array uint8 2D valide.
    """
    density = np.zeros((height, width), dtype=np.float32)

    if not positions:
        return np.zeros((height, width), dtype=np.uint8)

    for x, y in positions:
        xi = int(np.clip(x, 0, width  - 1))
        yi = int(np.clip(y, 0, height - 1))
        density[yi, xi] += 1.0

    # Flou gaussien
    ksize = sigma * 6
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(ksize, 3)

    density = cv2.GaussianBlur(density, (ksize, ksize), sigma)

    # Normalisation 0-255
    if density.max() > 0:
        density = density / density.max() * 255.0

    return density.astype(np.uint8)


# ─────────────────────────────────────────
# FOND TERRAIN
# ─────────────────────────────────────────
def draw_pitch(width, height, sport="football"):
    """
    Dessine un terrain simplifié comme fond de heatmap.
    """
    # Fond vert foncé
    pitch      = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:, :, 1] = 80  # canal vert

    color = (255, 255, 255)
    t     = 2

    if sport in ["football", "mini-foot"]:

        # Contour
        cv2.rectangle(pitch,
            (int(width * 0.02), int(height * 0.05)),
            (int(width * 0.98), int(height * 0.95)),
            color, t
        )
        # Ligne médiane
        cv2.line(pitch,
            (width // 2, int(height * 0.05)),
            (width // 2, int(height * 0.95)),
            color, t
        )
        # Cercle central
        cv2.circle(pitch,
            (width // 2, height // 2),
            int(min(width, height) * 0.12),
            color, t
        )
        # Surface gauche
        cv2.rectangle(pitch,
            (int(width * 0.02), int(height * 0.25)),
            (int(width * 0.18), int(height * 0.75)),
            color, t
        )
        # Surface droite
        cv2.rectangle(pitch,
            (int(width * 0.82), int(height * 0.25)),
            (int(width * 0.98), int(height * 0.75)),
            color, t
        )
        # But gauche
        cv2.rectangle(pitch,
            (int(width * 0.02), int(height * 0.40)),
            (int(width * 0.05), int(height * 0.60)),
            color, t
        )
        # But droit
        cv2.rectangle(pitch,
            (int(width * 0.95), int(height * 0.40)),
            (int(width * 0.98), int(height * 0.60)),
            color, t
        )

    elif sport == "basketball":

        # Contour
        cv2.rectangle(pitch,
            (int(width * 0.02), int(height * 0.05)),
            (int(width * 0.98), int(height * 0.95)),
            color, t
        )
        # Ligne médiane
        cv2.line(pitch,
            (width // 2, int(height * 0.05)),
            (width // 2, int(height * 0.95)),
            color, t
        )
        # Cercle central
        cv2.circle(pitch,
            (width // 2, height // 2),
            int(min(width, height) * 0.10),
            color, t
        )
        # Raquette gauche
        cv2.rectangle(pitch,
            (int(width * 0.02), int(height * 0.30)),
            (int(width * 0.19), int(height * 0.70)),
            color, t
        )
        # Raquette droite
        cv2.rectangle(pitch,
            (int(width * 0.81), int(height * 0.30)),
            (int(width * 0.98), int(height * 0.70)),
            color, t
        )
        # Panier gauche
        cv2.circle(pitch,
            (int(width * 0.05), height // 2),
            int(height * 0.05),
            color, t
        )
        # Panier droit
        cv2.circle(pitch,
            (int(width * 0.95), height // 2),
            int(height * 0.05),
            color, t
        )

    elif sport == "handball":

        # Contour
        cv2.rectangle(pitch,
            (int(width * 0.02), int(height * 0.05)),
            (int(width * 0.98), int(height * 0.95)),
            color, t
        )
        # Ligne médiane
        cv2.line(pitch,
            (width // 2, int(height * 0.05)),
            (width // 2, int(height * 0.95)),
            color, t
        )
        # Zone de but gauche
        cv2.ellipse(pitch,
            (int(width * 0.02), height // 2),
            (int(width * 0.18), int(height * 0.40)),
            0, -90, 90, color, t
        )
        # Zone de but droite
        cv2.ellipse(pitch,
            (int(width * 0.98), height // 2),
            (int(width * 0.18), int(height * 0.40)),
            0, 90, 270, color, t
        )

    elif sport == "rugby":

        # Contour
        cv2.rectangle(pitch,
            (int(width * 0.02), int(height * 0.05)),
            (int(width * 0.98), int(height * 0.95)),
            color, t
        )
        # Ligne médiane
        cv2.line(pitch,
            (width // 2, int(height * 0.05)),
            (width // 2, int(height * 0.95)),
            color, t
        )
        # Lignes des 22m
        cv2.line(pitch,
            (int(width * 0.22), int(height * 0.05)),
            (int(width * 0.22), int(height * 0.95)),
            color, t
        )
        cv2.line(pitch,
            (int(width * 0.78), int(height * 0.05)),
            (int(width * 0.78), int(height * 0.95)),
            color, t
        )
        # En-buts gauche
        cv2.rectangle(pitch,
            (int(width * 0.02), int(height * 0.05)),
            (int(width * 0.08), int(height * 0.95)),
            color, t
        )
        # En-buts droit
        cv2.rectangle(pitch,
            (int(width * 0.92), int(height * 0.05)),
            (int(width * 0.98), int(height * 0.95)),
            color, t
        )

    else:
        # Terrain générique pour sports non définis
        cv2.rectangle(pitch,
            (int(width * 0.02), int(height * 0.05)),
            (int(width * 0.98), int(height * 0.95)),
            color, t
        )
        cv2.line(pitch,
            (width // 2, int(height * 0.05)),
            (width // 2, int(height * 0.95)),
            color, t
        )

    return pitch


# ─────────────────────────────────────────
# HEATMAP PRINCIPALE
# ─────────────────────────────────────────
def generate_heatmap(
    events,
    output_path,
    width      = None,
    height     = None,
    sport      = "football",
    target     = "all",
    team       = None,
    colormap   = DEFAULT_COLORMAP,
    alpha      = 0.65,
    sigma      = 25
):
    """
    Génère une heatmap PNG superposée sur un terrain.

    Paramètres :
        events      : liste d'events du pipeline
        output_path : chemin PNG de sortie
        width/height: dimensions (défaut : config)
        sport       : football | basketball | handball | rugby | ...
        target      : all | ball | shot | goal | player
        team        : 0 | 1 | None
        colormap    : jet | hot | plasma | viridis
        alpha       : transparence heatmap (0-1)
        sigma       : rayon diffusion gaussienne
    """

    if not CV2_AVAILABLE:
        print("OpenCV non disponible — heatmap ignoree")
        return None

    width  = width  or config.FRAME_WIDTH
    height = height or config.FRAME_HEIGHT

    # Positions
    positions = extract_positions(events, target=target, team=team)
    print(f"  Heatmap '{target}' : {len(positions)} positions")

    # Carte de densité — toujours uint8
    density = build_density_map(positions, width, height, sigma=sigma)

    # Vérifications sécurité
    if density.dtype != np.uint8:
        density = density.astype(np.uint8)
    if len(density.shape) != 2:
        density = cv2.cvtColor(density, cv2.COLOR_BGR2GRAY)

    # Colorisation
    cmap    = COLORMAPS.get(colormap, cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(density, cmap)

    # Fond terrain
    pitch = draw_pitch(width, height, sport=sport)

    # Fusion heatmap + terrain
    mask    = density > 10
    output  = pitch.copy()

    if mask.any():
        blended        = cv2.addWeighted(colored, alpha, pitch, 1 - alpha, 0)
        output[mask]   = blended[mask]

    # Légende
    label = f"Heatmap {target.capitalize()}"
    if team is not None:
        label += f" Equipe {team + 1}"

    cv2.putText(
        output, label,
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, output)

    print(f"  Heatmap OK -> {output_path}")
    return output_path


# ─────────────────────────────────────────
# GÉNÉRATION MULTIPLE
# ─────────────────────────────────────────
def generate_all_heatmaps(
    events,
    output_dir,
    width  = None,
    height = None,
    sport  = "football"
):
    """
    Génère 4 heatmaps en une fois :
        global    -> toute l'activité
        tirs      -> positions des tirs
        equipe_A  -> activité équipe 0
        equipe_B  -> activité équipe 1
    """
    os.makedirs(output_dir, exist_ok=True)

    targets = [
        ("all",  None, "global"),
        ("shot", None, "tirs"),
        ("all",  0,    "equipe_A"),
        ("all",  1,    "equipe_B"),
    ]

    results = {}

    for target, team, name in targets:
        path   = os.path.join(output_dir, f"heatmap_{name}.png")
        result = generate_heatmap(
            events      = events,
            output_path = path,
            width       = width,
            height      = height,
            sport       = sport,
            target      = target,
            team        = team
        )
        if result:
            results[name] = result

    return results