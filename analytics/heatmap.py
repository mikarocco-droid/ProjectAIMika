# analytics/heatmap.py
# -*- coding: utf-8 -*-

import os
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV non installé — heatmap désactivée")

import config


# ─────────────────────────────────────────
# COLORMAPS
# ─────────────────────────────────────────
COLORMAPS = {
    "jet":     cv2.COLORMAP_JET if CV2_AVAILABLE else None,
    "hot":     cv2.COLORMAP_HOT if CV2_AVAILABLE else None,
    "plasma":  cv2.COLORMAP_PLASMA if CV2_AVAILABLE else None,
    "viridis": cv2.COLORMAP_VIRIDIS if CV2_AVAILABLE else None,
}

DEFAULT_COLORMAP = "jet"


# ─────────────────────────────────────────
# EXTRACTION POSITIONS AVANCÉE
# ─────────────────────────────────────────
def extract_positions(events, target="all", team=None, player_id=None, weighted=False):
    """
    Extraction intelligente des positions
    """

    positions = []

    for e in events:

        # ── filtres
        if team is not None and e.get("team") != team:
            continue

        if player_id is not None:
            if str(e.get("player")) != str(player_id):
                continue

        x, y = None, None

        # ── types
        if target == "shot" and e["type"] == "shot":
            x, y = e.get("x"), e.get("y")

        elif target == "goal" and e["type"] in ["goal", "score"]:
            x, y = e.get("x"), e.get("y")

        elif target == "pass" and e["type"] == "pass":
            x, y = e.get("x"), e.get("y")

        elif target == "touch" and e["type"] == "possession":
            x, y = e.get("x"), e.get("y")

        elif target == "all":
            x, y = e.get("x"), e.get("y")

        if x is None or y is None:
            continue

        weight = 1.0

        if weighted:
            if e["type"] == "shot":
                weight = 1 + e.get("xg", 0.1) * 3
            elif e["type"] == "goal":
                weight = 5
            elif e["type"] == "pass":
                weight = 1.2

        positions.append((float(x), float(y), weight))

    return positions


# ─────────────────────────────────────────
# DENSITY MAP AVANCÉE
# ─────────────────────────────────────────
def build_density_map(positions, width, height, sigma=25):

    density = np.zeros((height, width), dtype=np.float32)

    if not positions:
        return np.zeros((height, width), dtype=np.uint8)

    for x, y, w in positions:
        xi = int(np.clip(x, 0, width - 1))
        yi = int(np.clip(y, 0, height - 1))
        density[yi, xi] += w

    # Gaussian blur
    ksize = sigma * 6
    if ksize % 2 == 0:
        ksize += 1

    density = cv2.GaussianBlur(density, (ksize, ksize), sigma)

    # normalize
    if density.max() > 0:
        density = density / density.max() * 255

    return density.astype(np.uint8)


# ─────────────────────────────────────────
# TERRAIN PRO
# ─────────────────────────────────────────
def draw_pitch(width, height, sport="football"):

    pitch = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:, :, 1] = 90

    color = (255, 255, 255)
    t = 2

    # FOOTBALL
    if sport in ["football", "mini-foot"]:
        cv2.rectangle(pitch, (20, 20), (width-20, height-20), color, t)
        cv2.line(pitch, (width//2, 20), (width//2, height-20), color, t)
        cv2.circle(pitch, (width//2, height//2), 80, color, t)

        # surfaces
        cv2.rectangle(pitch, (20, height//3), (150, height*2//3), color, t)
        cv2.rectangle(pitch, (width-150, height//3), (width-20, height*2//3), color, t)

    # BASKET
    elif sport == "basketball":
        cv2.rectangle(pitch, (20, 20), (width-20, height-20), color, t)
        cv2.line(pitch, (width//2, 20), (width//2, height-20), color, t)
        cv2.circle(pitch, (width//2, height//2), 60, color, t)

    # GENERIC
    else:
        cv2.rectangle(pitch, (20, 20), (width-20, height-20), color, t)

    return pitch


# ─────────────────────────────────────────
# HEATMAP MULTI-LAYER
# ─────────────────────────────────────────
def generate_heatmap(
    events,
    output_path,
    width=None,
    height=None,
    sport="football",
    target="all",
    team=None,
    player_id=None,
    weighted=True,
    colormap=DEFAULT_COLORMAP,
    alpha=0.65,
    sigma=25
):

    if not CV2_AVAILABLE:
        return None

    width  = width or config.FRAME_WIDTH
    height = height or config.FRAME_HEIGHT

    positions = extract_positions(
        events,
        target=target,
        team=team,
        player_id=player_id,
        weighted=weighted
    )

    print(f"Heatmap {target} : {len(positions)} points")

    density = build_density_map(positions, width, height, sigma)

    cmap = COLORMAPS.get(colormap, cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(density, cmap)

    pitch = draw_pitch(width, height, sport)

    output = pitch.copy()
    mask = density > 10

    if mask.any():
        blended = cv2.addWeighted(colored, alpha, pitch, 1 - alpha, 0)
        output[mask] = blended[mask]

    # ── TITRE
    label = f"{target.upper()} HEATMAP"
    if player_id:
        label += f" PLAYER {player_id}"

    cv2.putText(output, label, (20, height-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, output)

    return output_path


# ─────────────────────────────────────────
# MULTI HEATMAPS GOD MODE
# ─────────────────────────────────────────
def generate_all_heatmaps(events, output_dir, width=None, height=None, sport="football"):

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    configs = [
        ("global", "all", None, None),
        ("shots", "shot", None, None),
        ("goals", "goal", None, None),
        ("team_A", "all", 0, None),
        ("team_B", "all", 1, None),
    ]

    for name, target, team, player in configs:
        path = os.path.join(output_dir, f"{name}.png")

        res = generate_heatmap(
            events=events,
            output_path=path,
            width=width,
            height=height,
            sport=sport,
            target=target,
            team=team,
            player_id=player
        )

        if res:
            results[name] = res

    return results


# ─────────────────────────────────────────
# HEATMAP JOUEUR (🔥 FEATURE PRO)
# ─────────────────────────────────────────
def generate_player_heatmaps(events, output_dir):

    players = set(str(e.get("player")) for e in events if e.get("player"))

    results = {}

    for pid in players:
        path = os.path.join(output_dir, f"player_{pid}.png")

        res = generate_heatmap(
            events,
            path,
            target="touch",
            player_id=pid
        )

        if res:
            results[pid] = res

    return results