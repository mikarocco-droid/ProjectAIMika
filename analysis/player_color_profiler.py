"""
analysis/player_color_profiler.py
----------------------------------
Outil de diagnostic de l'extraction couleur proxy.
Ne modifie RIEN au pipeline. Observe et mesure uniquement.

Pour chaque blob joueur détecté, compare 4 méthodes d'extraction :
  original            : couleur moyenne RGB sur tout le crop
  centered_crop       : médiane HSV sur torse (25%-75% vertical)
  centered_crop_no_grass : idem + filtre pixels pelouse/ombre
  hs_histogram        : pic du histogramme HS (teinte+saturation)

Produit un CSV avec une ligne par joueur/frame :
  t_s, frame_idx, player_idx,
  bbox_x, bbox_y, bbox_w, bbox_h, crop_area,
  [méthode]_H, [méthode]_S, [méthode]_V,
  [méthode]_valid_pixels, [méthode]_dominant_fraction,
  agreement_with_original,   # 1 si centered_crop_no_grass ~ original
  dominant_fraction_cng,     # fraction des pixels qui ont la couleur dominante
  valid_pixels_ratio,        # ratio pixels exploitables / total crop
  temporal_stability_est,    # estimé sur fenêtre glissante (rempli en post)
  team_original,             # équipe assignée par la règle R/B actuelle
  team_cng,                  # équipe assignée par centered_crop_no_grass
  agreement_team             # 1 si team_original == team_cng
"""

import cv2
import numpy as np
import csv
import math
import os
from pathlib import Path


# ── Helpers couleur ──────────────────────────────────────────────────────────

def _is_grass(h, s, v):
    """Pixel pelouse ou ombre de pelouse."""
    h, s, v = int(h), int(s), int(v)
    if 30 <= h <= 90 and s > 40 and v > 60: return True
    if 25 <= h <= 45 and v > 40: return True
    return False

def _assign_team_rule(H, S):
    """Règle actuelle rouge/bleu."""
    if H is None: return None
    if (H < 12 or H > 168) and S > 80: return 0
    if 95 <= H <= 140 and S > 40: return 1
    if 12 <= H <= 25 and S > 100: return 0
    return None

def _dominant_hsv_histogram(roi_hsv, bins_h=18, bins_s=4):
    """Pic du histogramme HS — robuste aux pixels minoritaires."""
    h_vals = roi_hsv[:, 0].astype(np.float32)
    s_vals = roi_hsv[:, 1].astype(np.float32)
    hist, xedges, yedges = np.histogram2d(
        h_vals, s_vals,
        bins=[bins_h, bins_s],
        range=[[0, 180], [0, 256]]
    )
    idx = np.unravel_index(hist.argmax(), hist.shape)
    h_center = (xedges[idx[0]] + xedges[idx[0]+1]) / 2
    s_center = (yedges[idx[1]] + yedges[idx[1]+1]) / 2
    dominant_count = hist[idx]
    total = len(roi_hsv)
    fraction = dominant_count / max(total, 1)
    # V moyen sur les pixels dans ce bin
    mask = ((h_vals >= xedges[idx[0]]) & (h_vals < xedges[idx[0]+1]) &
            (s_vals >= yedges[idx[1]]) & (s_vals < yedges[idx[1]+1]))
    v_mean = roi_hsv[mask, 2].mean() if mask.sum() > 0 else 0
    return int(h_center), int(s_center), int(v_mean), float(fraction)


# ── Extraction par méthode ───────────────────────────────────────────────────

def extract_original(img_bgr, x1, y1, x2, y2):
    """Couleur moyenne RGB sur tout le crop."""
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size < 50: return None
    b, g, r = roi.mean(axis=(0,1))[:3]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = int(np.median(roi_hsv[:,:,0]))
    S = int(np.median(roi_hsv[:,:,1]))
    V = int(np.median(roi_hsv[:,:,2]))
    return {"H": H, "S": S, "V": V, "valid_pixels": roi.shape[0]*roi.shape[1],
            "dominant_fraction": 1.0, "r": int(r), "g": int(g), "b": int(b)}

def extract_centered_crop(img_bgr, x1, y1, x2, y2):
    """Médiane HSV sur torse (25%-75% vertical)."""
    h_crop = y2 - y1
    yt = y1 + int(h_crop * 0.25)
    yb = y1 + int(h_crop * 0.75)
    roi = img_bgr[yt:yb, x1:x2]
    if roi.size < 50: return None
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    H = int(np.median(roi_hsv[:, 0]))
    S = int(np.median(roi_hsv[:, 1]))
    V = int(np.median(roi_hsv[:, 2]))
    return {"H": H, "S": S, "V": V, "valid_pixels": len(roi_hsv),
            "dominant_fraction": 1.0}

def extract_centered_crop_no_grass(img_bgr, x1, y1, x2, y2):
    """Médiane HSV sur torse, pixels pelouse filtrés."""
    h_crop = y2 - y1
    yt = y1 + int(h_crop * 0.25)
    yb = y1 + int(h_crop * 0.75)
    roi = img_bgr[yt:yb, x1:x2]
    if roi.size < 50: return None
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    total = len(roi_hsv)
    mask = np.array([not _is_grass(p[0], p[1], p[2]) and int(p[2]) > 30
                     for p in roi_hsv])
    valid = roi_hsv[mask]
    if len(valid) < 8: return None
    H = int(np.median(valid[:, 0]))
    S = int(np.median(valid[:, 1]))
    V = int(np.median(valid[:, 2]))
    return {"H": H, "S": S, "V": V,
            "valid_pixels": len(valid),
            "valid_pixels_ratio": len(valid) / max(total, 1),
            "dominant_fraction": len(valid) / max(total, 1)}

def extract_hs_histogram(img_bgr, x1, y1, x2, y2):
    """Pic histogramme HS sur torse, sans pelouse."""
    h_crop = y2 - y1
    yt = y1 + int(h_crop * 0.25)
    yb = y1 + int(h_crop * 0.75)
    roi = img_bgr[yt:yb, x1:x2]
    if roi.size < 50: return None
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    mask = np.array([not _is_grass(p[0], p[1], p[2]) and int(p[2]) > 30
                     for p in roi_hsv])
    valid = roi_hsv[mask]
    if len(valid) < 8: return None
    H, S, V, frac = _dominant_hsv_histogram(valid)
    return {"H": H, "S": S, "V": V,
            "valid_pixels": len(valid),
            "dominant_fraction": frac}

def _hue_diff(h1, h2):
    """Distance angulaire entre deux teintes (0-180)."""
    d = abs(int(h1) - int(h2))
    return min(d, 180 - d)


# ── Détection blobs (masque morphologique) ──────────────────────────────────

def detect_blobs(img, frame_w, frame_h, h_min=30):
    """
    Détection des blobs joueurs par masque morphologique.

    h_min : borne basse de Hue pour le masque pelouse/foreground.
        Valeur historique = 30 (pelouse d'été). Sur les matchs à pelouse
        d'hiver/brune, une valeur plus basse (ex: 20) capture beaucoup
        plus de pelouse — mais dégrade la détection sur pelouse d'été
        (mesuré : -70% de blobs joueurs sur Ster-Wanze avec h_min=20).
        Ne pas changer la valeur par défaut sans validation multi-matchs.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(cv2.inRange(hsv, np.array([h_min,40,40]),
                                             np.array([90,255,255])))
    mask[:int(frame_h*0.08), :] = 0
    mask[int(frame_h*0.90):, :] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k),
                            cv2.MORPH_OPEN, k)
    blobs = []
    for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area < 200 or area > 8000: continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w/max(h,1) > 3.0 or w/max(h,1) < 0.2: continue
        blobs.append((x, y, x+w, y+h, w*h))
    return blobs


# ── Profiler principal ───────────────────────────────────────────────────────

def profile_video(video_path, output_csv, sample_fps=1.0, max_minutes=30,
                  max_blobs_per_frame=40, h_min=30):
    """
    Analyse une vidéo frame par frame et produit un CSV de diagnostics couleur.

    Parameters
    ----------
    video_path   : chemin vers la vidéo
    output_csv   : chemin du CSV de sortie
    sample_fps   : frames analysées par seconde (défaut=1.0)
    max_minutes  : durée max analysée
    max_blobs_per_frame : limite de blobs par frame (évite les frames avec trop de bruit)
    """
    cap = cv2.VideoCapture(str(video_path))
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_s     = min(cap.get(cv2.CAP_PROP_FRAME_COUNT)/src_fps,
                    max_minutes * 60)
    step      = max(1, int(src_fps / sample_fps))

    print(f"  Vidéo : {Path(video_path).name}")
    print(f"  {fw}x{fh}  fps={src_fps:.1f}  analyse={max_s/60:.1f}min  step=1/{step}")

    fieldnames = [
        "t_s", "t_fmt", "frame_idx", "player_idx",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "crop_area",
        # original
        "orig_H", "orig_S", "orig_V", "orig_valid_px",
        # centered_crop
        "cc_H", "cc_S", "cc_V", "cc_valid_px",
        # centered_crop_no_grass
        "cng_H", "cng_S", "cng_V", "cng_valid_px", "cng_valid_ratio",
        # hs_histogram
        "hist_H", "hist_S", "hist_V", "hist_valid_px", "hist_dom_frac",
        # comparaisons
        "hue_diff_orig_cng",    # |orig_H - cng_H|
        "hue_diff_cc_cng",      # |cc_H  - cng_H|
        "hue_diff_cng_hist",    # |cng_H - hist_H|
        # équipes
        "team_orig", "team_cng", "team_hist",
        "agreement_orig_cng",   # team_orig == team_cng
        "agreement_cng_hist",   # team_cng  == team_hist
    ]

    rows_written = 0
    frame_idx = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            ret, img = cap.read()
            if not ret: break
            t_s = frame_idx / src_fps
            if t_s > max_s: break

            if frame_idx % step == 0:
                blobs = detect_blobs(img, fw, fh, h_min=h_min)

                if len(blobs) <= max_blobs_per_frame:
                    for pi, (x1, y1, x2, y2, area) in enumerate(blobs):
                        orig = extract_original(img, x1, y1, x2, y2)
                        cc   = extract_centered_crop(img, x1, y1, x2, y2)
                        cng  = extract_centered_crop_no_grass(img, x1, y1, x2, y2)
                        hist = extract_hs_histogram(img, x1, y1, x2, y2)

                        row = {
                            "t_s": round(t_s, 2),
                            "t_fmt": f"{int(t_s//60)}:{int(t_s%60):02d}",
                            "frame_idx": frame_idx,
                            "player_idx": pi,
                            "bbox_x1": x1, "bbox_y1": y1,
                            "bbox_x2": x2, "bbox_y2": y2,
                            "crop_area": area,
                        }

                        for key, res in [("orig", orig), ("cc", cc),
                                         ("cng", cng), ("hist", hist)]:
                            prefix = {"orig":"orig", "cc":"cc",
                                      "cng":"cng", "hist":"hist"}[key]
                            if res:
                                row[f"{prefix}_H"] = res.get("H")
                                row[f"{prefix}_S"] = res.get("S")
                                row[f"{prefix}_V"] = res.get("V")
                                row[f"{prefix}_valid_px"] = res.get("valid_pixels")
                                if prefix == "cng":
                                    row["cng_valid_ratio"] = res.get("valid_pixels_ratio", "")
                                if prefix == "hist":
                                    row["hist_dom_frac"] = res.get("dominant_fraction", "")
                            else:
                                row[f"{prefix}_H"] = ""
                                row[f"{prefix}_S"] = ""
                                row[f"{prefix}_V"] = ""
                                row[f"{prefix}_valid_px"] = 0
                                if prefix == "cng": row["cng_valid_ratio"] = 0
                                if prefix == "hist": row["hist_dom_frac"] = 0

                        # Comparaisons hue
                        o_H = orig["H"] if orig else None
                        c_H = cng["H"]  if cng  else None
                        h_H = hist["H"] if hist else None
                        cc_H = cc["H"]  if cc   else None

                        row["hue_diff_orig_cng"] = (
                            _hue_diff(o_H, c_H) if o_H is not None and c_H is not None else "")
                        row["hue_diff_cc_cng"] = (
                            _hue_diff(cc_H, c_H) if cc_H is not None and c_H is not None else "")
                        row["hue_diff_cng_hist"] = (
                            _hue_diff(c_H, h_H) if c_H is not None and h_H is not None else "")

                        # Équipes
                        t_orig = _assign_team_rule(o_H,  orig["S"] if orig else None)
                        t_cng  = _assign_team_rule(c_H,  cng["S"]  if cng  else None)
                        t_hist = _assign_team_rule(h_H,  hist["S"] if hist else None)
                        row["team_orig"]  = t_orig if t_orig is not None else ""
                        row["team_cng"]   = t_cng  if t_cng  is not None else ""
                        row["team_hist"]  = t_hist if t_hist is not None else ""
                        row["agreement_orig_cng"]  = int(t_orig == t_cng)  if (t_orig is not None and t_cng  is not None) else ""
                        row["agreement_cng_hist"]  = int(t_cng  == t_hist) if (t_cng  is not None and t_hist is not None) else ""

                        writer.writerow(row)
                        rows_written += 1

            frame_idx += 1

    cap.release()
    print(f"  ✅ {rows_written} joueurs analysés → {output_csv}")
    return rows_written


# ── Rapport de synthèse ──────────────────────────────────────────────────────

def build_report(csv_path, output_txt=None):
    """Lire le CSV et produire un rapport de synthèse."""
    import csv as csv_mod

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("CSV vide.")
        return

    def fval(r, k):
        v = r.get(k, "")
        try: return float(v)
        except: return None

    def ival(r, k):
        v = r.get(k, "")
        try: return int(v)
        except: return None

    n = len(rows)
    lines = []
    lines.append("=" * 60)
    lines.append("PLAYER COLOR PROFILER — Rapport de synthèse")
    lines.append("=" * 60)
    lines.append(f"Joueurs analysés : {n}")

    # Valid pixels ratio
    cng_ratios = [fval(r, "cng_valid_ratio") for r in rows
                  if fval(r, "cng_valid_ratio") is not None]
    if cng_ratios:
        lines.append(f"\nPixels exploitables (cng) :")
        lines.append(f"  moy={sum(cng_ratios)/len(cng_ratios):.2f}  "
                     f"med={sorted(cng_ratios)[len(cng_ratios)//2]:.2f}  "
                     f"<0.1: {sum(1 for v in cng_ratios if v < 0.1)} ({sum(1 for v in cng_ratios if v < 0.1)/n*100:.0f}%)")

    # Hue diffs
    for key, label in [("hue_diff_orig_cng", "orig vs cng"),
                        ("hue_diff_cc_cng",   "cc vs cng"),
                        ("hue_diff_cng_hist", "cng vs hist")]:
        vals = [fval(r, key) for r in rows if fval(r, key) is not None]
        if vals:
            big = sum(1 for v in vals if v > 20)
            lines.append(f"\nDiff teinte {label} (n={len(vals)}) :")
            lines.append(f"  moy={sum(vals)/len(vals):.1f}  "
                         f"med={sorted(vals)[len(vals)//2]:.1f}  "
                         f">20: {big} ({big/len(vals)*100:.0f}%)")

    # Agreement équipes
    for key, label in [("agreement_orig_cng", "orig==cng"),
                        ("agreement_cng_hist", "cng==hist")]:
        vals = [ival(r, key) for r in rows if ival(r, key) is not None]
        if vals:
            agree = sum(vals)
            lines.append(f"\nAccord équipe {label} : "
                         f"{agree}/{len(vals)} ({agree/len(vals)*100:.0f}%)")

    # Distribution teintes cng
    cng_H = [fval(r, "cng_H") for r in rows if fval(r, "cng_H") is not None]
    if cng_H:
        buckets = {"rouge(H<12/>168)": 0, "orange(12-25)": 0,
                   "jaune(25-45)": 0, "vert(45-90)": 0,
                   "cyan(90-95)": 0, "bleu(95-140)": 0,
                   "violet(140-168)": 0}
        for h in cng_H:
            if h < 12 or h > 168: buckets["rouge(H<12/>168)"] += 1
            elif h < 25:  buckets["orange(12-25)"] += 1
            elif h < 45:  buckets["jaune(25-45)"] += 1
            elif h < 90:  buckets["vert(45-90)"] += 1
            elif h < 95:  buckets["cyan(90-95)"] += 1
            elif h < 140: buckets["bleu(95-140)"] += 1
            else:         buckets["violet(140-168)"] += 1
        lines.append(f"\nDistribution teintes cng_H (n={len(cng_H)}) :")
        for k, v in buckets.items():
            bar = "█" * int(v/len(cng_H)*30)
            lines.append(f"  {k:20} {bar} {v} ({v/len(cng_H)*100:.0f}%)")

    report = "\n".join(lines)
    print(report)
    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✅ Rapport → {output_txt}")
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python player_color_profiler.py <video> <output.csv> [max_min]")
        sys.exit(1)
    video  = sys.argv[1]
    out    = sys.argv[2]
    maxmin = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    n = profile_video(video, out, sample_fps=1.0, max_minutes=maxmin)
    if n > 0:
        build_report(out, out.replace(".csv", "_report.txt"))