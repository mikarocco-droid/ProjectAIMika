"""
player_color_profiler.py — Étape 2 de la feuille de route proxy couleur.

Compare 4 méthodes d'extraction de couleur en A/B (au lieu de remplacer
silencieusement l'ancienne par la nouvelle) :

    original                 baseline d'avant cette session — TOUJOURS la méthode
                              utilisée par le pipeline de production (voir
                              _extract_dominant_color() dans notebook_kickoff_only.py)
    centered_crop             + recentrage horizontal seul
    centered_crop_no_grass    + torse resserré + suppression ombre
    hs_histogram               + mode d'histogramme HS au lieu de la médiane

Ce module est un LABORATOIRE : il calcule et compare les 4 méthodes, mais ne
change rien à ce que le pipeline utilise réellement. Tant qu'on n'a pas de
résultats sur plusieurs matchs réels confirmant qu'une variante fait mieux,
la production reste sur 'original'.

Pour chaque méthode, mesure :
    - valid_pixels_ratio moyen (le crop/filtre rejette-t-il trop de pixels ?)
    - dominant_fraction moyen (confiance : une couleur nette à 81% n'est pas la
      même chose qu'un crop ambigu à 41%/38%, même avec un bon valid_pixels_ratio)
    - dispersion (écart-type circulaire de la teinte H)
    - temporal_stability_estimated_h (voir avertissement ci-dessous)

IMPORTANT — deux limites assumées explicitement :

1. `tracker_id` dans _extract_player_proxies() n'est PAS une identité persistante
   (c'est juste l'index du joueur dans la frame courante, remis à 0 à chaque
   frame). Pour approcher la stabilité temporelle, ce module reconstruit des
   "tracklets" par appariement au plus proche voisin en position entre frames
   consécutives. C'est une MÉTRIQUE DIAGNOSTIQUE, pas un vrai tracker — d'où le
   suffixe `_estimated` sur tous les champs qui en dépendent. Ne jamais utiliser
   ces valeurs pour décider d'une identité de joueur ou d'une équipe.

2. Le synthétique valide seulement que le code tourne et que les métriques sont
   cohérentes — il ne valide en rien la qualité de l'algorithme sur de vraies
   images de match.

Usage (dans notebook_kickoff_only.py, après extraction des frames) :
    from analysis.player_color_profiler import build_color_report
    build_color_report(frames_data, output_dir=OUTPUT_DIR, match_id=match_id)

Produit :
    - {match_id}_player_colors.csv     (une ligne par observation x méthode)
    - {match_id}_color_scatter.png     (grille 2x2, une case par méthode)
    - Un tableau comparatif imprimé dans les logs.
"""

import csv
import math


METHOD_NAMES = ["original", "centered_crop", "centered_crop_no_grass", "hs_histogram"]

# Distance normalisée max entre deux frames consécutives pour considérer que
# c'est le même joueur (pseudo-tracking, cf. avertissement en tête de fichier)
_TRACKLET_MAX_DIST = 0.08


def _color_bucket(h, s, v):
    """
    Réduit une couleur HSV à une étiquette grossière (8 teintes + neutres) pour
    pouvoir comparer si deux méthodes 'sont d'accord' sur la couleur perçue,
    sans exiger une égalité exacte de H/S/V qui n'arrivera presque jamais.
    Purement diagnostique — ne sert à aucune décision de team assignment.
    """
    if h is None:
        return None
    if s < 25:
        if v < 60:   return "noir"
        if v > 180:  return "blanc"
        return "gris"
    # 8 secteurs de teinte de ~22.5 (échelle OpenCV 0-179)
    sector = int(((h + 11) % 180) // 22.5) % 8
    return ["rouge", "orange", "jaune", "vert", "cyan", "bleu", "violet", "magenta"][sector]


def collect_color_samples(frames_data, max_samples_per_method=300, min_valid_pixels=8):
    """
    Retourne une liste d'observations à plat :
    {t_s, cx, cy, method, H, S, V, pixels_total, pixels_kept, valid_pixels_ratio,
     dominant_fraction, second_fraction, agreement_with_original}

    `agreement_with_original` est calculé UNE FOIS par observation (joueur x frame),
    en comparant le bucket de couleur des 4 méthodes entre elles — donc identique
    sur les 4 lignes (une par méthode) issues de la même observation. Valeur de
    1 à 4 (voir _color_bucket). None si 'original' elle-même a échoué (H=None) —
    pas de référence à comparer.

    ATTENTION à l'interprétation : c'est un indicateur de DIVERGENCE par rapport
    à la méthode historique, pas de JUSTESSE. 'original' n'est pas une vérité
    terrain — si les 3 autres méthodes s'accordent entre elles contre 'original',
    ça peut vouloir dire que ce sont elles qui ont raison. Un score bas dit
    "il y a désaccord ici, à regarder", pas "les nouvelles méthodes se trompent".

    Sous-échantillonne UNIFORMÉMENT DANS LE TEMPS par méthode (pas les N premiers)
    pour rester représentatif sur un match entier.
    """
    per_method = {m: [] for m in METHOD_NAMES}

    for fd in frames_data:
        t_s = fd.get("t_s", fd.get("frame", 0))
        for p in fd.get("players", []):
            variants = p.get("dominant_color_variants")
            cx, cy = p.get("center", (None, None))
            if not variants:
                continue

            # Bucket par méthode, calculé une fois pour toutes les 4 lignes de
            # cette observation (agreement_with_original est une propriété de
            # l'observation, pas de la méthode individuelle)
            buckets = {}
            for method in METHOD_NAMES:
                v = variants.get(method)
                if v and v.get("H") is not None:
                    buckets[method] = _color_bucket(v["H"], v["S"], v["V"])
                else:
                    buckets[method] = None

            original_bucket = buckets.get("original")
            if original_bucket is None:
                agreement = None
            else:
                agreement = sum(1 for m in METHOD_NAMES if buckets.get(m) == original_bucket)

            for method in METHOD_NAMES:
                v = variants.get(method)
                if not v or v.get("H") is None:
                    continue
                if v.get("pixels_kept", 0) < min_valid_pixels:
                    continue
                per_method[method].append({
                    "t_s": t_s, "cx": cx, "cy": cy, "method": method,
                    "H": v["H"], "S": v["S"], "V": v["V"],
                    "pixels_total": v["pixels_total"],
                    "pixels_kept": v["pixels_kept"],
                    "valid_pixels_ratio": v["valid_pixels_ratio"],
                    "dominant_fraction": v.get("dominant_fraction"),
                    "second_fraction": v.get("second_fraction"),
                    "agreement_with_original": agreement,
                })

    all_samples = []
    for method, samples in per_method.items():
        if len(samples) <= max_samples_per_method:
            all_samples.extend(samples)
        else:
            step = len(samples) / max_samples_per_method
            all_samples.extend(samples[int(i * step)] for i in range(max_samples_per_method))
    return all_samples


def _circular_std_h(h_values, h_max=180.0):
    """Écart-type circulaire de la teinte (H est un angle : 179 et 0 sont voisins,
    pas aux deux extrémités d'un axe linéaire — un std() classique surestimerait
    la dispersion des teintes rouges, qui enjambent la coupure 0/179)."""
    if len(h_values) < 2:
        return 0.0
    angles = [(h / h_max) * 2 * math.pi for h in h_values]
    sum_cos = sum(math.cos(a) for a in angles)
    sum_sin = sum(math.sin(a) for a in angles)
    n = len(angles)
    r = math.sqrt(sum_cos**2 + sum_sin**2) / n
    r = min(r, 0.999999)
    return math.sqrt(-2 * math.log(r)) * (h_max / (2 * math.pi))


def _build_tracklets(frames_data, method, max_dist=_TRACKLET_MAX_DIST):
    """
    Pseudo-tracking par plus proche voisin en position entre frames consécutives
    (voir avertissement en tête de fichier — tracker_id n'est pas persistant).
    Retourne une liste de tracklets, chacun étant une liste de valeurs H dans
    l'ordre temporel pour un même joueur approximatif.
    """
    frames_sorted = sorted(frames_data, key=lambda fd: fd.get("t_s", 0))
    active_tracklets = []
    finished_tracklets = []

    for fd in frames_sorted:
        detections = []
        for p in fd.get("players", []):
            variants = p.get("dominant_color_variants")
            if not variants:
                continue
            v = variants.get(method)
            if not v or v.get("H") is None:
                continue
            cx, cy = p.get("center", (None, None))
            if cx is None:
                continue
            detections.append((cx, cy, v["H"]))

        used = set()
        new_active = []
        for tl in active_tracklets:
            lx, ly = tl["last_pos"]
            best_i, best_d = None, max_dist
            for i, (cx, cy, h) in enumerate(detections):
                if i in used:
                    continue
                d = math.hypot(cx - lx, cy - ly)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                cx, cy, h = detections[best_i]
                tl["h_values"].append(h)
                tl["last_pos"] = (cx, cy)
                used.add(best_i)
                new_active.append(tl)
            else:
                finished_tracklets.append(tl)
        for i, (cx, cy, h) in enumerate(detections):
            if i not in used:
                new_active.append({"last_pos": (cx, cy), "h_values": [h]})
        active_tracklets = new_active

    finished_tracklets.extend(active_tracklets)
    return [tl["h_values"] for tl in finished_tracklets if len(tl["h_values"]) >= 3]


def compute_agreement_distribution(samples):
    """
    agreement_with_original est une propriété PAR OBSERVATION (joueur x frame),
    pas par méthode — une seule valeur par observation, dupliquée sur ses 4 lignes
    dans le CSV. On calcule donc sa distribution sur les observations uniques
    (dédupliquées via t_s+cx+cy), pas sur toutes les lignes du CSV.
    """
    seen = set()
    values = []
    for s in samples:
        key = (s["t_s"], s["cx"], s["cy"])
        if key in seen:
            continue
        seen.add(key)
        if s.get("agreement_with_original") is not None:
            values.append(s["agreement_with_original"])

    if not values:
        return None
    dist = {k: values.count(k) for k in range(1, 5)}
    n = len(values)
    return {"n_observations": n,
            "distribution_pct": {k: round(100 * v / n, 1) for k, v in dist.items()}}


def compare_methods(frames_data, samples):
    """Tableau comparatif par méthode : ratio de pixels gardés, dispersion H,
    confiance (dominant_fraction), stabilité temporelle ESTIMÉE (voir avertissement
    en tête de fichier — approximation par pseudo-tracking, pas un vrai tracker)."""
    report = {}
    for method in METHOD_NAMES:
        m_samples = [s for s in samples if s["method"] == method]
        if not m_samples:
            report[method] = None
            continue

        mean_ratio = sum(s["valid_pixels_ratio"] for s in m_samples) / len(m_samples)
        frac_samples = [s["dominant_fraction"] for s in m_samples if s.get("dominant_fraction") is not None]
        mean_dominant_fraction = sum(frac_samples) / len(frac_samples) if frac_samples else None
        h_values = [s["H"] for s in m_samples]
        dispersion = _circular_std_h(h_values)

        tracklets = _build_tracklets(frames_data, method)
        if tracklets:
            stabilities = [_circular_std_h(tl) for tl in tracklets]
            mean_stability = sum(stabilities) / len(stabilities)
        else:
            mean_stability = None

        report[method] = {
            "n_samples": len(m_samples),
            "mean_valid_pixels_ratio": round(mean_ratio, 3),
            "mean_dominant_fraction": round(mean_dominant_fraction, 3) if mean_dominant_fraction is not None else None,
            "dispersion_h": round(dispersion, 1),
            # ESTIMÉE : pseudo-tracking par plus proche voisin, pas un vrai tracker.
            # Ne jamais utiliser cette valeur pour décider d'une identité de joueur.
            "temporal_stability_estimated_h": round(mean_stability, 1) if mean_stability is not None else None,
            "n_tracklets_estimated": len(tracklets),
        }
    return report


def _hsv_to_rgb_hex(h, s, v):
    h_norm = (h / 179.0) * 360.0
    s_norm = s / 255.0
    v_norm = v / 255.0
    c = v_norm * s_norm
    x = c * (1 - abs((h_norm / 60.0) % 2 - 1))
    m = v_norm - c
    if h_norm < 60:    r, g, b = c, x, 0
    elif h_norm < 120: r, g, b = x, c, 0
    elif h_norm < 180: r, g, b = 0, c, x
    elif h_norm < 240: r, g, b = 0, x, c
    elif h_norm < 300: r, g, b = x, 0, c
    else:              r, g, b = c, 0, x
    r, g, b = int((r+m)*255), int((g+m)*255), int((b+m)*255)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_color_csv(samples, output_path):
    fieldnames = ["match_id", "t_s", "cx", "cy", "method", "H", "S", "V",
                  "pixels_total", "pixels_kept", "valid_pixels_ratio",
                  "dominant_fraction", "second_fraction", "agreement_with_original"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({k: s.get(k) for k in fieldnames})
    return output_path


def _print_comparison_table(report):
    print(f"  [COLOR PROFILER] Comparaison des 4 méthodes :")
    print(f"  {'méthode':26s} {'n':>5s} {'ratio_gardé':>12s} {'dominant_frac':>14s} {'dispersion_H':>13s} {'stab_H_estimée':>15s} {'n_tracklets':>12s}")
    for method in METHOD_NAMES:
        r = report.get(method)
        if r is None:
            print(f"  {method:26s} {'--- aucune donnée exploitable ---':>70s}")
            continue
        stab = f"{r['temporal_stability_estimated_h']:.1f}" if r['temporal_stability_estimated_h'] is not None else "n/a"
        dom_frac = f"{r['mean_dominant_fraction']*100:.0f}%" if r['mean_dominant_fraction'] is not None else "n/a"
        print(f"  {method:26s} {r['n_samples']:>5d} {r['mean_valid_pixels_ratio']*100:>11.1f}% "
              f"{dom_frac:>14s} {r['dispersion_h']:>13.1f} {stab:>15s} {r['n_tracklets_estimated']:>12d}")
    print(f"  Lecture : ratio_gardé haut = crop propre. dominant_frac haut = couleur "
          f"nette et fiable (ex: 81% dans le bin dominant = confiance forte ; 41% = "
          f"crop ambigu même si le ratio_gardé est bon). dispersion_H bas = couleurs "
          f"concentrées. stab_H_estimée bas = couleur cohérente d'une frame à l'autre "
          f"— ESTIMÉE via pseudo-tracking (plus proche voisin), PAS un vrai tracker, "
          f"à ne jamais utiliser pour décider d'une identité de joueur.")


def build_color_report(frames_data, output_dir, match_id, max_samples_per_method=300):
    """Point d'entrée principal : CSV comparatif + scatter A/B + tableau texte."""
    samples = collect_color_samples(frames_data, max_samples_per_method=max_samples_per_method)
    for s in samples:
        s["match_id"] = match_id

    csv_path = f"{output_dir}/{match_id}_player_colors.csv"
    write_color_csv(samples, csv_path)
    print(f"  [COLOR PROFILER] {len(samples)} échantillons (toutes méthodes) -> {csv_path}")

    if not samples:
        print(f"  [COLOR PROFILER] Aucun échantillon exploitable — "
              f"vérifier que dominant_color_variants est bien peuplé dans frames_data")
        return {"csv_path": csv_path, "scatter_path": None, "n_samples": 0, "report": {}}

    report = compare_methods(frames_data, samples)
    _print_comparison_table(report)

    agreement_dist = compute_agreement_distribution(samples)
    if agreement_dist:
        print(f"  [COLOR PROFILER] Distribution de agreement_with_original "
              f"(n={agreement_dist['n_observations']} observations) :")
        for k in [4, 3, 2, 1]:
            pct = agreement_dist["distribution_pct"].get(k, 0.0)
            print(f"    {k}/4 méthodes d'accord : {pct:.1f}%")
        print(f"  Lecture : agreement_with_original mesure un DÉSACCORD avec la "
              f"méthode historique, PAS une justesse — 'original' n'est pas une "
              f"vérité terrain. Un fort % à 1-2/4 signale des joueurs ambigus à "
              f"regarder (maillots sombres/verts, ombres, occlusions), mais si les "
              f"3 autres méthodes s'accordent entre elles CONTRE 'original', c'est "
              f"peut-être 'original' qui se trompe, pas les 3 autres. Ne pas lire "
              f"un score bas comme \"les nouvelles méthodes sont mauvaises\".")

    scatter_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes_flat = axes.flatten()

        for ax, method in zip(axes_flat, METHOD_NAMES):
            m_samples = [s for s in samples if s["method"] == method]
            if not m_samples:
                ax.set_title(f"{method} — aucune donnée")
                continue
            colors = [_hsv_to_rgb_hex(s["H"], s["S"], s["V"]) for s in m_samples]
            h_vals = [s["H"] for s in m_samples]
            s_vals = [s["S"] for s in m_samples]
            ax.scatter(h_vals, s_vals, c=colors, s=30, edgecolors="black", linewidths=0.3)
            r = report.get(method)
            if r and r['temporal_stability_estimated_h'] is not None:
                subtitle = (f"ratio={r['mean_valid_pixels_ratio']*100:.0f}% "
                            f"dom_frac={r['mean_dominant_fraction']*100:.0f}% "
                            f"disp={r['dispersion_h']:.0f} "
                            f"stab~={r['temporal_stability_estimated_h']:.0f}")
            elif r:
                subtitle = f"ratio={r['mean_valid_pixels_ratio']*100:.0f}%"
            else:
                subtitle = ""
            ax.set_title(f"{method}\n{subtitle}")
            ax.set_xlabel("Teinte (H)")
            ax.set_ylabel("Saturation (S)")
            ax.set_xlim(0, 179)
            ax.set_ylim(0, 255)

        plt.suptitle(f"{match_id} — comparaison A/B des 4 méthodes d'extraction couleur")
        plt.tight_layout()
        scatter_path = f"{output_dir}/{match_id}_color_scatter.png"
        plt.savefig(scatter_path, dpi=120)
        plt.close(fig)
        print(f"  [COLOR PROFILER] Scatter comparatif -> {scatter_path}")

    except ImportError:
        print(f"  [COLOR PROFILER] matplotlib indisponible — CSV + tableau seuls produits")

    return {"csv_path": csv_path, "scatter_path": scatter_path,
            "n_samples": len(samples), "report": report,
            "agreement_distribution": agreement_dist}
