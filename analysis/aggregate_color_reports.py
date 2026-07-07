"""
aggregate_color_reports.py — Étape "mesure" de la feuille de route proxy couleur.

Ce module ne développe rien de nouveau et ne décide rien : il assemble les CSV
produits par player_color_profiler.py sur plusieurs matchs et affiche des
distributions comparatives (médiane/IQR, pas seulement des moyennes — les
moyennes peuvent masquer des comportements très différents entre matchs).

Usage :
    python aggregate_color_reports.py match1_player_colors.csv match2_player_colors.csv ...

ou en important :
    from analysis.aggregate_color_reports import aggregate
    result = aggregate(["match1_player_colors.csv", "match2_player_colors.csv", ...])

Produit :
    - un tableau imprimé : par méthode, médiane + IQR de valid_pixels_ratio,
      dominant_fraction, dispersion_H (recalculée depuis les H bruts) ;
    - la distribution pooled de agreement_with_original ;
    - un boxplot comparatif par méthode (si matplotlib disponible) ;
    - PAS de recommandation automatique — la lecture reste manuelle, volontairement.

Rappel (cf. player_color_profiler.py) : agreement_with_original mesure un
désaccord avec la méthode historique, pas une justesse. Ne pas l'interpréter
comme "les nouvelles méthodes sont mauvaises" si le score est bas.
"""

import csv
import math
import sys
from collections import defaultdict


METHOD_NAMES = ["original", "centered_crop", "centered_crop_no_grass", "hs_histogram"]


def load_csvs(paths):
    """Charge et concatène plusieurs {match}_player_colors.csv. Tolère les
    fichiers sans colonne match_id (générés avant cet ajout) en la déduisant
    du nom de fichier."""
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("match_id"):
                    row["match_id"] = path.rsplit("/", 1)[-1].replace("_player_colors.csv", "")
                rows.append(row)
    return rows


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _median_iqr(values):
    values = sorted(v for v in values if v is not None)
    n = len(values)
    if n == 0:
        return None, None, None
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    q1 = values[int(n * 0.25)]
    q3 = values[min(int(n * 0.75), n - 1)]
    return round(median, 3), round(q1, 3), round(q3, 3)


def _circular_std_h(h_values, h_max=180.0):
    if len(h_values) < 2:
        return 0.0
    angles = [(h / h_max) * 2 * math.pi for h in h_values]
    sum_cos = sum(math.cos(a) for a in angles)
    sum_sin = sum(math.sin(a) for a in angles)
    n = len(angles)
    r = min(math.sqrt(sum_cos**2 + sum_sin**2) / n, 0.999999)
    return math.sqrt(-2 * math.log(r)) * (h_max / (2 * math.pi))


def aggregate(csv_paths, output_dir=None):
    rows = load_csvs(csv_paths)
    n_matches = len({r["match_id"] for r in rows})
    print(f"[AGGREGATE] {len(rows)} lignes chargées depuis {len(csv_paths)} fichiers "
          f"({n_matches} matchs distincts)")

    per_method = defaultdict(list)
    for r in rows:
        per_method[r["method"]].append(r)

    summary = {}
    for method in METHOD_NAMES:
        m_rows = per_method.get(method, [])
        if not m_rows:
            summary[method] = None
            continue
        ratios = [_to_float(r["valid_pixels_ratio"]) for r in m_rows]
        fracs = [_to_float(r["dominant_fraction"]) for r in m_rows]
        h_vals = [_to_float(r["H"]) for r in m_rows if r.get("H")]

        ratio_med, ratio_q1, ratio_q3 = _median_iqr(ratios)
        frac_med, frac_q1, frac_q3 = _median_iqr(fracs)
        dispersion = round(_circular_std_h([h for h in h_vals if h is not None]), 1)

        summary[method] = {
            "n": len(m_rows),
            "valid_pixels_ratio_median": ratio_med, "valid_pixels_ratio_iqr": (ratio_q1, ratio_q3),
            "dominant_fraction_median": frac_med, "dominant_fraction_iqr": (frac_q1, frac_q3),
            "dispersion_h_pooled": dispersion,
        }

    # distribution pooled de agreement_with_original (dédupliquée par observation :
    # match_id + t_s + cx + cy, comme dans player_color_profiler.py)
    seen = set()
    agreements = []
    for r in rows:
        key = (r["match_id"], r["t_s"], r["cx"], r["cy"])
        if key in seen:
            continue
        seen.add(key)
        a = _to_float(r.get("agreement_with_original"))
        if a is not None:
            agreements.append(int(a))
    agreement_dist = None
    if agreements:
        n = len(agreements)
        agreement_dist = {k: round(100 * agreements.count(k) / n, 1) for k in range(1, 5)}

    _print_summary(summary, agreement_dist, n_matches)

    scatter_path = None
    if output_dir:
        scatter_path = _plot_boxplots(per_method, output_dir)

    return {"summary": summary, "agreement_distribution": agreement_dist,
            "n_matches": n_matches, "n_rows": len(rows), "boxplot_path": scatter_path}


def _print_summary(summary, agreement_dist, n_matches):
    print(f"\n[AGGREGATE] Synthèse sur {n_matches} match(s) — médiane [IQR 25-75%], PAS la moyenne :")
    print(f"  {'méthode':26s} {'n':>6s} {'ratio_gardé':>22s} {'dominant_frac':>22s} {'dispersion_H':>13s}")
    for method in METHOD_NAMES:
        s = summary.get(method)
        if s is None:
            print(f"  {method:26s} --- aucune donnée ---")
            continue
        ratio_str = f"{s['valid_pixels_ratio_median']:.2f} [{s['valid_pixels_ratio_iqr'][0]:.2f}-{s['valid_pixels_ratio_iqr'][1]:.2f}]"
        frac_str = f"{s['dominant_fraction_median']:.2f} [{s['dominant_fraction_iqr'][0]:.2f}-{s['dominant_fraction_iqr'][1]:.2f}]"
        print(f"  {method:26s} {s['n']:>6d} {ratio_str:>22s} {frac_str:>22s} {s['dispersion_h_pooled']:>13.1f}")

    if agreement_dist:
        print(f"\n[AGGREGATE] Distribution pooled de agreement_with_original :")
        for k in [4, 3, 2, 1]:
            print(f"    {k}/4 : {agreement_dist.get(k, 0.0):.1f}%")
        print(f"  Rappel : ceci mesure un désaccord avec 'original', pas une justesse.")

    print(f"\n[AGGREGATE] Aucune décision automatique — lecture manuelle requise avant "
          f"de considérer un changement de méthode en production.")


def _plot_boxplots(per_method, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        data_ratio = [[_to_float(r["valid_pixels_ratio"]) for r in per_method.get(m, [])] for m in METHOD_NAMES]
        data_frac = [[_to_float(r["dominant_fraction"]) for r in per_method.get(m, [])] for m in METHOD_NAMES]
        data_ratio = [[v for v in d if v is not None] for d in data_ratio]
        data_frac = [[v for v in d if v is not None] for d in data_frac]

        axes[0].boxplot(data_ratio, tick_labels=METHOD_NAMES, showfliers=False)
        axes[0].set_title("valid_pixels_ratio par méthode (toutes matchs)")
        axes[0].tick_params(axis='x', rotation=20)

        axes[1].boxplot(data_frac, tick_labels=METHOD_NAMES, showfliers=False)
        axes[1].set_title("dominant_fraction par méthode (toutes matchs)")
        axes[1].tick_params(axis='x', rotation=20)

        plt.tight_layout()
        path = f"{output_dir}/aggregate_color_boxplots.png"
        plt.savefig(path, dpi=120)
        plt.close(fig)
        print(f"\n[AGGREGATE] Boxplots -> {path}")
        return path
    except ImportError:
        print(f"\n[AGGREGATE] matplotlib indisponible — pas de boxplot, tableau texte seul")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggregate_color_reports.py match1_player_colors.csv match2_player_colors.csv ...")
        sys.exit(1)
    aggregate(sys.argv[1:], output_dir=".")
