"""
Figures 5 and 7 — Supervised recovery of the latent archetypes and the
distributional signature of their most discriminative features.

Design
------
  * Targets: K-Means archetype labels recovered in Figure 2 (identical
    partition; the modelling pipeline is imported, not duplicated).
  * Split: 70 / 30 train-test, stratified on the class distribution.
  * Model: RandomForestClassifier(n_estimators = 100, max_depth = 5).
    The standardiser is fitted on the training half only, so the reported
    test accuracy is genuinely out-of-sample.
  * Figure 5: row-normalised confusion matrix (left) and horizontal Gini
    feature importance, mean decrease in impurity (right).
  * Figure 7: 2 x 2 raincloud matrices for the four most important features —
    half-violin kernel-density contour, compact white boxplot (median + IQR),
    and jittered strip of individual observations.

Rationale
---------
If a supervised learner can recover the unsupervised partition from the
engineered features alone at accuracy far above the majority-class baseline,
the archetypes are a structured property of the feature space rather than an
artefact of the initialisation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe raster backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
FIG2_PATH = HERE / "fig2_pca_fancy_biplot.py"

FIG5_PNG = HERE / "fig5_supervised_evaluation.png"
FIG7_PNG = HERE / "fig7_raincloud_distributions.png"

TEST_SIZE = 0.30
RANDOM_STATE = 20261103
N_TREES = 100
MAX_DEPTH = 5

COLOR_TEXT = "#22252A"
COLOR_MUTED = "#6B717A"
COLOR_AXIS = "#6E6E6E"
COLOR_GRID = "#D8D8D8"

CLUSTER_COLORS: dict[str, str] = {
    "hub": "#2b5c8f",
    "protected": "#e7298a",
    "peripheral": "#238b45",
}
CLUSTER_SHORT: dict[str, str] = {"hub": "Hubs", "protected": "Protected", "peripheral": "Peripheral"}

FEATURE_LABELS: dict[str, str] = {}  # populated from the Figure 2 module at runtime


def load_fig2_pipeline():
    """Import the Figure 2 module so data, features and labels stay identical."""
    spec = importlib.util.spec_from_file_location("fig2_pipeline_rf", FIG2_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fig2_pipeline_rf"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Supervised evaluation layer
# --------------------------------------------------------------------------- #

def train_random_forest(feats: pd.DataFrame, labels: np.ndarray,
                       label_map: dict[int, str]) -> dict:
    """Stratified 70/30 split, leakage-free scaling, RF fit and evaluation."""
    x_train, x_test, y_train, y_test = train_test_split(
        feats.to_numpy(dtype=float), labels,
        test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE,
    )

    scaler = StandardScaler().fit(x_train)  # fitted on training data only
    train_scaled, test_scaled = scaler.transform(x_train), scaler.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(train_scaled, y_train)

    predictions = model.predict(test_scaled)
    accuracy = float(accuracy_score(y_test, predictions))
    majority_baseline = float(max(np.bincount(y_test, minlength=3)) / len(y_test))

    print(f"[rf] split: train n = {len(y_train)}, test n = {len(y_test)} "
          f"(stratified {int((1 - TEST_SIZE) * 100)} / {int(TEST_SIZE * 100)})")
    print(f"[rf] out-of-sample test accuracy = {accuracy * 100:.1f}% | "
          f"majority-class baseline on test set = {majority_baseline * 100:.1f}%")
    print("[rf] classification report (test set):")
    print(class_report(y_test, predictions, label_map))

    return {
        "model": model,
        "accuracy": accuracy,
        "majority_baseline": majority_baseline,
        "y_test": y_test,
        "predictions": predictions,
        "importances": pd.Series(model.feature_importances_, index=feats.columns),
        "n_test": int(len(y_test)),
    }


def class_report(y_test: np.ndarray, predictions: np.ndarray,
                 label_map: dict[int, str]) -> str:
    """Text classification report whose class names match the ascending ids."""
    ids = sorted(int(v) for v in np.unique(y_test))
    names = [CLUSTER_SHORT[label_map[cid]] for cid in ids]
    return classification_report(y_test, predictions, target_names=names, digits=3)


def normalized_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                         label_map: dict[int, str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Row-normalised confusion matrix (true-class conditional shares).

    Rows and columns follow ascending cluster id; the class names are derived
    from the same ordering so that labels cannot drift out of alignment.
    """
    ids = sorted(int(v) for v in np.unique(y_true))
    counts = confusion_matrix(y_true, y_pred, labels=ids)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = counts / counts.sum(axis=1, keepdims=True)
    names = [CLUSTER_SHORT[label_map[int(cid)]] for cid in ids]
    return counts, np.nan_to_num(shares), names


# --------------------------------------------------------------------------- #
# Figure 5 — dual-panel supervised evaluation
# --------------------------------------------------------------------------- #

def _init_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _minimal_axis(ax, title: str, subtitle: str = "") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3.2, color=COLOR_AXIS)
    if title:
        ax.set_title(title, loc="left", fontsize=10.6, fontweight="bold",
                     color=COLOR_TEXT, pad=13)
    if subtitle:
        ax.text(0.0, 1.042, subtitle, transform=ax.transAxes, fontsize=7.9,
                color=COLOR_MUTED, va="bottom", ha="left")


def _text_color_for(value: float, cmap_name: str) -> str:
    """Pick black or white labels from the luminance of a [0, 1] colormap cell."""
    red, green, blue, _ = plt.get_cmap(cmap_name)(value)
    return COLOR_TEXT if 0.299 * red + 0.587 * green + 0.114 * blue > 0.55 else "white"


def draw_confusion_panel(ax, counts: np.ndarray, shares: np.ndarray,
                         class_names: list[str]) -> None:
    """Left panel: row-normalised confusion matrix with exact shares."""
    p = len(class_names)
    grid = np.arange(p + 1)
    ax.pcolormesh(grid, grid, shares, cmap="Blues", vmin=0.0, vmax=1.0,
                  edgecolors="white", linewidth=1.8, shading="flat")
    ax.set_xlim(0, p)
    ax.set_ylim(p, 0)
    ax.set_xticks(grid[:-1] + 0.5, class_names, fontsize=9)
    ax.set_yticks(grid[:-1] + 0.5, class_names, fontsize=9)
    ax.tick_params(length=0, labelcolor=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Predicted archetype", labelpad=6)
    ax.set_ylabel("True archetype", labelpad=6)

    for i in range(p):
        for j in range(p):
            share, count = shares[i, j], counts[i, j]
            ax.text(j + 0.5, i + 0.5 - 0.09, f"{share * 100:.1f}%",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=_text_color_for(share, "Blues"))
            ax.text(j + 0.5, i + 0.5 + 0.14, f"n = {count}",
                    ha="center", va="center", fontsize=7.2,
                    color=_text_color_for(share, "Blues"))


def draw_importance_panel(ax, importances: pd.Series) -> list[str]:
    """Right panel: horizontal Gini importance with value labels at bar tips."""
    ranked = importances.sort_values(ascending=True)
    y_pos = np.arange(len(ranked))
    names = [FEATURE_LABELS[c] for c in ranked.index]
    values = ranked.to_numpy(dtype=float)

    bars = ax.barh(y_pos, values, height=0.62, color="#5B7B95", zorder=3)
    bars[-1].set_color("#2b5c8f")     # strongest driver
    bars[-2].set_color("#e7298a")     # second driver, echoed in Fig. 7

    for yi, v in zip(y_pos, values):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=8.2, fontweight="bold", color=COLOR_TEXT)

    ax.set_yticks(y_pos, names, fontsize=8.6)
    ax.set_xlim(0, float(values.max()) * 1.14)
    ax.set_xlabel("Gini importance (mean decrease in impurity, normalised)", labelpad=6)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return list(ranked.index[::-1])  # descending order for Figure 7


def compose_fig5(result: dict, label_map: dict[int, str],
                 output: Path = FIG5_PNG) -> Path:
    """Assemble and persist the dual-panel supervised evaluation figure."""
    _init_rc()
    counts, shares, class_names = normalized_confusion(
        result["y_test"], result["predictions"], label_map)

    fig, (ax_c, ax_i) = plt.subplots(1, 2, figsize=(12.6, 5.6), dpi=300,
                                     gridspec_kw={"width_ratios": [1.0, 1.25],
                                                  "wspace": 0.38})

    draw_confusion_panel(ax_c, counts, shares, class_names)
    _minimal_axis(
        ax_c,
        f"Out-of-sample test accuracy = {result['accuracy'] * 100:.1f}%",
    )

    ordered_features = draw_importance_panel(ax_i, result["importances"])
    _minimal_axis(
        ax_i,
        f"Random Forest drivers (n_estimators = {N_TREES}, max_depth = {MAX_DEPTH})",
    )

    fig.text(0.015, 0.995,
             "Figure 5. Supervised recovery of the unsupervised archetypes",
             fontsize=12.6, fontweight="bold", color=COLOR_TEXT, va="top", ha="left")
    fig.text(0.015, 0.958,
             f"Row-normalised confusion matrix on {result['n_test']} held-out economies "
             f"(majority-class baseline = {result['majority_baseline'] * 100:.1f}%); "
             "features standardised on the training split only, so the estimate is leakage-free",
             fontsize=8.2, color=COLOR_MUTED, va="top", ha="left")

    # Explicit geometry: the reserved band between 0.86 and 0.94 belongs to the
    # panel titles, so no text can collide regardless of label lengths.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.115, wspace=0.42)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[save] supervised evaluation written -> {output} (300 DPI, PNG)")
    return output, ordered_features


# --------------------------------------------------------------------------- #
# Figure 7 — raincloud distribution matrix
# --------------------------------------------------------------------------- #

def draw_raincloud(ax, values: np.ndarray, labels: np.ndarray, group_ids: list[int],
                   colors: list[str], tick_names: list[str],
                   jitter_width: float = 0.075, violin_width: float = 0.46) -> None:
    """One panel: half-violin density, compact white boxplot, jittered strip."""
    rng = np.random.default_rng(RANDOM_STATE)

    lo = min(float(values[labels == g].min()) for g in group_ids)
    hi = max(float(values[labels == g].max()) for g in group_ids)
    span = hi - lo or 1.0
    y_lo, y_hi = lo - 0.05 * span, hi + 0.09 * span

    for slot, g in enumerate(group_ids):
        v = values[labels == g]
        center = float(slot)
        color = colors[slot]

        # --- Layer 1: half-violin (KDE contour), extending rightward ---
        # Support is clipped to the observed range so the density does not
        # flare into a vertical wall beyond the data boundary.
        kde = stats.gaussian_kde(v, bw_method=0.32)
        grid_y = np.linspace(float(v.min()), float(v.max()), 220)
        dens = kde(grid_y)
        dens_scaled = dens / (dens.max() or 1.0) * violin_width
        base = center - 0.42
        ax.fill_betweenx(grid_y, base, base + dens_scaled, color=color,
                         alpha=0.30, zorder=3)
        ax.plot(base + dens_scaled, grid_y, color=color, linewidth=0.9, zorder=4)

        # --- Layer 2: compact white boxplot (median and IQR) ---
        box = ax.boxplot([v], positions=[center], widths=0.13, patch_artist=True,
                         showfliers=False, zorder=5,
                         boxprops=dict(facecolor="white", edgecolor=COLOR_AXIS, linewidth=0.9),
                         whiskerprops=dict(color=COLOR_AXIS, linewidth=0.9),
                         capprops=dict(color=COLOR_AXIS, linewidth=0.9),
                         medianprops=dict(color=color, linewidth=2.0))

        # --- Layer 3: jittered strip of individual observations ---
        jitter = center + 0.24 + rng.uniform(-jitter_width, jitter_width, size=v.shape)
        ax.scatter(jitter, v, s=9, color=color, alpha=0.55,
                   edgecolors="white", linewidths=0.3, zorder=6)

    ax.set_xlim(-0.62, len(group_ids) - 1 + 0.68)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(np.arange(len(group_ids)), tick_names, fontsize=8.8)
    ax.tick_params(axis="y", labelsize=8.0, length=3.0, color=COLOR_AXIS)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
        ax.spines[side].set_linewidth(0.8)


def compose_fig7(feats: pd.DataFrame, labels: np.ndarray, label_map: dict[int, str],
                 top4: list[str], output: Path = FIG7_PNG) -> Path:
    """Assemble the 2 x 2 raincloud matrix for the four leading features."""
    _init_rc()
    ids = sorted(label_map)
    colors = [CLUSTER_COLORS[label_map[cid]] for cid in ids]
    n_by_cluster = pd.Series(labels).value_counts().sort_index()
    ticks = [f"{CLUSTER_SHORT[label_map[cid]]}\n(n = {int(n_by_cluster[cid])})" for cid in ids]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), dpi=300)
    for ax, feature in zip(axes.ravel(), top4):
        draw_raincloud(ax, feats[feature].to_numpy(dtype=float), labels,
                       ids, colors, ticks)
        ax.set_ylabel(FEATURE_LABELS[feature], fontsize=9.4, labelpad=6)
        # Rank badge: Gini importance order carried over from Figure 5.
        ax.text(1.0, 1.055, f"importance rank #{top4.index(feature) + 1}",
                transform=ax.transAxes, fontsize=7.8, color=COLOR_MUTED,
                va="bottom", ha="right")

    fig.text(0.015, 0.985,
             "Figure 7. Distributional signature of the four most discriminative features",
             fontsize=12.6, fontweight="bold", color=COLOR_TEXT, va="top", ha="left")
    fig.text(0.015, 0.955,
             "Raincloud layout per archetype: half-violin kernel density estimate (left), "
             "compact white boxplot with median and IQR (centre), jittered strip of individual economies (right)",
             fontsize=8.2, color=COLOR_MUTED, va="top", ha="left")

    # Class identity is already encoded by the x-axis tick labels and the
    # layer colours, so a separate legend would be redundant ink.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.938))
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[save] raincloud matrix written -> {output} (300 DPI, PNG)")
    return output


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> tuple[Path, Path]:
    pipeline = load_fig2_pipeline()
    globals()["FEATURE_LABELS"] = pipeline.FEATURE_LABELS

    cross = pipeline.extract_cross_section()
    feats = pipeline.engineer_features(cross)
    model = pipeline.fit_models(feats)

    result = train_random_forest(feats, model.labels, model.label_map)
    _, ordered_features = compose_fig5(result, model.label_map)

    top4 = ordered_features[:4]
    print(f"[raincloud] leading features by Gini importance: "
          f"{', '.join(FEATURE_LABELS[f] for f in top4)}")
    f7 = compose_fig7(feats, model.labels, model.label_map, top4)
    return FIG5_PNG, f7


if __name__ == "__main__":
    main()
