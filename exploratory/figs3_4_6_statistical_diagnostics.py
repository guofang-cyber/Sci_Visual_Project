"""
Figures 3, 4 and 6 — Non-parametric statistical diagnostics of the 2019
trade-policy archetypes recovered by K-Means (identical partition to Figure 2).

Shared upstream
---------------
The design matrix and cluster partition are produced by the Figure 2 module
(extract_cross_section -> engineer_features -> fit_models), so every inference
below operates on exactly the same 226 economies, 8 engineered features and
3 latent archetypes displayed in the biplot.

Deliverables
------------
  fig6_hierarchical_clustermap.png : Spearman rank-correlation matrix of the
      engineered features, reordered by Ward-linkage hierarchical clustering
      with dual dendrograms, crisp white cell gridlines and a diverging
      coolwarm scale anchored at zero.
  fig3_effect_size_ranking.png    : Kruskal-Wallis global tests across the
      three archetypes for each feature, reported as Epsilon-squared,
      eps^2 = (H - k + 1) / (n - k), ranked in a horizontal bar chart with
      exact value labels at the bar tips and conventional magnitude cutoffs.
  fig4_pairwise_dunn_heatmap.png  : Dunn post-hoc pairwise comparisons
      (mid-rank based, tie-corrected standard error) with Benjamini-Hochberg
      adjustment applied over the full family of 24 hypotheses, rendered as
      -log10(p_adj) on a diverging scale centred at the alpha = 0.05 boundary.

Aesthetics follow S. Few: maximal data-ink, redundant chartjunk removed, a
single grid dimension per panel, exact value labels, restrained colour.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe raster backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
FIG2_PATH = HERE / "fig2_pca_fancy_biplot.py"

FIG6_PNG = HERE / "fig6_hierarchical_clustermap.png"
FIG3_PNG = HERE / "fig3_effect_size_ranking.png"
FIG4_PNG = HERE / "fig4_pairwise_dunn_heatmap.png"

COLOR_TEXT = "#22252A"
COLOR_MUTED = "#6B717A"
COLOR_AXIS = "#6E6E6E"
COLOR_GRID = "#D8D8D8"
COLOR_BAR = "#2b5c8f"
COLOR_ACCENT = "#e7298a"   # leads the ranking: emphasised, not decorative
ALPHA_SIG = 0.05

# Epsilon-squared magnitude cutoffs, conventionally paralleling eta-squared.
MAGNITUDE_CUTOFFS: tuple[tuple[float, str], ...] = ((0.01, "small"), (0.09, "medium"), (0.25, "large"))

# Compact archetype names for the pairwise contrast headers.
SHORT_NAMES: dict[str, str] = {
    "hub": "Hubs",
    "protected": "Protected",
    "peripheral": "Peripheral",
}

FEATURE_LABELS: dict[str, str] = {}  # populated from the Figure 2 module at runtime


def load_fig2_pipeline():
    """Import the Figure 2 module so the modelling pipeline is shared verbatim."""
    spec = importlib.util.spec_from_file_location("fig2_pipeline", FIG2_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fig2_pipeline"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Statistical layer
# --------------------------------------------------------------------------- #

def epsilon_squared(h_stat: float, n: int, k: int) -> float:
    """Epsilon-squared effect size for a Kruskal-Wallis test.

    eps^2 = (H - k + 1) / (n - k), floored at zero. It rescales the rank
    statistic so that it is comparable across designs with different numbers
    of groups and observations, unlike the raw H.
    """
    return max(0.0, (h_stat - k + 1) / (n - k))


def kruskal_wallis_table(feats: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Kruskal-Wallis global test plus effect size for every engineered feature."""
    groups_ids = np.unique(labels)
    k = int(groups_ids.size)
    rows = []
    for col in feats.columns:
        arrays = [feats.loc[labels == cid, col].to_numpy(dtype=float) for cid in groups_ids]
        res = stats.kruskal(*arrays)
        rows.append({
            "feature": col,
            "H": float(res.statistic),
            "p_value": float(res.pvalue),
            "epsilon_sq": epsilon_squared(float(res.statistic), int(len(feats)), k),
        })
    table = pd.DataFrame(rows).sort_values("epsilon_sq", ascending=False).reset_index(drop=True)

    print(f"[kruskal] global tests (Kruskal-Wallis, k = {k}, n = {len(feats)}):")
    view = table.copy()
    view["feature"] = view["feature"].map(lambda c: FEATURE_LABELS.get(c, c))
    print(view.round(4).to_string(index=False))
    return table


def dunn_test(values: np.ndarray, labels: np.ndarray) -> dict[tuple[int, int], float]:
    """Dunn pairwise comparisons for one variable, tie-corrected.

    z_ij = (Rbar_i - Rbar_j) / sqrt( N(N+1)/12 * C * (1/n_i + 1/n_j) ),
    where C = 1 - sum(t^3 - t)/(N^3 - N) is the mid-rank tie correction and
    t the size of each tie block. Two-sided p-values from the normal reference.
    """
    n_total = int(len(values))
    ranks = stats.rankdata(values)
    _, tie_sizes = np.unique(values, return_counts=True)
    tie_term = float((tie_sizes ** 3 - tie_sizes).sum())
    correction = 1.0 - tie_term / (n_total ** 3 - n_total)

    group_ids = sorted(int(g) for g in np.unique(labels))
    mean_rank = {g: float(ranks[labels == g].mean()) for g in group_ids}
    n_group = {g: int((labels == g).sum()) for g in group_ids}

    base_variance = n_total * (n_total + 1) / 12.0 * correction
    out: dict[tuple[int, int], float] = {}
    for a, b in combinations(group_ids, 2):
        se = np.sqrt(base_variance * (1.0 / n_group[a] + 1.0 / n_group[b]))
        z = (mean_rank[a] - mean_rank[b]) / se
        out[(a, b)] = float(2.0 * stats.norm.sf(abs(z)))
    return out


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values, monotone and clipped to [0, 1]."""
    m = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    ranked = np.asarray(p_values, dtype=float)[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Shared chrome
# --------------------------------------------------------------------------- #

def _init_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 11.6,
        "axes.labelsize": 9.6,
    })


def _minimal_axis(ax, title: str, subtitle: str = "") -> None:
    """Remove redundant spines; attach a title block only when one is given."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3.2, color=COLOR_AXIS)
    if title:
        ax.set_title(title, loc="left", fontsize=11.6, fontweight="bold",
                     color=COLOR_TEXT, pad=20)
    if subtitle:
        ax.text(0.0, 1.042, subtitle, transform=ax.transAxes, fontsize=8.2,
                color=COLOR_MUTED, va="bottom", ha="left")


# --------------------------------------------------------------------------- #
# Figure 6 — Spearman clustermap with dual dendrograms
# --------------------------------------------------------------------------- #

def clustermap_ordering(rho: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Ward-linkage ordering of the correlation distance matrix."""
    distance = 1.0 - rho
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0  # enforce exact symmetry for squareform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_matrix = linkage(squareform(distance, checks=False), method="ward")
    leaves = dendrogram(z_matrix, no_labels=True)["leaves"]
    return z_matrix, leaves


def plot_clustermap(feats: pd.DataFrame, output: Path = FIG6_PNG) -> Path:
    """Feature rank-correlation clustermap: dual dendrograms + annotated matrix."""
    rho = feats.corr(method="spearman").to_numpy(dtype=float)
    full_labels = list(feats.columns)
    z_matrix, order = clustermap_ordering(rho)
    rho_ord = rho[np.ix_(order, order)]
    names_ord = [FEATURE_LABELS[full_labels[i]] for i in order]
    p = len(full_labels)

    fig = plt.figure(figsize=(9.6, 8.2), dpi=300)

    # Explicit rectangles: the gap between the dendrogram panels and the matrix
    # is reserved for the tick labels, which gridspec wspace cannot size.
    m_left, m_bottom, m_w, m_h = 0.315, 0.135, 0.435, 0.575
    ax_m = fig.add_axes([m_left, m_bottom, m_w, m_h])
    ax_dc = fig.add_axes([m_left, m_bottom + m_h + 0.045, m_w, 0.125])
    ax_dr = fig.add_axes([0.022, m_bottom, 0.105, m_h])

    # --- Dendrograms: uniform restrained strokes, no chartbook framing ---
    for ax_d, orient in ((ax_dc, "top"), (ax_dr, "left")):
        dendrogram(z_matrix, no_labels=True, orientation=orient, ax=ax_d,
                   color_threshold=-1.0, link_color_func=lambda _: "#3D4551",
                   above_threshold_color="#3D4551")
        ax_d.set_xticks([])
        ax_d.set_yticks([])
        for spine in ax_d.spines.values():
            spine.set_visible(False)
        ax_d.tick_params(length=0)

    # --- Correlation matrix: white cell borders give the crisp gridlines ---
    grid = np.arange(p + 1)
    mesh = ax_m.pcolormesh(grid, grid, rho_ord, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                           edgecolors="white", linewidth=1.2, shading="flat")
    ax_m.set_xlim(0, p)
    ax_m.set_ylim(p, 0)  # row 0 at the top, matching the printed matrix order
    ax_m.set_xticks(grid[:-1] + 0.5)
    ax_m.set_yticks(grid[:-1] + 0.5)
    ax_m.set_xticklabels(names_ord, rotation=52, ha="right", fontsize=7.9)
    ax_m.set_yticklabels(names_ord, fontsize=7.9)
    ax_m.tick_params(length=0, labelcolor=COLOR_TEXT)
    for spine in ax_m.spines.values():
        spine.set_visible(False)

    for i in range(p):
        for j in range(p):
            value = rho_ord[i, j]
            ax_m.text(j + 0.5, i + 0.5, f"{value:.2f}", ha="center", va="center",
                      fontsize=7.0,
                      fontweight="bold" if i == j else "normal",
                      color="white" if abs(value) > 0.70 else COLOR_TEXT)

    cbar = fig.colorbar(mesh, ax=ax_m, fraction=0.046, pad=0.028)
    cbar.set_label("Spearman rank correlation $\\rho$", fontsize=9.2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8.2, length=3, color=COLOR_AXIS)

    # Title block in figure space, clear of the dendrogram strokes.
    fig.text(m_left, 0.965,
             "Figure 6. Hierarchical clustermap of the feature Spearman correlation matrix",
             fontsize=12.6, fontweight="bold", color=COLOR_TEXT, va="top", ha="left")
    fig.text(m_left, 0.928,
             f"2019 cross-sectional engineered features (n = {len(feats)} economies); "
             "rows and columns reordered by Ward-linkage dendrograms on the distance 1 \u2212 $\\rho$",
             fontsize=8.4, color=COLOR_MUTED, va="top", ha="left")

    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[save] clustermap written -> {output} (300 DPI, PNG)")
    return output


# --------------------------------------------------------------------------- #
# Figure 3 — Epsilon-squared effect-size ranking
# --------------------------------------------------------------------------- #

def plot_effect_sizes(kw_table: pd.DataFrame, n_obs: int, k_groups: int,
                      output: Path = FIG3_PNG) -> Path:
    """Ranked horizontal bars of Kruskal-Wallis epsilon-squared values."""
    data = kw_table.reset_index(drop=True)
    y_pos = np.arange(len(data))[::-1]
    values = data["epsilon_sq"].to_numpy(dtype=float)
    names = [FEATURE_LABELS[c] for c in data["feature"]]

    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=300)
    bars = ax.barh(y_pos, values, height=0.60, color=COLOR_BAR, zorder=3)
    bars[0].set_color(COLOR_ACCENT)  # strongest discriminator, hue not size

    for cut, tag in MAGNITUDE_CUTOFFS:
        ax.axvline(cut, color="#AEB4BB", linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
        ax.text(cut, len(data) - 0.34, tag, fontsize=7.2, color=COLOR_MUTED,
                ha="center", va="bottom")

    for yi, v, h, p in zip(y_pos, values, data["H"], data["p_value"]):
        ax.text(v + 0.010, yi, f"{v:.3f}", va="center", ha="left", fontsize=8.6,
                fontweight="bold", color=COLOR_TEXT)
        ax.text(0.004, yi, f"H = {h:,.1f}   $p$ {'< 0.001' if p < 1e-3 else f'= {p:.3f}'}",
                va="center", ha="left", fontsize=7.2, color="#FFFFFF" if v > 0.06 else COLOR_MUTED,
                zorder=4)

    ax.set_yticks(y_pos, names, fontsize=9.2)
    ax.set_xlim(0, float(values.max()) * 1.16)
    ax.set_ylim(-0.7, len(data) - 0.1)
    ax.set_xlabel("Epsilon-squared effect size  $\\varepsilon^2 = (H - k + 1)/(n - k)$")
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    n_sig = int((data["p_value"] < ALPHA_SIG).sum())
    _minimal_axis(ax, "", "")
    fig.text(0.02, 0.995, "Figure 3. Which features drive the archetype separation",
             fontsize=12.4, fontweight="bold", color=COLOR_TEXT, va="top", ha="left")
    fig.text(0.02, 0.952,
             f"Kruskal-Wallis global tests across the three latent clusters "
             f"(k = {k_groups}, n = {n_obs} economies) | "
             f"{n_sig}/{len(data)} tests significant at $\\alpha$ = {ALPHA_SIG}",
             fontsize=8.2, color=COLOR_MUTED, va="top", ha="left")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.905))
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[save] effect-size ranking written -> {output} (300 DPI, PNG)")
    return output


# --------------------------------------------------------------------------- #
# Figure 4 — Dunn post-hoc pairwise heatmap
# --------------------------------------------------------------------------- #

def build_dunn_matrix(feats: pd.DataFrame, labels: np.ndarray,
                      label_map: dict[int, str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Compute BH-adjusted Dunn p-values and the -log10 transform."""
    group_ids = sorted(label_map)
    pairs = list(combinations(group_ids, 2))
    pair_headers = [f"{SHORT_NAMES[label_map[a]]} vs\n{SHORT_NAMES[label_map[b]]}"
                    for a, b in pairs]
    feature_names = [FEATURE_LABELS[c] for c in feats.columns]

    raw_p: dict[tuple[str, tuple[int, int]], float] = {}
    for col in feats.columns:
        for pair, p in dunn_test(feats[col].to_numpy(dtype=float), labels).items():
            raw_p[(col, pair)] = p

    keys = list(raw_p)
    adjusted = benjamini_hochberg(np.array([raw_p[k] for k in keys]))
    p_adj = dict(zip(keys, adjusted))

    minus_log = np.array([[-np.log10(max(p_adj[(c, pr)], 1e-16)) for pr in pairs]
                          for c in feats.columns])
    prob = np.array([[p_adj[(c, pr)] for pr in pairs] for c in feats.columns])
    return minus_log, prob, feature_names, pair_headers


def _text_color_for(value: float, cmap_name: str, norm) -> str:
    """Choose black or white text from the luminance of the cell background."""
    red, green, blue, _ = plt.get_cmap(cmap_name)(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return COLOR_TEXT if luminance > 0.55 else "white"


def plot_dunn_heatmap(feats: pd.DataFrame, labels: np.ndarray,
                      label_map: dict[int, str], output: Path = FIG4_PNG) -> Path:
    """Pairwise archetype contrasts as -log10(BH-adjusted p)."""
    minus_log, prob, feature_names, pair_headers = build_dunn_matrix(feats, labels, label_map)
    n_rows, n_cols = minus_log.shape
    n_family = int(n_rows * n_cols)

    fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=300)

    # Diverging scale centred exactly on the significance boundary.
    log_alpha = -np.log10(ALPHA_SIG)
    vmax = float(np.ceil(minus_log.max()))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=log_alpha, vmax=vmax)
    mesh = ax.pcolormesh(np.arange(n_cols + 1), np.arange(n_rows + 1), minus_log,
                         cmap="RdBu_r", norm=norm, edgecolors="white",
                         linewidth=1.6, shading="flat")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(pair_headers, fontsize=8.8)
    ax.set_yticklabels(feature_names, fontsize=8.8)
    ax.tick_params(length=0, labelcolor=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n_rows):
        for j in range(n_cols):
            value = minus_log[i, j]
            significant = prob[i, j] < ALPHA_SIG
            # Cells at the numerical floor: adjusted p underflowed below 1e-16,
            # so the transform is reported as censored, not as an exact 16.00.
            text = (f"$\\geq${value:.0f}" if value >= 15.99 else f"{value:.2f}")
            ax.text(j + 0.5, i + 0.5, text + ("*" if significant else ""),
                    ha="center", va="center", fontsize=8.4,
                    fontweight="bold" if significant else "normal",
                    color=_text_color_for(value, "RdBu_r", norm))

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.032)
    cbar.set_label("$-\\log_{10}$ of Benjamini-Hochberg adjusted $p$", fontsize=9.2)
    # Reference line at the significance boundary; explained in the subtitle.
    cbar.ax.axhline(log_alpha, color=COLOR_TEXT, linewidth=1.0)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8.2, length=3, color=COLOR_AXIS)

    n_sig = int((prob < ALPHA_SIG).sum())
    _minimal_axis(ax, "", "")
    # Title block in figure space: avoids the pad-vs-subtitle collision seen
    # when both are attached to a short axes.
    fig.text(0.02, 0.995, "Figure 4. Dunn post-hoc contrasts between archetypes",
             fontsize=12.4, fontweight="bold", color=COLOR_TEXT, va="top", ha="left")
    fig.text(0.02, 0.947,
             f"Rank-based pairwise tests with tie-corrected standard errors, BH-adjusted over the "
             f"full family of {n_family} hypotheses | {n_sig}/{n_family} contrasts at $p_{{adj}} < {ALPHA_SIG}$ "
             f"(colourbar tick line) | cells marked $\\geq$ reached the numerical floor $p_{{adj}} < 10^{{-16}}$",
             fontsize=8.2, color=COLOR_MUTED, va="top", ha="left")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.905))
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[dunn] significant BH-adjusted contrasts: {n_sig}/{n_family}")
    print(f"[save] Dunn heatmap written -> {output} (300 DPI, PNG)")
    return output


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> tuple[Path, Path, Path]:
    _init_rc()

    pipeline = load_fig2_pipeline()
    globals()["FEATURE_LABELS"] = pipeline.FEATURE_LABELS

    cross = pipeline.extract_cross_section()
    feats = pipeline.engineer_features(cross)
    model = pipeline.fit_models(feats)

    f6 = plot_clustermap(feats)
    kw_table = kruskal_wallis_table(feats, model.labels)
    k_groups = int(len(np.unique(model.labels)))
    f3 = plot_effect_sizes(kw_table, int(len(feats)), k_groups)
    f4 = plot_dunn_heatmap(feats, model.labels, model.label_map)
    return f6, f3, f4


if __name__ == "__main__":
    main()
