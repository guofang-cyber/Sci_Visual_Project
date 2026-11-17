"""
Figure 2 — Latent archetype structure of world trade and tariff policy.

Pipeline
--------
  1. Cross-sectional extraction: baseline year 2019, macro-regional aggregates
     and uninhabited territorial claims removed.
  2. Log-transformation of scale variables and econometric feature engineering
     (8 analysis variables).
  3. Standardisation with StandardScaler (mean 0, unit variance).
  4. K-Means partitioning (k = 3) to recover latent archetypes.
  5. Two-component PCA, used for visual embedding only — the partition is
     estimated in the standardised 8-D space, never on the 2-D projection.
  6. Two-panel publication figure at 350 DPI:
       Panel A: PCA biplot with eigenvector loading arrows and feature labels.
       Panel B: convex-hull topological envelopes, jittered scatter with micro
                white borders, and collision-avoiding pointer-line annotations
                for representative anchor economies.

Methodological hygiene
----------------------
* Degenerate columns (zero variance or saturated) are discarded rather than
  imputed, since they carry no cross-sectional information.
* Scale variables are log-transformed; policy rates are kept in percentage
  points so that the standardised features remain interpretable.
* Loadings follow the conventional biplot construction: feature-component
  correlations scaled by the component singular values.
* Archetype names are derived from the empirical centroid profile, so the
  interpretation is a consequence of the data and not imposed on it.
* Anchor economies are selected by a composite of outlierness and absolute
  trade size, which prevents micro-territories from dominating the annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe raster backend, reproducible output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Paths are repo-relative so the script runs after a fresh clone.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
INPUT_CSV = REPO / "data" / "34_years_world_export_import_dataset.csv"
OUTPUT_PNG = HERE / "fig2_pca_fancy_biplot.png"

BASELINE_YEAR = 2019
N_CLUSTERS = 3
RANDOM_STATE = 20261103

PARTNER_COL = "Partner Name"
YEAR_COL = "Year"

# Supranational / macro-regional / memo aggregates. In a cross-section these
# rows occupy the extreme upper-right of the trade-volume axis and would drag
# every cluster centroid, so they are excluded before partitioning. Uninhabited
# sovereignty claims are dropped as well: they are not economies, and their
# degenerate tariff architecture creates high-leverage outliers.
MACRO_AGGREGATE_PARTNERS: frozenset[str] = frozenset(
    {
        "World", "East Asia & Pacific", "Europe & Central Asia",
        "Latin America & Caribbean", "Middle East & North Africa",
        "North America", "South Asia", "Sub-Saharan Africa",
        "Pacific Islands", "Other Asia, nes",
        "Bunkers", "Free Zones", "Special Categories", "Unspecified",
        "Br. Antr. Terr", "Fr. So. Ant. Tr", "Bouvet Island",
        "Heard Island and McDonald Isla", "South Georgia and the South Sa",
        "British Indian Ocean Ter.", "Tokelau", "Pitcairn",
    }
)

# Columns discarded as degenerate in the 2019 cross-section:
#   Export/Import Product Share -> constant 100.0 (zero variance)
#   MFN MinRate                 -> constant 0.0   (zero variance)
#   MFN Duty Free Imports       -> constant 0.0   (zero variance)
#   MFN MaxRate                 -> saturated at 3000 in >50% of observations
DEGENERATE_COLUMNS: frozenset[str] = frozenset(
    {
        "Export Product Share (%)", "Import Product Share (%)",
        "MFN MinRate (%)", "MFN Duty Free Imports (US$ Thousand)", "MFN MaxRate (%)",
    }
)

EXPORT_COL = "Export (US$ Thousand)"
IMPORT_COL = "Import (US$ Thousand)"
MFN_SIMPLE = "MFN Simple Average (%)"
MFN_WEIGHTED = "MFN Weighted Average (%)"
MFN_SPECIFIC_SHARE = "MFN Specific Tariff Lines Share (%)"
MFN_TOTAL_LINES = "MFN Total Tariff Lines"
AHS_WEIGHTED = "AHS Weighted Average (%)"
AHS_DUTYFREE_SHARE = "AHS Duty Free Tariff Lines Share (%)"

FEATURE_LABELS: dict[str, str] = {
    "log_trade_volume":        "ln Total trade volume",
    "trade_balance_ratio":     "Trade balance ratio",
    "log_tariff_lines":        "ln Tariff schedule lines",
    "mfn_simple":              "MFN simple bound rate",
    "mfn_weighted":            "MFN weighted bound rate",
    "applied_weighted":        "Applied weighted rate (AHS)",
    "applied_duty_free_share": "Applied duty-free line share",
    "specific_duty_share":     "Specific-duty line share",
}


@dataclass(frozen=True)
class Archetype:
    """A recovered latent cluster together with its interpretive reading."""

    key: str
    name: str
    color: str


# Perceptually balanced palette: hue separation survives greyscale printing
# and the common colour-vision deficiencies.
ARCHETYPES: tuple[Archetype, ...] = (
    Archetype("hub",        "Trade-Integrated Hubs",     "#2b5c8f"),
    Archetype("protected",  "Protected Economies",       "#e7298a"),
    Archetype("peripheral", "Peripheral / Low-Capacity", "#238b45"),
)

COLOR_GRID = "#DCDCDC"
COLOR_AXIS = "#6E6E6E"
COLOR_TEXT = "#22252A"
COLOR_MUTED = "#6B717A"
COLOR_LOADING = "#37414C"

# Minimum bilateral trade volume (US$ thousand) required for an entity to be
# annotated: US$ 5 bn, which keeps callouts on materially sized economies
# without excluding mid-sized open economies.
ANCHOR_MIN_VOLUME = 5.0e6
ANCHORS_PER_CLUSTER = 4

# Compact display names for entities whose official label is too long to fit a
# callout box without crowding the panel.
DISPLAY_NAMES: dict[str, str] = {
    "Russian Federation": "Russia",
    "United Arab Emirates": "UAE",
    "Congo, Dem. Rep.": "DR Congo",
    "Syrian Arab Republic": "Syria",
    "Iran, Islamic Rep.": "Iran",
    "Korea, Rep.": "Korea",
    "Egypt, Arab Rep.": "Egypt",
    "Central African Republic": "C.A.R.",
    "Equatorial Guinea": "Eq. Guinea",
    "Bosnia and Herzegovina": "Bosnia & Herz.",
    "Dominican Republic": "Dom. Rep.",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "St. Vincent and the Grenadines": "St. Vincent",
    "Micronesia, Fed. Sts.": "Micronesia",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Czech Republic": "Czechia",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Sao Tome and Principe": "São Tomé",
    "North Macedonia": "N. Macedonia",
    "West Bank and Gaza": "West Bank & Gaza",
}


def display_name(entity: str) -> str:
    """Return the shortened label rendered inside the figure."""
    return DISPLAY_NAMES.get(entity, entity)


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #

def extract_cross_section(path: Path = INPUT_CSV, year: int = BASELINE_YEAR) -> pd.DataFrame:
    """Return the partner-level cross-section for the baseline year."""
    panel = pd.read_csv(path)
    panel[PARTNER_COL] = panel[PARTNER_COL].astype(str).str.strip()
    panel[YEAR_COL] = panel[YEAR_COL].astype(int)

    cross = panel.loc[
        (panel[YEAR_COL] == year)
        & (~panel[PARTNER_COL].isin(MACRO_AGGREGATE_PARTNERS))
    ].copy()

    needed = [
        EXPORT_COL, IMPORT_COL, MFN_SIMPLE, MFN_WEIGHTED, MFN_SPECIFIC_SHARE,
        MFN_TOTAL_LINES, AHS_WEIGHTED, AHS_DUTYFREE_SHARE,
    ]
    # Coercion turns silent parse failures into NaN, which the completeness
    # filter below removes rather than imputing.
    cross[needed] = cross[needed].apply(pd.to_numeric, errors="coerce")

    before = len(cross)
    cross = cross.dropna(subset=needed)
    cross = cross.loc[(cross[EXPORT_COL] + cross[IMPORT_COL]) > 0.0]
    cross = cross.set_index(PARTNER_COL).sort_index()

    print(f"[section] baseline year {year}: {before} candidate rows -> {len(cross)} usable economies "
          f"({before - len(cross)} dropped for missing or zero-volume records)")
    print(f"[section] aggregate / non-economic entities excluded: {len(MACRO_AGGREGATE_PARTNERS)}")
    print(f"[section] degenerate columns discarded: {', '.join(sorted(DEGENERATE_COLUMNS))}")
    return cross


def engineer_features(cross: pd.DataFrame) -> pd.DataFrame:
    """Build the eight analysis variables from the raw cross-section."""
    export = cross[EXPORT_COL].astype(float)
    import_ = cross[IMPORT_COL].astype(float)
    volume = export + import_

    feats = pd.DataFrame(index=cross.index)

    # --- Scale and orientation block. Log10 because bilateral volumes span
    #     roughly 4.5 orders of magnitude and are strongly right-skewed. ---
    feats["log_trade_volume"] = np.log10(volume)
    feats["trade_balance_ratio"] = (export - import_) / volume

    # --- Tariff architecture block. Schedule depth is logged; policy rates
    #     are retained in percentage points for interpretability. ---
    feats["log_tariff_lines"] = np.log10(cross[MFN_TOTAL_LINES].astype(float))
    feats["mfn_simple"] = cross[MFN_SIMPLE].astype(float)
    feats["mfn_weighted"] = cross[MFN_WEIGHTED].astype(float)
    feats["applied_weighted"] = cross[AHS_WEIGHTED].astype(float)
    feats["applied_duty_free_share"] = cross[AHS_DUTYFREE_SHARE].astype(float)
    feats["specific_duty_share"] = cross[MFN_SPECIFIC_SHARE].astype(float)

    print("[features] engineered design matrix (descriptive statistics):")
    print(feats.rename(columns=FEATURE_LABELS)
          .describe(percentiles=[0.05, 0.5, 0.95]).round(3).to_string())

    redundancy = feats.corr(method="spearman").abs().where(
        ~np.eye(feats.shape[1], dtype=bool)
    )
    print(f"[features] multicollinearity screen: highest |rho| = "
          f"{redundancy.max().max():.2f} "
          f"({FEATURE_LABELS[redundancy.max().idxmax()]})")
    return feats


# --------------------------------------------------------------------------- #
# Modelling layer
# --------------------------------------------------------------------------- #

@dataclass
class ClusterModel:
    """Fitted partition, PCA embedding and the derived archetype reading."""

    features: pd.DataFrame
    scaled: np.ndarray
    labels: np.ndarray
    pca: PCA
    scores: np.ndarray
    centroids_raw: pd.DataFrame
    silhouette: float
    label_map: dict[int, str] = field(default_factory=dict)

    @property
    def variance_explained(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    def cluster_id(self, key: str) -> int:
        """Return the K-Means cluster id carrying the given archetype key."""
        return [cid for cid, name in self.label_map.items() if name == key][0]


def fit_models(feats: pd.DataFrame) -> ClusterModel:
    """Standardise the design matrix, partition it, and project for display."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats.to_numpy(dtype=float))

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        init="k-means++",
        n_init=30,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    labels = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    scores = pca.fit_transform(scaled)

    sil = float(silhouette_score(scaled, labels))

    centroids_raw = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feats.columns,
    )
    centroids_raw.index = centroids_raw.index.astype(int)
    centroids_raw.index.name = "cluster_id"
    sizes = pd.Series(labels).value_counts().sort_index()
    centroids_raw["n_entities"] = sizes.reindex(centroids_raw.index).to_numpy()

    var = pca.explained_variance_ratio_
    print(f"[model] K-Means k={N_CLUSTERS} | silhouette = {sil:.3f} | inertia = {kmeans.inertia_:.1f}")
    print(f"[model] PCA variance explained: PC1 {var[0] * 100:.1f}%, PC2 {var[1] * 100:.1f}% "
          f"(cumulative {var.sum() * 100:.1f}%)")
    print("[model] centroid profile in transformed feature units:")
    print(centroids_raw.rename(columns=FEATURE_LABELS).round(2).to_string())

    # Archetype reading driven by the centroid profile. The largest-scale
    # cluster is the hub group. Of the two remaining, the one with the higher
    # MFN weighted bound rate is the protected group: the bound ceiling is the
    # legal market-access commitment and the standard protectionism metric.
    # The residual cluster is peripheral / low-capacity, typically signalled
    # by a heavy specific-duty share (weak customs valuation capacity).
    scale_axis = centroids_raw["log_trade_volume"]
    bound_axis = centroids_raw["mfn_weighted"]
    hub_id = int(scale_axis.idxmax())
    remainder = [int(c) for c in centroids_raw.index if int(c) != hub_id]
    protected_id = int(bound_axis.loc[remainder].idxmax())
    peripheral_id = int([c for c in centroids_raw.index if int(c) not in (hub_id, protected_id)][0])

    label_map = {hub_id: "hub", protected_id: "protected", peripheral_id: "peripheral"}
    print("[model] archetype assignment from centroids: "
          + " | ".join(
              f"{name} = cluster {cid} (n = {int(sizes.loc[cid])}, "
              f"ln volume {scale_axis.loc[cid]:.2f}, bound rate {bound_axis.loc[cid]:.2f}%)"
              for cid, name in label_map.items()
          ))

    return ClusterModel(feats, scaled, labels, pca, scores, centroids_raw, sil, label_map)


# --------------------------------------------------------------------------- #
# Anchor selection and collision-avoiding annotation engine
# --------------------------------------------------------------------------- #

def _volume_of(feats: pd.DataFrame, entity: str) -> float:
    """Total bilateral volume (US$ thousand) implied by ln trade volume."""
    return 10.0 ** float(feats.loc[entity, "log_trade_volume"])


def select_anchor_entities(model: ClusterModel, feats: pd.DataFrame) -> list[str]:
    """Pick representative anchor economies per archetype.

    Ranking combines outlierness from the cluster centroid in PCA space with
    absolute trade size; the volume floor keeps micro-territories out of the
    callouts while retaining mid-sized open economies.
    """
    index = list(feats.index)
    scores, labels = model.scores, model.labels

    anchors: list[str] = []
    for arc in ARCHETYPES:
        cid = model.cluster_id(arc.key)
        mask = np.flatnonzero(labels == cid)
        if mask.size == 0:
            continue
        pts = scores[mask]
        dist = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
        dist_n = (dist - dist.min()) / (np.ptp(dist) or 1.0)

        vol = np.array([_volume_of(feats, index[i]) for i in mask])
        log_vol = np.log10(np.maximum(vol, 1.0))
        vol_n = (log_vol - log_vol.min()) / (np.ptp(log_vol) or 1.0)
        eligible = vol >= ANCHOR_MIN_VOLUME
        if eligible.sum() == 0:
            eligible = np.ones_like(eligible, dtype=bool)

        composite = np.where(eligible, 0.55 * dist_n + 0.45 * vol_n, -1.0)
        order = mask[np.argsort(-composite)]
        anchors += [index[i] for i in order][:ANCHORS_PER_CLUSTER]
    return anchors


def _bbox_overlaps(a, b, pad: float = 1.0) -> bool:
    """Return True when two display-space bounding boxes intersect."""
    return not (
        a.x1 + pad < b.x0 or b.x1 + pad < a.x0
        or a.y1 + pad < b.y0 or b.y1 + pad < a.y0
    )


def _occupied_display_boxes(ax: Axes, renderer) -> tuple[list, np.ndarray]:
    """Collect every element a new callout must not overprint."""
    text_boxes = [t.get_window_extent(renderer) for t in ax.texts]
    marker_points: list[np.ndarray] = []
    for artist in ax.collections:
        offsets = artist.get_offsets()
        if len(offsets):
            marker_points.append(ax.transData.transform(np.asarray(offsets, dtype=float)))
    points_display = np.vstack(marker_points) if marker_points else np.empty((0, 2))
    return text_boxes, points_display


def _marker_hit(box, points_display: np.ndarray, pad: float = 3.0) -> int:
    """Count scatter markers falling inside a candidate label box."""
    if points_display.size == 0:
        return 0
    inside = (
        (points_display[:, 0] >= box.x0 - pad) & (points_display[:, 0] <= box.x1 + pad)
        & (points_display[:, 1] >= box.y0 - pad) & (points_display[:, 1] <= box.y1 + pad)
    )
    return int(inside.sum())


def annotate_without_collisions(
    ax: Axes,
    points: dict[str, tuple[float, float]],
    n_cand: int = 108,
    r_min_pt: float = 54.0,
    r_max_pt: float = 132.0,
    fontsize: float = 8.0,
) -> None:
    """Greedy callout placement on an expanding polar lattice.

    Each anchor is offered candidate positions at increasing radius and
    rotating bearing; the position minimising overlap with existing text and
    with the scatter cloud is kept, and a thin pointer line connects the
    callout back to its observation.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    occupied, points_display = _occupied_display_boxes(ax, renderer)

    for label, (x_data, y_data) in points.items():
        anchor = ax.transData.transform((x_data, y_data))
        best: tuple[float, tuple[float, float]] | None = None

        for ring in range(8):
            r = r_min_pt + (r_max_pt - r_min_pt) * ring / 7.0
            for j in range(n_cand):
                theta_deg = 360.0 * j / n_cand
                theta = np.deg2rad(theta_deg)
                cx, cy = inv.transform(
                    (anchor[0] + r * np.cos(theta), anchor[1] + r * np.sin(theta))
                )
                if not (xlim[0] <= cx <= xlim[1] and ylim[0] <= cy <= ylim[1]):
                    continue

                probe = ax.text(cx, cy, label, fontsize=fontsize, ha="center",
                                va="center", color=COLOR_TEXT)
                box = probe.get_window_extent(renderer)
                probe.remove()

                text_hits = sum(_bbox_overlaps(box, o) for o in occupied)
                marker_hits = _marker_hit(box, points_display)
                cost = (text_hits * 120.0 + marker_hits * 18.0 + ring * 0.6
                        + abs(90.0 - theta_deg) * 0.005)
                if best is None or cost < best[0]:
                    best = (cost, (cx, cy))
                if text_hits == 0 and marker_hits == 0 and ring >= 2:
                    break
            if best is not None and best[0] < 1.0:
                break

        if best is None:
            continue
        _, (cx, cy) = best
        probe = ax.text(cx, cy, label, fontsize=fontsize)
        occupied.append(probe.get_window_extent(renderer))
        probe.remove()

        ax.annotate(
            label,
            xy=(x_data, y_data), xycoords="data",
            xytext=(cx, cy), textcoords="data",
            fontsize=fontsize, color=COLOR_TEXT, ha="center", va="center",
            zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E4E6E9",
                      lw=0.5, alpha=0.88),
            arrowprops=dict(
                arrowstyle="-", color=COLOR_MUTED, lw=0.6,
                shrinkA=1.0, shrinkB=4.0, connectionstyle="arc3,rad=0.0",
            ),
        )


# --------------------------------------------------------------------------- #
# Rendering layer
# --------------------------------------------------------------------------- #

def _style_panel(ax: Axes, title: str, subtitle: str) -> None:
    """Shared Few-style chrome: single grid dimension, no redundant spines."""
    ax.grid(True, which="major", color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=3.2, color=COLOR_AXIS, labelsize=8.6,
                   top=False, right=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.set_title(title, loc="left", fontsize=10.6, fontweight="bold",
                 color=COLOR_TEXT, pad=20)
    ax.text(0.0, 1.032, subtitle, transform=ax.transAxes, fontsize=7.8,
            color=COLOR_MUTED, va="bottom", ha="left")


def _jittered_scores(scores: np.ndarray) -> np.ndarray:
    """Deterministic micro-jitter separating coincident observations without
    displacing them beyond the marker radius."""
    rng = np.random.default_rng(RANDOM_STATE)
    out = scores.copy()
    for axis in range(scores.shape[1]):
        out[:, axis] += rng.normal(0.0, 0.012 * (np.ptp(scores[:, axis]) or 1.0),
                                   scores.shape[0])
    return out


def draw_pca_biplot(ax: Axes, model: ClusterModel) -> None:
    """Panel A — PCA scores with eigenvector loading vectors and labels."""
    scores, labels = model.scores, model.labels
    var = model.variance_explained

    for arc in ARCHETYPES:
        cid = model.cluster_id(arc.key)
        mask = labels == cid
        if mask.sum() == 0:
            continue
        ax.scatter(scores[mask, 0], scores[mask, 1], s=27, c=arc.color,
                   alpha=0.72, edgecolors="white", linewidths=0.45, zorder=3)

    # Zero reference crosshair: makes sign and magnitude of scores readable.
    ax.axhline(0.0, color="#B4B9BF", linewidth=0.75, zorder=2)
    ax.axvline(0.0, color="#B4B9BF", linewidth=0.75, zorder=2)

    # --- Loading vectors: correlations scaled by component singular values ---
    loadings = model.pca.components_.T * np.sqrt(model.pca.explained_variance_)[None, :]
    half_x = np.ptp(scores[:, 0]) / 2.0
    half_y = np.ptp(scores[:, 1]) / 2.0
    xlim, ylim = half_x * 1.46, half_y * 1.46
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    unit = loadings / np.abs(loadings).max()
    for i, (fname, short) in enumerate(FEATURE_LABELS.items()):
        dx = unit[i, 0] * xlim * 0.80
        dy = unit[i, 1] * ylim * 0.80
        strength = float(abs(loadings[i, 0]) + abs(loadings[i, 1]))
        ax.annotate(
            "", xy=(dx, dy), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_LOADING,
                            lw=0.9 + 1.2 * strength, alpha=0.93, shrinkB=1.5),
            zorder=6,
        )
        ax.text(dx * 1.13, dy * 1.13, short, fontsize=7.8, color="#1F2933",
                ha="center", va="center", fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.78))

    ax.set_xlabel(f"Principal component 1  ({var[0] * 100:.1f}% of variance)", labelpad=7)
    ax.set_ylabel(f"Principal component 2  ({var[1] * 100:.1f}% of variance)", labelpad=7)
    _style_panel(
        ax, "A.  PCA biplot of the 2019 trade-policy feature space",
        "Arrows are eigenvector loading vectors, scaled to feature-component correlation",
    )


def draw_hull_envelopes(ax: Axes, model: ClusterModel, feats: pd.DataFrame) -> None:
    """Panel B — convex-hull envelopes, jittered points, annotated anchors."""
    scores, labels = model.scores, model.labels
    jpts = _jittered_scores(scores)

    pad_x, pad_y = np.ptp(scores[:, 0]) * 0.15, np.ptp(scores[:, 1]) * 0.17
    ax.set_xlim(scores[:, 0].min() - pad_x, scores[:, 0].max() + pad_x * 2.3)
    ax.set_ylim(scores[:, 1].min() - pad_y, scores[:, 1].max() + pad_y)

    for arc in ARCHETYPES:
        cid = model.cluster_id(arc.key)
        mask = labels == cid
        if mask.sum() == 0:
            continue
        pts = jpts[mask]

        # --- Topological envelope: convex hull over the jittered members ---
        if mask.sum() >= 3:
            try:
                hull = ConvexHull(pts, qhull_options="QJ Pp")
                verts = pts[hull.vertices]
                centre = verts.mean(axis=0)
                inset = centre + 0.965 * (verts - centre)  # optical inset
                ax.add_patch(MplPolygon(
                    inset, closed=True, facecolor=arc.color, alpha=0.075,
                    edgecolor=arc.color, linewidth=1.15, zorder=1.5,
                ))
            except (QhullError, ValueError):
                pass  # degenerate configuration: envelope omitted, points remain

        # --- Cluster centroid ---
        c = pts.mean(axis=0)
        ax.scatter(c[0], c[1], marker="X", s=120, c=arc.color,
                   edgecolors="white", linewidths=1.0, zorder=6)

        # --- Jittered members with micro white borders ---
        ax.scatter(pts[:, 0], pts[:, 1], s=21, c=arc.color, alpha=0.82,
                   edgecolors="white", linewidths=0.38, zorder=4)

    # --- Representative anchor economies, collision-avoided callouts ---
    index = list(feats.index)
    named: dict[str, tuple[float, float]] = {}
    for entity in select_anchor_entities(model, feats):
        pos = index.index(entity)
        named[display_name(entity)] = (jpts[pos, 0], jpts[pos, 1])
    annotate_without_collisions(ax, named)

    ax.set_xlabel("Principal component 1", labelpad=7)
    ax.set_ylabel("Principal component 2", labelpad=7)
    _style_panel(
        ax, "B.  Latent archetypes as convex-hull topological envelopes",
        "Partition estimated in the standardised 8-D space; projection is for display only",
    )


def compose_figure(model: ClusterModel, feats: pd.DataFrame) -> Figure:
    """Assemble the two-panel publication figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.8, 6.7), dpi=350)

    draw_pca_biplot(ax_a, model)
    draw_hull_envelopes(ax_b, model, feats)

    sizes = pd.Series(model.labels).value_counts()
    handles = [
        Line2D([], [], marker="o", ls="none", ms=6.4, mfc=arc.color, mec="white",
               mew=0.6,
               label=f"{arc.name}  (n = {int(sizes[model.cluster_id(arc.key)])})")
        for arc in ARCHETYPES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.8, handlelength=1.3, columnspacing=2.6,
               bbox_to_anchor=(0.5, 0.006))

    fig.text(0.013, 0.981,
             "Figure 2. Latent archetype structure of world trade and tariff policy, "
             "cross-sectional baseline 2019",
             ha="left", va="top", fontsize=12.8, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.013, 0.942,
             f"{len(feats)} economies | {len(FEATURE_LABELS)} log-transformed and engineered features, "
             f"standardised to mean 0 and s.d. 1 | K-Means k = {N_CLUSTERS}, "
             f"silhouette = {model.silhouette:.3f} | PCA retains "
             f"{model.variance_explained.sum() * 100:.1f}% of total variance",
             ha="left", va="top", fontsize=8.1, color=COLOR_MUTED)

    # Explicit placement keeps title, panels and legend deterministically spaced.
    fig.subplots_adjust(left=0.062, right=0.988, top=0.862, bottom=0.118, wspace=0.235)
    return fig


def save_figure(fig: Figure, output: Path = OUTPUT_PNG) -> Path:
    """Persist the figure at print-resolution raster quality."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[save] two-panel figure written -> {output} (350 DPI, PNG)")
    return output


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> Path:
    cross = extract_cross_section()
    feats = engineer_features(cross)
    model = fit_models(feats)
    fig = compose_figure(model, feats)
    return save_figure(fig)


if __name__ == "__main__":
    main()
