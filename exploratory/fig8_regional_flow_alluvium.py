"""
Figure 8 — Geographic reorientation of world merchandise trade flows,
2000 -> 2019 (alluvial / ribbon-Sankey view).

Narrative role
--------------
Act II (Relational Capital Flow) of the briefing: how the material flow of
world merchandise trade was reallocated across the seven macro-regions of the
planet between the pre-financial-crisis baseline (2000) and the pre-pandemic
cross-section used for the archetype analysis (2019).

Design notes
------------
* Unlike the longitudinal trend (Figure 1), where macro-regional aggregate
  rows must be removed to avoid double counting, this exhibit measures the
  regions themselves; the aggregate rows are therefore the intended unit of
  observation. The seven region nodes carry the full ribbon flow and are
  exactly conserved between the two columns (no residual node is needed).
* Ribbon width is proportional to gross flow (exports + imports); each node
  is internally stacked into export (steel blue) and import (ochre) so the
  instrument split is legible without a second diagram.
* Nodes in both columns are ordered by the 2019 magnitude, so any ribbon that
  visibly flares or narrows is a statement about re-weighting of that region
  in world trade, and crossings of ribbons are suppressed by construction.
* Bezier ribbons are hand-built cubic paths (no third-party Sankey package),
  keeping the figure fully reproducible with matplotlib alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Paths are repo-relative so the script runs after a fresh clone.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
INPUT_CSV = REPO / "data" / "34_years_world_export_import_dataset.csv"
OUTPUT_PNG = HERE / "fig8_regional_flow_alluvium.png"

BASE_YEAR, END_YEAR = 2000, 2019
THOUSAND_USD_TO_TRILLION_USD = 1.0e-9

PARTNER_COL = "Partner Name"
YEAR_COL = "Year"
EXPORT_COL = "Export (US$ Thousand)"
IMPORT_COL = "Import (US$ Thousand)"

# The seven inhabited macro-regions of the dataset (Pacific Islands ceased in
# 1991 and is excluded; memo categories are not regions).
REGIONS: list[str] = [
    "Europe & Central Asia",
    "East Asia & Pacific",
    "North America",
    "Latin America & Caribbean",
    "Middle East & North Africa",
    "South Asia",
    "Sub-Saharan Africa",
]

REGION_COLORS: dict[str, str] = {
    "Europe & Central Asia":       "#4C72B0",
    "East Asia & Pacific":         "#C44E52",
    "North America":               "#55A868",
    "Latin America & Caribbean":   "#8172B2",
    "Middle East & North Africa":  "#CCB974",
    "South Asia":                  "#64B5CD",
    "Sub-Saharan Africa":          "#9A6E5A",
}

COLOR_EXPORT_NODE = "#1F4E79"
COLOR_IMPORT_NODE = "#C0762B"
COLOR_TEXT = "#22252A"
COLOR_MUTED = "#6B717A"
COLOR_AXIS = "#6E6E6E"

NODE_WIDTH = 0.016          # fraction of x-axis span
GAP_EXPORT_IMPORT = 0.35   # intra-node separation, in flow units
REGION_GAP = 5.2         # inter-region separation, in flow units (clears two-line labels)


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #

def regional_flows(year: int) -> pd.DataFrame:
    """Export/import gross flows (trillion USD) for the seven macro-regions."""
    panel = pd.read_csv(INPUT_CSV)
    panel[PARTNER_COL] = panel[PARTNER_COL].astype(str).str.strip()
    rows = panel.loc[
        (panel[YEAR_COL] == year) & (panel[PARTNER_COL].isin(REGIONS)),
        [PARTNER_COL, EXPORT_COL, IMPORT_COL],
    ].set_index(PARTNER_COL)
    out = rows.mul(THOUSAND_USD_TO_TRILLION_USD)
    out.columns = ["export", "import"]
    out["total"] = out["export"] + out["import"]
    return out.reindex(REGIONS).sort_values("total", ascending=False)


# --------------------------------------------------------------------------- #
# Alluvial geometry
# --------------------------------------------------------------------------- #

def _ribbon(x0, x1, h0, h1, y0, y1):
    """Cubic-Bezier closed ribbon between two vertical segments.

    (y0, h0) is the left span, (y1, h1) the right span; control points sit at
    the horizontal midpoint to give the classic smooth Sankey waist.
    """
    xm = (x0 + x1) / 2.0
    verts = [
        (x0, y0), (xm, y0), (xm, y1), (x1, y1),          # top edge
        (x1, y1 + h1), (xm, y1 + h1), (xm, y0 + h0), (x0, y0 + h0),  # bottom edge
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(verts, codes))


def stack_spans(table: pd.DataFrame, x, ax, y_top):
    """Draw one column of nodes; return per-region slab spans in flow units."""
    spans = {}
    for region, row in table.iterrows():
        h_exp, h_imp = row["export"], row["import"]
        y_exp_top = y_top
        y_imp_top = y_exp_top - h_exp - GAP_EXPORT_IMPORT
        # export slab
        ax.add_patch(plt.Rectangle((x - NODE_WIDTH / 2, y_exp_top - h_exp),
                                   NODE_WIDTH, h_exp, facecolor=COLOR_EXPORT_NODE,
                                   edgecolor="white", linewidth=0.8, zorder=5))
        # import slab
        ax.add_patch(plt.Rectangle((x - NODE_WIDTH / 2, y_imp_top - h_imp),
                                   NODE_WIDTH, h_imp, facecolor=COLOR_IMPORT_NODE,
                                   edgecolor="white", linewidth=0.8, zorder=5))
        spans[region] = dict(y_exp=y_exp_top - h_exp, h_exp=h_exp,
                             y_imp=y_imp_top - h_imp, h_imp=h_imp)
        y_top = y_imp_top - h_imp - REGION_GAP  # inter-region separation
    return spans


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def compose_figure(output: Path = OUTPUT_PNG) -> Path:
    start = regional_flows(BASE_YEAR)
    end = regional_flows(END_YEAR)
    # Common vertical canvas sized to the larger column (2019) plus gaps.
    canvas = float(end['total'].sum() + 7 * GAP_EXPORT_IMPORT + 7 * REGION_GAP) + 6.0
    order = list(end.index)
    start = start.reindex(order)
    end = end.reindex(order)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(12.4, 7.4), dpi=300)
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(0, canvas + 6.5)
    ax.axis("off")

    x0, x1 = 0.06, 0.94
    s_spans = stack_spans(start, x0, ax, canvas)
    e_spans = stack_spans(end, x1, ax, canvas)

    # --- Ribbons: export layer and import layer per region, tapering ---
    for region in order:
        color = REGION_COLORS[region]
        for key, alpha in (("exp", 0.38), ("imp", 0.30)):
            s, e = s_spans[region], e_spans[region]
            patch = _ribbon(x0 + NODE_WIDTH / 2, x1 - NODE_WIDTH / 2,
                            s[f"h_{key}"], e[f"h_{key}"],
                            s[f"y_{key}"], e[f"y_{key}"])
            patch.set(facecolor=color, edgecolor="none", alpha=alpha, zorder=2)
            ax.add_patch(patch)

    # --- Region labels: left column = 2000 values, right = 2019 + shift ---
    for region in order:
        s_row, e_row = start.loc[region], end.loc[region]
        yc_s = (s_spans[region]["y_exp"] + s_spans[region]["h_exp"]
                + s_spans[region]["y_imp"]) / 2.0
        yc_e = (e_spans[region]["y_exp"] + e_spans[region]["h_exp"]
                + e_spans[region]["y_imp"]) / 2.0
        color = REGION_COLORS[region]

        share0 = s_row["total"] / start["total"].sum() * 100.0
        share1 = e_row["total"] / end["total"].sum() * 100.0
        d_share = share1 - share0
        growth = e_row["total"] / s_row["total"]

        ax.text(x0 - NODE_WIDTH / 2 - 0.012, yc_s, f"{region}",
                ha="right", va="center", fontsize=8.6, fontweight="bold", color=color)
        ax.text(x0 - NODE_WIDTH / 2 - 0.012, yc_s - 2.3,
                f"{s_row['total']:.1f} T USD   ({share0:.1f}%)",
                ha="right", va="center", fontsize=7.6, color=COLOR_MUTED)

        ax.text(x1 + NODE_WIDTH / 2 + 0.012, yc_e, f"{region}",
                ha="left", va="center", fontsize=8.6, fontweight="bold", color=color)
        sign = "+" if d_share >= 0 else "\u2212"
        ax.text(x1 + NODE_WIDTH / 2 + 0.012, yc_e - 2.3,
                f"{e_row['total']:.1f} T USD   ({share1:.1f}%)   "
                f"{sign}{abs(d_share):.1f} pp share   {growth:.1f}\u00d7",
                ha="left", va="center", fontsize=7.6, color=COLOR_MUTED)

    # --- Column headers and legend chrome ---
    ax.text(x0, canvas + 2.2, f"{BASE_YEAR} gross flow", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    ax.text(x1, canvas + 2.2, f"{END_YEAR} gross flow", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=COLOR_TEXT)

    for sx, color, label in ((0.395, COLOR_EXPORT_NODE, "Export component"),
                             (0.545, COLOR_IMPORT_NODE, "Import component")):
        ax.add_patch(plt.Rectangle((sx, 1.0), 0.013, 2.2,
                                   facecolor=color, edgecolor="none", zorder=6))
        ax.text(sx + 0.019, 2.1, label, fontsize=7.8, color=COLOR_MUTED,
                va="center", ha="left", zorder=6)

    fig.text(0.015, 0.988,
             "Figure 8. Geographic reorientation of world merchandise trade flows, 2000 to 2019",
             fontsize=13.0, fontweight="bold", color=COLOR_TEXT, va="top", ha="left")
    fig.text(0.015, 0.947,
             "Alluvial allocation of gross bilateral flow (exports + imports, trillion USD current values) "
             "across the seven macro-regions; ribbon width proportional to flow, nodes stacked into export and import components",
             fontsize=8.3, color=COLOR_MUTED, va="top", ha="left")
    fig.text(0.015, 0.012,
             "Note: macro-regional aggregates are the unit of observation here; "
             "regions ordered by 2019 share. Residual memo categories (<6% of world flow) excluded.",
             fontsize=7.3, color="#8A8F98", va="bottom", ha="left")

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    # Console summary for the slide bullets.
    for region in order:
        s, e = start.loc[region], end.loc[region]
        print(f"[flow] {region:<28} 2000 {s['total']:5.2f} T ({s['total']/start['total'].sum()*100:4.1f}%)"
              f"  ->  2019 {e['total']:5.2f} T ({e['total']/end['total'].sum()*100:4.1f}%)  "
              f"shift {(e['total']/end['total'].sum()-s['total']/start['total'].sum())*100:+5.1f} pp  "
              f"growth {e['total']/s['total']:.1f}x")
    print(f"[save] alluvium written -> {output} (300 DPI, PNG)")
    return output


if __name__ == "__main__":
    compose_figure()
