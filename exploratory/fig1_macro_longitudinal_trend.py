"""
Figure 1 — Longitudinal trajectory of world merchandise trade (1988-2021).

Design contract
---------------
* Data-ink maximisation (S. Few): no chartbook junk, no dual axes, no
  redundant legend, single grid dimension, direct end-of-line labelling.
* Macro-regional aggregates are stripped before aggregation so that each
  dollar of trade is counted exactly once (prevents cluster / scale distortion).
* Output: 300 DPI PNG, `fig1_macro_longitudinal_trend.png`.

Author: econometrics & visual analytics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe, reproducible raster output

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Paths are repo-relative so the script runs after a fresh clone.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
INPUT_CSV = REPO / "data" / "34_years_world_export_import_dataset.csv"
OUTPUT_PNG = HERE / "fig1_macro_longitudinal_trend.png"

EXPORT_COL = "Export (US$ Thousand)"
IMPORT_COL = "Import (US$ Thousand)"
PARTNER_COL = "Partner Name"
YEAR_COL = "Year"

# Thousand USD -> Trillion USD: 1 US$ thousand = 1e3 USD, 1 trillion USD = 1e12 USD,
# hence the conversion factor is 1e3 / 1e12 = 1e-9.
THOUSAND_USD_TO_TRILLION_USD = 1.0e-9

# Supranational / macro-regional / memo aggregates. Retaining any of these
# alongside their member economies double-counts volume and inflates the
# right-hand end of the sample, which distorts every downstream cluster
# solution and panel regression.
MACRO_AGGREGATE_PARTNERS: frozenset[str] = frozenset(
    {
        "World",
        "East Asia & Pacific",
        "Europe & Central Asia",
        "Latin America & Caribbean",
        "Middle East & North Africa",
        "North America",
        "South Asia",
        "Sub-Saharan Africa",
        "Pacific Islands",
        "Other Asia, nes",
        # Non-territorial memo items: not resident economies, no macro-region label.
        "Bunkers",
        "Free Zones",
        "Special Categories",
        "Unspecified",
    }
)


@dataclass(frozen=True)
class ExogenousShock:
    """A documented global shock to be rendered as a shaded span."""

    label: str
    start_year: int
    end_year: int

    @property
    def midpoint(self) -> float:
        return self.start_year + (self.end_year - self.start_year) / 2.0


SHOCKS: tuple[ExogenousShock, ...] = (
    ExogenousShock("ASIAN\nFINANCIAL\nCRISIS", 1997, 1998),
    ExogenousShock("GLOBAL\nFINANCIAL\nCRISIS", 2008, 2009),
    ExogenousShock("TRADE\nSLOWDOWN", 2014, 2016),
    ExogenousShock("COVID-19\nPANDEMIC", 2020, 2021),
)

# Muted, colour-blind-safe palette; hue carries the series, not the ranking.
COLOR_EXPORT = "#1F4E79"   # deep steel blue
COLOR_IMPORT = "#C0762B"   # burnt ochre
COLOR_GAP = "#9AA7B4"      # neutral slate for the surplus band
COLOR_SPAN = "#5A6472"     # neutral grey-blue for shock shading
COLOR_GRID = "#D8D8D8"
COLOR_AXIS = "#6E6E6E"
COLOR_TEXT = "#22252A"


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #

def load_panel(path: Path = INPUT_CSV) -> pd.DataFrame:
    """Read the raw partner x year panel and normalise key fields."""
    frame = pd.read_csv(path)
    frame[PARTNER_COL] = frame[PARTNER_COL].astype(str).str.strip()
    frame[YEAR_COL] = frame[YEAR_COL].astype(int)
    frame[[EXPORT_COL, IMPORT_COL]] = (
        frame[[EXPORT_COL, IMPORT_COL]].apply(pd.to_numeric, errors="coerce")
    )
    return frame


def drop_macro_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only resident-economy rows; report what was removed."""
    is_aggregate = frame[PARTNER_COL].isin(MACRO_AGGREGATE_PARTNERS)
    cleaned = frame.loc[~is_aggregate].copy()

    removed = sorted(cleaned_parts for cleaned_parts in frame.loc[is_aggregate, PARTNER_COL].unique())
    print(
        f"[filter] rows retained {len(cleaned):,} / {len(frame):,} | "
        f"partner entities retained {cleaned[PARTNER_COL].nunique():,}"
    )
    print(f"[filter] macro-regional / memo aggregates removed ({len(removed)}): {', '.join(removed)}")
    return cleaned


def aggregate_longitudinal_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the panel to world totals in trillion USD per calendar year."""
    annual = (
        frame.groupby(YEAR_COL, as_index=True)[[EXPORT_COL, IMPORT_COL]]
        .sum()
        .reindex(sorted(frame[YEAR_COL].unique()), fill_value=0.0)
    )
    annual["Export (Trillion USD)"] = annual[EXPORT_COL] * THOUSAND_USD_TO_TRILLION_USD
    annual["Import (Trillion USD)"] = annual[IMPORT_COL] * THOUSAND_USD_TO_TRILLION_USD
    annual["Trade surplus (Trillion USD)"] = (
        annual["Export (Trillion USD)"] - annual["Import (Trillion USD)"]
    )
    return annual


def print_diagnostics(annual: pd.DataFrame) -> None:
    """Emit the summary statistics quoted by the figure annotations."""
    for series in ("Export", "Import"):
        column = f"{series} (Trillion USD)"
        first, last = annual[column].iloc[0], annual[column].iloc[-1]
        cagr = (last / first) ** (1.0 / (annual.index[-1] - annual.index[0])) - 1.0
        growth = annual[column].pct_change()
        print(
            f"[diag] {series:<7}| peak {annual[column].max():.2f} T USD in "
            f"{int(annual[column].idxmax())} | latest {last:.2f} T USD | "
            f"CAGR {cagr * 100:.1f}% | sharpest y/y drop "
            f"{growth.min() * 100:.1f}% in {int(growth.idxmin())}"
        )
    gap = annual["Trade surplus (Trillion USD)"]
    print(f"[diag] sample: {int(annual.index.min())}-{int(annual.index.max())} | "
          f"export > import in {int((gap > 0).sum())}/{len(gap)} years | "
          f"mean gap {gap.mean():.2f} T USD | widest gap {gap.max():.2f} T USD "
          f"({int(gap.idxmax())}) | narrowest {gap.min():.2f} T USD ({int(gap.idxmin())})")


# --------------------------------------------------------------------------- #
# Rendering layer
# --------------------------------------------------------------------------- #

def build_figure(annual: pd.DataFrame) -> tuple[Figure, Axes]:
    """Compose the publication-grade dual-trajectory chart."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.titlesize": 13.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=300)

    export = annual["Export (Trillion USD)"]
    impo = annual["Import (Trillion USD)"]
    year = annual.index.to_series().astype(float)

    # Axis ranges are fixed up-front so that span labels can reference them.
    x_lo, x_hi = int(annual.index.min()) - 1.2, int(annual.index.max()) + 3.4
    # Upper headroom keeps the rotated shock captions clear of the trajectories.
    y_hi = 26.0
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.0, y_hi)

    # --- Shaded exogenous shocks (painted first: they sit behind the ink) ---
    for shock in SHOCKS:
        ax.axvspan(
            shock.start_year - 0.5,
            shock.end_year + 0.5,
            color=COLOR_SPAN,
            alpha=0.09,
            linewidth=0.0,
            zorder=0,
        )
        ax.axvline(
            shock.start_year - 0.5,
            color=COLOR_SPAN,
            alpha=0.28,
            linewidth=0.7,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        ax.text(
            shock.midpoint,
            y_hi - 0.35,
            shock.label,
            rotation=90,
            va="top",
            ha="center",
            fontsize=7.4,
            fontweight="bold",
            color="#3D4551",
            linespacing=1.25,
            zorder=6,
        )

    # --- Surplus band: encodes export-minus-import without a third axis ---
    ax.fill_between(
        year, export, impo,
        where=export >= impo,
        color=COLOR_GAP, alpha=0.16, linewidth=0.0, zorder=2,
        interpolate=True,
    )

    # --- Dual trajectory trendlines ---
    ax.plot(year, impo, color=COLOR_IMPORT, linewidth=1.9,
            marker="o", markersize=3.0, markerfacecolor="white",
            markeredgewidth=1.0, zorder=4, label="Import")
    ax.plot(year, export, color=COLOR_EXPORT, linewidth=2.2,
            marker="o", markersize=3.4, markerfacecolor="white",
            markeredgewidth=1.1, zorder=5, label="Export")

    # --- Direct end-of-line labelling (removes the legend) ---
    for series, color, dy in ((export, COLOR_EXPORT, 0.42), (impo, COLOR_IMPORT, -0.60)):
        ax.annotate(
            f"{series.name.split(' (')[0]}\n{series.iloc[-1]:.2f} T USD",
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(series.index[-1] + 0.55, series.iloc[-1] + dy),
            fontsize=8.6, fontweight="bold", color=color,
            va="center", ha="left", linespacing=1.35,
        )

    # --- One quantitative callout on the deepest reversal (2008-09 collapse) ---
    drop_pct = export.pct_change().min() * 100.0
    drop_year = int(export.pct_change().idxmin())
    ax.annotate(
        f"{drop_pct:.1f}% y/y\n(Great Trade Collapse)",
        xy=(drop_year, float(export.loc[drop_year])),
        xytext=(drop_year - 2.6, float(export.loc[drop_year]) - 3.9),
        fontsize=8.2, color="#7A2E1D", fontweight="bold",
        ha="center", va="top", linespacing=1.35,
        arrowprops=dict(arrowstyle="->", color="#7A2E1D", lw=0.8,
                        shrinkA=2, shrinkB=5,
                        connectionstyle="arc3,rad=-0.25"),
    )

    # --- Axes, grid and spines (Few-style minimal chrome) ---
    ax.set_ylabel("Merchandise trade volume (trillion USD, current values)", labelpad=8)
    ax.set_xlabel("")

    ax.yaxis.grid(True, which="major", color=COLOR_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=3.5, color=COLOR_AXIS,
                   which="both", top=False, right=False)
    ax.set_xticks(range(int(annual.index.min()), int(annual.index.max()) + 1, 2))
    ax.set_yticks([0, 4, 8, 12, 16, 20, 24])

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
        ax.spines[side].set_linewidth(0.8)

    fig.text(
        0.005, 0.012,
        "Source: author's calculations on the 34-year world export/import panel. "
        "Shaded spans mark four exogenous global shocks.",
        fontsize=7.4, color="#8A8F98", ha="left", va="bottom",
    )

    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.895))

    # --- Title block placed in figure space, anchored to the axes' left edge ---
    fig.canvas.draw()
    anchor_x = ax.get_position().x0
    fig.text(
        anchor_x, 0.975,
        "Figure 1. Longitudinal trajectory of world merchandise trade, 1988–2021",
        fontsize=13.5, fontweight="bold", color=COLOR_TEXT, ha="left", va="top",
    )
    fig.text(
        anchor_x, 0.928,
        "World totals aggregated over resident economies only; macro-regional aggregates excluded "
        "to prevent double counting",
        fontsize=8.8, color="#666C76", ha="left", va="top",
    )
    return fig, ax


def render(annual: pd.DataFrame, output: Path = OUTPUT_PNG) -> Path:
    """Render and persist the figure at print-resolution raster quality."""
    fig, _ = build_figure(annual)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"[save] publication-grade figure written -> {output} (300 DPI, PNG)")
    return output


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> Path:
    panel = load_panel(INPUT_CSV)
    resident = drop_macro_aggregates(panel)
    annual = aggregate_longitudinal_totals(resident)
    print_diagnostics(annual)
    return render(annual)


if __name__ == "__main__":
    main()
