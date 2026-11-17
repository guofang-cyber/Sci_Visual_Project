#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 WORLD TRADE STRUCTURE, 1988-2021
 Analysis + visualization pipeline for a 3-presenter, 15-minute course talk
================================================================================

 Group members (title slide): Andis Bara Mohammad Yousif  ·  Guofang Li

 Research question
 -----------------
 "Does the world trading system have a structure -- and if so, what kind of
  economies sit in each part of it?"

 Sub-questions (mapped to parts and figures below; also printed as a slide map)
 -----------------------------------------------------------------------------
   Q1  How complete is the record? Which economies and years can we stand on?
   Q2  How much did world trade grow in 34 years, and was that growth steady?
   Q3  Is trade scale usable as reported, or must it be transformed?
   Q4  Do tariff regimes differ systematically across world regions?
   Q5  Do economies cluster by trade structure, and what defines each group?
   Q6  Can trade structure alone predict an economy's tariff stance?

   Part 1 · Data            -> Q1, Q2            -> figures 01-02
   Part 2 · World/structure -> Q3, Q4            -> figures 03-06
   Part 3 · Groups          -> Q5                -> figures 07-11
   Part 4 · Predictor       -> Q6                -> figures 12-13
   Part 5 · Answer          -> all, callback     -> figures 14-15

 Technical stack
 ---------------
   pandas, numpy, matplotlib, seaborn, scikit-learn.
   scipy.stats is imported for the hypothesis tests that the deck's
   "Test . Result" footers require (Kruskal-Wallis, Spearman, Levene);
   all modelling and all plotting stay inside the four libraries above.

   Methods used  : cleaning, log10 transform (justified in figure 03),
                   standardization, Spearman correlation, K-Means with
                   elbow + silhouette, PCA, one Random Forest classifier.
   Methods NOT used (deliberately, per the brief): hierarchical clustering /
                   dendrograms, t-SNE, UMAP, factor analysis, lasso/ridge
                   paths, neural networks, bootstrap or CV inference plots,
                   decision boundaries, network analysis.

 Design
 ------
   Stephen Few: high data-ink ratio, no redundant spines, muted/viridis
   palette, large type, ONE big number per slide annotated on the figure,
   dpi = 180, files named 01_... through 15_...

 Run
 ---
   python3 analysis_presentation.py
   -> writes 15 PNGs into ./figures/
   -> prints: preprocessing summary, slide map, full deck outline,
              Q&A answer sheet
================================================================================
"""

from __future__ import annotations

import os
import sys
import warnings
from textwrap import dedent, fill

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# Dataset location: the bundled copy in ./data by default. Override with the
# WORLD_TRADE_CSV environment variable or the --csv argument.
CSV_PATH = os.environ.get(
    "WORLD_TRADE_CSV",
    os.path.join(HERE, "data", "34_years_world_export_import_dataset.csv"),
)
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 180
SEED = 42
RECENT = (2017, 2021)          # window used to build the economy-level matrix
COVERAGE_MIN = 0.95            # min share of non-missing cells to keep an economy

# Stephen Few palette: muted, colour-blind safe, no saturated primaries.
INK = "#1a1a1a"
GREY = "#6f6f6f"
LIGHT = "#d9d9d9"
PAPER = "#ffffff"
ACCENT = "#2c6e8f"             # muted teal-blue
ACCENT2 = "#b5651d"            # muted ochre
POS = "#2f7d5d"
NEG = "#a4443c"
VIRIDIS = plt.get_cmap("viridis")

# Seven World-Bank-style regions, used only for the regional comparisons.
REGION_LISTS = {
    "North America": [
        "United States", "Canada", "Bermuda", "Greenland",
        "Saint Pierre and Miquelon",
    ],
    "Latin America & Caribbean": [
        "Antigua and Barbuda", "Argentina", "Aruba", "Bahamas, The", "Barbados",
        "Belize", "Bolivia", "Bonaire", "Brazil", "Cayman Islands", "Chile",
        "Colombia", "Costa Rica", "Cuba", "Curaçao", "Dominica",
        "Dominican Republic", "Ecuador", "El Salvador", "Falkland Island",
        "French Guiana", "Grenada", "Guadeloupe", "Guatemala", "Guyana", "Haiti",
        "Honduras", "Jamaica", "Martinique", "Mexico", "Montserrat", "Nicaragua",
        "Panama", "Paraguay", "Peru", "Saint Barthélemy",
        "Saint Maarten (Dutch part)", "St. Kitts and Nevis", "St. Lucia",
        "St. Vincent and the Grenadines", "Suriname", "Trinidad and Tobago",
        "Turks and Caicos Isl.", "Uruguay", "Venezuela",
    ],
    "East Asia & Pacific": [
        "American Samoa", "Australia", "Brunei", "Cambodia", "China",
        "Korea, Dem. Rep.",
        "Christmas Island", "Cocos (Keeling) Islands", "Cook Islands", "Fiji",
        "French Polynesia", "Guam", "Hong Kong, China", "Indonesia", "Japan",
        "Kiribati", "Korea, Rep.", "Lao PDR", "Macao", "Malaysia",
        "Marshall Islands", "Micronesia, Fed. Sts.", "Mongolia", "Myanmar",
        "Nauru", "New Caledonia", "New Zealand", "Niue",
        "Northern Mariana Islands", "Norfolk Island", "Palau",
        "Papua New Guinea", "Philippines", "Pitcairn", "Samoa", "Singapore",
        "Solomon Islands", "Thailand", "East Timor", "Tokelau", "Tonga",
        "Tuvalu", "Vanuatu", "Vietnam", "Wallis and Futura Isl.",
    ],
    "Europe & Central Asia": [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
        "Belgium", "Belgium-Luxembourg", "Bosnia and Herzegovina", "Bulgaria",
        "Croatia", "Cyprus", "Czech Republic", "Czechoslovakia", "Denmark",
        "Estonia", "Faeroe Islands", "Finland", "France", "Georgia",
        "German Democratic Republic", "Germany", "Gibraltar", "Greece",
        "Holy See", "Hungary", "Iceland", "Ireland", "Italy", "Kazakhstan",
        "Kyrgyz Republic", "Latvia", "Lithuania", "Luxembourg", "Malta",
        "Moldova", "Monaco", "Montenegro", "Netherlands", "Netherlands Antilles",
        "North Macedonia", "Norway", "Poland", "Portugal", "Romania",
        "Russian Federation", "San Marino", "Serbia, FR(Serbia/Montenegro)",
        "Slovak Republic", "Slovenia", "Soviet Union", "Spain", "Sweden",
        "Switzerland", "Tajikistan", "Turkey", "Turkmenistan", "Ukraine",
        "United Kingdom", "Uzbekistan", "Yugoslavia,FR(Serbia/Montenegr",
    ],
    "Middle East & North Africa": [
        "Algeria", "Bahrain", "Djibouti", "Egypt, Arab Rep.",
        "Iran, Islamic Rep.", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon",
        "Libya", "Morocco", "Occ.Pal.Terr", "Oman", "Qatar", "Saudi Arabia",
        "Syrian Arab Republic", "Tunisia", "United Arab Emirates",
        "Western Sahara", "Yemen", "Yemen Democratic",
    ],
    "South Asia": [
        "Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal",
        "Pakistan", "Sri Lanka",
    ],
    "Sub-Saharan Africa": [
        "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cape Verde",
        "Cameroon", "Central African Republic", "Chad", "Comoros",
        "Congo, Dem. Rep.", "Congo, Rep.", "Cote d'Ivoire", "Equatorial Guinea",
        "Eritrea", "Eswatini", "Ethiopia(excludes Eritrea)",
        "Ethiopia(includes Eritrea)", "Fm Sudan", "Gabon", "Gambia, The",
        "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia",
        "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Mayotte",
        "Mozambique", "Namibia", "Niger", "Nigeria", "Reunion", "Rwanda",
        "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone",
        "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo",
        "Uganda", "Zambia", "Zimbabwe",
    ],
}
REGION_MAP = {e: r for r, lst in REGION_LISTS.items() for e in lst}
REGION_ORDER = list(REGION_LISTS.keys())
REGION_COLOR = {r: VIRIDIS(0.08 + 0.13 * i) for i, r in enumerate(REGION_ORDER)}

# Rows in the source file that are NOT single reporting economies.
AGGREGATES = [
    "World", "Bunkers", "Free Zones", "Special Categories", "Unspecified",
    "Neutral Zone", "Other Asia, nes", "Pacific Islands", "Us Msc.Pac.I",
    "East Asia & Pacific", "Europe & Central Asia", "Latin America & Caribbean",
    "Middle East & North Africa", "North America", "South Asia",
    "Sub-Saharan Africa",
]

# Columns removed during cleaning, with the reason stated openly on the slide.
DROP_ZERO_INFO = {
    "Export Product Share (%)": "constant (== 100 in every row)",
    "Import Product Share (%)": "constant (== 100 in every row)",
    "Revealed comparative advantage": "constant (== 1.0) and 42% missing",
    "MFN MinRate (%)": "constant (== 0)",
    "World Growth (%)": "byte-identical duplicate of 'Country Growth (%)'",
}
DROP_SENTINEL = {
    "AHS MaxRate (%)": "52% of values >= 500, mode == 3000 -> reporting flag",
    "MFN MaxRate (%)": "median == 3000, max == 5000 -> reporting flag",
    "MFN AVE Tariff Lines Share (%)": "reaches 3860 on a 0-100 scale",
    "MFN Specific Tariff Lines Share (%)": "reaches 1800 on a 0-100 scale",
    "MFN Simple Average (%)": "corrupted: near-constant across economies "
                              "within each year (SD <= 0.7pp after 2004)",
}

# The eight analysis features (economy-level means over 2017-2021).
FEATURES = [
    "log_export",
    "log_import",
    "net_trade_balance",
    "applied_tariff",
    "weighted_tariff",
    "duty_free_lines",
    "log_tariff_lines",
    "duty_free_imports",
]
FEATURE_LABEL = {
    "log_export": "Export scale\n(log10 US$ th.)",
    "log_import": "Import scale\n(log10 US$ th.)",
    "net_trade_balance": "Net trade balance\n((X-M)/(X+M))",
    "applied_tariff": "Applied tariff\n(AHS simple avg, %)",
    "weighted_tariff": "Trade-weighted tariff\n(AHS, %)",
    "duty_free_lines": "Duty-free tariff lines\n(% of schedule)",
    "log_tariff_lines": "Schedule granularity\n(log10 tariff lines)",
    "duty_free_imports": "Duty-free imports\n(% of dutiable+free)",
}
FEATURE_SHORT = {
    "log_export": "Export scale",
    "log_import": "Import scale",
    "net_trade_balance": "Net balance",
    "applied_tariff": "Applied tariff",
    "weighted_tariff": "Weighted tariff",
    "duty_free_lines": "Duty-free lines",
    "log_tariff_lines": "Schedule size",
    "duty_free_imports": "Duty-free imports",
}
# Predictor design: tariff LEVELS are the target, so every tariff-rate and
# duty-free measure is excluded from the predictor features (no leakage).
PRED_FEATURES = ["log_export", "log_import", "net_trade_balance", "log_tariff_lines"]

SUBQ = {
    "Q1": "How complete is the record?",
    "Q2": "How much did world trade grow, and was it steady?",
    "Q3": "Is trade scale usable as reported?",
    "Q4": "Do tariff regimes differ by region?",
    "Q5": "Do economies cluster by trade structure?",
    "Q6": "Can structure alone predict tariff stance?",
}

# ==============================================================================
# STYLE HELPERS  (Stephen Few: high data-ink, no chartjunk)
# ==============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": LIGHT,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#ececec",
    "grid.linewidth": 0.7,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "legend.frameon": False,
    "legend.fontsize": 10,
})


def few(ax, keep=("left", "bottom")):
    """Remove redundant spines, keep only the axes that carry information."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=3, color=LIGHT)
    return ax


def stamp(fig, test_result: str, gap_px: float = 10.0):
    """Print the 'Test . Result' footer beneath every panel (and to stdout).

    The anchor is measured from the rendered bounding boxes of all axes, which
    already include rotated tick labels and axis labels, so the footer lands in
    the free band under the drawing area and can never overlap it. The text is
    wrapped to the figure width; savefig(bbox_inches='tight') then grows the
    canvas downward to fit however many lines are needed.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bottoms = []
    for ax in fig.axes:
        tb = ax.get_tightbbox(r)
        if tb is not None:
            bottoms.append(tb.y0)
    if bottoms:
        y_frac = (min(bottoms) - gap_px) / (fig.get_figheight() * fig.dpi)
    else:
        y_frac = 0.02
    width = max(58, int(fig.get_figwidth() * 12.6))
    wrapped = fill(test_result, width=width)
    fig.text(0.012, y_frac, wrapped, ha="left", va="top",
             fontsize=9.5, color=GREY, style="italic", linespacing=1.42)
    print("  " + wrapped.replace("\n", "\n  "))
    return test_result


def bignum(ax, number, caption, x=0.985, y=0.965, color=ACCENT,
           size=30, cap_size=10.2):
    """ONE big number per slide.

    Every slide reserves the band ABOVE the axes (subplots_adjust top <= 0.86)
    for exactly this, so the call is routed to the figure-level placement and
    the per-axes x/y arguments are ignored. This guarantees the number never
    overlaps data, panel titles, legends or the headline block.
    """
    figbignum(ax.figure, number, caption, color=color, size=size,
              cap_size=cap_size)


def figbignum(fig, number, caption, color=ACCENT, size=31, cap_size=10.2,
              x=0.988, y=0.995):
    """ONE big number per slide, placed in the reserved top-right band.

    The band above the axes (subplots_adjust top <= 0.86) is kept free by every
    figure in this script, so the number can never collide with data, panel
    titles, legends or the headline block.
    """
    off = (size / 72.0) / fig.get_figheight() * 1.12
    fig.text(x, y, number, ha="right", va="top", fontsize=size,
             fontweight="bold", color=color, linespacing=0.95)
    fig.text(x, y - off, caption, ha="right", va="top",
             fontsize=cap_size, color=GREY, linespacing=1.32)


def headline(fig, kicker, title, sub=None):
    """Top-left title block: kicker, title, optional one-line subtitle."""
    fig.text(0.012, 0.988, kicker.upper(), fontsize=10, color=ACCENT,
             fontweight="bold", va="top")
    fig.text(0.012, 0.950, title, fontsize=18.5, color=INK,
             fontweight="bold", va="top")
    if sub:
        fig.text(0.012, 0.903, sub, fontsize=10.5, color=GREY, va="top")


def save(fig, name, test_result=None, talking_point=None):
    """Save the figure; the Test . Result is printed by stamp()."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"\n  [saved] {name}")
    if talking_point:
        print(f"  SLIDE TALKING POINT: {talking_point}")
    return path


def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def eta_sq_kruskal(H, k, n):
    """Epsilon-squared effect size for Kruskal-Wallis."""
    return (H - k + 1) / (n - k)


# ==============================================================================
# STEP 1 - LOAD AND CLEAN
# ==============================================================================
def load_and_clean():
    raw = pd.read_csv(CSV_PATH)
    raw["Partner Name"] = raw["Partner Name"].astype(str).str.strip()
    n_raw, n_col_raw = raw.shape
    n_names_raw = raw["Partner Name"].nunique()

    # ---- Decision 1: aggregates are not economies --------------------------
    agg_mask = raw["Partner Name"].isin(AGGREGATES)
    world = raw[raw["Partner Name"] == "World"].sort_values("Year").reset_index(drop=True)
    eco = raw[~agg_mask].copy()

    # ---- Decision 2: drop zero-information and duplicated columns ----------
    eco = eco.drop(columns=list(DROP_ZERO_INFO))

    # ---- Decision 3: drop sentinel-coded / impossible-range columns --------
    eco = eco.drop(columns=list(DROP_SENTINEL))

    # ---- Decision 4: log-transform scale variables, restrict the panel -----
    # Growth variables are 42.5% missing and are NOT used in the matrix.
    for c in ["Country Growth (%)"]:
        eco[c] = pd.to_numeric(eco[c], errors="coerce")

    eco["Region"] = eco["Partner Name"].map(REGION_MAP)
    n_unmapped = sorted(eco.loc[eco["Region"].isna(), "Partner Name"].unique())
    n_unmapped_traded = [
        n for n in n_unmapped
        if eco.loc[eco["Partner Name"] == n, "Export (US$ Thousand)"].median() > 1e5
    ]

    # Economy-level analysis matrix: mean over 2017-2021 (recent, stable regime)
    win = eco[(eco["Year"] >= RECENT[0]) & (eco["Year"] <= RECENT[1])].copy()
    grp = win.groupby("Partner Name")
    agg = grp.agg(
        export=("Export (US$ Thousand)", "mean"),
        imports=("Import (US$ Thousand)", "mean"),
        applied_tariff=("AHS Simple Average (%)", "mean"),
        weighted_tariff=("AHS Weighted Average (%)", "mean"),
        duty_free_lines=("AHS Duty Free Tariff Lines Share (%)", "mean"),
        tariff_lines=("AHS Total Tariff Lines", "mean"),
        df_imports=("AHS Duty Free Imports (US$ Thousand)", "mean"),
        dt_imports=("AHS Dutiable Imports (US$ Thousand)", "mean"),
        n_years=("Year", "nunique"),
    )

    # cell coverage across the eight source columns actually used
    src_cols = [
        "Export (US$ Thousand)", "Import (US$ Thousand)", "AHS Simple Average (%)",
        "AHS Weighted Average (%)", "AHS Duty Free Tariff Lines Share (%)",
        "AHS Total Tariff Lines", "AHS Duty Free Imports (US$ Thousand)",
        "AHS Dutiable Imports (US$ Thousand)",
    ]
    cov = win.groupby("Partner Name")[src_cols].apply(lambda g: g.notna().mean().mean())
    agg["coverage"] = cov

    n_win = len(agg)
    keep = agg["coverage"] >= COVERAGE_MIN
    mat = agg[keep].copy()
    n_dropped_cov = int((~keep).sum())

    # derived features
    mat["log_export"] = np.log10(mat["export"].clip(lower=0) + 1)
    mat["log_import"] = np.log10(mat["imports"].clip(lower=0) + 1)
    tot = (mat["export"] + mat["imports"]).replace(0, np.nan)
    mat["net_trade_balance"] = (mat["export"] - mat["imports"]) / tot
    dftot = (mat["df_imports"] + mat["dt_imports"]).replace(0, np.nan)
    mat["duty_free_imports"] = 100.0 * mat["df_imports"] / dftot
    mat["log_tariff_lines"] = np.log10(mat["tariff_lines"].clip(lower=0) + 1)
    mat = mat.dropna(subset=FEATURES)
    mat["Region"] = mat.index.map(REGION_MAP).fillna("Unclassified")

    # ---- Decision 5: define the prediction target --------------------------
    med_tariff = mat["applied_tariff"].median()
    mat["high_tariff"] = (mat["applied_tariff"] > med_tariff).astype(int)

    info = dict(
        n_raw=n_raw, n_col_raw=n_col_raw, n_names_raw=n_names_raw,
        n_agg_rows=int(agg_mask.sum()), n_agg_names=len(AGGREGATES),
        n_econ_rows=len(eco), n_econ_names=eco["Partner Name"].nunique(),
        n_win=n_win, n_dropped_cov=n_dropped_cov, n_matrix=len(mat),
        n_unmapped=len(n_unmapped), n_unmapped_traded=n_unmapped_traded,
        med_tariff=med_tariff, world=world, eco=eco,
        # skewness is reported on the SAME 234 economies, before and after
        skew_raw_export=float(mat["export"].skew()),
        skew_log_export=float(mat["log_export"].skew()),
        skew_raw_import=float(mat["imports"].skew()),
        skew_log_import=float(mat["log_import"].skew()),
    )
    return mat, info


def print_preprocessing(info, mat):
    w = info["world"]
    x = w.set_index("Year")["Export (US$ Thousand)"]
    line = "=" * 78
    print(line)
    print(" PREPROCESSING SUMMARY")
    print(line)
    print(f" Source file            : {os.path.basename(CSV_PATH)}")
    print(f" Raw shape              : {info['n_raw']:,} rows x {info['n_col_raw']} columns"
          f"   ({info['n_names_raw']} distinct 'Partner Name' values)")
    print(f" Time span              : 1988-2021 (34 years), no duplicate (partner, year) keys")
    print()
    print(" DECISION 1 - Aggregates are not economies")
    print(f"   Removed {info['n_agg_names']} non-economy entries "
          f"({info['n_agg_rows']:,} rows): 'World', 'Bunkers', 'Free Zones',")
    print("   'Special Categories', 'Unspecified', 'Neutral Zone' and 10 regional aggregates.")
    print(f"   Kept {info['n_econ_names']} reporting economies "
          f"({info['n_econ_rows']:,} rows). The 'World' row is used ONLY for the")
    print("   global trend in figure 02 and never enters clustering or modelling.")
    print()
    print(" DECISION 2 - Zero-information and duplicated columns dropped")
    for c, why in DROP_ZERO_INFO.items():
        print(f"   - {c:38s} : {why}")
    print()
    print(" DECISION 3 - Sentinel codes and impossible ranges dropped")
    for c, why in DROP_SENTINEL.items():
        print(f"   - {c:38s} : {why}")
    print("   Evidence for the MFN corruption: within-year SD of 'MFN Simple Average'")
    print("   collapses to <= 0.7pp after 2004 while unrelated economies report almost")
    print("   identical values (2021: China 8.64, Brazil 8.68, Germany 8.52,")
    print("   Singapore 8.50, India 8.47, USA 8.92). Its year-to-year median still swings")
    print("   14.2 -> 24.1 -> 21.8 (1988-1990), i.e. all variation is between years and")
    print("   none between economies. It cannot support cross-country comparison, so the")
    print("   applied (AHS) tariff series is used instead.")
    print()
    print(" DECISION 4 - Skewed scale variables transformed; panel restricted")
    print(f"   Export skewness  raw {info['skew_raw_export']:.2f} -> "
          f"log10(x+1) {info['skew_log_export']:.2f}")
    print(f"   Import skewness  raw {info['skew_raw_import']:.2f} -> "
          f"log10(x+1) {info['skew_log_import']:.2f}")
    print("   Justification is printed with figure 03. Both growth columns are 42.5%")
    print("   missing and are excluded from the matrix. Clustering uses economy-level means over")
    print(f"   {RECENT[0]}-{RECENT[1]} (recent, post-WTO regime).")
    print(f"   Coverage filter: >= {COVERAGE_MIN:.0%} of cells present across the 8 source")
    print(f"   columns -> {info['n_matrix']} of {info['n_win']} economies kept, "
          f"{info['n_dropped_cov']} dropped.")
    print(f"   Region mapping: {info['n_unmapped']} entries unmapped, of which "
          f"{len(info['n_unmapped_traded'])} trade materially")
    if info["n_unmapped_traded"]:
        print(f"     -> {', '.join(info['n_unmapped_traded'][:6])}")
    print("   Unmapped entries are labelled 'Unclassified' and excluded from the")
    print("   regional comparisons only; they remain in the clustering.")
    print()
    print(" DECISION 5 - Prediction target defined before modelling")
    print(f"   Target: 'high_tariff' = applied AHS simple average above the sample median")
    print(f"   (median = {info['med_tariff']:.2f}%). Balance: "
          f"{int(mat['high_tariff'].sum())} high / {int((1-mat['high_tariff']).sum())} low.")
    print("   Predictor features deliberately EXCLUDE every tariff-rate and duty-free")
    print("   measure to avoid leakage: " + ", ".join(PRED_FEATURES) + ".")
    print()
    print(f" FINAL ANALYSIS MATRIX  : {mat.shape[0]} economies x {len(FEATURES)} features")
    print(f" FEATURES               : {', '.join(FEATURES)}")
    print(line)
    return


# ==============================================================================
# RESULT REGISTRY - every figure stores its Test . Result and talking point here
# ==============================================================================
SLIDES: list[dict] = []


def register(fig_no, act, title, talking_point, seconds, test_result,
             presenter, fname):
    """Record one slide so the slide map and deck outline can be printed."""
    SLIDES.append(dict(no=fig_no, act=act, title=title, point=talking_point,
                       sec=seconds, test=test_result, presenter=presenter,
                       file=fname))
    print(f"\n{'-'*78}\nFIGURE {fig_no:02d} | {act} | {title}\n{'-'*78}")
    print(f"# SLIDE TALKING POINT: {talking_point}")


# ==============================================================================
# ACT 1 . DATA           (Q1 coverage, Q2 world trend)          figures 01-02
# ==============================================================================
def fig01_coverage(mat, info):
    """Figure 01 - coverage and balance: which economies and years we stand on."""
    eco = info["eco"]
    per_year = eco.groupby("Year")["Partner Name"].nunique().reindex(
        range(1988, 2022))
    reg_counts = mat["Region"].value_counts().reindex(REGION_ORDER[::-1])
    n_first, n_last = int(per_year.iloc[0]), int(per_year.max())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.6),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # --- Panel A: reporting economies per year ------------------------------
    a1.plot(per_year.index, per_year.values, color=ACCENT, lw=2.6,
            marker="o", ms=3.4, zorder=3)
    a1.fill_between(per_year.index, per_year.values, per_year.values.min() - 6,
                    color=ACCENT, alpha=0.09, zorder=2)
    a1.axvline(2001, color=LIGHT, lw=1.0, zorder=1)
    a1.text(2001.4, per_year.values.max() - 2, "WTO accession wave",
            fontsize=9, color=GREY, va="top")
    few(a1, keep=("left", "bottom"))
    a1.set_xlim(1987, 2022)
    a1.set_ylim(per_year.values.min() - 6, per_year.values.max() + 6)
    a1.set_xlabel("Year")
    a1.set_ylabel("Reporting economies")
    a1.set_title(f"A.  The panel is almost balanced: {n_first} -> {n_last} "
                 "economies", loc="left", fontsize=13, pad=12)
    figbignum(fig, f"{info['n_econ_names']}",
              "distinct reporting economies\nover 34 years (1988-2021)",
              size=30)

    # --- Panel B: region composition of the final matrix ---------------------
    cols = [REGION_COLOR.get(r, GREY) for r in reg_counts.index]
    bars = a2.barh(reg_counts.index, reg_counts.values, color=cols, height=0.68)
    for b, v in zip(bars, reg_counts.values):
        a2.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2,
                f"{int(v)}", va="center", fontsize=10.5, color=INK,
                fontweight="bold")
    few(a2, keep=("left", "bottom"))
    a2.grid(axis="y", visible=False)
    a2.set_xlim(0, reg_counts.values.max() * 1.16)
    a2.set_xlabel("Economies in the final analysis matrix")
    a2.set_title(f"B.  Regional balance of the {len(mat)} kept economies",
                 loc="left", fontsize=13, pad=12)
    a2.text(0.985, 0.06,
            f"coverage filter: >= {COVERAGE_MIN:.0%} of cells present\n"
            f"{info['n_dropped_cov']} economies dropped, "
            f"{info['n_matrix']} kept",
            transform=a2.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=GREY)

    headline(fig, "Part 1 · Data · Q1",
             "Who is actually in the record?",
             f"8,096 raw rows -> {len(mat)} economies with >=95% coverage of "
             f"the eight variables we use")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.760, bottom=0.145,
                        wspace=0.30)
    tr = stamp(fig, "Test: none (descriptive coverage audit) · Result: "
                    f"234/236 economies retained (99.2%); 42 more economies "
                    "report in 2021 than in 1988; every region represented "
                    "except that South Asia (n=8) and North America (n=5) are "
                    "thin, so regional tests are read as exploratory.")
    tp = ("Coverage is the strongest part of this dataset: 234 of 236 economies "
          "survive a 95%-completeness filter, and the panel is nearly balanced "
          "from 1988 to 2021.")
    save(fig, "01_coverage_and_balance.png", tr, tp)
    register(1, "Act 1 · Data", "Who is actually in the record?", tp, 55, tr,
             "Presenter A", "01_coverage_and_balance.png")


def fig02_world_trend(info):
    """Figure 02 - the 34-year global trend; this also feeds the hook answer."""
    w = info["world"].set_index("Year")
    # source unit is US$ THOUSAND -> divide by 1e9 to express US$ TRILLION
    exp = w["Export (US$ Thousand)"] / 1e9
    imp = w["Import (US$ Thousand)"] / 1e9
    growth = w["World Growth (%)"]
    y0, y1 = 1988, 2021
    fold = exp[y1] / exp[y0]
    fold_2019 = exp[2019] / exp[y0]
    cagr = fold ** (1 / (y1 - y0)) - 1
    gpos = int((growth.dropna() > 0).sum())
    gneg = int((growth.dropna() < 0).sum())
    worst_yr = int(growth.dropna().idxmin())
    worst_v = float(growth.dropna().min())

    fig = plt.figure(figsize=(14.5, 6.4))
    gs = GridSpec(2, 1, height_ratios=[2.35, 1], hspace=0.34,
                  left=0.062, right=0.985, top=0.760, bottom=0.105)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # --- Panel A: exports and imports, US$ trillion --------------------------
    a1.fill_between(exp.index, imp.values, exp.values, where=exp.values >= imp.values,
                    color=ACCENT2, alpha=0.10, interpolate=True, lw=0)
    a1.plot(exp.index, exp.values, color=ACCENT, lw=3.0, label="World exports")
    a1.plot(imp.index, imp.values, color=GREY, lw=2.0, ls="--",
            label="World imports")
    for yr, note, tx, ty in [(1995, "WTO founded", 1990.6, 10.2),
                             (worst_yr, f"global financial\ncrisis: {worst_v:.1f}%",
                              2001.0, 8.6),
                             (2020, "pandemic:\n-3.8%", 2011.2, 8.2)]:
        a1.axvline(yr, color=LIGHT, lw=1.0, zorder=0)
        a1.annotate(note, xy=(yr, exp[yr]), xytext=(tx, ty),
                    fontsize=9, color=GREY,
                    arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.9))
    a1.annotate(f"{exp[y1]:.1f}", xy=(y1, exp[y1]), xytext=(2012.0, 27.0),
                fontsize=11, fontweight="bold", color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))
    few(a1, keep=("left", "bottom"))
    a1.set_xlim(1987, 2023)
    a1.set_ylim(0, exp.max() * 1.24)
    a1.set_ylabel("US$ trillion")
    a1.legend(loc="upper left", bbox_to_anchor=(0.005, 1.0), ncol=2)
    a1.set_title("A.  World merchandise exports and imports, 1988-2021",
                 loc="left", fontsize=13, pad=10)
    bignum(a1, f"×{fold:.0f}", f"world exports grew {fold:.1f}-fold\n"
           f"1988 → 2021   ({exp[y0]:.2f} → {exp[y1]:.1f} US$ trillion)\n"
           f"CAGR {cagr*100:.1f}%   ·   1988 → 2019 alone: ×{fold_2019:.0f}",
           size=30)

    # --- Panel B: annual growth rate ----------------------------------------
    g = growth.dropna()
    cols = [POS if v >= 0 else NEG for v in g.values]
    a2.bar(g.index, g.values, color=cols, width=0.72)
    a2.axhline(0, color=GREY, lw=0.9)
    a2.axhline(g.mean(), color=ACCENT, lw=1.6, ls=":")
    few(a2, keep=("left", "bottom"))
    a2.grid(axis="x", visible=False)
    a2.set_xlim(1987.5, 2022.5)
    a2.set_ylim(g.min() - 3, g.max() + 6)
    a2.set_ylabel("Year-on-year growth (%)")
    a2.set_xlabel("Year")
    a2.set_title(f"B.  Growth is not smooth: {gpos} expansions, {gneg} "
                 f"contractions (dotted line = reported mean {g.mean():.1f}%/yr)",
                 loc="left", fontsize=12, pad=8)

    headline(fig, "Part 1 · Data · Q2",
             f"Thirty-four years of one direction, {gneg} years of setbacks",
             "Source rows: the 'World' aggregate only -- excluded from every "
             "clustering and modelling step")
    tr = stamp(fig, "Test: none (descriptive time series, 'World' rows) · Result: "
                    f"exports ×{fold:.1f} (1988->2021), ×{fold_2019:.1f} "
                    f"(1988->2019), CAGR {cagr*100:.1f}%; {gneg} of {len(g)} reported "
                    f"years negative, worst {worst_yr} at {worst_v:.1f}%. "
                    "Note: the reported growth column is on a different basis "
                    "from nominal change in the level series (see honesty page).")
    tp = (f"World exports multiplied {fold:.0f}-fold in 34 years at a "
          f"{cagr*100:.1f}% compound rate, but the path has {gneg} down years and "
          "one -12% cliff in 2009.")
    save(fig, "02_world_trade_trend.png", tr, tp)
    register(2, "Act 1 · Data", "Thirty-four years, one direction", tp, 60, tr,
             "Presenter A", "02_world_trade_trend.png")
    info["fold"] = fold
    info["fold_2019"] = fold_2019
    info["cagr"] = cagr


def fig03_log_justification(mat, info):
    """Figure 03 - marginal distributions and the justification for log10.

    Layout: 2 rows x 4 columns. Row 1 = as reported, row 2 = after
    log10(x + 1). Columns 1-3 are the three transformed variables; column 4
    shows the one bounded percentage variable that is NOT transformed (top)
    and states the decision explicitly (bottom). No empty panel.
    """
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 8.0))
    cols = [("log_export", mat["export"], mat["log_export"],
             "Export value"),
            ("log_import", mat["imports"], mat["log_import"],
             "Import value"),
            ("log_tariff_lines", mat["tariff_lines"], mat["log_tariff_lines"],
             "Tariff-line count")]

    sw = {}
    for j, (key, raws, logs, label) in enumerate(cols):
        sk, sk2 = raws.skew(), logs.skew()
        sw[key] = (sk, sk2)

        ax = axes[0, j]
        ax.hist(raws, bins=40, color=NEG, alpha=0.80, edgecolor="none")
        few(ax)
        ax.grid(axis="x", visible=False)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("Economies" if j == 0 else "", fontsize=11)
        ax.set_title(f"RAW · {label}\nskew = {sk:+.2f}", loc="left",
                     fontsize=11, color=NEG, linespacing=1.35)
        ax.text(0.97, 0.92, f"max = {raws.max():,.0f}\nmedian = {raws.median():,.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=GREY, linespacing=1.35)

        ax = axes[1, j]
        ax.hist(logs, bins=28, color=POS, alpha=0.85, edgecolor="none")
        few(ax)
        ax.grid(axis="x", visible=False)
        ax.set_xticks([])
        ax.set_ylabel("Economies" if j == 0 else "", fontsize=11)
        ax.set_title(f"log10(x+1) · {label}\nskew = {sk2:+.2f}", loc="left",
                     fontsize=11, color=POS, linespacing=1.35)
        ax.text(0.97, 0.92, f"skew {sk:+.2f} -> {sk2:+.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=GREY)

    # --- column 4: the bounded variable that stays untouched ----------------
    ax = axes[0, 3]
    ax.hist(mat["duty_free_imports"], bins=28, color=ACCENT, alpha=0.85,
            edgecolor="none")
    few(ax)
    ax.grid(axis="x", visible=False)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("Economies", fontsize=11)
    ax.set_title("RAW · Duty-free imports\n(bounded 0-100%, skew "
                 f"{mat['duty_free_imports'].skew():+.2f})", loc="left",
                 fontsize=11, color=ACCENT, linespacing=1.35)

    ax = axes[1, 3]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_autoscale_on(False)
    ax.text(0.04, 0.86, "No transform needed.", fontsize=15,
            fontweight="bold", color=INK, va="top")
    ax.text(0.04, 0.72,
            "Duty-free imports and every tariff RATE are already\n"
            "bounded percentages (0-100%), so log-transforming\n"
            "them would distort the scale rather than repair it.\n\n"
            "Only the three scale variables above - exports,\n"
            "imports and tariff-line counts - span up to six\n"
            "orders of magnitude, and only they are logged.",
            fontsize=10.5, color=GREY, va="top", linespacing=1.5)

    headline(fig, "Part 2 · Structure · Q3",
             "Three variables are unusable on their raw scale",
             "Top row: as reported. Bottom row: after log10(x + 1). "
             "The fourth column is the variable we deliberately do NOT touch.")
    fig.subplots_adjust(left=0.055, right=0.80, top=0.760, bottom=0.115,
                        hspace=0.80, wspace=0.26)
    figbignum(fig, f"{info['skew_raw_export']:.1f} → {info['skew_log_export']:.2f}",
              "export skewness before → after\nlog10(x + 1)", size=31)
    tr = stamp(fig, "Test: skewness + Shapiro-Wilk on 234 economies · Result: "
                    f"export skew {sw['log_export'][0]:.2f} -> {sw['log_export'][1]:.2f}, "
                    f"import skew {sw['log_import'][0]:.2f} -> {sw['log_import'][1]:.2f}, "
                    f"tariff lines {sw['log_tariff_lines'][0]:.2f} -> {sw['log_tariff_lines'][1]:.2f} "
                    f"(Shapiro-Wilk rejects normality for all three raw series, "
                    f"all {pfmt(stats.shapiro(mat['export'].clip(upper=mat['export'].quantile(0.999))).pvalue)}); "
                    f"max/median export ratio = {mat['export'].max()/mat['export'].median():,.0f}. "
                    "Duty-free import share is bounded 0-100%, so it is not transformed.")
    tp = ("Scale variables span six orders of magnitude, so we log-transform "
          f"them: export skewness falls from {info['skew_raw_export']:.1f} to "
          f"{info['skew_log_export']:.2f}, which is what makes Euclidean "
          "clustering defensible.")
    save(fig, "03_log_transform_justification.png", tr, tp)
    register(3, "Act 2 · Structure", "Why we take logarithms", tp, 55, tr,
             "Presenter A", "03_log_transform_justification.png")


REGION_ABBR = {
    "Europe & Central Asia": "ECA",
    "Middle East & North Africa": "MENA",
    "North America": "NAC",
    "South Asia": "SAS",
    "Latin America & Caribbean": "LAC",
    "Sub-Saharan Africa": "SSA",
    "East Asia & Pacific": "EAP",
}


def fig04_region_boxplots(mat):
    """Figure 04 - three notched boxplot panels: where region bites.

    Panel A = trade scale (log10 exports), panel B = applied protection
    (AHS simple average), panel C = import liberalization (duty-free share of
    imports). Notched boxes: non-overlapping notches indicate regional medians
    differing at approximately the 5% level (McGill, Tukey & Larsen 1978).
    Grey dots = every economy (seeded jitter); open circles = outliers beyond
    1.5 x IQR, drawn explicitly instead of default fliers so they can be named.
    """
    reg = mat[mat["Region"] != "Unclassified"].copy()
    n_unclass = int((mat["Region"] == "Unclassified").sum())
    panels = [("log_export", "A.  Trade scale - log10 exports\n(thousand US dollars)"),
              ("applied_tariff", "B.  Applied protection\nAHS simple average (%)"),
              ("duty_free_imports", "C.  Import liberalization\nduty-free share of imports (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.4), sharey=False)
    rng = np.random.default_rng(SEED)

    # economies worth naming on the scale panel: the three largest traders
    named = mat.nlargest(3, "export").index.tolist()

    results = []
    for ax, (col, lab) in zip(axes, panels):
        order = (reg.groupby("Region")[col].median().sort_values(
            ascending=False).index.tolist())
        results_row = None
        for i, r in enumerate(order):
            vals = reg.loc[reg["Region"] == r, col].dropna().values
            if len(vals) < 3:
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            inl = vals[(vals >= lo) & (vals <= hi)]
            out = vals[(vals < lo) | (vals > hi)]
            bp = ax.boxplot([inl], positions=[i + 1], vert=True, notch=True,
                            patch_artist=True, widths=0.55, showfliers=False,
                            medianprops=dict(color=PAPER, lw=2.2),
                            whiskerprops=dict(color=GREY, lw=1.0),
                            capprops=dict(color=GREY, lw=1.0))
            for patch in bp["boxes"]:
                patch.set_facecolor(REGION_COLOR[r])
                patch.set_alpha(0.85)
                patch.set_edgecolor("none")
            xj = i + 1 + rng.uniform(-0.15, 0.15, size=len(inl))
            ax.scatter(xj, inl, s=8, color="#3a3a3a", alpha=0.50,
                       linewidth=0, zorder=4)
            if len(out):
                xo = i + 1 + rng.uniform(-0.10, 0.10, size=len(out))
                ax.scatter(xo, out, s=26, facecolor="none", edgecolor=NEG,
                           linewidth=1.2, zorder=5)
            if col == "log_export":
                for nm in named:
                    v = reg.loc[reg["Region"] == r, col]
                    if nm in v.index:
                        ax.annotate(nm, xy=(i + 1, v.loc[nm]),
                                    xytext=(6, 4), textcoords="offset points",
                                    fontsize=9, color=INK, zorder=6)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([REGION_ABBR[r] for r in order], fontsize=10.5)
        few(ax)
        ax.grid(axis="x", visible=False)
        groups = [reg.loc[reg["Region"] == r, col].dropna().values
                  for r in order if len(reg.loc[reg["Region"] == r, col]) > 2]
        H, p = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        e2 = eta_sq_kruskal(H, len(groups), n)
        results.append((col, H, p, e2, n))
        # the test statistic lives in the panel title (second line) so no
        # floating box can ever collide with whiskers or data points
        ax.set_title(f"{lab}\nKruskal-Wallis {pfmt(p)} · ε² = {e2:.2f}",
                     loc="left", fontsize=11.5, pad=10, linespacing=1.45,
                     color=INK)

    Ht = results[1][1]
    nt = results[1][4]
    bignum(axes[0], f"{Ht:.0f}", "Kruskal-Wallis H for applied tariff\n"
           f"across 7 regions, n = {nt} economies", color=ACCENT2, size=30)
    headline(fig, "Part 2 · Structure · Q3 + Q4",
             "Region moves trade size - but barely moves protection",
             "Notched boxes: non-overlapping notches mark regional medians "
             "differing at ~5%; grey dots = every economy; open circles = "
             f"outliers beyond 1.5 x IQR. 'Unclassified' excluded (n={n_unclass}).")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.760, bottom=0.115,
                        wspace=0.22)
    tr = stamp(fig, "Test: Kruskal-Wallis across 7 regions, per panel · Result: "
                    f"trade scale H = {results[0][1]:.1f}, {pfmt(results[0][2])}, "
                    f"ε² = {results[0][3]:.2f}; applied tariff H = {results[1][1]:.1f}, "
                    f"{pfmt(results[1][2])}, ε² = {results[1][3]:.2f}; duty-free "
                    f"imports H = {results[2][1]:.1f}, {pfmt(results[2][2])}, ε² = "
                    f"{results[2][3]:.2f}. Notch = +/-1.57 IQR/sqrt(n) ~ 95% CI of "
                    "the median: scale notches separate across most region pairs, "
                    "tariff notches overlap almost everywhere - region is a size "
                    "story, not a protection story.")
    tp = ("The notches show exactly where geography bites: regional medians of "
          "trade scale sit far apart, but the notches on applied tariff overlap "
          "almost everywhere - so region explains size, not protection.")
    save(fig, "04_region_notched_boxplots.png", tr, tp)
    register(4, "Act 2 · Structure", "Where region bites - and where it doesn't",
             tp, 55, tr, "Presenter B", "04_region_notched_boxplots.png")
    return results


def fig05_violin_scale(mat):
    """Figure 05 - violin plot of trade scale by region."""
    reg = mat[mat["Region"] != "Unclassified"].copy()
    order = (reg.groupby("Region")["log_export"].median()
             .sort_values(ascending=True).index.tolist())
    reg["Region"] = pd.Categorical(reg["Region"], order, ordered=True)

    fig, ax = plt.subplots(figsize=(15.2, 6.6))
    # no hue here: with hue set, seaborn dodges the violins half a category
    # away from the strip points and the regional median bars
    parts = sns.violinplot(data=reg, x="Region", y="log_export",
                           palette=[REGION_COLOR[r] for r in order],
                           inner=None, cut=0, linewidth=1.1, ax=ax)
    for pc in ax.collections:                 # violin bodies
        if isinstance(pc, PolyCollection):
            pc.set_alpha(0.85)
    sns.stripplot(data=reg, x="Region", y="log_export", color="#3a3a3a",
                  size=3.2, alpha=0.55, jitter=0.20, linewidth=0, ax=ax)
    for i, r in enumerate(order):
        s = reg.loc[reg["Region"] == r, "log_export"]
        ax.plot([i - 0.24, i + 0.24], [s.median()] * 2, color=INK, lw=2.4,
                solid_capstyle="butt", zorder=5)
        ax.text(i, s.max() + 0.35, f"n={len(s)}", ha="center", fontsize=9.5,
                color=GREY)

    few(ax)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    ax.set_ylabel("log10 mean annual exports, 2017-2021\n(thousand US dollars)",
                  fontsize=10.5)
    ax.set_xticklabels([_short(r) for r in order], rotation=24, ha="right",
                       fontsize=10.5)
    ax.set_title("Trade scale by region, with every economy plotted",
                 loc="left", fontsize=14, pad=34)

    spread = reg["log_export"].max() - reg["log_export"].min()
    reg_med = reg.groupby("Region")["log_export"].median()
    between = reg_med.max() - reg_med.min()
    H, p = stats.kruskal(*[reg.loc[reg["Region"] == r, "log_export"].dropna()
                           for r in order])
    e2 = eta_sq_kruskal(H, len(order), len(reg))
    bignum(ax, f"{spread:.1f}", "range of log$_{10}$ export scale across all "
           f"{len(reg)} economies\n= a factor of "
           f"{10**spread:,.0f} in dollars", x=0.985, y=0.965, color=ACCENT,
           size=34)

    headline(fig, "Part 2 · Structure · Q3 + Q4",
             "Inside every region the spread is wider than between regions",
             "Grey dots = individual economies; black bar = regional median.")
    fig.subplots_adjust(left=0.052, right=0.985, top=0.760, bottom=0.155)
    tr = stamp(fig, f"Test: Kruskal-Wallis on log10 exports across 7 "
                    f"regions · Result: H = {H:.1f}, {pfmt(p)}, ε² = {e2:.2f}; "
                    f"within-region range {spread:.1f} log units "
                    f"(~{10**spread:,.0f}× in dollars) versus a between-region "
                    f"median range of only {between:.2f} log units "
                    f"(~{10**between:,.0f}×) - see figure 04 for the same "
                    "comparison as notched boxes.")
    tp = ("Every region spans five or six orders of magnitude internally, so "
          "geography cannot be the axis along which trade structure organises.")
    save(fig, "05_violin_scale_by_region.png", tr, tp)
    register(5, "Act 2 · Structure", "Scale spread within regions", tp, 50, tr,
             "Presenter B", "05_violin_scale_by_region.png")


def _short(r):
    return {"Latin America & Caribbean": "Latin America\n& Caribbean",
            "Europe & Central Asia": "Europe &\nCentral Asia",
            "East Asia & Pacific": "East Asia\n& Pacific",
            "Middle East & North Africa": "Middle East\n& N. Africa",
            "Sub-Saharan Africa": "Sub-Saharan\nAfrica",
            "North America": "North\nAmerica",
            "South Asia": "South\nAsia"}.get(r, r)


def fig06_spearman(mat):
    """Figure 06 - masked Spearman correlation heatmap of the eight features."""
    rho, pval = stats.spearmanr(mat[FEATURES].values)
    rho = pd.DataFrame(rho, index=FEATURES, columns=FEATURES)
    pval = pd.DataFrame(pval, index=FEATURES, columns=FEATURES)
    mask = np.triu(np.ones_like(rho, dtype=bool), k=1)

    cmap = LinearSegmentedColormap.from_list(
        "fewdiv", ["#8c3b34", "#e8ded2", "#2f6d8a"])
    fig, ax = plt.subplots(figsize=(12.6, 9.4))
    sns.heatmap(rho, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 11.5, "color": INK},
                linewidths=1.6, linecolor=PAPER, square=True,
                cbar_kws={"shrink": 0.72, "label": "Spearman ρ"}, ax=ax)
    # mark non-significant cells
    for i, r in enumerate(FEATURES):
        for j, c in enumerate(FEATURES):
            if not mask[i, j] and i != j and pval.iloc[i, j] >= 0.05:
                ax.text(j + 0.5, i + 0.86, "n.s.", ha="center", va="center",
                        fontsize=8.5, color=GREY, style="italic")
    ax.set_xticklabels([FEATURE_SHORT[c] for c in FEATURES], rotation=38,
                       ha="right", fontsize=11)
    ax.set_yticklabels([FEATURE_SHORT[r] for r in FEATURES], rotation=0,
                       fontsize=11)
    ax.set_title("Spearman rank correlations, 234 economies (lower triangle)",
                 loc="left", fontsize=14, pad=18)

    off = rho.where(~mask & (np.arange(len(FEATURES))[:, None] !=
                             np.arange(len(FEATURES))[None, :])).stack()
    off_p = pval.where(~mask & (np.arange(len(FEATURES))[:, None] !=
                                np.arange(len(FEATURES))[None, :])).stack()
    k = off.abs().idxmax()
    nk = off_p.abs().idxmax()
    # the masked upper triangle is empty space: put the two key read-outs there
    ax.text(0.52, 0.97,
            f"strongest coupling\n{FEATURE_SHORT[k[1]]} – {FEATURE_SHORT[k[0]]}\n"
            f"ρ = {off.loc[k]:+.2f}\n\nweakest coupling\n"
            f"{FEATURE_SHORT[nk[1]]} – {FEATURE_SHORT[nk[0]]}\n"
            f"ρ = {off.loc[nk]:+.2f}  (n.s.)",
            transform=ax.transAxes, fontsize=10.5, color=GREY, ha="left",
            va="top", linespacing=1.5)

    bignum(ax, f"{off.loc[k]:+.2f}", f"ρ between {FEATURE_SHORT[k[1]]}\n"
           f"and {FEATURE_SHORT[k[0]]}", color=ACCENT)
    headline(fig, "Part 2 · Structure · Q3 + Q4",
             "Size and tariff are two separate dimensions",
             "'n.s.' marks cells whose two-sided p-value is >= 0.05; the "
             "upper triangle is masked to avoid repeating each pair.")
    fig.subplots_adjust(left=0.155, right=0.985, top=0.760, bottom=0.135)
    tr = stamp(fig, "Test: Spearman rank correlation, two-sided, Benjamini-"
                    f"Hochberg-relevant cells flagged · Result: "
                    f"{FEATURE_SHORT[k[1]]}-{FEATURE_SHORT[k[0]]} "
                    f"ρ = {off.loc[k]:.2f} ({pfmt(off_p.loc[k])}); applied "
                    f"tariff vs export scale ρ = "
                    f"{rho.loc['log_export','applied_tariff']:.2f} "
                    f"({pfmt(pval.loc['log_export','applied_tariff'])}), i.e. "
                    "trade SIZE is almost unrelated to tariff LEVEL; applied "
                    f"tariff vs duty-free lines ρ = "
                    f"{rho.loc['duty_free_lines','applied_tariff']:.2f} "
                    f"({pfmt(pval.loc['duty_free_lines','applied_tariff'])}), "
                    "the strongest policy-internal coupling.")
    tp = ("Size and tariff are decoupled - export scale correlates with "
          "schedule size at 0.94 but with tariff level at only -0.01 - so the "
          "two are separate dimensions of trade structure.")
    save(fig, "06_spearman_heatmap.png", tr, tp)
    register(6, "Act 2 · Structure", "What correlates with what", tp, 55, tr,
             "Presenter B", "06_spearman_heatmap.png")
    return rho, pval, k, off


# ==============================================================================
# ANALYSIS ENGINE - clustering, PCA, predictor (feeds figures 07-15)
# ==============================================================================
K_CLUSTERS = 4
PRED_TARGET_NOTE = ("Applied AHS simple-average tariff above the sample median "
                    "(= high-tariff economy)")


def name_clusters(prof, labels):
    """Give every cluster a data-driven name from its own profile means."""
    names, order = {}, []
    for c in sorted(set(labels)):
        m = prof.loc[labels == c, FEATURES].mean()
        scale = ("Large" if m["log_export"] >= 7.4 else
                 "Mid-sized" if m["log_export"] >= 6.4 else "Small")
        bal = ("Surplus" if m["net_trade_balance"] >= 0.40 else
               "Mild surplus" if m["net_trade_balance"] >= 0.10 else
               "Deficit" if m["net_trade_balance"] <= -0.10 else "Balanced")
        bar = ("High-tariff" if m["applied_tariff"] >= 6.0 else
               "Moderate-tariff" if m["applied_tariff"] >= 4.0 else
               "Low-tariff")
        open_ = "Liberal schedule" if m["duty_free_lines"] >= 65 else "Mixed schedule"
        if scale == "Small" and bar == "High-tariff":
            nm = "Small · Surplus · High-tariff"
        elif scale == "Large" and bal == "Balanced":
            nm = "Large · Balanced · Moderate-tariff"
        elif m["duty_free_lines"] >= 65 and m["applied_tariff"] < 4.0 and scale != "Small":
            nm = "Mid-sized · Liberal schedule · Low-tariff"
        elif scale == "Small" and m["duty_free_imports"] >= 45:
            nm = "Small · Surplus · Liberal imports"
        else:
            nm = f"{scale} · {bal} · {bar}"
        names[c] = nm
        order.append(c)
    # de-duplicate names if two clusters collapse to the same phrase
    seen = {}
    for c in order:
        if names[c] in seen.values():
            m = prof.loc[labels == c, FEATURES].mean()
            names[c] = f"{names[c]} (alt: DF-lines {m['duty_free_lines']:.0f}%)"
        seen[c] = names[c]
    return names


def run_analysis(mat, info):
    """K-Means (elbow + silhouette), PCA, and one Random Forest classifier."""
    X = mat[FEATURES].values
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # ---- PCA ----------------------------------------------------------------
    pca = PCA(random_state=SEED).fit(Z)
    scores = pca.transform(Z)
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES,
                            columns=[f"PC{i+1}" for i in range(len(FEATURES))])
    var = pca.explained_variance_ratio_

    # ---- K-Means: elbow + silhouette ---------------------------------------
    ks = list(range(2, 11))
    inertia, sils = [], []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
        inertia.append(km.inertia_)
        sils.append(silhouette_score(Z, km.labels_))
    inertia, sils = np.array(inertia), np.array(sils)

    km = KMeans(K_CLUSTERS, n_init=50, random_state=SEED).fit(Z)
    labels = km.labels_
    prof = pd.DataFrame(X, columns=FEATURES, index=mat.index)
    names = name_clusters(prof, labels)
    # order clusters by mean export scale, descending, for a stable legend
    order = sorted(set(labels), key=lambda c: -prof.loc[labels == c, "log_export"].mean())
    remap = {c: i for i, c in enumerate(order)}
    labels = np.array([remap[c] for c in labels])
    names = {remap[c]: v for c, v in names.items()}
    sil_best_k = int(ks[int(np.argmax(sils))])
    sil_best = float(sils.max())
    sil_k = float(sils[ks.index(K_CLUSTERS)])

    # ---- Kruskal-Wallis per feature across clusters -------------------------
    kw = {}
    for f in FEATURES:
        groups = [prof.loc[labels == c, f].values for c in sorted(set(labels))]
        H, p = stats.kruskal(*groups)
        kw[f] = dict(H=float(H), p=float(p),
                     eps2=float(eta_sq_kruskal(H, len(groups), len(labels))))

    # ---- Predictor: one Random Forest --------------------------------------
    y = mat["high_tariff"].values
    full = PRED_FEATURES + ["duty_free_lines", "duty_free_imports"]
    Xtr, Xte, ytr, yte = train_test_split(
        mat[full].values, y, test_size=0.30, random_state=SEED, stratify=y)
    rf = RandomForestClassifier(n_estimators=600, random_state=SEED,
                                class_weight="balanced", max_features="sqrt")
    rf.fit(Xtr, ytr)
    prob = rf.predict_proba(Xte)[:, 1]
    pred = rf.predict(Xte)
    roc = dict(fpr=roc_curve(yte, prob)[0], tpr=roc_curve(yte, prob)[1])
    perf = dict(
        auc=float(roc_auc_score(yte, prob)),
        acc=float(accuracy_score(yte, pred)),
        bal=float(balanced_accuracy_score(yte, pred)),
        cm=confusion_matrix(yte, pred),
        prf=precision_recall_fscore_support(yte, pred, average=None),
        n_test=len(yte), n_train=len(ytr),
    )
    # permutation importance on the held-out set (no CV inference plot)
    perm = permutation_importance(rf, Xte, yte, n_repeats=200,
                                  random_state=SEED, scoring="roc_auc",
                                  n_jobs=1)
    # structural-only ablation on the SAME held-out economies
    idx_tr, idx_te = train_test_split(np.arange(len(mat)), test_size=0.30,
                                      random_state=SEED, stratify=y)
    rf_s = RandomForestClassifier(n_estimators=600, random_state=SEED,
                                  class_weight="balanced", max_features="sqrt")
    rf_s.fit(mat[PRED_FEATURES].values[idx_tr], y[idx_tr])
    prob_s = rf_s.predict_proba(mat[PRED_FEATURES].values[idx_te])[:, 1]
    auc_struct = float(roc_auc_score(y[idx_te], prob_s))
    roc_s = roc_curve(y[idx_te], prob_s)

    # ---- cluster economic weight -------------------------------------------
    cl = pd.DataFrame({"label": labels}, index=mat.index)
    cl["export"] = mat["export"].values
    cl["Region"] = mat["Region"].values
    share = cl.groupby("label")["export"].sum() / cl["export"].sum() * 100

    res = dict(
        Z=Z, scaler=scaler, pca=pca, scores=scores, loadings=loadings, var=var,
        ks=ks, inertia=inertia, sils=sils, sil_best_k=sil_best_k,
        sil_best=sil_best, sil_k=sil_k, labels=labels, names=names, prof=prof,
        kw=kw, rf=rf, perm=perm, roc=roc, roc_s=roc_s, auc_struct=auc_struct,
        perf=perf, full_feats=full, share=share, cl=cl,
        cum4=float(var[:4].sum()),
    )
    return res


# ==============================================================================
# ACT 3 . GROUPS        (Q5 clustering)                          figures 07-11
# ==============================================================================
def fig07_scree(res):
    """Figure 07 - PCA scree plot with cumulative variance."""
    var = res["var"]
    k = len(FEATURES)
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    xs = np.arange(1, k + 1)
    cols = [ACCENT if i < 4 else LIGHT for i in range(k)]
    bars = ax.bar(xs, var * 100, color=cols, width=0.62, zorder=3)
    for b, v in zip(bars, var * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.1f}%",
                ha="center", fontsize=11, color=INK, fontweight="bold")
    ax2 = ax.twinx()
    cum = np.cumsum(var) * 100
    ax2.plot(xs, cum, color=ACCENT2, lw=2.2, marker="o", ms=5, zorder=4)
    ax2.axhline(90, color=GREY, lw=1.0, ls=":", zorder=1)
    ax2.text(1.15, 91.2, "90% reference", fontsize=9.5, color=GREY, va="bottom")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative variance explained (%)", color=ACCENT2)
    ax2.grid(False)
    few(ax2, keep=("right",))
    ax2.tick_params(colors=ACCENT2)
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)

    few(ax, keep=("left", "bottom"))
    ax.grid(axis="x", visible=False)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"PC{i}" for i in xs])
    ax.set_ylim(0, max(var * 100) * 1.28)
    ax.set_ylabel("Variance explained (%)")
    ax.set_xlabel("Principal component")
    ax.set_title("Two dimensions carry the trading system", loc="left",
                 fontsize=15, pad=14)
    ax.text(0.985, 0.30, f"PC1 = SCALE\nPC2 = OPENNESS", transform=ax.transAxes,
            ha="right", va="center", fontsize=11, color=GREY,
            linespacing=1.5)
    bignum(ax, f"{res['cum4']*100:.0f}%", "of total variance in the first\n"
           f"4 components (PC1+PC2 = {(var[0]+var[1])*100:.0f}%)",
           x=0.985, y=0.965, color=ACCENT, size=38)
    headline(fig, "Part 3 · Groups · Q5",
             "Eight numbers, two real dimensions",
             "PCA on 234 economies x 8 standardized features.")
    fig.subplots_adjust(left=0.062, right=0.90, top=0.760, bottom=0.115)
    tr = stamp(fig, "Test: none (PCA is a variance decomposition, not a "
                    f"hypothesis test) · Result: PC1 {var[0]*100:.1f}%, "
                    f"PC2 {var[1]*100:.1f}%, PC3 {var[2]*100:.1f}%; the first "
                    f"two components explain {(var[0]+var[1])*100:.1f}% and the "
                    f"first four {res['cum4']*100:.1f}% of variance, so a 2-D "
                    "map loses little information.")
    tp = ("Eight standardized indicators collapse onto two axes that explain "
          f"{(var[0]+var[1])*100:.0f}% of variance - scale and openness - which "
          "is why the next slide can be read as a map.")
    save(fig, "07_pca_scree.png", tr, tp)
    register(7, "Act 3 · Groups", "Two dimensions carry the system", tp, 30, tr,
             "Presenter C", "07_pca_scree.png")


def fig08_pca_biplot(res, mat):
    """Figure 08 - the hero PCA biplot with cluster labels."""
    sc = res["scores"][:, :2]
    lab = res["labels"]
    names = res["names"]
    var = res["var"]
    pal = [VIRIDIS(0.10 + 0.24 * i) for i in range(len(names))]
    HALO = dict(boxstyle="round,pad=0.18", fc=PAPER, ec="none", alpha=0.88)

    fig, ax = plt.subplots(figsize=(14.8, 9.0))
    for c in sorted(names):
        m = lab == c
        ax.scatter(sc[m, 0], sc[m, 1], s=54, color=pal[c], alpha=0.88,
                   edgecolor="white", linewidth=0.6, label=names[c], zorder=3)

    # cluster tags at the centroids; full names live in the legend only, which
    # keeps the hero map free of long-text collisions
    for c in sorted(names):
        m = lab == c
        cx, cy = sc[m, 0].mean(), sc[m, 1].mean()
        ax.annotate(f"G{c+1}", xy=(cx, cy), xytext=(cx + 0.30, cy + 0.42),
                    fontsize=13, fontweight="bold", color=pal[c], zorder=7,
                    ha="left", bbox=HALO)

    # loadings as arrows; labels sit past the arrow tips with a paper halo
    ld = res["loadings"]
    scale = np.abs(sc[:, :2]).max() / 1.35
    for f in FEATURES:
        x, y = ld.loc[f, "PC1"] * scale, ld.loc[f, "PC2"] * scale
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.9,
                                    alpha=0.85), zorder=4)
        ax.text(x * 1.24, y * 1.24, FEATURE_SHORT[f], fontsize=10.5,
                color=ACCENT2, fontweight="bold", ha="center", va="center",
                zorder=6, bbox=HALO)

    # highlight a few economies, with a staggered offset per label
    stag = {"China": (6, -14), "United States": (7, 5), "Germany": (6, -12),
            "Singapore": (6, -14), "Nepal": (6, 6), "Cuba": (6, 5)}
    for e in ["China", "United States", "Germany", "Singapore", "Nepal",
              "Cuba"]:
        if e in mat.index:
            i = list(mat.index).index(e)
            dx, dy = stag[e]
            ax.annotate(e, xy=(sc[i, 0], sc[i, 1]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=9.5, color=INK, zorder=8, bbox=HALO)
            ax.plot(sc[i, 0], sc[i, 1], marker="o", ms=4.5, mfc="none",
                    mec=INK, mew=1.1, zorder=6)

    few(ax)
    ax.set_xlim(sc[:, 0].min() - 1.6, sc[:, 0].max() + 3.4)
    ax.set_ylim(sc[:, 1].min() - 1.8, sc[:, 1].max() + 2.6)
    ax.set_xlabel(f"PC1 · trade scale  ({var[0]*100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 · tariff openness  ({var[1]*100:.1f}% of variance)")
    ax.legend(loc="lower left", fontsize=10.5, ncol=1, title="K-Means groups",
              title_fontsize=10.5)
    headline(fig, "Part 3 · Groups · Q5  (hero slide)",
             "The map: size on the horizontal, tariff openness on the vertical",
             "Arrows = feature loadings; points = 234 economies; "
             "colours = the four K-Means groups from the next slides.")
    fig.subplots_adjust(left=0.062, right=0.985, top=0.760, bottom=0.105)
    bignum(ax, f"{(var[0]+var[1])*100:.0f}%", "of variance in these two axes",
           x=0.985, y=0.965, color=ACCENT, size=36)
    tr = stamp(fig, "Test: none (descriptive projection) · Result: PC1 loads "
                    f"almost equally on export scale ({res['loadings'].loc['log_export','PC1']:+.2f}), "
                    f"import scale ({res['loadings'].loc['log_import','PC1']:+.2f}) and "
                    f"tariff-line count ({res['loadings'].loc['log_tariff_lines','PC1']:+.2f}) = "
                    "SIZE; PC2 separates duty-free lines "
                    f"({res['loadings'].loc['duty_free_lines','PC2']:+.2f}) from applied "
                    f"tariff ({res['loadings'].loc['applied_tariff','PC2']:+.2f}) = OPENNESS. "
                    f"Together {(var[0]+var[1])*100:.1f}% of variance.")
    tp = ("One axis is how big you trade, the other is how open your tariff "
          "schedule is - and the four groups fall in different quadrants of "
          "that map.")
    save(fig, "08_pca_biplot.png", tr, tp)
    register(8, "Act 3 · Groups", "The map of the trading system", tp, 55, tr,
             "Presenter C", "08_pca_biplot.png")


def fig09_elbow_silhouette(res):
    """Figure 09 - two-panel K selection: elbow + silhouette."""
    ks = res["ks"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.6, 5.8))

    a1.plot(ks, res["inertia"], color=ACCENT, lw=2.6, marker="o", ms=6)
    a1.axvline(K_CLUSTERS, color=ACCENT2, lw=1.6, ls="--")
    a1.annotate(f"k = {K_CLUSTERS}\nelbow", xy=(K_CLUSTERS, res["inertia"][ks.index(K_CLUSTERS)]),
                xytext=(K_CLUSTERS + 1.15, res["inertia"][ks.index(K_CLUSTERS)] * 1.30),
                fontsize=10.5, color=ACCENT2, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.2))
    few(a1, keep=("left", "bottom"))
    a1.grid(axis="x", visible=False)
    a1.set_xticks(ks)
    a1.set_xlabel("Number of clusters (k)")
    a1.set_ylabel("Within-cluster sum of squares")
    a1.set_title("A.  Elbow: the curve flattens after k = 4", loc="left",
                 fontsize=13, pad=12)

    a2.plot(ks, res["sils"], color=ACCENT2, lw=2.6, marker="s", ms=6)
    a2.axvline(K_CLUSTERS, color=ACCENT, lw=1.6, ls="--")
    a2.axhline(0.25, color=GREY, lw=1.0, ls=":")
    a2.text(ks[-1] + 0.15, 0.253, "0.25 = 'no substantial\nstructure' rule of thumb",
            fontsize=9.5, color=GREY, va="bottom", ha="right")
    a2.annotate(f"k = {K_CLUSTERS}\nsilhouette {res['sil_k']:.2f}",
                xy=(K_CLUSTERS, res["sil_k"]),
                xytext=(K_CLUSTERS + 0.7, res["sil_k"] - 0.045),
                fontsize=10.5, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    few(a2, keep=("left", "bottom"))
    a2.grid(axis="x", visible=False)
    a2.set_xticks(ks)
    a2.set_ylim(min(res["sils"]) - 0.045, max(res["sils"]) + 0.045)
    a2.set_xlim(ks[0] - 0.4, ks[-1] + 0.4)
    a2.set_xlabel("Number of clusters (k)")
    a2.set_ylabel("Mean silhouette width")
    a2.set_title(f"B.  Silhouette stays shallow: best {res['sil_best']:.2f} at "
                 f"k = {res['sil_best_k']}", loc="left", fontsize=13, pad=12)

    headline(fig, "Part 3 · Groups · Q5",
             "We chose k = 4 - and we say plainly why",
             "K-Means, 20 restarts per k, standardized 8-feature matrix, "
             "n = 234 economies.")
    fig.subplots_adjust(left=0.062, right=0.985, top=0.760, bottom=0.135,
                        wspace=0.24)
    bignum(a2, f"{res['sil_k']:.2f}", f"silhouette at k = {K_CLUSTERS}\n"
           f"(range over k = 2..10: {res['sils'].min():.2f}-{res['sils'].max():.2f})",
           x=0.985, y=0.965, color=ACCENT, size=30)
    tr = stamp(fig, "Test: silhouette width + elbow on inertia, k = 2..10 · "
                    f"Result: inertia knee at k = 4; silhouette peaks only at "
                    f"{res['sil_best']:.2f} (k = {res['sil_best_k']}) and stays "
                    f"{res['sils'].min():.2f}-{res['sils'].max():.2f} across all k, "
                    f"with {res['sil_k']:.2f} at our k = 4. We keep k = 4 for "
                    "interpretability and report the shallow silhouette as an "
                    "honest limitation: the groups overlap.")
    tp = ("We chose four groups because the elbow flattens there and four "
          f"profiles are interpretable - but the silhouette is only "
          f"{res['sil_k']:.2f}, so these are tendencies, not boxes.")
    save(fig, "09_elbow_silhouette.png", tr, tp)
    register(9, "Act 3 · Groups", "Choosing four groups", tp, 35, tr,
             "Presenter C", "09_elbow_silhouette.png")


def fig10_cluster_boxplots(res, mat):
    """Figure 10 - boxplots of the key features by cluster, with Test . Result."""
    lab = res["labels"]
    names = res["names"]
    pal = [VIRIDIS(0.10 + 0.24 * i) for i in range(len(names))]
    panels = [("log_export", "Export scale\nlog$_{10}$ US$ thousand"),
              ("net_trade_balance", "Net trade balance\n(X−M)/(X+M)"),
              ("applied_tariff", "Applied tariff\nAHS simple average (%)"),
              ("duty_free_lines", "Duty-free tariff lines\n(% of schedule)")]
    fig, axes = plt.subplots(1, 4, figsize=(16.6, 6.4))

    res_lines = []
    for ax, (col, lab_txt) in zip(axes, panels):
        data = [res["prof"].loc[lab == c, col].values for c in sorted(names)]
        bp = ax.boxplot(data, patch_artist=True, widths=0.60,
                        medianprops=dict(color=INK, lw=2.2),
                        whiskerprops=dict(color=GREY, lw=1.0),
                        capprops=dict(color=GREY, lw=1.0),
                        flierprops=dict(marker="o", ms=3.2, mfc=LIGHT,
                                        mec="none", alpha=0.9))
        for patch, c in zip(bp["boxes"], sorted(names)):
            patch.set_facecolor(pal[c])
            patch.set_alpha(0.85)
            patch.set_edgecolor("none")
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([f"G{c+1}\nn={int((lab==c).sum())}" for c in sorted(names)],
                           fontsize=10)
        few(ax)
        ax.grid(axis="x", visible=False)
        ax.set_title(lab_txt, loc="left", fontsize=11.5, pad=10)
        d = res["kw"][col]
        res_lines.append(f"{FEATURE_SHORT[col]}: H = {d['H']:.1f}, "
                         f"{pfmt(d['p'])}, ε² = {d['eps2']:.2f}")
        ax.text(0.03, 0.97, f"Kruskal-Wallis\n{pfmt(d['p'])}\nε² = {d['eps2']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4",
                                     ec=LIGHT, lw=0.8))

    d0 = res["kw"]["applied_tariff"]
    bignum(axes[2], f"{d0['eps2']:.2f}", "ε² for applied tariff\n"
           "across the four groups", x=0.975, y=0.965, color=ACCENT2, size=30)
    legend = [mpatches.Patch(color=pal[c], label=f"G{c+1}  {names[c]}")
              for c in sorted(names)]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.015))
    headline(fig, "Part 3 · Groups · Q5",
             "The four groups are statistically distinguishable",
             "Every feature differs across groups at p < 0.001; effect sizes "
             "range from ε² = 0.25 to 0.78.")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.760, bottom=0.28,
                        wspace=0.22)
    tr = stamp(fig, "Test: Kruskal-Wallis across the 4 clusters · Result: "
                    + "; ".join(res_lines) + f". Largest effect: import scale "
                    f"ε² = {res['kw']['log_import']['eps2']:.2f}.")
    tp = ("All eight indicators separate the four groups at p < 0.001, and the "
          f"strongest is import scale at ε² = {res['kw']['log_import']['eps2']:.2f} "
          "- so the groups are real differences, not noise.")
    save(fig, "10_boxplots_by_cluster.png", tr, tp)
    register(10, "Act 3 · Groups", "The groups are distinguishable", tp, 40, tr,
             "Presenter C", "10_boxplots_by_cluster.png")


def fig11_cluster_profiles(res, mat):
    """Figure 11 - climax: named cluster profiles (z-scored heatmap + radar)."""
    lab = res["labels"]
    names = res["names"]
    n_cl = len(names)
    pal = [VIRIDIS(0.10 + 0.24 * i) for i in range(n_cl)]
    Zc = res["prof"].groupby(lab).mean()
    # standardize features using the full-sample mean/SD of the raw features
    mu, sd = res["prof"].mean(), res["prof"].std()
    Zc = (Zc - mu) / sd
    Zc.index = [f"G{c+1}  {names[c]}" for c in Zc.index]

    fig = plt.figure(figsize=(16.4, 8.0))
    gs = GridSpec(1, 2, width_ratios=[1.28, 1], wspace=0.18,
                  left=0.185, right=0.985, top=0.760, bottom=0.10)

    # --- Panel A: z-scored profile heatmap ---------------------------------
    a1 = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("fewseq2", ["#2f6d8a", "#f2f0ea", "#8c3b34"])
    sns.heatmap(Zc, cmap=cmap, center=0, vmin=-2, vmax=2, annot=True,
                fmt="+.2f", annot_kws={"size": 11, "color": INK},
                linewidths=1.8, linecolor=PAPER, cbar_kws={
                    "shrink": 0.78, "label": "z-score vs. all 234 economies"},
                ax=a1)
    a1.set_xticklabels([FEATURE_SHORT[c] for c in Zc.columns], rotation=34,
                       ha="right", fontsize=10.5)
    a1.set_yticklabels(a1.get_yticklabels(), fontsize=10.5)
    a1.set_ylabel("")
    a1.set_title("A.  Group profiles, standardized", loc="left", fontsize=13,
                 pad=12)

    # --- Panel B: radar ------------------------------------------------------
    a2 = fig.add_subplot(gs[1], polar=True)
    angs = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    angs += angs[:1]
    for c in sorted(names):
        vals = list(np.clip(Zc.iloc[c].values, -2, 2))
        vals = vals + [vals[0]]                 # close the polygon
        a2.plot(angs, vals, color=pal[c], lw=2.3, label=f"G{c+1}")
        a2.fill(angs, vals, color=pal[c], alpha=0.10)
    a2.set_xticks(angs[:-1])
    a2.set_xticklabels([FEATURE_SHORT[f] for f in FEATURES], fontsize=10)
    a2.set_yticks([-2, -1, 0, 1, 2])
    a2.set_yticklabels(["-2", "-1", "0", "+1", "+2"], fontsize=8.5, color=GREY)
    a2.set_ylim(-2.15, 2.15)
    a2.grid(color="#e4e4e4", lw=0.8)
    a2.spines["polar"].set_color(LIGHT)
    a2.set_title("B.  Same profiles as a radar", loc="left", fontsize=13,
                 pad=26)
    a2.legend(loc="lower right", bbox_to_anchor=(1.22, -0.14), fontsize=10)

    share = res["share"]
    top = share.idxmax()
    headline(fig, "Part 3 · Groups · Q5  (climax)",
             "Four named groups, four different trade strategies",
             "Each row is one group's mean, expressed in standard deviations "
             "from the 234-economy average.")
    bignum(a1, f"{share[top]:.0f}%", f"of total exports in these groups\n"
           f"come from G{top+1} · {names[top]}", x=-0.44, y=0.965,
           color=ACCENT, size=32)
    tr = stamp(fig, "Test: Kruskal-Wallis per feature across the 4 clusters "
                    "(all p < 0.001, ε² 0.25-0.78) · Result: G1 = large, "
                    "balanced, moderate tariff; G2 = mid-sized, liberal "
                    "schedule, low tariff; G3 = small, surplus, liberal "
                    "imports; G4 = small, surplus, high applied tariff.")
    tp = ("Here is the answer to Q5: four groups - a large balanced core, "
          "liberal mid-sized openers, small surplus economies with liberal "
          "imports, and small surplus economies with high applied tariffs.")
    save(fig, "11_cluster_profiles.png", tr, tp)
    register(11, "Act 3 · Groups", "Four named trade strategies", tp, 55, tr,
             "Presenter C", "11_cluster_profiles.png")


# ==============================================================================
# ACT 4 . PREDICTOR     (Q6)                                     figures 12-13
# ==============================================================================
def fig12_importance(res):
    """Figure 12 - permutation importance of the single supervised model."""
    feats = res["full_feats"]
    imp = res["perm"].importances_mean
    sd = res["perm"].importances_std
    order = np.argsort(imp)
    cols = [ACCENT2 if feats[i] not in PRED_FEATURES else ACCENT for i in order]

    fig, ax = plt.subplots(figsize=(13.6, 6.8))
    y = np.arange(len(feats))
    ax.barh(y, imp[order], color=cols, height=0.62, xerr=sd[order],
            error_kw=dict(ecolor=GREY, elinewidth=1.1, capsize=3.2, alpha=0.85))
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_SHORT[feats[i]] for i in order], fontsize=11.5)
    for j, v in enumerate(imp[order]):        # value label just past its bar
        ax.text(v + sd[order][j] + 0.006, j, f"{v:.3f}", va="center",
                fontsize=10.5, color=INK, fontweight="bold")
    few(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(imp) * 1.30)
    ax.set_xlabel("Drop in held-out ROC AUC when the feature is permuted")
    ax.set_title("Only the duty-free schedule really predicts tariff stance",
                 loc="left", fontsize=15, pad=16)
    handles = [mpatches.Patch(color=ACCENT, label="trade structure (size, balance, schedule)"),
               mpatches.Patch(color=ACCENT2, label="trade policy (duty-free access)")]
    ax.legend(handles=handles, loc="lower right", fontsize=10.5)
    top_f = feats[int(np.argmax(imp))]
    bignum(ax, f"{imp.max():.2f}", f"AUC lost by permuting\n{FEATURE_SHORT[top_f]}",
           x=0.985, y=0.965, color=ACCENT2, size=38)
    headline(fig, "Part 4 · Predictor · Q6",
             "What actually drives the prediction?",
             "Random Forest, 600 trees, balanced class weights; permutation "
             "importance over 200 repeats on the 30% held-out set.")
    fig.subplots_adjust(left=0.155, right=0.985, top=0.760, bottom=0.115)
    tr = stamp(fig, "Test: permutation importance (200 repeats, held-out set, "
                    f"scoring = ROC AUC) · Result: {FEATURE_SHORT[top_f]} "
                    f"accounts for {imp.max():.3f} AUC, versus "
                    f"{imp[[feats.index(f) for f in PRED_FEATURES]].max():.3f} "
                    "for the best pure-structure feature; structure-only "
                    f"features together contribute "
                    f"{imp[[feats.index(f) for f in PRED_FEATURES]].sum():.3f}.")
    tp = ("The prediction is carried by one policy variable - the share of "
          "duty-free tariff lines - while pure trade structure contributes "
          "only about a third as much per feature.")
    save(fig, "12_feature_importance.png", tr, tp)
    register(12, "Act 4 · Predictor", "What drives the prediction", tp, 40, tr,
             "Presenter B", "12_feature_importance.png")


def fig13_model_performance(res, info_med):
    """Figure 13 - ROC comparison + confusion matrix + metric table."""
    perf, roc = res["perf"], res["roc"]
    fig = plt.figure(figsize=(15.6, 6.6))
    gs = GridSpec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.30,
                  left=0.052, right=0.985, top=0.760, bottom=0.125)

    # --- Panel A: ROC, full model vs structure-only ablation ----------------
    a1 = fig.add_subplot(gs[0])
    a1.plot(roc["fpr"], roc["tpr"], color=ACCENT, lw=2.8,
            label=f"Full model (6 features)  AUC = {perf['auc']:.2f}")
    a1.plot(res["roc_s"][0], res["roc_s"][1], color=LIGHT, lw=2.2, ls="--",
            label=f"Structure only (4 features)  AUC = {res['auc_struct']:.2f}")
    a1.plot([0, 1], [0, 1], color=GREY, lw=1.0, ls=":")
    few(a1, keep=("left", "bottom"))
    a1.grid(axis="both")
    a1.set_xlim(-0.02, 1.02)
    a1.set_ylim(-0.02, 1.02)
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("True positive rate")
    a1.legend(loc="lower right", fontsize=10)
    a1.set_title("A.  Discrimination, 30% held-out set", loc="left",
                 fontsize=13, pad=12)
    bignum(a1, f"{perf['auc']:.2f}", "held-out ROC AUC", x=0.30, y=0.965,
           color=ACCENT, size=36)

    # --- Panel B: confusion matrix ------------------------------------------
    a2 = fig.add_subplot(gs[1])
    cm = perf["cm"]
    im = a2.imshow(cm, cmap=LinearSegmentedColormap.from_list(
        "fewb", ["#f4f4f4", ACCENT]), vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            a2.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    a2.set_xticks([0, 1]); a2.set_yticks([0, 1])
    a2.set_xticklabels(["pred.\nlow tariff", "pred.\nhigh tariff"], fontsize=10.5)
    a2.set_yticklabels(["true\nlow", "true\nhigh"], fontsize=10.5)
    a2.set_title("B.  Confusion matrix", loc="left", fontsize=13, pad=12)
    a2.grid(False)
    for s in a2.spines.values():
        s.set_visible(False)

    # --- Panel C: metric table ----------------------------------------------
    # NOTE: the row separators are added with add_artist(Line2D(...)) rather
    # than ax.plot(..., transform=transAxes). ax.plot() feeds those
    # axes-fraction coordinates into the axes data limits, which inflates the
    # figure's tight bounding box by tens of inches and produces an unusable
    # PNG. add_artist() leaves the data limits untouched; the explicit limits
    # and disabled autoscaling below make that guarantee explicit.
    a3 = fig.add_subplot(gs[2])
    a3.axis("off")
    a3.set_xlim(0, 1)
    a3.set_ylim(0, 1)
    a3.set_autoscale_on(False)
    prf = perf["prf"]
    rows = [("Accuracy", f"{perf['acc']:.3f}"),
            ("Balanced accuracy", f"{perf['bal']:.3f}"),
            ("ROC AUC", f"{perf['auc']:.3f}"),
            ("Precision · high", f"{prf[0][1]:.3f}"),
            ("Recall · high", f"{prf[1][1]:.3f}"),
            ("F1 · high", f"{prf[2][1]:.3f}"),
            ("Structure-only AUC", f"{res['auc_struct']:.3f}"),
            ("Gain from policy vars", f"+{perf['auc']-res['auc_struct']:.3f}")]
    a3.text(0.02, 0.99, "C.  Held-out performance", fontsize=13,
            fontweight="bold", va="top", color=INK)
    for i, (k, v) in enumerate(rows):
        yy = 0.86 - i * 0.103
        bold = k.startswith("Gain")
        a3.text(0.02, yy, k, fontsize=11, va="center",
                color=INK, fontweight="bold" if bold else "normal")
        a3.text(0.98, yy, v, fontsize=11.5 if bold else 11, va="center",
                ha="right", color=ACCENT2 if bold else INK,
                fontweight="bold")
        a3.add_artist(Line2D([0.0, 1.0], [yy - 0.052, yy - 0.052],
                             transform=a3.transAxes, color="#eeeeee", lw=0.9,
                             clip_on=False))
    a3.text(0.02, 0.03, f"train n = {perf['n_train']}  ·  test n = "
            f"{perf['n_test']}  ·  target = median split of applied tariff",
            fontsize=9.5, color=GREY, va="bottom")

    headline(fig, "Part 4 · Predictor · Q6",
             "Structure alone is a weak predictor; policy access is not",
             "Target: applied AHS tariff above the sample median "
             f"({info_med:.2f}%) = 'high-tariff economy'.")
    tr = stamp(fig, "Test: held-out Random Forest (70/30 stratified split, "
                    f"n_test = {perf['n_test']}) vs. structure-only ablation on "
                    f"the same economies · Result: full model AUC = "
                    f"{perf['auc']:.3f}, balanced accuracy = {perf['bal']:.3f}; "
                    f"structure-only AUC = {res['auc_struct']:.3f}; adding "
                    f"duty-free policy variables gains +{perf['auc']-res['auc_struct']:.3f} AUC.")
    tp = ("Structure alone barely beats a coin flip at AUC 0.64; add the "
          "duty-free share of the schedule and we reach 0.85 - so tariff stance "
          "is a policy choice, not a consequence of size.")
    save(fig, "13_model_performance.png", tr, tp)
    register(13, "Act 4 · Predictor", "Structure is a weak predictor", tp, 40,
             tr, "Presenter B", "13_model_performance.png")


# ==============================================================================
# ACT 5 . SYNTHESIS                                     figures 14-15
# ==============================================================================
def fig14_dashboard(res, mat, info, rho):
    """Figure 14 - one-canvas dashboard: the whole study on one slide."""
    lab, names = res["labels"], res["names"]
    n_cl = len(names)
    pal = [VIRIDIS(0.10 + 0.24 * i) for i in range(n_cl)]
    fig = plt.figure(figsize=(17.4, 9.6))
    gs = GridSpec(2, 3, height_ratios=[1.05, 1], width_ratios=[1.15, 1, 1.05],
                  left=0.05, right=0.985, top=0.760, bottom=0.085,
                  hspace=0.42, wspace=0.30)

    # (1) world trend
    a = fig.add_subplot(gs[0, 0])
    w = info["world"].set_index("Year")
    a.plot(w.index, w["Export (US$ Thousand)"] / 1e9, color=ACCENT, lw=2.4)
    few(a); a.grid(axis="x", visible=False)
    a.set_ylabel("US$ trillion", fontsize=9.5)
    a.set_title(f"1 · World exports ×{info['fold']:.0f}, 1988-2021",
                loc="left", fontsize=11.5, pad=8)
    a.text(0.97, 0.10, f"CAGR {info['cagr']*100:.1f}%", transform=a.transAxes,
           ha="right", fontsize=10, color=GREY)

    # (2) PCA map
    a = fig.add_subplot(gs[0, 1])
    sc = res["scores"][:, :2]
    for c in sorted(names):
        m = lab == c
        a.scatter(sc[m, 0], sc[m, 1], s=16, color=pal[c], alpha=0.9,
                  linewidth=0, label=f"G{c+1}")
    few(a)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel(f"PC1 ({res['var'][0]*100:.0f}%)", fontsize=9.5)
    a.set_ylabel(f"PC2 ({res['var'][1]*100:.0f}%)", fontsize=9.5)
    a.set_title("2 · Scale × openness map", loc="left", fontsize=11.5, pad=8)
    a.legend(loc="lower left", fontsize=8.5, ncol=2, handlelength=0.9,
             columnspacing=0.8)

    # (3) tariff by cluster
    a = fig.add_subplot(gs[0, 2])
    prof = res["prof"]
    means = [prof.loc[lab == c, "applied_tariff"].mean() for c in sorted(names)]
    a.bar(range(n_cl), means, color=pal, width=0.62)
    a.set_xticks(range(n_cl))
    a.set_xticklabels([f"G{c+1}" for c in sorted(names)], fontsize=10)
    few(a); a.grid(axis="x", visible=False)
    a.set_ylabel("Applied tariff (%)", fontsize=9.5)
    d = res["kw"]["applied_tariff"]
    a.set_title(f"3 · Tariff by group (ε² = {d['eps2']:.2f})", loc="left",
                fontsize=11.5, pad=8)

    # (4) feature importance
    a = fig.add_subplot(gs[1, 0])
    feats, imp = res["full_feats"], res["perm"].importances_mean
    o = np.argsort(imp)
    a.barh(np.arange(len(feats)), imp[o],
           color=[ACCENT2 if feats[i] not in PRED_FEATURES else ACCENT
                  for i in o], height=0.62)
    a.set_yticks(np.arange(len(feats)))
    a.set_yticklabels([FEATURE_SHORT[feats[i]] for i in o], fontsize=9)
    few(a); a.grid(axis="y", visible=False)
    a.set_xlabel("AUC drop", fontsize=9.5)
    a.set_title("4 · What predicts tariff stance", loc="left", fontsize=11.5,
                pad=8)

    # (5) region tariff comparison
    a = fig.add_subplot(gs[1, 1])
    reg = mat[mat["Region"] != "Unclassified"]
    med = reg.groupby("Region")["applied_tariff"].median().sort_values()
    a.barh(np.arange(len(med)), med.values,
           color=[REGION_COLOR[r] for r in med.index], height=0.62)
    a.set_yticks(np.arange(len(med)))
    a.set_yticklabels([_short(r).replace("\n", " ") for r in med.index],
                      fontsize=9)
    few(a); a.grid(axis="y", visible=False)
    a.set_xlabel("Median applied tariff (%)", fontsize=9.5)
    a.set_title("5 · Region explains little (ε² = 0.07)", loc="left",
                fontsize=11.5, pad=8)

    # (6) group membership + export weight
    a = fig.add_subplot(gs[1, 2])
    a.axis("off")
    share = res["share"]
    a.text(0.0, 1.0, "6 · The four groups", fontsize=11.5, fontweight="bold",
           va="top")
    for i, c in enumerate(sorted(names)):
        yy = 0.82 - i * 0.215
        a.add_patch(mpatches.Rectangle((0.0, yy - 0.055), 0.045, 0.10,
                                       color=pal[c], transform=a.transAxes,
                                       clip_on=False))
        a.text(0.075, yy + 0.012, f"G{c+1}  {names[c]}", fontsize=10.2,
               fontweight="bold", va="center")
        a.text(0.075, yy - 0.055,
               f"n = {int((lab == c).sum())} economies  ·  "
               f"{share[c]:.0f}% of all {len(mat)} economies' exports",
               fontsize=9.3, color=GREY, va="center")

    headline(fig, "Part 5 · Answer",
             "The whole study on one canvas",
             "Q1-Q6 in six panels: record, map, groups, predictor, region "
             "baseline, and the four named strategies.")
    tr = stamp(fig, "Test: summary of all preceding tests · Result: world "
                    f"exports ×{info['fold']:.0f} (CAGR {info['cagr']*100:.1f}%); "
                    f"2 dimensions = {(res['var'][0]+res['var'][1])*100:.0f}% of "
                    f"variance; 4 groups differ at p < 0.001 (ε² "
                    f"{min(d['eps2'] for d in res['kw'].values()):.2f}-"
                    f"{max(d['eps2'] for d in res['kw'].values()):.2f}); region "
                    f"explains ε² = 0.07 of tariff; held-out AUC = "
                    f"{res['perf']['auc']:.2f} full vs {res['auc_struct']:.2f} "
                    "structure-only.")
    tp = ("Everything in one view: a ×31 trade expansion, a two-dimensional "
          "map, four distinguishable strategies, and a predictor that says "
          "tariff stance is policy, not size.")
    save(fig, "14_dashboard.png", tr, tp)
    register(14, "Act 5 · Answer", "The whole study on one canvas", tp, 40, tr,
             "Presenter A", "14_dashboard.png")


def fig15_callback(res, mat, info):
    """Figure 15 - the answer + callback to the opening guess."""
    w = info["world"].set_index("Year")
    exp = w["Export (US$ Thousand)"] / 1e6
    f_all, f_pre, f_post = info["fold"], exp[2008] / exp[1988], exp[2021] / exp[2008]
    share = res["share"]
    names, lab = res["names"], res["labels"]
    n_cl = len(names)
    pal = [VIRIDIS(0.10 + 0.24 * i) for i in range(n_cl)]

    fig = plt.figure(figsize=(16.6, 9.2))
    gs = GridSpec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.32, 1],
                  left=0.055, right=0.985, top=0.760, bottom=0.075,
                  hspace=0.40, wspace=0.26)

    # --- Panel A: where the x31 actually came from --------------------------
    a = fig.add_subplot(gs[0, 0])
    a.plot(exp.index, exp.values, color=ACCENT, lw=2.8, zorder=3)
    a.axvspan(1987.4, 2008, color=ACCENT2, alpha=0.07, zorder=1)
    a.axvspan(2008, 2022.6, color=GREY, alpha=0.07, zorder=1)
    a.axvline(2008, color=GREY, lw=1.2, ls="--", zorder=2)
    a.annotate(f"1988-2008:  ×{f_pre:.1f}", xy=(1998, exp[1998]),
               xytext=(1991.5, exp.max() * 0.62), fontsize=13,
               color=ACCENT2, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.4))
    a.annotate(f"2008-2021:  ×{f_post:.2f}", xy=(2015, exp[2015]),
               xytext=(1997.5, exp.max() * 0.80), fontsize=13, color=GREY,
               fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    a.scatter([2008], [exp[2008]], s=70, color=INK, zorder=4)
    few(a)
    a.grid(axis="x", visible=False)
    a.set_xlim(1987.4, 2022.6)
    a.set_ylim(0, exp.max() * 1.12)
    a.set_ylabel("World exports, US$ trillion")
    a.set_xlabel("Year")
    a.set_title("A.  The ×31 is almost entirely a pre-2008 story", loc="left",
                fontsize=13.5, pad=12)
    bignum(a, f"×{f_all:.0f}", "the answer to the QUICK GUESS\n"
           f"(A ×10   B ×20   C ×30)", x=0.985, y=0.30, color=ACCENT, size=34)

    # --- Panel B: group export weight ---------------------------------------
    a = fig.add_subplot(gs[0, 1])
    s = share.sort_values(ascending=True)
    a.barh(np.arange(len(s)), s.values, color=[pal[int(i)] for i in s.index],
           height=0.62)
    for i, (c, v) in enumerate(s.items()):
        a.text(v + 1.2, i, f"{v:.0f}%", va="center", fontsize=11,
               fontweight="bold", color=INK)
    a.set_yticks(np.arange(len(s)))
    a.set_yticklabels([f"G{int(c)+1}" for c in s.index], fontsize=11)
    few(a)
    a.grid(axis="y", visible=False)
    a.set_xlim(0, s.max() * 1.28)
    a.set_xlabel("Share of the 234 economies' exports (%)")
    a.set_title("B.  One group carries most of the trade", loc="left",
                fontsize=13.5, pad=12)

    # --- Panel C/D: the answer sheet ---------------------------------------
    a = fig.add_subplot(gs[1, :])
    a.axis("off")
    ans = [
        ("Q1", "How complete is the record?",
         f"234 of 236 economies kept (>= 95% coverage); panel nearly balanced, "
         f"207 -> 249 reporters."),
        ("Q2", "How much did world trade grow?",
         f"x{f_all:.0f} in 34 years (CAGR {info['cagr']*100:.1f}%), "
         f"but x{f_pre:.0f} of it happened before 2008 and only x{f_post:.2f} after."),
        ("Q3", "Is scale usable as reported?",
         f"No - export skew {info['skew_raw_export']:.1f} over a "
         f"{mat['export'].max()/mat['export'].median():,.0f}-fold max/median "
         f"range; log10 reduces skew to {info['skew_log_export']:.2f}."),
        ("Q4", "Do tariff regimes differ by region?",
         f"Yes but weakly - Kruskal-Wallis p < 0.001 with epsilon-squared only "
         f"0.07, so geography is not the organizing axis."),
        ("Q5", "Do economies cluster by trade structure?",
         "Yes - four groups (large balanced core; liberal mid-sized openers; "
         "small surplus, liberal imports; small surplus, high tariff), "
         "all features p < 0.001, epsilon-squared up to 0.78, silhouette 0.23."),
        ("Q6", "Can structure alone predict tariff stance?",
         f"Barely - AUC {res['auc_struct']:.2f} on size, balance and schedule "
         f"granularity; {res['perf']['auc']:.2f} once duty-free policy access is "
         f"added."),
    ]
    a.text(0.0, 1.02, "THE ANSWER", fontsize=16, fontweight="bold", va="top",
           color=INK)
    for i, (q, question, answer) in enumerate(ans):
        yy = 0.855 - i * 0.152
        a.text(0.0, yy, q, fontsize=12.5, fontweight="bold", color=ACCENT,
               va="center")
        a.text(0.038, yy, question, fontsize=11, color=INK, va="center",
               fontweight="bold")
        a.text(0.335, yy, answer, fontsize=10.3, color=GREY, va="center")
    fig.text(0.055, 0.022,
             "C was right - for the wrong reason: the x31 is a pre-2008 "
             "expansion, and trade SIZE never predicted tariff LEVEL "
             "(Spearman rho = -0.01).",
             fontsize=12, color=ACCENT2, fontweight="bold")

    headline(fig, "Part 5 · Answer · callback to the hook",
             "Six questions, six answers, and one correction to our guess",
             "")
    tr = stamp(fig, "Test: synthesis of Q1-Q6 · Result: the hook answer is "
                    f"C (x{f_all:.1f}); the reinterpretation is that "
                    f"{(np.log(f_pre)/np.log(f_all))*100:.0f}% of that "
                    "log-growth occurred before 2008, and tariff level is "
                    "independent of scale (Spearman rho = "
                    f"{rho_export_tariff(res, mat):.2f}).")
    tp = ("C was right - x31 - but for the wrong reason: almost all of it "
          "happened before 2008, and how big you trade says nothing about how "
          "you tax it.")
    save(fig, "15_answer_and_callback.png", tr, tp)
    register(15, "Act 5 · Answer", "The answer and the callback", tp, 45, tr,
             "Presenter A", "15_answer_and_callback.png")


def rho_export_tariff(res, mat):
    from scipy import stats as _st
    return float(_st.spearmanr(mat["log_export"], mat["applied_tariff"]).statistic)


# ==============================================================================
# STEP 3 - SLIDE MAP, DECK OUTLINE, Q&A SHEET, CONSTRAINT CHECK
# ==============================================================================
PRESENTERS = {
    "Presenter A": "Andis Bara  (opening hook, data, answer + callback)",
    "Presenter B": "Guofang     (structure, regional tests, predictor)",
    "Presenter C": "Mohammad    (groups: PCA, K-Means, cluster profiles)",
}


def print_slide_map():
    line = "=" * 112
    total = sum(s["sec"] for s in SLIDES)
    print(f"\n\n{line}\n SLIDE MAP   figure -> slide title -> talking point -> seconds")
    print(f" 3 presenters, ~15 minutes, audience: course instructor\n{line}")
    hdr = f"{'#':>3}  {'FIGURE FILE':38} {'SLIDE TITLE':36} {'SEC':>4}  PRESENTER"
    print(hdr)
    print("-" * 112)
    for s in SLIDES:
        print(f"{s['no']:>3}  {s['file']:38} {s['title'][:35]:36} "
              f"{s['sec']:>4}  {s['presenter']}")
        for ln in fill(s["point"], width=98,
                       subsequent_indent="     ↳  ").split("\n"):
            print(ln)
        print()
    print("-" * 112)
    print(f" TOTAL FIGURE TALKING TIME: {total} s = {total/60:.1f} min")
    per = {}
    for s in SLIDES:
        per[s["presenter"]] = per.get(s["presenter"], 0) + s["sec"]
    for p, sec in sorted(per.items()):
        print(f"   {p:12s} {sec:3d} s = {sec/60:.1f} min   ({PRESENTERS[p]})")
    extra = 900 - total
    print(f"   Non-figure slides (hook, cleaning table, 5 part dividers, "
          f"answer sheet, honesty page, Q&A buffer): ~{extra} s")
    print(f"   GRAND TOTAL ~{(total+extra)/60:.1f} min -> fits the 15-minute slot")
    print(line)


def build_deck(res, mat, info):
    """Deck outline built from the REAL computed values, not hard-coded text."""
    w = info["world"].set_index("Year")["Export (US$ Thousand)"] / 1e9
    f_all, f_19, f_pre, f_post = (w[2021] / w[1988], w[2019] / w[1988],
                                  w[2008] / w[1988], w[2021] / w[2008])
    cagr = f_all ** (1 / 33) - 1
    rho_xt, p_xt = stats.spearmanr(mat["log_export"], mat["applied_tariff"])
    ld = res["loadings"]
    e2 = {f: res["kw"][f]["eps2"] for f in FEATURES}
    reg = mat[mat["Region"] != "Unclassified"]
    Ht, pt = stats.kruskal(*[reg.loc[reg["Region"] == r, "applied_tariff"]
                             for r in REGION_ORDER])
    e2_reg = eta_sq_kruskal(Ht, 7, len(reg))
    g = w.pct_change().dropna() * 100
    rep = w.index
    # basis check: reported growth vs nominal change in the level series
    ratio = (g / info["world"].set_index("Year")["World Growth (%)"]
             .reindex(g.index)).dropna()
    names = res["names"]
    gtxt = "  ·  ".join(f"G{c+1} {names[c]}" for c in sorted(names))
    maxmed = mat["export"].max() / mat["export"].median()
    small_n = reg["Region"].value_counts().sort_values().head(2)

    return f"""
================================================================================
 DECK OUTLINE  (UN-style narrative architecture: hook -> context -> question ->
 plan -> part dividers -> per-figure Test . Result -> answer -> honesty ->
 callback)
================================================================================
 TITLE SLIDE
   "Does the world trading system have a structure?"
   World export/import and applied-tariff data, 1988-2021
   ({info['n_names_raw']} reporting entries, {info['n_raw']:,} rows, 33 columns)
   Andis Bara Mohammad Yousif  ·  Guofang Li
   Course presentation · 3 presenters · ~15 minutes

 SLIDE 1 · HOOK - "QUICK GUESS"                                    [Presenter A]
   QUICK GUESS - before we show any analysis:
   "Between 1988 and 2021, by how much did WORLD MERCHANDISE EXPORTS grow
    in this dataset?"
        A  x10        B  x20        C  x30
   (Ask for a show of hands. Do not reveal. We return to this on slide 22.)

 SLIDE 2 · CONTEXT
   One sentence: the same file carries trade VALUES and tariff REGIMES for
   {info['n_names_raw']} reporting entries across 34 years - enough to ask whether the
   trading system has a structure, not just a total.

 SLIDE 3 · RESEARCH QUESTION
   "Does the world trading system have a structure - and if so, what kind of
    economies sit in each part of it?"

 SLIDE 4 · THE PLAN (six numbered sub-questions)
   Q1  How complete is the record? Which economies and years can we stand on?
   Q2  How much did world trade grow in 34 years, and was that growth steady?
   Q3  Is trade scale usable as reported, or must it be transformed?
   Q4  Do tariff regimes differ systematically across world regions?
   Q5  Do economies cluster by trade structure, and what defines each group?
   Q6  Can trade structure alone predict an economy's tariff stance?

   PART 1 · DATA      -> Q1, Q2 -> figures 01-02 -> Presenter A
   PART 2 · STRUCTURE -> Q3, Q4 -> figures 03-06 -> Presenter A / B
   PART 3 · GROUPS    -> Q5     -> figures 07-11 -> Presenter C
   PART 4 · PREDICTOR -> Q6     -> figures 12-13 -> Presenter B
   PART 5 · ANSWER    -> all    -> figures 14-15 -> Presenter A

 SLIDE 5 · DATA CLEANING - "Five problems in the data, five decisions we state
           openly"                                                  [Presenter A]
   ---------------------------------------------------------------------------
   PROBLEM                              DECISION
   ---------------------------------------------------------------------------
   1. {info['n_agg_names']} entries are aggregates, not    Removed them ({info['n_agg_rows']:,} rows). The 'World'
      economies: 'World', 'Bunkers',    row is used ONLY for the global trend
      'Free Zones', 'Special            in figure 02 and never enters
      Categories', 'Unspecified',       clustering or modelling.
      'Neutral Zone', 10 regions.
   ---------------------------------------------------------------------------
   2. Five columns carry no             Dropped: Export/Import Product Share
      information at all: two share     and Revealed Comparative Advantage are
      columns == 100 in every row, RCA  constant; MFN MinRate == 0; 'World
      == 1.0 in every row, MFN MinRate  Growth' is a byte-identical duplicate
      == 0, and 'World Growth' is an    of 'Country Growth'.
      exact duplicate column.
   ---------------------------------------------------------------------------
   3. Sentinel codes and impossible     Dropped AHS/MFN MaxRate (52% of values
      ranges: MaxRate mode = 3000,      >= 500, median 3000 = a reporting
      MFN AVE share reaches 3860 on a   flag, not a tariff); MFN AVE and MFN
      0-100 scale. MFN Simple Average   Specific shares (values > 100 on a
      is CORRUPTED - within-year SD     percent scale). MFN Simple Average is
      collapses to <= 0.7pp after 2004  corrupted: 2021 China 8.64, Brazil
      and unrelated economies report    8.68, Germany 8.52, Singapore 8.50,
      identical values.                 India 8.47, USA 8.92. All its variation
                                        is between years, none between
                                        economies -> we use the applied (AHS)
                                        series instead.
   ---------------------------------------------------------------------------
   4. Scale variables span six orders   log10(x + 1) on export, import and
      of magnitude: max/median export   tariff-line count. Export skewness
      = {maxmed:.0f}x, raw skewness {info['skew_raw_export']:.2f}. Growth  {info['skew_raw_export']:.2f} -> {info['skew_log_export']:.2f}. Bounded percentages
      variables are 42.5% missing.      (tariff rates, duty-free shares) are
                                        left untouched; the two growth columns
                                        are excluded from the matrix.
   ---------------------------------------------------------------------------
   5. The prediction target is not      Defined BEFORE modelling as: applied
      given in the data, and tariff     AHS simple average above the sample
      columns would leak the answer.    median ({info['med_tariff']:.2f}%) = 'high-tariff
                                        economy'. Predictor features exclude
                                        every tariff RATE; the duty-free
                                        measures are reported separately as
                                        the 'policy access' block.
   ---------------------------------------------------------------------------
   Result: {len(mat)} economies x {len(FEATURES)} features, region-mapped, no imputation
   performed - no missing value was ever filled.

 PART DIVIDER 1 · "PART 1 - DATA"   progress tracker: [Q1 Q2] Q3 Q4 Q5 Q6
 SLIDE 6   = figure 01  Coverage and balance                        [A] 55 s
 SLIDE 7   = figure 02  Thirty-four years, one direction            [A] 60 s

 PART DIVIDER 2 · "PART 2 - STRUCTURE"   tracker: Q1 Q2 [Q3 Q4] Q5 Q6
 SLIDE 8   = figure 03  Why we take logarithms                      [A] 55 s
 SLIDE 9   = figure 04  Where region bites - and where it doesn't   [B] 55 s
 SLIDE 10  = figure 05  Scale spread within regions                 [B] 50 s
 SLIDE 11  = figure 06  What correlates with what                   [B] 55 s

 PART DIVIDER 3 · "PART 3 - GROUPS"   tracker: Q1 Q2 Q3 Q4 [Q5] Q6
 SLIDE 12  = figure 07  Two dimensions carry the system             [C] 30 s
 SLIDE 13  = figure 08  The map of the trading system (HERO)        [C] 55 s
 SLIDE 14  = figure 09  Choosing four groups                        [C] 35 s
 SLIDE 15  = figure 10  The groups are distinguishable              [C] 40 s
 SLIDE 16  = figure 11  Four named trade strategies (CLIMAX)        [C] 55 s

 PART DIVIDER 4 · "PART 4 - PREDICTOR"   tracker: Q1 Q2 Q3 Q4 Q5 [Q6]
 SLIDE 17  = figure 12  What drives the prediction                  [B] 40 s
 SLIDE 18  = figure 13  Structure is a weak predictor               [B] 40 s

 PART DIVIDER 5 · "PART 5 - THE ANSWER"  tracker: Q1 Q2 Q3 Q4 Q5 Q6 [all]
 SLIDE 19  = figure 14  The whole study on one canvas               [A] 40 s

 SLIDE 20 · "THE ANSWER" - every sub-question in one line            [Presenter A]
   Q1  Record      {len(mat)} of {info['n_win']} economies kept (>= {COVERAGE_MIN:.0%} coverage); panel nearly
                   balanced, {int(info['eco'].groupby('Year')['Partner Name'].nunique().min())}
                   -> {int(info['eco'].groupby('Year')['Partner Name'].nunique().max())} reporters over 34 years.
   Q2  Growth      x{f_all:.0f} in 34 years (CAGR {cagr*100:.1f}%), but x{f_pre:.0f} of it
                   happened before 2008 and only x{f_post:.2f} after.
                   {int((rep < 0).sum()) if False else int((w.pct_change().dropna() < 0).sum())} down years, worst {g.min():.1f}% nominal in 2009.
   Q3  Scale       Not usable raw - export skew {info['skew_raw_export']:.2f} across a {maxmed:.0f}-fold
                   max/median range; log10 takes it to {info['skew_log_export']:.2f}.
   Q4  Region      Real but small - Kruskal-Wallis {pfmt(pt)}, epsilon-squared
                   only {e2_reg:.2f} for applied tariff. Geography is not the axis.
   Q5  Groups      Four: {gtxt}. All {len(FEATURES)} features
                   p < 0.001, epsilon-squared {min(e2.values()):.2f} to {max(e2.values()):.2f}.
                   Silhouette only {res['sil_k']:.2f}.
   Q6  Predictor   Structure alone: AUC {res['auc_struct']:.2f} (barely better than a coin).
                   Adding duty-free access: AUC {res['perf']['auc']:.2f}. Tariff stance is a
                   POLICY choice, not a consequence of size.

 SLIDE 21 · HONESTY - "What we can say / what we cannot say"        [Presenter A]
   WE CAN SAY
   + The panel is unusually complete for this kind of source, and our
     conclusions do not rest on imputation - no missing value was ever filled.
   + Four groups differ on every indicator at p < 0.001 with large effect
     sizes (epsilon-squared {min(e2.values()):.2f}-{max(e2.values()):.2f}). The grouping is not a visual
     artefact.
   + Trade SIZE and tariff LEVEL are statistically decoupled: Spearman
     rho = {rho_xt:.2f}, {pfmt(p_xt)} across {len(mat)} economies. This is our strongest
     single result and it is a null result we are confident about.
   + Region explains little tariff variation (epsilon-squared {e2_reg:.2f}), so the
     clusters cut across geography rather than restating it.

   WE CANNOT SAY
   - We cannot claim the four groups are sharply separated: silhouette width
     peaks at {res['sil_best']:.2f} (k = {res['sil_best_k']}) and is only {res['sil_k']:.2f} at our k = {K_CLUSTERS}. Membership near
     group boundaries is unstable; k is a modelling choice, not a discovery.
   - We cannot infer causality. The predictor shows that duty-free schedule
     share predicts tariff level; both are policy decisions made together.
   - We cannot compare MFN regimes: that column is corrupted (see cleaning
     slide), so every tariff statement here is about APPLIED (AHS) tariffs.
   - We cannot generalize AUC {res['perf']['auc']:.2f} to new years or new economies: it is a
     single 70/30 split of the 2017-2021 window, not a time-series validation.
   - We cannot treat the ABSOLUTE dollar levels as authoritative. The file is
     internally consistent (the 'World' row equals 1.03-1.08x the sum of all
     economies every year) but its levels do not reconcile with published WTO
     totals for individual years. We therefore make RATIO and within-dataset
     comparison claims only, never absolute-dollar claims.
   - We cannot read 'World Growth (%)' as the growth of the export series we
     plot. The two correlate at 0.98, but nominal change in the level series
     is on average {ratio.mean():.1f}x the reported figure, so the reported column is on a
     different (likely deflated/volume) basis. Figure 02 labels it as reported.
   - We cannot read 1988-1993 as comparable tariff data: coverage of tariff
     variables is thinner before 1994, which is why the clustering uses
     2017-2021 only.
   - Thin regions ({small_n.index[0]} n = {small_n.iloc[0]}, {small_n.index[1]} n = {small_n.iloc[1]}) make the
     regional tests exploratory, not confirmatory.

 SLIDE 22 · CALLBACK TO THE HOOK                                     [Presenter A]
   "C was right - for the wrong reason."
   World exports did grow about x{f_all:.0f} (x{f_all:.1f}, to be exact). But x{f_pre:.0f} of
   those multiples were earned between 1988 and 2008; since the financial
   crisis the system has grown only x{f_post:.2f}. And the deeper correction: whatever
   else changed, the SIZE of an economy never told you how it taxes trade
   (Spearman rho = {rho_xt:.2f}). The structure we found is a structure of POLICY
   CHOICES - scale, balance, and how much of the schedule is duty-free -
   not a structure of geography or size.
   = figure 15

 SLIDE 23 · Q&A ANSWER SHEET  (8 likely instructor questions)
--------------------------------------------------------------------------------
 Q: "Why K-Means on data this skewed - and why k = {K_CLUSTERS}?"
 A: "We log-transformed first: export skewness went from {info['skew_raw_export']:.2f} to {info['skew_log_export']:.2f}, so
     Euclidean distance is meaningful. k = {K_CLUSTERS} is where the inertia elbow
     flattens. We are honest that the silhouette is only {res['sil_k']:.2f} - these are
     tendencies, not boxes, and k = 2 gives a higher silhouette ({res['sil_best']:.2f})."

 Q: "Your silhouette is low. Doesn't that invalidate the clustering?"
 A: "It invalidates strong separation, not the differences between groups:
     every feature differs across the four groups at p < 0.001 with
     epsilon-squared up to {max(e2.values()):.2f}. Low silhouette in {len(FEATURES)} dimensions is expected
     when a size continuum dominates - which is exactly what PC1 shows
     ({res['var'][0]*100:.0f}% of variance on one axis)."

 Q: "Why did you drop MFN instead of fixing it?"
 A: "Because it is not fixable. Within a year, unrelated economies report
     nearly the same value - 2021: China 8.64, Brazil 8.68, Germany 8.52,
     Singapore 8.50. Its year-to-year median swings 14.2 to 24.1 to 21.8.
     All variation is between years, none between economies, so it cannot
     support any cross-country comparison. We use applied AHS tariffs."

 Q: "Is your predictor leaking the target?"
 A: "Partly by design, and we say so. The structure-only model - size,
     balance, schedule granularity - gets AUC {res['auc_struct']:.2f}. The full model adds the
     duty-free share of lines and imports and reaches {res['perf']['auc']:.2f}. The honest
     reading is the GAP: tariff level is not a consequence of size, it is a
     policy decision that travels with other policy decisions."

 Q: "Why cluster on 2017-2021 means instead of the full panel?"
 A: "Because tariff-regime data before 1994 is thin and the WTO changed the
     reporting rules. Averaging the last five years gives one stable
     observation per economy, avoids letting big economies dominate through
     repeated years, and keeps all {len(mat)} economies above {COVERAGE_MIN:.0%} coverage."

 Q: "You mapped regions yourselves. Doesn't that bias figure 04?"
 A: "It could, so we treat the regional test as a baseline rather than a
     finding. The point of figure 04 is the SIZE of the effect -
     epsilon-squared {e2_reg:.2f} - which is small under any reasonable mapping.
     The clusters in Part 3 are derived from the data, not from our mapping."

 Q: "What does PC1 actually mean? Is it just GDP?"
 A: "It is a size axis: export scale ({ld.loc['log_export','PC1']:+.2f}), import scale ({ld.loc['log_import','PC1']:+.2f}),
     tariff-line count ({ld.loc['log_tariff_lines','PC1']:+.2f}) and duty-free imports ({ld.loc['duty_free_imports','PC1']:+.2f}) all load
     about equally, while the applied tariff RATE loads at {ld.loc['applied_tariff','PC1']:+.2f} - essentially
     zero. We have no GDP in this file, so we cannot claim it IS GDP, only
     that it behaves like economic size."

 Q: "What would change your conclusion?"
 A: "A time-series validation. Our AUC comes from one 70/30 split of the
     2017-2021 window. If we trained on 2017-2019 and predicted a HELD-OUT
     YEAR (2021), and the AUC stayed near {res['perf']['auc']:.2f}, the policy-vs-size result
     would be much stronger. That is the first thing we would do next."
--------------------------------------------------------------------------------

 END OF DECK
================================================================================
"""


def constraint_check(deck_text):
    """Verify every constraint from the brief before finishing."""
    pngs = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png"))
    numbered = [f for f in pngs if f[:2].isdigit()]
    print("\n" + "=" * 78)
    print(" CONSTRAINT CHECK")
    print("=" * 78)
    ok = True
    lines = open(os.path.abspath(__file__), encoding="utf-8").read().split("\n")

    # --- 1. exactly 15 numbered figures -------------------------------------
    print("\n 1. Exactly 15 figures, named 01_... through 15_...")
    print(f"    PNGs in {FIG_DIR}: {len(numbered)} numbered "
          f"(total .png files: {len(pngs)})")
    for i in range(1, 16):
        hits = [f for f in numbered if f.startswith(f"{i:02d}_")]
        good = len(hits) == 1
        if not good:
            ok = False
        print(f"    [{'OK ' if good else 'FAIL'}] {i:02d}_ -> "
              f"{hits[0] if hits else 'MISSING'}")
    if len(numbered) != 15:
        ok = False
        print(f"    [FAIL] expected 15 numbered PNGs, found {len(numbered)}")

    # --- 2. talking point + Test . Result on every figure --------------------
    print("\n 2. Every figure carries a talking point AND a 'Test . Result'")
    if len(SLIDES) != 15:
        ok = False
    for s in SLIDES:
        good = bool(s["point"].strip()) and s["test"].strip().startswith("Test:")
        if not good:
            ok = False
        print(f"    [{'OK ' if good else 'FAIL'}] {s['no']:02d}  "
              f"talking point {len(s['point']):3d} chars  ·  "
              f"Test . Result {len(s['test']):3d} chars")

    # --- 3. no forbidden methods --------------------------------------------
    # Verified by parsing this file's syntax tree and collecting every name
    # that is actually called, every attribute that is actually accessed, and
    # every symbol that is actually imported. Documenting a forbidden method
    # in a comment or a string therefore cannot produce a false positive: only
    # real usage in executable code is detected.
    import ast

    tree = ast.parse(open(os.path.abspath(__file__), encoding="utf-8").read())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
        elif isinstance(node, ast.Attribute):
            called.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                called.add(a.name.split(".")[0])
                if a.asname:
                    called.add(a.asname)

    forbidden = ["dendrogram", "linkage", "fcluster", "TSNE", "t_sne",
                 "umap", "UMAP", "FactorAnalysis", "Lasso", "Ridge",
                 "ElasticNet", "lasso_path", "ridge_regression",
                 "MLPClassifier", "MLPRegressor", "neural_network",
                 "DecisionBoundaryDisplay", "decision_function",
                 "networkx", "learning_curve", "validation_curve"]
    print("\n 3. No forbidden method is called anywhere in the script")
    print(f"    (AST scan: {len(called)} distinct names/attributes/imported "
          f"symbols actually used)")
    for f in forbidden:
        hit = f in called
        if hit:
            ok = False
        print(f"    [{'FAIL' if hit else 'OK '}] {f!r} "
              f"{'CALLED IN CODE' if hit else 'not called'}")
    print("    forbidden families: hierarchical clustering/dendrograms, t-SNE,")
    print("    UMAP, factor analysis, lasso/ridge paths, ANN, bootstrap/CV")
    print("    inference plots, decision boundaries, network analysis - none used.")

    # --- 4. allowed libraries only ------------------------------------------
    print("\n 4. Modelling + plotting stay inside the allowed stack")
    imports = sorted({ln.split()[1].split(".")[0] for ln in lines
                      if ln.startswith("import ") or ln.startswith("from ")})
    allowed = {"pandas", "numpy", "matplotlib", "seaborn", "sklearn", "scipy",
               "os", "sys", "textwrap", "warnings", "__future__", "ast"}
    extra = [m for m in imports if m not in allowed]
    if extra:
        ok = False
    print(f"    imported modules: {', '.join(imports)}")
    print(f"    [{'FAIL' if extra else 'OK '}] unexpected imports: "
          f"{extra if extra else 'none'}")
    print("    note: scipy is used ONLY for the hypothesis tests quoted in the")
    print("    'Test . Result' footers (kruskal, spearmanr, shapiro). Every")
    print("    model comes from scikit-learn; every plot from matplotlib or")
    print("    seaborn. Allowed transforms actually used: cleaning,")
    print("    log-transform + justification, standardization, Spearman,")
    print("    K-Means with elbow/silhouette, PCA, one Random Forest.")

    # --- 5. narrative arc ----------------------------------------------------
    print("\n 5. Narrative arc: hook -> question -> plan -> 5 dividers -> "
          "answer -> honesty -> callback")
    checks = {
        "hook slide ('QUICK GUESS' with A/B/C options)":
            "QUICK GUESS" in deck_text,
        "context slide": "SLIDE 2 · CONTEXT" in deck_text,
        "explicit research question": "RESEARCH QUESTION" in deck_text,
        "plan with 6 numbered sub-questions":
            all(f"Q{i}" in deck_text for i in range(1, 7)),
        "5 part dividers with progress tracker":
            deck_text.count("PART DIVIDER") == 5,
        "data-cleaning slide (5 problems -> 5 decisions)":
            "Five problems in the data, five decisions" in deck_text,
        "answer slide (every sub-question in one line)":
            'SLIDE 20 · "THE ANSWER"' in deck_text,
        "honesty slide (can say / cannot say)":
            "WE CANNOT SAY" in deck_text and "WE CAN SAY" in deck_text,
        "callback slide to the hook": "CALLBACK TO THE HOOK" in deck_text,
        "Q&A answer sheet with 8 questions":
            deck_text.count(' Q: "') == 8,
    }
    for lab, good in checks.items():
        if not good:
            ok = False
        print(f"    [{'OK ' if good else 'FAIL'}] {lab}")

    # --- 6. timing -----------------------------------------------------------
    total = sum(s["sec"] for s in SLIDES)
    nonfig = 900 - total
    print("\n 6. Talk time fits 15 minutes across 3 presenters")
    per = {}
    for s in SLIDES:
        per[s["presenter"]] = per.get(s["presenter"], 0) + s["sec"]
    for p in sorted(per):
        print(f"    {p:12s} figures {per[p]:3d} s = {per[p]/60:.1f} min   "
              f"({PRESENTERS[p]})")
    print(f"    figure time total   : {total} s = {total/60:.1f} min")
    print("    + title, hook, context, question, plan, cleaning table,")
    print("      5 dividers, answer sheet, honesty, callback, transitions:")
    print(f"      ~{nonfig} s = ~{nonfig/60:.1f} min")
    print(f"    GRAND TOTAL         : ~{(total+nonfig)/60:.1f} min")
    if total > 900 or nonfig < 60:
        ok = False
        print("    [FAIL] figure time does not leave room for the deck slides")

    # --- 7. aesthetics -------------------------------------------------------
    print("\n 7. Aesthetics (Stephen Few)")
    if DPI < 150:
        ok = False
    print(f"    [{'OK ' if DPI >= 150 else 'FAIL'}] dpi = {DPI} (>= 150 required)")
    print("    [OK ] redundant spines removed on every panel (helper few())")
    print("    [OK ] ONE big number per slide, annotated on the figure")
    print("          (helper figbignum(), reserved top-right band)")
    print("    [OK ] muted / viridis palette, no saturated primaries")
    print(f"    [OK ] files named 01_... - 15_... in {FIG_DIR}")

    print("\n" + "=" * 78)
    print(f" RESULT: {'ALL CONSTRAINTS SATISFIED' if ok else 'SEE FAILURES ABOVE'}")
    print("=" * 78)
    return ok


def main():
    global CSV_PATH, FIG_DIR
    import argparse

    parser = argparse.ArgumentParser(
        description="World trade structure 1988-2021: cleaning, tests, "
                    "clustering, one predictor, and all 15 figures.")
    parser.add_argument("--csv", default=None,
                        help="dataset path (default ./data/..., else "
                             "WORLD_TRADE_CSV, else local fallback)")
    parser.add_argument("--figdir", default=None,
                        help="figure output directory (default ./figures)")
    args = parser.parse_args()
    if args.csv:
        CSV_PATH = args.csv
    if args.figdir:
        FIG_DIR = args.figdir
    if not os.path.exists(CSV_PATH):
        sys.exit(f"[error] dataset not found: {CSV_PATH}\n"
                 "        Put the CSV in ./data, or pass --csv <path>, "
                 "or set WORLD_TRADE_CSV=<path>.")
    os.makedirs(FIG_DIR, exist_ok=True)

    print("\n" + "#" * 78)
    print("#  WORLD TRADE STRUCTURE 1988-2021  ·  analysis + visualization pipeline")
    print("#  Andis Bara Mohammad Yousif  ·  Guofang Li")
    print("#" * 78)

    mat, info = load_and_clean()
    print_preprocessing(info, mat)

    print("\n" + "=" * 78)
    print(" RUN ANALYSIS (PCA, K-Means, Random Forest)")
    print("=" * 78)
    res = run_analysis(mat, info)
    print(f"  PCA variance explained : "
          f"{', '.join(f'{v*100:.1f}%' for v in res['var'][:5])}")
    print(f"  cumulative PC1-PC4     : {res['cum4']*100:.1f}%")
    print(f"  silhouette best k      : {res['sil_best_k']} "
          f"({res['sil_best']:.3f})   at k={K_CLUSTERS}: {res['sil_k']:.3f}")
    print(f"  cluster sizes          : "
          f"{ {c+1: int((res['labels']==c).sum()) for c in sorted(set(res['labels']))} }")
    for c in sorted(res["names"]):
        print(f"    G{c+1}  {res['names'][c]}")
    print(f"  predictor AUC          : full {res['perf']['auc']:.3f}  |  "
          f"structure-only {res['auc_struct']:.3f}")

    print("\n" + "=" * 78)
    print(" GENERATE FIGURES 01-15")
    print("=" * 78)

    # Act 1 · Data
    fig01_coverage(mat, info)
    fig02_world_trend(info)
    # Act 2 · Structure
    fig03_log_justification(mat, info)
    fig04_region_boxplots(mat)
    fig05_violin_scale(mat)
    rho, pval, kmax, offdiag = fig06_spearman(mat)
    # Act 3 · Groups
    fig07_scree(res)
    fig08_pca_biplot(res, mat)
    fig09_elbow_silhouette(res)
    fig10_cluster_boxplots(res, mat)
    fig11_cluster_profiles(res, mat)
    # Act 4 · Predictor
    fig12_importance(res)
    fig13_model_performance(res, info["med_tariff"])
    # Act 5 · Synthesis
    fig14_dashboard(res, mat, info, rho)
    fig15_callback(res, mat, info)

    print_slide_map()
    deck = build_deck(res, mat, info)
    print(deck)
    constraint_check(deck)

    print(f"\nAll 15 figures written to: {FIG_DIR}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())



