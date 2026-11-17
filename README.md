# Does the World Trading System Have a Structure?

**Trade values and applied tariffs of 234 economies, 1988–2021 — cleaned, tested, clustered, and predicted in five acts.**

> Course group presentation · Andis Bara · Mohammad Yousif · Guofang Li
> One runnable script → 15 figures from the bundled dataset; the final 28-slide deck ships under `slides/`.

---

## The research question

> **Does the world trading system have a structure — and if so, what kind of economies sit in each part of it?**

It is decomposed into six sub-questions, each answered by specific exhibits and each carrying a printed `Test · Result` line:

| | Sub-question | Answer (one line) |
|---|---|---|
| **Q1** | How complete is the record? | 234 of 236 economies survive a ≥95 % completeness filter; the panel grows from 192 reporters in 1988 to 234 in 2021. |
| **Q2** | How much did world trade grow, and was it steady? | ×30.9 in 34 years (CAGR 11.0 %) — but ×22.7 of that before 2008 and only ×1.36 after. |
| **Q3** | Is trade scale usable as reported? | No. Export skewness 7.01 over a 330-fold max/median range; `log10(x+1)` takes it to −0.39. |
| **Q4** | Do tariff regimes differ by region? | Real but small: Kruskal–Wallis p = 0.002 with ε² = 0.07. Geography is not the organising axis. |
| **Q5** | Do economies cluster by trade structure? | Yes — four named strategies, all 8 features p < 0.001, ε² 0.25–0.78. Silhouette is only 0.23, so tendencies, not boxes. |
| **Q6** | Can structure alone predict tariff stance? | Barely (AUC 0.638). Duty-free policy access lifts it to 0.854 — stance is a policy choice, not a consequence of size. |

---

## Headline results

### 1. The strongest result is a null result
Trade **size** and tariff **level** are statistically decoupled:

```
Spearman ρ(log10 exports, applied tariff) = −0.020   p = 0.759   n = 234
```

Size does hang together internally — exports↔imports ρ = 0.953, exports↔tariff-line count ρ = 0.936 — and so does policy (duty-free lines↔applied tariff ρ = −0.729). The two blocks simply do not touch.

### 2. Four trade strategies, one dominant group

| Group | Label | n | Share of exports |
|---|---|---|---|
| **G1** | Large · Balanced · Moderate-tariff | 71 | **90.9 %** |
| **G2** | Mid-sized · Liberal schedule · Low-tariff | 73 | 8.0 % |
| **G3** | Small · Surplus · Liberal imports | 57 | 0.6 % |
| **G4** | Small · Surplus · High-tariff | 33 | 0.4 % |

All eight indicators separate the groups at p < 0.001 (ε² from 0.25 for weighted tariff up to 0.78 for import scale). PCA gives the map they live on: **PC1 = scale (48.0 %)**, **PC2 = openness (22.6 %)**, 70.7 % combined.

### 3. Tariff stance is policy, not size
A single Random Forest (600 trees, balanced weights), target = applied AHS tariff above the sample median of 4.26 %:

| Feature set | Held-out ROC AUC |
|---|---|
| Structure only (scale, balance, schedule granularity) | **0.638** |
| + duty-free policy access | **0.854** |

Permutation importance puts duty-free tariff lines at 0.296 AUC, versus 0.039 for the best pure-structure feature.

---

## Data cleaning: five problems, five stated decisions

| # | Problem | Decision |
|---|---|---|
| 1 | **Aggregates are not economies** — `World`, `Bunkers`, `Free Zones`, `Special Categories`, `Unspecified`, `Neutral Zone` + 10 regional aggregates (16 entries, 482 rows) | Removed. The `World` row feeds only the global trend (Exhibit 2) and never enters clustering or modelling. |
| 2 | **Zero-information columns** — two share columns equal 100 in every row; RCA equals 1.0; `MFN MinRate` equals 0; `World Growth` duplicates `Country Growth` byte-for-byte | Dropped all five. |
| 3 | **Sentinel codes + a corrupted series** — `MaxRate` mode = 3000 (a reporting flag); `MFN AVE` share reaches 3860 on a 0–100 scale; `MFN Simple Average` is near-identical across unrelated economies within each year | Dropped. Evidence for the MFN corruption: 2021 values are China 8.64, Brazil 8.68, Germany 8.52, Singapore 8.50, India 8.47, USA 8.92 — within-year SD collapses to ≤0.7 pp after 2004, while its year-to-year median swings 14.2 → 24.1 → 21.8. All variation is *between years*, none *between economies*, so it cannot support cross-country comparison. **Every tariff statement here uses the applied (AHS) series.** |
| 4 | **Scale spans six orders of magnitude** — export skewness 7.01, max/median ratio 330×; both growth columns 42.5 % missing | `log10(x+1)` on exports, imports and tariff-line count; bounded percentages left untouched; growth columns excluded from the matrix. |
| 5 | **The target is not given — and tariffs would leak it** | Defined *before* modelling: applied AHS above the sample median = high-tariff (117 high / 117 low). The structure-only feature set contains **no tariff rate at all**. |

**Result: an analysis matrix of 234 economies × 8 features, region-mapped, with no imputation — not one missing value was ever filled.**

---

## Reproducing everything

```bash
git clone <your-repo-url> && cd <repo-name>
pip install -r requirements.txt

python analysis_presentation.py
```

That single command prints the preprocessing summary, writes all **15 figures** to `figures/`, prints the slide map (figure → title → talking point → seconds), the full 30-slide deck outline, the Q&A answer sheet, and a **constraint self-check** that ends with `RESULT: ALL CONSTRAINTS SATISFIED`.

The dataset is bundled in `data/`. To point the script elsewhere:

```bash
python analysis_presentation.py --csv /path/to/dataset.csv --figdir out
# or: WORLD_TRADE_CSV=/path/to/dataset.csv python analysis_presentation.py
```

A complete run log is in [`output/run_log.txt`](output/run_log.txt). Every number quoted above appears there.

---

## Repository layout

```
.
├── analysis_presentation.py     ← the whole pipeline: cleaning → tests → PCA →
│                                   K-Means → Random Forest → 15 figures + deck text
├── requirements.txt
├── data/
│   └── 34_years_world_export_import_dataset.csv   (8,096 rows × 33 cols, 1988–2021)
├── figures/                     ← the 15 exhibits, 01_ … 15_ , dpi 180
├── slides/
│   ├── World_trade.pptx                           (final 28-slide presented version)
│   └── World_trade.pdf                            (PDF export of the same deck)
├── output/
│   └── run_log.txt              ← full console output of one successful run
└── exploratory/                 ← early iterations (see the note below)
```

### The five-act structure of the 15 exhibits

| Act | Exhibits | Content |
|---|---|---|
| **I · Data** | 01–02 | Coverage and balance of the panel; the 34-year global trend (feeds the opening hook's answer) |
| **II · Structure** | 03–06 | Log-transform justification; notched boxplots by region; within-region scale spread; masked Spearman heatmap |
| **III · Groups** | 07–11 | PCA scree; the biplot (hero); elbow + silhouette; boxplots by cluster; named cluster profiles (climax) |
| **IV · Predictor** | 12–13 | Permutation importance; ROC + confusion matrix + metric table |
| **V · Answer** | 14–15 | One-canvas dashboard; the answer visualisation supporting the callback |

---

## What we can say — and what we cannot

**We can say**
- The conclusions rest on no imputation; coverage is the strongest part of this source.
- The four groups differ on every indicator at p < 0.001 with ε² = 0.25–0.78 — not a visual artefact.
- Trade size and tariff level are decoupled (ρ = −0.020, p = 0.759).
- Region explains only ε² = 0.07 of tariff variation, so the clusters cut across geography rather than restating it.

**We cannot say**
- That the groups are **sharply separated**. Silhouette is 0.23 at our k = 4 (it peaks at 0.30 for k = 2). k is a modelling choice we defend with the Kruskal–Wallis evidence, not a discovery.
- Anything **causal**. Duty-free share predicts tariff level because both are decided together.
- Anything about **MFN regimes** — that column is corrupted, so every tariff claim here is *applied* (AHS).
- That **AUC 0.854 generalises**. It is one 70/30 split of the 2017–2021 window, not a held-out year.
- **Absolute-dollar claims.** The file is internally consistent (the `World` row equals 1.03–1.08× the sum of all economies every year) but its levels do not reconcile with published WTO totals for individual years. We therefore make **ratio and within-dataset comparison claims only**.
- That `World Growth (%)` is the growth of the plotted export series. They correlate at 0.98, but nominal change in the level series averages 2.3× the reported figure, so that column is on a different (likely deflated/volume) basis.
- That 1988–1993 is comparable tariff data — coverage is thinner before 1994, which is why clustering uses 2017–2021.
- That the regional tests are confirmatory: thin cells (North America n = 5, South Asia n = 8) keep them exploratory.

---

## Methods and scope

**Stack:** pandas · numpy · matplotlib · seaborn · scikit-learn. `scipy.stats` is used only for the hypothesis tests quoted in the `Test · Result` footers (Kruskal–Wallis, Spearman, Shapiro–Wilk).

**Used:** cleaning, `log10` transform with justification, standardization, Spearman correlation, K-Means with elbow + silhouette, PCA, one Random Forest classifier, permutation importance.

**Deliberately not used:** hierarchical clustering / dendrograms, t-SNE, UMAP, factor analysis, lasso/ridge paths, neural networks, bootstrap or CV inference plots, decision boundaries, network analysis. The build script enforces this with an AST scan of its own source — it fails the constraint check if any of these is called.

**Design:** Stephen Few — high data-ink ratio, no redundant spines, muted/viridis palette, one big number per exhibit, dpi 180.

---

## ⚠️ About `exploratory/`

This directory holds **early iterations that were not used for any result in the deck or the figures**. Two of the scripts there (`figs3_4_6_statistical_diagnostics.py`, producing `fig6_hierarchical_clustermap.png`, and `fig8_regional_flow_alluvium.py`) use methods **explicitly forbidden by the course brief** — Ward-linkage hierarchical clustering with dendrograms, and an alluvial/Sankey flow diagram. They are kept for transparency of process only; the final analysis relies on none of them.

---

## Dataset

`34_years_world_export_import_dataset.csv` — world export/import values and applied & MFN tariff regimes, 1988–2021, 8,096 rows × 33 columns, 265 distinct reporting entries.

## Presentation

[`slides/World_trade.pptx`](slides/World_trade.pptx) is the **final presented version (28 slides)**, with [`slides/World_trade.pdf`](slides/World_trade.pdf) as its PDF export. It follows the same narrative arc as the figures: opening hook → context → research question → plan → cleaning table → five part dividers with progress trackers → 15 exhibits → the answer → honesty page → callback to the hook → Q&A answer sheet. Each exhibit slide carries its speaker notes.

## Licence

See [`LICENSE`](LICENSE).
