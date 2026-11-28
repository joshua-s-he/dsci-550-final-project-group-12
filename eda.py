"""Step 3: exploratory analysis. Writes figures/ (PNGs for slides and report) and figures/eda_summary.txt.

Uses the small feature set over the full history (train + validation). Run baselines.py first
so the cumulative-CORR figure has results/per_date_corr_baselines_small.csv to read.
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")
VERSION, DATE, TARGET = "v6.0", "date", "target"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left", "lines.linewidth": 2,
})
os.makedirs("figures", exist_ok=True)
notes = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


with open(f"{VERSION}/features.json") as f:
    features = [c for c in json.load(f)["feature_sets"]["small"] if c != "feature_country"]
cols = ["numerai_ticker", DATE, TARGET, *features]
df = pd.concat([
    pd.read_parquet(f"{VERSION}/train.parquet", columns=cols).assign(split="train"),
    pd.read_parquet(f"{VERSION}/validation.parquet", columns=cols).assign(split="validation"),
]).dropna(subset=[TARGET])
df[DATE] = pd.to_datetime(df[DATE])
notes.append(f"rows {len(df):,}; weeks {df[DATE].nunique()}; {df[DATE].min().date()} to {df[DATE].max().date()}")

# 1. target distribution: binned, centred on 0.5
shares = df[TARGET].round(4).value_counts(normalize=True).sort_index()
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar([f"{v:g}" for v in shares.index], shares.values, color=BLUE, width=0.6)
ax.set(title="Target is ranked into five fixed bins (5 / 20 / 50 / 20 / 5%)",
       xlabel="target value", ylabel="share of rows")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.grid(axis="x", visible=False)
save(fig, "fig1_target_distribution")
notes.append("target shares: " + ", ".join(f"{k:g}={v:.1%}" for k, v in shares.items()))

# 2. universe size over time
n = df.groupby(DATE).size()
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(n.index, n.values, color=BLUE)
ax.axvline(df.loc[df.split == "validation", DATE].min(), color=MUTED, ls="--", lw=1)
ax.text(df.loc[df.split == "validation", DATE].min(), n.min(), "  validation starts", color=MUTED, va="bottom")
ax.set(title="Stocks per week grow over time", ylabel="stocks per week")
save(fig, "fig2_stocks_per_week")
notes.append(f"stocks per week: min {n.min():,}, median {int(n.median()):,}, max {n.max():,}")

# per-week correlation of every feature with the target (the basis for figs 3 and 4)
fc = df.groupby(DATE).apply(lambda d: d[features].corrwith(d[TARGET]))

# 3. feature stability: early training years vs validation years
early = fc[fc.index < "2011-01-01"].mean()
late = fc[fc.index >= "2019-01-01"].mean()
flipped = (early * late < 0)
fig, ax = plt.subplots(figsize=(5, 4.6))
ax.scatter(early[~flipped], late[~flipped], s=24, color=BLUE, label="same sign")
ax.scatter(early[flipped], late[flipped], s=24, color=ORANGE, label="sign flipped")
lim = max(early.abs().max(), late.abs().max()) * 1.1
ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="mean corr with target, 2003-2010",
       ylabel="mean corr with target, 2019-2026",
       title=f"{flipped.sum()} of {len(features)} features flip sign between eras")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig3_feature_stability")
notes.append(f"features flipping sign 2003-10 vs 2019-26: {flipped.sum()}/{len(features)}; "
             f"corr of early vs late means {early.corr(late):.2f}")

# 4. regime change: 52-week rolling corr of the three strongest features over the full history
top = fc.mean().abs().sort_values(ascending=False).index[:3]
roll = fc[top].rolling(52).mean()
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for c, colr in zip(top, [BLUE, ORANGE, AQUA]):
    ax.plot(roll.index, roll[c], color=colr, label=c.replace("feature_", ""))
ax.axhline(0, color=MUTED, lw=1)
ax.set(title="Even the strongest features drift and reverse over time", ylabel="52-week rolling corr")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save(fig, "fig4_feature_drift")
notes.append("top-3 features by mean |corr|: " + ", ".join(f"{c} {fc[c].mean():+.4f}" for c in top))
notes.append("share of weeks where per-week feature corr is positive (mean over features): "
             f"{(fc > 0).mean().mean():.2f}")

# 5. cumulative CORR of the baselines over the validation years
path = "results/per_date_corr_baselines_small.csv"
if os.path.exists(path):
    pc = pd.read_csv(path, index_col=0, parse_dates=True)
    names = {"pred_benchmark": ("Numerai benchmark", BLUE), "pred_lgbm_example": ("LightGBM example", ORANGE),
             "pred_ridge": ("Ridge", AQUA), "pred_xgb": ("XGBoost", YELLOW),
             "pred_single_feature": ("Best single feature", MAGENTA)}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for c, (label, colr) in names.items():
        s = pc[c].cumsum()
        ax.plot(s.index, s.values, color=colr, label=label)
        ax.annotate(f"{s.iloc[-1]:.2f}", (s.index[-1], s.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set(title="Cumulative validation CORR: our baselines vs Numerai's benchmark",
           ylabel="cumulative CORR (sum over weeks)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig5_baselines_cumulative_corr")

with open("figures/eda_summary.txt", "w") as f:
    f.write("\n".join(notes) + "\n")
print("\n".join(notes))
