# dsci-550-final-project-group-12

DSCI 550 Project 2: an end-to-end study on the [Numerai Tournament](https://numer.ai/tournament), predicting the relative future returns of stocks from obfuscated features.

## Group Members
Daniel Vasquez, Joshua He, William Khuu, Krish Kalawadia

## The problem
- **Data:** Numerai dataset v6.0. Each row is one stock in one week, with up to 1,040 anonymized integer features (0-4). Feature sets: small (100), medium (351), all (1,040).
- **Target:** `target` (identical to `target_jupiter_60`) ranks each stock's return over the next 60 business days into five bins (0, 0.25, 0.5, 0.75, 1 at 5/20/50/20/5%).
- **History:** 1,235 weeks, 2003 to 2026. Training data runs to Nov 2018; validation covers Jan 2019 onward.
- **Metric:** Numerai's official CORR (rank-gaussianized correlation with the target), computed per week and then averaged. v6.0 has no meta model file, so MMC can't be computed offline.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
On macOS, LightGBM may also need `brew install libomp`.

Data downloads automatically to `v6.0/` (about 3.6 GB, git-ignored) the first time a script runs.

## Pipeline
Run from the repository root, in this order:

| Script | What it does | Output |
| --- | --- | --- |
| `evaluate.py` | Shared harness: time-ordered train/validation split with a 90-day gap (so 60-day targets can't leak), per-week CORR, Sharpe, max drawdown, feature exposure. Run directly for an XGBoost smoke test. | `results/summary_xgb.csv` |
| `baselines.py [small\|medium\|all]` | Best single feature, ridge, XGBoost, Numerai's example LightGBM, and Numerai's benchmark model, all scored the same way. | `results/summary_baselines_<set>.csv` |
| `eda.py` | Target distribution, stocks per week, feature stability across eras, feature drift, baseline cumulative CORR. | `figures/*.png`, `figures/eda_summary.txt` |
| `experiments.py --features <set> --target <t> --every <n>` | One LightGBM experiment per call (feature set, training target, training-week subsample). Saves validation predictions for ensembling. | `results/experiments.csv`, `results/preds/` |

Earlier starter scripts: `numerapi_setup.py` (data download) and `xgboost_model.py` (first XGBoost model, writes live predictions).

`results/` is git-ignored; the summary tables are reported below and in the final report.

## Results
Validation CORR, Jan 2019 to Aug 2026 (400 weeks). Filled in as experiments finish.

| Model | Features | Mean CORR | Sharpe |
| --- | --- | --- | --- |
| _pending_ | | | |

## Timeline
- Code and presentation due: Dec 6, 2026
- Final presentations: Dec 7 and 9, 2026
- Final report due: Dec 14, 2026
