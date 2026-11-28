"""Step 2: baselines, all scored on the same validation dates with evaluate.py.

1. best single feature  - the feature with the highest mean per-date corr on train (a floor)
2. ridge                - linear model on all features of the set
3. xgboost              - Josh's starter settings (200 trees, depth 5)
4. lgbm_example         - Numerai's documented example LightGBM settings
5. benchmark            - Numerai's own lgbm_jupiter_60 predictions (all features, same 60-day target; a ceiling)

Usage: python baselines.py [small|medium|all]
"""

import os
import sys
import time
import warnings

import lightgbm as lgb
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from evaluate import DATE_COL, TARGET, add_benchmark, evaluate, load_data

warnings.filterwarnings("ignore")  # some features are constant on some dates; their corr is NaN and skipped

feature_set = sys.argv[1] if len(sys.argv) > 1 else "small"
train, valid, features = load_data(feature_set)
valid = add_benchmark(valid).dropna(subset=["pred_benchmark"])
fit_seconds = {}


def timed(name, fn):
    t = time.time()
    valid[name] = fn()
    fit_seconds[name] = round(time.time() - t)
    print(f"{name}: {fit_seconds[name]}s", flush=True)


# 1. best single feature, picked on train only (sampled dates keep it fast)
def best_feature():
    dates = train[DATE_COL].drop_duplicates().iloc[::4]
    sample = train[train[DATE_COL].isin(dates)]
    corrs = sample.groupby(DATE_COL).apply(lambda d: d[features].corrwith(d[TARGET], method="spearman")).mean()
    best = corrs.abs().idxmax()
    print(f"best single feature: {best} (train corr {corrs[best]:.4f})")
    return valid[best] * (1 if corrs[best] > 0 else -1)


timed("pred_single_feature", best_feature)
# 2. ridge: float32 on every other date - sklearn otherwise copies the int8 data to float64 and runs out of memory
def ridge():
    t = train[train[DATE_COL].isin(train[DATE_COL].drop_duplicates().iloc[::2])]
    model = Ridge(alpha=1.0).fit(t[features].astype("float32"), t[TARGET].astype("float32"))
    return model.predict(valid[features].astype("float32"))


timed("pred_ridge", ridge)
timed("pred_xgb", lambda: XGBRegressor(max_depth=5, n_estimators=200, n_jobs=-1, tree_method="hist")
      .fit(train[features], train[TARGET]).predict(valid[features]))
timed("pred_lgbm_example", lambda: lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=2**5 - 1,
    colsample_bytree=0.1, verbose=-1).fit(train[features], train[TARGET]).predict(valid[features]))

pred_cols = ["pred_single_feature", "pred_ridge", "pred_xgb", "pred_lgbm_example", "pred_benchmark"]
summary, corr = evaluate(valid, pred_cols, features)
summary["fit_seconds"] = pd.Series(fit_seconds)

os.makedirs("results", exist_ok=True)
summary.to_csv(f"results/summary_baselines_{feature_set}.csv")
corr.to_csv(f"results/per_date_corr_baselines_{feature_set}.csv")
