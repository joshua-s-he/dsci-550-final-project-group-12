"""Evaluation harness for the Numerai v6 dataset.

Usage from another script:
    from evaluate import load_data, evaluate
    train, valid, features = load_data("small")
    model.fit(train[features], train["target"])
    valid["pred_xgb"] = model.predict(valid[features])
    evaluate(valid, ["pred_xgb"], features)

Run directly (python evaluate.py) to score the current XGBoost model as a smoke test.
"""

import json
import os

import numpy as np
import pandas as pd
from numerapi import NumerAPI
from numerai_tools.scoring import correlation_contribution, max_feature_correlation, numerai_corr

VERSION = "v6.0"
DATE_COL = "date"
TARGET = "target"
GAP_DAYS = 35  # target looks 20 business days ahead; drop train rows this close to validation start
napi = NumerAPI()


def _download(name):
    path = f"{VERSION}/{name}"
    if not os.path.exists(path):
        napi.download_dataset(path)
    return path


def load_data(feature_set="small", targets=(TARGET,)):
    """Load train/validation with a time gap between them so the forward-looking target can't leak."""
    with open(_download("features.json")) as f:
        features = [c for c in json.load(f)["feature_sets"][feature_set] if c != "feature_country"]
    cols = ["numerai_ticker", DATE_COL, *targets, *features]

    train = pd.read_parquet(_download("train.parquet"), columns=cols).dropna(subset=[TARGET])
    valid = pd.read_parquet(_download("validation.parquet"), columns=cols).dropna(subset=[TARGET])
    for df in (train, valid):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    cutoff = valid[DATE_COL].min() - pd.Timedelta(days=GAP_DAYS)
    train = train[train[DATE_COL] < cutoff]
    print(f"train: {len(train):,} rows, {train[DATE_COL].nunique()} dates, ends {train[DATE_COL].max().date()}")
    print(f"valid: {len(valid):,} rows, {valid[DATE_COL].nunique()} dates, "
          f"{valid[DATE_COL].min().date()} to {valid[DATE_COL].max().date()}")
    print(f"features: {len(features)} ({feature_set})")
    return train, valid, features


def per_date_corr(df, pred_cols):
    """Numerai's official CORR, computed separately for each date. Returns dates x models."""
    return df.groupby(DATE_COL).apply(lambda d: numerai_corr(d[pred_cols], d[TARGET]))


def per_date_mmc(df, pred_cols, meta_col="numerai_meta_model"):
    """MMC per date; needs the meta model column merged into df (see add_meta_model)."""
    d = df.dropna(subset=[meta_col])
    return d.groupby(DATE_COL).apply(lambda g: correlation_contribution(g[pred_cols], g[meta_col], g[TARGET]))


def add_meta_model(df):
    """Merge Numerai's historical meta model predictions into the validation frame, if available."""
    try:
        mm = pd.read_parquet(_download("meta_model.parquet")).reset_index()
    except Exception as e:
        print(f"meta model not available, skipping MMC ({e})")
        return df
    meta_col = next(c for c in mm.columns if c.startswith("numerai_meta_model"))
    if DATE_COL in mm.columns:
        mm[DATE_COL] = pd.to_datetime(mm[DATE_COL])
    keys = [k for k in ("numerai_ticker", DATE_COL) if k in mm.columns]
    return df.merge(mm[keys + [meta_col]].rename(columns={meta_col: "numerai_meta_model"}), on=keys, how="left")


def max_drawdown(scores):
    cum = scores.cumsum()
    return (cum.cummax() - cum).max()


def feature_exposure(df, pred_col, features, n_dates=50):
    """Mean over sampled dates of the largest |corr| between predictions and any single feature."""
    dates = df[DATE_COL].drop_duplicates().sort_values()
    dates = dates.iloc[np.linspace(0, len(dates) - 1, min(n_dates, len(dates))).astype(int)]
    vals = [abs(max_feature_correlation(g[pred_col], g[features])[1])
            for _, g in df[df[DATE_COL].isin(dates)].groupby(DATE_COL)]
    return float(np.mean(vals))


def evaluate(df, pred_cols, features, mmc=False):
    """Print and return a summary table: one row per model."""
    corr = per_date_corr(df, pred_cols)
    summary = pd.DataFrame({
        "corr_mean": corr.mean(),
        "corr_std": corr.std(),
        "sharpe": corr.mean() / corr.std(),
        "max_drawdown": corr.apply(max_drawdown),
        "pct_dates_positive": (corr > 0).mean(),
        "feature_exposure": [feature_exposure(df, c, features) for c in pred_cols],
    })
    if mmc and "numerai_meta_model" in df:
        m = per_date_mmc(df, pred_cols)
        summary["mmc_mean"] = m.mean()
        summary["mmc_sharpe"] = m.mean() / m.std()
    print(summary.round(4).to_string())
    return summary, corr


if __name__ == "__main__":
    from xgboost import XGBRegressor

    train, valid, features = load_data("small")
    model = XGBRegressor(max_depth=5, n_estimators=200, n_jobs=-1)
    model.fit(train[features], train[TARGET])
    valid["pred_xgb"] = model.predict(valid[features])

    valid = add_meta_model(valid)
    summary, corr = evaluate(valid, ["pred_xgb"], features, mmc=True)
    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/summary_xgb.csv")
    corr.to_csv("results/per_date_corr_xgb.csv")
