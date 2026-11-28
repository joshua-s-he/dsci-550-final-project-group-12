"""Model using the Numerai v6 dataset."""

import json
import pandas as pd
from xgboost import XGBRegressor

# choose a set of numeric features
with open("v6.0/features.json") as f:
  feature_set = json.load(f)["feature_sets"]["all"]
feature_set = [f for f in feature_set if f != "feature_country"]

# load training data and live data
train = pd.read_parquet("v6.0/train.parquet", columns=feature_set + ["target"])
train = train.dropna(subset=["target"])
live = pd.read_parquet(
  "v6.0/live.parquet", columns=["numerai_ticker"] + feature_set
).set_index("numerai_ticker")

# train a model to predict the target
model = XGBRegressor(max_depth=5, n_estimators=200)
model.fit(train[feature_set], train["target"])

# predict live data and save a submission file
predictions = pd.Series(model.predict(live[feature_set]), index=live.index)
predictions.rank(pct=True).to_frame("prediction").to_csv("predictions.csv")