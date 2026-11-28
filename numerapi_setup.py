from numerapi import NumerAPI
import json
import pandas as pd

version = "v6.0"
napi = NumerAPI()

napi.download_dataset(f"{version}/features.json")
with open(f"{version}/features.json") as f:
    metadata = json.load(f)

# Currently only includes 'small' to use less memory and to train faster
# Change to 'medium' or 'all' to use medium/full feature set
features = [
    col for col in metadata["feature_sets"]["all"]
    if col != "feature_country"
]

napi.download_dataset(f"{version}/live.parquet")
live = pd.read_parquet(
    f"{version}/live.parquet",
    columns=["numerai_ticker"] + features,
).set_index("numerai_ticker")

napi.download_dataset(f"{version}/train.parquet")
train = pd.read_parquet(
    f"{version}/train.parquet",
    columns=["numerai_ticker", "date", "target"] + features,
).dropna(subset=["target"])

napi.download_dataset(f"{version}/validation.parquet")
validation = pd.read_parquet(
    f"{version}/validation.parquet",
    columns=["numerai_ticker", "date", "target"] + features,
).dropna(subset=["target"])