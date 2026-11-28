from numerapi import NumerAPI
import json
import pandas as pd

version = "v6.0"
napi = NumerAPI()

napi.download_dataset(f"{version}/features.json")
with open(f"{version}/features.json") as f:
    metadata = json.load(f)

features = [
    col for col in metadata["feature_sets"]["small"]
    if col != "feature_country"
]

napi.download_dataset(f"{version}/live.parquet")
live = pd.read_parquet(
    f"{version}/live.parquet",
    columns=["numerai_ticker"] + features,
).set_index("numerai_ticker")

napi.download_dataset(f"{version}/train.parquet")
data = pd.read_parquet(
    f"{version}/train.parquet",
    columns=["numerai_ticker", "date", "target"] + features,
).dropna(subset=["target"])