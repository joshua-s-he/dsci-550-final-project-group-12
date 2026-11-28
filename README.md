# dsci-550-final-project-group-12
## Group Members: 
Daniel Vasquez, Joshua He, William Khuu, Krish Kalawadia


## Setup

From the repository root, install the required packages:

```bash
python -m pip install numerapi
python -m pip install xgboost
python -m pip install numerai-tools pyarrow
```

## Evaluation

`python evaluate.py` trains the XGBoost baseline on the small feature set, scores it on the validation dates with Numerai's official CORR (and MMC when the meta model file is available), and writes `results/summary_xgb.csv` and `results/per_date_corr_xgb.csv`.

Other scripts import `load_data` and `evaluate` from `evaluate.py` so every model is scored the same way.

