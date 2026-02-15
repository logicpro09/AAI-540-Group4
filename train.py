import os
import json
import joblib
import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


TARGET_COL = "extreme_precip_tomorrow"
TIME_COL = "date"
ID_COL = "station_id"

NUM_COLS = ["prcp_lag_1", "prcp_roll_7", "TMAX", "TMIN"]
CAT_COLS = [ID_COL]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)


def time_split(df: pd.DataFrame, train_quantile: float = 0.8):
    split_date = df[TIME_COL].quantile(train_quantile)
    train_df = df[df[TIME_COL] <= split_date].copy()
    test_df = df[df[TIME_COL] > split_date].copy()
    return train_df, test_df, split_date


def build_model():
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, NUM_COLS),
            ("cat", categorical, CAT_COLS),
        ]
    )

    clf = LogisticRegression(max_iter=200, class_weight="balanced")

    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    # roc_auc requires both classes present
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = None
    out["threshold"] = float(threshold)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to extreme_precip_model_data.csv")
    ap.add_argument("--outdir", default="outputs", help="Where to write artifacts")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("model", exist_ok=True)

    df = load_data(args.data)
    train_df, test_df, split_date = time_split(df, 0.8)

    X_train = train_df[NUM_COLS + CAT_COLS]
    y_train = train_df[TARGET_COL].astype(int).values

    X_test = test_df[NUM_COLS + CAT_COLS]
    y_test = test_df[TARGET_COL].astype(int).values

    pipe = build_model()
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob, threshold=args.threshold)

    joblib.dump(pipe, "model/model.joblib")

    meta = {
        "split_date": str(split_date),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features_num": NUM_COLS,
        "features_cat": CAT_COLS,
        "target": TARGET_COL,
        "metrics": metrics,
    }

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
