import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

TARGET_COL = "extreme_precip_tomorrow"
TIME_COL = "date"
ID_COL = "station_id"

NUM_COLS = ["prcp_lag_1", "prcp_roll_7", "TMAX", "TMIN"]
CAT_COLS = [ID_COL]


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)


def time_based_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    split_date = df[TIME_COL].quantile(1 - test_ratio)
    test_df = df[df[TIME_COL] > split_date].copy()
    return test_df, split_date


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = None

    # PR-AUC
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["pr_auc"] = None

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    return metrics, cm, report


def threshold_search(y_true, y_prob, step=0.05):
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}

    thresholds = [round(i * step, 2) for i in range(1, int(1 / step))]
    for t in thresholds:
        m, _, _ = compute_metrics(y_true, y_prob, threshold=t)
        if m["f1"] > best["f1"]:
            best = {
                "threshold": float(t),
                "f1": float(m["f1"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
            }
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to saved model joblib file")
    ap.add_argument("--data", required=True, help="CSV path containing features + target")
    ap.add_argument("--outdir", default="outputs", help="Output folder")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Size of time-based test split")
    ap.add_argument("--min_f1", type=float, default=0.20, help="Validation gate: minimum F1")
    ap.add_argument("--min_recall", type=float, default=0.50, help="Validation gate: minimum Recall")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Optional: force a threshold (otherwise best-F1 threshold search is used)")
    ap.add_argument("--search_step", type=float, default=0.05,
                    help="Threshold search step size (e.g., 0.05 checks 0.05..0.95)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    model = joblib.load(args.model)

    # Load data
    df = load_df(args.data)

    # Get test data
    test_df, split_date = time_based_test_split(df, test_ratio=args.test_ratio)

    required = set(NUM_COLS + CAT_COLS + [TARGET_COL])
    missing = sorted(list(required - set(test_df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    X_test = test_df[NUM_COLS + CAT_COLS]
    y_test = test_df[TARGET_COL].astype(int).values

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    if args.threshold is None:
        best = threshold_search(y_test, y_prob, step=args.search_step)
        chosen_threshold = best["threshold"]
        print("Best threshold search (by F1):", json.dumps(best, indent=2))
    else:
        chosen_threshold = args.threshold
        best = None
        print(f"Using user-provided threshold: {chosen_threshold}")

    metrics, cm, report = compute_metrics(y_test, y_prob, threshold=chosen_threshold)

    payload = {
        "split_date": str(split_date),
        "n_test": int(len(test_df)),
        "best_threshold_search": best,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_columns": {"numeric": NUM_COLS, "categorical": CAT_COLS},
        "target_column": TARGET_COL,
        "model_path": args.model,
        "data_path": args.data,
    }

    out_path = os.path.join(args.outdir, "evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nFinal metrics at chosen threshold:")
    print(json.dumps(metrics, indent=2))

    print("\nConfusion Matrix:")
    print(np.array(cm))

    #Validation gate
    if metrics["f1"] < args.min_f1 or metrics["recall"] < args.min_recall:
        raise SystemExit(
            f"\nVALIDATION FAILED: F1={metrics['f1']:.3f}, Recall={metrics['recall']:.3f} "
            f"(min_f1={args.min_f1}, min_recall={args.min_recall})"
        )

    print(f"\nalidation passed. Wrote: {out_path}")


if __name__ == "__main__":
    main()
