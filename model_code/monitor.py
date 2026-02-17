import os
import json
import argparse
import numpy as np
import pandas as pd

TIME_COL = "date"
ID_COL = "station_id"
NUM_FEATURES = ["prcp_lag_1", "prcp_roll_7", "TMAX", "TMIN"]


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) for numeric drift detection.
    Rule of thumb:
      <0.1 = no/minor drift
      0.1-0.2 = moderate drift
      >0.2 = significant drift
    """
    e = expected.dropna().astype(float).values
    a = actual.dropna().astype(float).values
    if len(e) < 10 or len(a) < 10:
        return 0.0

    quantiles = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0

    def proportions(x):
        counts, _ = np.histogram(x, bins=quantiles)
        props = counts / max(counts.sum(), 1)
        # prevent zeros
        props = np.where(props == 0, 1e-6, props)
        return props

    e_p = proportions(e)
    a_p = proportions(a)

    return float(np.sum((e_p - a_p) * np.log(e_p / a_p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline CSV (training/reference)")
    ap.add_argument("--new", required=True, help="New CSV (recent/scoring batch)")
    ap.add_argument("--outdir", default="outputs", help="Output folder")
    ap.add_argument("--psi_threshold", type=float, default=0.2, help="PSI drift threshold to trigger retrain")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    baseline = pd.read_csv(args.baseline)
    new_data = pd.read_csv(args.new)

    if TIME_COL in baseline.columns:
        baseline[TIME_COL] = pd.to_datetime(baseline[TIME_COL], errors="coerce")
    if TIME_COL in new_data.columns:
        new_data[TIME_COL] = pd.to_datetime(new_data[TIME_COL], errors="coerce")

    drift = {}
    missing = []
    for f in NUM_FEATURES:
        if f not in baseline.columns or f not in new_data.columns:
            missing.append(f)
            continue
        drift[f] = psi(baseline[f], new_data[f])

    retrain_needed = any(v > args.psi_threshold for v in drift.values())

    report = {
        "baseline_path": args.baseline,
        "new_path": args.new,
        "psi_threshold": args.psi_threshold,
        "missing_features": missing,
        "psi": drift,
        "retrain_needed": retrain_needed,
        "notes": "PSI computed on numeric features. Use a truly new batch for meaningful drift detection."
    }

    out_path = os.path.join(args.outdir, "monitor_report.json")
    with open(out_path, "w") as fp:
        json.dump(report, fp, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
