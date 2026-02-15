import argparse
import joblib
import pandas as pd

NUM_COLS = ["prcp_lag_1", "prcp_roll_7", "TMAX", "TMIN"]
CAT_COLS = ["station_id"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/model.joblib")
    ap.add_argument("--input", required=True, help="CSV with feature columns")
    ap.add_argument("--output", default="outputs/predictions.csv")
    args = ap.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)

    X = df[NUM_COLS + CAT_COLS]
    prob = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["extreme_precip_prob"] = prob
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
