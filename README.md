## AAI-540 Group 4 — Extreme Precipitation Modeling + MLOps (SageMaker)
Final Project Repo for USD MSAAI 540  

**Group Members:**  
Ahmed Salem  
Victor Salcedo  

---
## Project Description

This project builds an end-to-end machine learning system to predict extreme rain events using historical weather data.  

We engineer time-series features from NOAA precipitation records to model the probability of extreme precipitation days. The workflow includes model training, threshold tuning, validation gates, batch inference, drift monitoring (PSI), and retraining trigger logic — forming a complete ML + MLOps pipeline implemented in SageMaker.

---
## Project Overview

This repository contains a complete end-to-end workflow for:

- Feature engineering for extreme precipitation prediction (time-series)  
- Model training (baseline)  
- Model evaluation with threshold tuning and validation gates  
- Batch inference (offline scoring)  
- Monitoring (feature drift using PSI) and retraining trigger logic  

---

## Repository Structure

```text
AAI-540-Group4/
│
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_Load_Processed_Data.ipynb
│   ├── 3_Create_Athena_DB.ipynb
│   ├── 4_Register_CSV_Athena.ipynb
│   ├── 5_ConvertS3csv_toParquet.ipynb
│   ├── 6_Create_Feature_Store.ipynb
│   ├── 7a_Log_Regression_Model_Training.ipynb
│   ├── 7b_Random_Forest_Model_Training.ipynb
│   ├── 8a_Model_Monitoring_Normal.ipynb
│   ├── 8b_Model_Monitoring_Alarm.ipynb
│   ├── 9_CI_CD_Pipelines.ipynb
│   └── README.md
│
├── data/
│   ├── extreme_precip_model_data.csv
│   └── new_scoring_data.csv
│
├── model/
│   ├── model.joblib
│   └── model (1).tar.gz
│
├── model_code/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── monitor.py
│   ├── test.py
│   └── untitled.txt
│
├── outputs/
│   ├── metrics.json
│   ├── evaluation_metrics.json
│   ├── predictions.csv
│   └── monitor_report.json
│
├── .gitignore
└── README.md
```


- `extreme_precip_model_data.csv` — Engineered dataset (features + target)
- `train.py` — Train model using time-based split and save artifacts
- `evaluate.py` — Evaluate model, auto-search best threshold (by F1), write metrics JSON, apply validation gate
- `inference.py` — Run batch inference and write predictions CSV
- `monitor.py` — Monitor drift (PSI) and output retrain-needed flag
- `model/` — Saved trained model artifact(s)
- `outputs/` — Metrics, predictions, and monitoring reports

---

## Requirements

In SageMaker Studio (or any Python 3 environment):

python3 -m pip install -U pandas numpy scikit-learn joblib

## Full Pipeline (Notebook)

The full end-to-end MLOps deliverables are implemented and documented within the `notebooks/` directory. These notebooks walk through data preparation, feature engineering, model training, monitoring, and CI/CD pipeline construction in SageMaker.

Core components include:

- Feature engineering for extreme precipitation prediction (time-series)  
- Model training (baseline)  
- Model evaluation with threshold tuning and validation gates  
- Batch inference (offline scoring)  
- Monitoring (feature drift using PSI) and retraining trigger logic  
- CI/CD pipeline implementation
  
## Run Full Pipeline (One Command)

You can run the entire ML + MLOps workflow in a single command:

- Train model
- Evaluate (threshold tuning + validation gate)
- Run batch inference
- Create new scoring batch
- Run drift monitoring



From the project root:

```bash
python3 train.py --data extreme_precip_model_data.csv --outdir outputs --threshold 0.5 && \
python3 evaluate.py --model model/model.joblib --data extreme_precip_model_data.csv --outdir outputs --min_f1 0.10 --min_recall 0.35 && \
python3 inference.py --model model/model.joblib --input extreme_precip_model_data.csv --output outputs/predictions.csv && \
python3 - << 'PY'
import pandas as pd
df = pd.read_csv("extreme_precip_model_data.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date")
new = df.tail(int(len(df)*0.2)).copy()
new.to_csv("new_scoring_data.csv", index=False)
print("Created new_scoring_data.csv", new.shape)
PY
python3 monitor.py --baseline extreme_precip_model_data.csv --new new_scoring_data.csv --outdir outputs
