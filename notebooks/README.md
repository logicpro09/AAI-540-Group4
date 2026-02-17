# Notebooks — AAI-540 Group 4  
Extreme Precipitation Modeling + MLOps Pipeline

This directory contains the primary deliverables for the AAI-540 final project.  
The notebooks implement a complete end-to-end ML + MLOps workflow using AWS services (S3, Athena, SageMaker Feature Store, Model Monitor, and Pipelines).

The notebooks are designed to be executed sequentially.

---

## Execution Order

### 1. `1_EDA.ipynb`
Exploratory Data Analysis.
- Initial inspection of precipitation dataset
- Data cleaning validation
- Feature distribution analysis
- Target class balance evaluation

---

### 2. `2_Load_Processed_Data.ipynb`
Loads processed dataset for cloud integration.
- Prepares data for S3 upload
- Validates schema consistency
- Establishes dataset structure for downstream steps

---

### 3. `3_Create_Athena_DB.ipynb`
Creates AWS Athena database.
- Configures query environment
- Connects S3 data lake to Athena
- Enables SQL-based exploration

---

### 4. `4_Register_CSV_Athena.ipynb`
Registers CSV dataset in Athena.
- Defines table schema
- Enables structured SQL queries over S3 data

---

### 5. `5_ConvertS3csv_toParquet.ipynb`
Optimizes data format.
- Converts CSV to Parquet
- Improves query performance
- Reduces storage and scanning cost

---

### 6. `6_Create_Feature_Store.ipynb`
Creates SageMaker Feature Store.
- Defines feature group
- Registers engineered features
- Prepares data for model training and inference

---

### 7a. `7a_Log_Regression_Model_Training.ipynb`
Baseline model training.
- Time-based train/test split
- Logistic regression implementation
- Performance evaluation

---

### 7b. `7b_Random_Forest_Model_Training.ipynb`
Tree-based model training.
- Random Forest implementation
- Model comparison against baseline
- Performance metrics analysis

---

### 8a. `8a_Model_Monitoring_Normal.ipynb`
Model monitoring — normal distribution scenario.
- Establishes baseline statistics
- Computes drift metrics
- Validates monitoring workflow under expected conditions

---

### 8b. `8b_Model_Monitoring_Alarm.ipynb`
Model monitoring — alarm scenario.
- Simulates feature drift
- Triggers monitoring alert logic
- Evaluates retraining criteria

---

### 9. `9_CI_CD_Pipelines.ipynb`
CI/CD orchestration.
- Defines SageMaker Pipeline
- Automates training, evaluation, and deployment workflow
- Simulates production-ready ML lifecycle management

---

## Summary

Together, these notebooks represent a full machine learning lifecycle:

Data Exploration → Data Lake Integration → Feature Engineering → Model Training → Monitoring → CI/CD Automation

They collectively simulate a production-grade ML system for extreme precipitation prediction.

