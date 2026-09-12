# Bank-Marketing-MLOps-Pipeline-


An end-to-end machine learning pipeline that automates data ingestion,
validation, preprocessing, model training, monitoring, and deployment for a
bank marketing prediction model — orchestrated with Apache Airflow and
tracked with MLflow.

## Overview

The pipeline takes raw bank marketing data through a full production-style
MLOps workflow: schema setup → data validation → preprocessing → model
training (XGBoost, tuned with Optuna) → drift monitoring → deployment behind
a FastAPI service, with a Streamlit dashboard for interactive predictions.

## Architecture

```
Airflow DAG (bank_ml_full_pipeline_dag)
  │
  ├─ activate_env / start_docker / docker_exec   → environment + DB checks
  ├─ mlflow_task                                  → starts MLflow tracking server
  ├─ create_bank_bigtable      (schema_table.py)  → DB schema setup
  ├─ ge_validate_bank_data     (ge_validation.py) → Great Expectations validation
  ├─ data_injection            (data_injection.py)→ loads data into MySQL
  ├─ data_preprocessing        (preprocessing.py) → cleaning & feature prep
  ├─ model_training            (model_train.py)   → XGBoost + Optuna tuning, logged to MLflow
  ├─ monitor_data_drift_task   (datadrift.py)      → data drift checks (Evidently)
  └─ monitor_concept_drift_task (conceptdrift.py) → concept drift checks

FastAPI (api/app.py)       → serves predictions from the MLflow "Production" model
Streamlit (steamlit/main.py) → dashboard UI that calls the FastAPI service
```

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | Apache Airflow |
| Modeling | XGBoost, Scikit-learn, Optuna |
| Data validation | Great Expectations |
| Drift monitoring | Evidently |
| Experiment tracking | MLflow |
| Database / caching | MySQL (MariaDB), Redis |
| Serving | FastAPI |
| Dashboard | Streamlit |

## Project Structure

```
├── dags/bank.py            # Airflow DAG definition
├── api/app.py               # FastAPI prediction service
├── steamlit/main.py         # Streamlit dashboard
├── schema_table.py          # DB schema creation
├── data_injection.py        # Loads data into MySQL
├── preprocessing.py         # Data cleaning & feature engineering
├── model_train.py           # XGBoost training + Optuna hyperparameter search
├── model_deploy.py          # Model deployment/aliasing in MLflow
├── ge_validation.py         # Great Expectations data validation
├── datadrift.py             # Data drift monitoring (Evidently)
├── conceptdrift.py          # Concept drift monitoring
├── airflow.cfg               # Airflow configuration
├── webserver_config.py       # Airflow webserver configuration
├── requirements.txt
└── SETUP.md                  # Full step-by-step setup guide
```

## Setup

Full setup steps (system packages, Airflow install, Docker containers for
MySQL/Redis, running the DAG, MLflow UI, FastAPI, and Streamlit) are in
[`SETUP.md`](./SETUP.md) — follow it in order the first time you set this up
on a new machine.

Quick summary once everything is installed and containers are running:

```bash
# Terminal A
airflow webserver --port 8080

# Terminal B
airflow scheduler

# Trigger the pipeline
airflow dags trigger bank_ml_full_pipeline_dag

# Once training has completed at least once:
cd api && uvicorn app:app --reload --port 8000        # prediction API
cd steamlit && streamlit run main.py                   # dashboard
```

- Airflow UI: `http://localhost:8080`
- MLflow UI: `http://localhost:5000`
- API health check: `http://localhost:8000/health`
- Streamlit dashboard: `http://localhost:8501`

## Key Features

- Automated data validation and schema checks before training
- Hyperparameter-tuned XGBoost model (Optuna, 50 trials)
- Experiment tracking and model versioning via MLflow
- Automated data drift and concept drift monitoring
- Model serving through a FastAPI prediction endpoint
- Redis caching layer for repeated prediction requests
- Interactive Streamlit dashboard for making predictions

## What I Learned

This project helped me understand how machine learning systems can move
beyond experimentation into structured and automated production workflows —
from data validation and orchestration to monitoring for drift after
deployment.

## Note

Some configuration values (database credentials, absolute file paths) are
currently hardcoded for a specific local setup, as detailed in `SETUP.md`.
Update these to match your own environment before running.
