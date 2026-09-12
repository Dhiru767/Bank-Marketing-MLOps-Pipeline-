import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)
import pickle
import logging
import redis
from airflow.exceptions import AirflowException
import numpy as np
import os
import mlflow
import mlflow.xgboost
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(
    train_arrow_file,
    test_arrow_file,
    model_path,
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    mlflow_experiment_name="Bank_Model_Training",
    registered_model_name="Bank_XGB_Model",
    model_alias="Production"
):
    
    logger.info("Starting model training with Redis caching, MLflow tracking, and model registration")

    try:
        # Validate inputs
        if not all([train_arrow_file, test_arrow_file, model_path]):
            raise ValueError("train_arrow_file, test_arrow_file, and model_path must be provided")
        if not os.path.exists(train_arrow_file):
            raise ValueError(f"Training file {train_arrow_file} does not exist")
        if not os.path.exists(test_arrow_file):
            raise ValueError(f"Test file {test_arrow_file} does not exist")
        if not os.path.isdir(os.path.dirname(model_path)):
            raise ValueError(f"Model directory {os.path.dirname(model_path)} does not exist")

        # === Connect to Redis and clear cache ===
        try:
            redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)
            redis_client.ping()  # Test connection
            logger.info("Clearing all data from Redis")
            redis_client.flushdb()  # Clears all keys in the current Redis database
            logger.info("Redis cache cleared successfully")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise AirflowException(f"Redis connection error: {str(e)}")

        model_key = "bank_model:xgb"
        train_key, test_key = "bank_data:X_train", "bank_data:X_test"
        ytrain_key, ytest_key = "bank_data:y_train", "bank_data:y_test"

        # === Load data from Parquet files (no Redis cache check since cleared) ===
        try:
            logger.info("Loading data from Parquet files")
            df_train = pd.read_parquet(train_arrow_file)
            df_test = pd.read_parquet(test_arrow_file)
            logger.info(f"Loaded {len(df_train)} rows from {train_arrow_file}")
            logger.info(f"Loaded {len(df_test)} rows from {test_arrow_file}")

            # Validate data
            for df, name in [(df_train, train_arrow_file), (df_test, test_arrow_file)]:
                if 'y' not in df.columns:
                    raise ValueError(f"Parquet file {name} must contain 'y' column")
                df.columns = [str(col) for col in df.columns]

            # Prepare features and target
            X_train = df_train.drop(['y', 'id'], axis=1, errors='ignore')
            y_train = df_train['y']
            X_test = df_test.drop(['y', 'id'], axis=1, errors='ignore')
            y_test = df_test['y']

            # Cache data in Redis
            try:
                redis_client.set(train_key, pickle.dumps(X_train))
                redis_client.set(test_key, pickle.dumps(X_test))
                redis_client.set(ytrain_key, pickle.dumps(y_train))
                redis_client.set(ytest_key, pickle.dumps(y_test))
                logger.info("Cached train/test data in Redis")
            except redis.RedisError as e:
                logger.error(f"Failed to cache data in Redis: {str(e)}")
                raise AirflowException(f"Redis caching error: {str(e)}")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise AirflowException(f"Data loading error: {str(e)}")

        # === Set up MLflow ===
        mlflow.set_experiment(mlflow_experiment_name)
        with mlflow.start_run() as parent_run:
            logger.info(f"Started MLflow parent run with ID: {parent_run.info.run_id}")

            # === Define Optuna objective ===
            def objective(trial):
                param = {
                    "verbosity": 0,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "use_label_encoder": False,
                    "tree_method": "hist",
                    "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
                    "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    "subsample": trial.suggest_float("subsample", 0.3, 1.0),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                }

                # Start a nested MLflow run for each trial
                with mlflow.start_run(nested=True):
                    for key, value in param.items():
                        log_param(key, value)

                    model = xgb.XGBClassifier(**param)
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
                    log_metric("trial_accuracy", scores.mean())
                    return scores.mean()

            # === Run Optuna ===
            try:
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=50)
                logger.info(f"Best Optuna trial: {study.best_trial.value}")

                # Log best parameters to MLflow in the parent run
                best_params = study.best_params
                for key, value in best_params.items():
                    log_param(f"best_{key}", value)
                log_metric("best_cv_accuracy", study.best_trial.value)

                # Train the final model with the best parameters
                best_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, **best_params)
                best_model.fit(X_train, y_train)

                # === Save model to Redis ===
                try:
                    redis_client.set(model_key, pickle.dumps(best_model))
                    logger.info("Trained model cached in Redis")
                except redis.RedisError as e:
                    logger.error(f"Failed to cache model in Redis: {str(e)}")
                    raise AirflowException(f"Redis model caching error: {str(e)}")
            except Exception as e:
                logger.error(f"Optuna optimization failed: {str(e)}")
                raise AirflowException(f"Optuna error: {str(e)}")

            # === Log and register model to MLflow ===
            try:
                mlflow.xgboost.log_model(
                    xgb_model=best_model,
                    artifact_path="xgboost_model",
                    registered_model_name=registered_model_name
                )
                logger.info(f"XGBoost model logged to MLflow and registered as {registered_model_name}")

                # === Assign alias to the latest model version ===
                client = MlflowClient()
                latest_versions = client.get_latest_versions(registered_model_name)
                if latest_versions:
                    latest_version = latest_versions[0].version
                    client.set_registered_model_alias(
                        name=registered_model_name,
                        alias=model_alias,
                        version=latest_version
                    )
                    logger.info(f"Assigned alias '{model_alias}' to model version {latest_version} of {registered_model_name}")
                else:
                    logger.warning(f"No versions found for model {registered_model_name}")
            except Exception as e:
                logger.error(f"MLflow model registration failed: {str(e)}")
                raise AirflowException(f"MLflow registration error: {str(e)}")

            # === Evaluate on train and test sets ===
            try:
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                train_report = classification_report(y_train, y_train_pred, output_dict=True)
                test_report = classification_report(y_test, y_test_pred, output_dict=True)

                train_cm = confusion_matrix(y_train, y_train_pred)
                test_cm = confusion_matrix(y_test, y_test_pred)

                train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
                test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

                train_ap = average_precision_score(y_train, best_model.predict_proba(X_train)[:, 1])
                test_ap = average_precision_score(y_test, best_model.predict_proba(X_test)[:, 1])
            except Exception as e:
                logger.error(f"Model evaluation failed: {str(e)}")
                raise AirflowException(f"Evaluation error: {str(e)}")

            # === Save and log classification reports as artifacts ===
            try:
                train_report_file = "train_classification_report.txt"
                test_report_file = "test_classification_report.txt"
                with open(train_report_file, "w") as f:
                    f.write(classification_report(y_train, y_train_pred))
                with open(test_report_file, "w") as f:
                    f.write(classification_report(y_test, y_test_pred))
                log_artifact(train_report_file)
                log_artifact(test_report_file)
                os.remove(train_report_file)
                os.remove(test_report_file)
                logger.info("Classification reports saved and logged to MLflow")
            except Exception as e:
                logger.error(f"Failed to save/log classification reports: {str(e)}")
                raise AirflowException(f"Artifact logging error: {str(e)}")

            # === Log metrics to MLflow ===
            try:
                log_metric("train_roc_auc", train_roc_auc)
                log_metric("test_roc_auc", test_roc_auc)
                log_metric("train_avg_precision", train_ap)
                log_metric("test_avg_precision", test_ap)

                for label in train_report.keys():
                    if isinstance(train_report[label], dict):
                        for metric, value in train_report[label].items():
                            log_metric(f"train_{label}_{metric}", value)
                        for metric, value in test_report[label].items():
                            log_metric(f"test_{label}_{metric}", value)
            except Exception as e:
                logger.error(f"Failed to log metrics to MLflow: {str(e)}")
                raise AirflowException(f"Metric logging error: {str(e)}")

            # === Log confusion matrices as artifacts ===
            try:
                train_cm_file = "train_confusion_matrix.txt"
                test_cm_file = "test_confusion_matrix.txt"
                np.savetxt(train_cm_file, train_cm, fmt="%d")
                np.savetxt(test_cm_file, test_cm, fmt="%d")
                log_artifact(train_cm_file)
                log_artifact(test_cm_file)
                os.remove(train_cm_file)
                os.remove(test_cm_file)
                logger.info("Confusion matrices saved and logged to MLflow")
            except Exception as e:
                logger.error(f"Failed to save/log confusion matrices: {str(e)}")
                raise AirflowException(f"Confusion matrix logging error: {str(e)}")

            # === Save model to file and log as artifact ===
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                log_artifact(model_path)
                logger.info(f"Model saved to {model_path} and logged to MLflow as a file artifact")
            except Exception as e:
                logger.error(f"Failed to save/log model file: {str(e)}")
                raise AirflowException(f"Model saving error: {str(e)}")

            logger.info("Model training/evaluation completed successfully")
            return True

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise AirflowException(f"Training error: {str(e)}")