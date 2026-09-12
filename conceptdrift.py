import pandas as pd
from sqlalchemy import create_engine
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset
from evidently import ColumnMapping
from pathlib import Path
import logging
from airflow.exceptions import AirflowException
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_concept_drift(
    db_url: str = "mysql+pymysql://bishnu:bishnu%40pass123@localhost:3306/bank",
    reference_table: str = "bank_train_data",
    current_table: str = "new_bank_data",
    reports_folder_path: str = "/home/dhiraj-bohara/projects/MLops/reports"
):
    """
    Monitor concept drift between reference and current datasets using Evidently.

    Args:
        db_url (str): Database connection URL
        reference_table (str): Reference table in MySQL
        current_table (str): Current data table in MySQL
        reports_folder_path (str): Folder path to save reports/plots

    Returns:
        dict: Summary of target drift results

    Raises:
        AirflowException: On critical errors
    """
    try:
        # Create reports folder
        reports_folder = Path(reports_folder_path)
        reports_folder.mkdir(parents=True, exist_ok=True)
        report_html = reports_folder / "concept_drift_report.html"
        report_csv = reports_folder / "concept_drift_report.csv"
        target_dist_plot = reports_folder / "target_distribution.png"

        # Define expected columns
        expected_cols = [
            'age', 'job', 'marital', 'education', 'campaign', 'previous',
            'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m',
            'nr_employed', 'contact_flag', 'default_flag', 'housing_flag',
            'loan_flag', 'y'
        ]
        numeric_cols = [
            'age', 'campaign', 'previous', 'emp_var_rate', 'cons_price_idx',
            'cons_conf_idx', 'euribor3m', 'nr_employed',
            'contact_flag', 'default_flag', 'housing_flag', 'loan_flag', 'y'
        ]
        categorical_cols = ['job', 'marital', 'education']

        # Connect to DB
        engine = create_engine(db_url)

        # Load reference data
        try:
            with engine.connect() as conn:
                reference = pd.read_sql(f"SELECT * FROM {reference_table} LIMIT 1000", conn)
                logger.info(f"Loaded reference dataset: {reference.shape}")
        except Exception as e:
            raise AirflowException(f"Failed to load reference data from table '{reference_table}': {e}")

        # Load current data
        try:
            with engine.connect() as conn:
                current = pd.read_sql(f"SELECT * FROM {current_table} LIMIT 1000", conn)
                logger.info(f"Loaded current dataset: {current.shape}")
        except Exception as e:
            raise AirflowException(f"Failed to load current data from table '{current_table}': {e}")

        # Validate datasets
        if reference.empty or current.empty:
            raise AirflowException("Reference or current dataset is empty.")

        # Check missing columns
        missing_ref = [c for c in expected_cols if c not in reference.columns]
        missing_cur = [c for c in expected_cols if c not in current.columns]
        if missing_ref or missing_cur:
            raise AirflowException(f"Missing columns - Reference: {missing_ref}, Current: {missing_cur}")

        # Select expected columns
        reference = reference[expected_cols]
        current = current[expected_cols]

        # Convert types
        for col in numeric_cols:
            reference[col] = pd.to_numeric(reference[col], errors="coerce")
            current[col] = pd.to_numeric(current[col], errors="coerce")
        for col in categorical_cols:
            reference[col] = reference[col].astype(str)
            current[col] = current[col].astype(str)

        # Fill missing values
        reference[numeric_cols] = reference[numeric_cols].fillna(reference[numeric_cols].median())
        current[numeric_cols] = current[numeric_cols].fillna(current[numeric_cols].median())
        for col in categorical_cols:
            reference[col] = reference[col].fillna("Unknown")
            current[col] = current[col].fillna("Unknown")

        # Validate target column
        if current['y'].isnull().all():
            raise AirflowException("Current dataset 'y' column is empty.")
        if current['y'].isnull().any():
            logger.warning("Filling missing values in 'y' with mode.")
            current['y'] = current['y'].fillna(current['y'].mode()[0])

        # Evidently column mapping
        column_mapping = ColumnMapping(
            target="y",
            numerical_features=[
                'age', 'campaign', 'previous', 'emp_var_rate', 'cons_price_idx',
                'cons_conf_idx', 'euribor3m', 'nr_employed'
            ],
            categorical_features=[
                'job', 'marital', 'education',
                'contact_flag', 'default_flag', 'housing_flag', 'loan_flag'
            ]
        )

        # Run drift report
        report = Report(metrics=[TargetDriftPreset()])
        report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
        report.save_html(report_html)

        # Extract summary
        drift_results = report.as_dict()
        drift_summary = {
            "target_drift_detected": drift_results['metrics'][0]['result']['drift_detected'],
            "target_drift_score": drift_results['metrics'][0]['result']['drift_score']
        }
        pd.DataFrame([drift_summary]).to_csv(report_csv, index=False)

        # Plot target distribution
        plt.figure(figsize=(8, 6))
        sns.histplot(reference['y'], stat="probability", label="Reference", color="blue", bins=2, alpha=0.5)
        sns.histplot(current['y'], stat="probability", label="Current", color="orange", bins=2, alpha=0.5)
        plt.title("Target Variable Distribution (y)")
        plt.legend()
        plt.savefig(target_dist_plot, bbox_inches='tight')
        plt.close()

        logger.info(f"Drift report saved: {report_html}")
        logger.info(f"Summary saved: {report_csv}")
        logger.info(f"Plot saved: {target_dist_plot}")

        return drift_summary

    except AirflowException as ae:
        logger.error(f"Concept drift monitoring failed: {ae}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise AirflowException(f"Unexpected error: {e}")
