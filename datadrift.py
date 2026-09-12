import pandas as pd
from sqlalchemy import create_engine
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
from pathlib import Path
import logging
from airflow.exceptions import AirflowException
from airflow.utils.email import send_email

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_data_drift():
    """
    Monitor data drift in the bank dataset by comparing feature distributions
    between reference (historical) and current (new) data from MySQL tables.
    Uses up to 1000 rows for both datasets, checks drift against a 0.5 threshold,
    and sends an email if the threshold is exceeded.
    """
    try:

        db_url = "mysql+pymysql://bishnu:bishnu%40pass123@localhost:3306/bank"
        reference_table = "bank_train_data"
        current_table = "new_bank_data"
        reports_folder = Path("/home/dhiraj-bohara/projects/MLops/reports")
        report_html = reports_folder / "data_drift_report.html"
        report_csv = reports_folder / "data_drift_report.csv"
        DRIFT_THRESHOLD = 0.5
        MIN_ROWS = 10  # Minimum rows required for drift analysis

        reports_folder.mkdir(parents=True, exist_ok=True)

        # Create SQLAlchemy engine
        engine = create_engine(db_url)

        # Load reference data (limit to 1000 rows)
        reference = pd.read_sql(f"SELECT * FROM {reference_table} LIMIT 1000", con=engine)
        logger.info(f"Loaded reference dataset with {len(reference)} rows from MySQL table {reference_table}")
        logger.info(f"Reference dataset columns: {list(reference.columns)}")

        # Validate reference dataset
        if len(reference) < MIN_ROWS:
            raise AirflowException(f"Reference dataset has insufficient rows: {len(reference)}")

        # Load current data (limit to 1000 rows, most recent first)
        current = pd.read_sql(f"SELECT * FROM {current_table} LIMIT 1000", con=engine)
        if len(current) < MIN_ROWS:
            raise AirflowException(f"Current dataset has insufficient rows: {len(current)}")
        logger.info(f"Loaded current dataset with {len(current)} rows from MySQL table {current_table}")
        logger.info(f"Current dataset columns: {list(current.columns)}")

        # Ensure consistent column names
        reference.columns = [str(col) for col in reference.columns]
        current.columns = [str(col) for col in current.columns]

        # Define expected columns
        expected_cols = [
            'age', 'job', 'marital', 'education', 'campaign', 'previous',
            'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m',
            'nr_employed', 'contact_flag', 'default_flag', 'housing_flag', 'loan_flag'
        ]

        # Validate columns
        missing_ref_cols = [col for col in expected_cols if col not in reference.columns]
        missing_curr_cols = [col for col in expected_cols if col not in current.columns]
        if missing_ref_cols:
            raise AirflowException(f"Missing columns in reference dataset: {missing_ref_cols}")
        if missing_curr_cols:
            raise AirflowException(f"Missing columns in current dataset: {missing_curr_cols}")

        # Select expected columns
        reference = reference[expected_cols]
        current = current[expected_cols]

        # Ensure correct data types
        numeric_cols = [
            'age', 'campaign', 'previous', 'emp_var_rate', 'cons_price_idx',
            'cons_conf_idx', 'euribor3m', 'nr_employed', 'contact_flag',
            'default_flag', 'housing_flag', 'loan_flag'
        ]
        categorical_cols = ['job', 'marital', 'education']
        for col in numeric_cols:
            reference[col] = pd.to_numeric(reference[col], errors='coerce')
            current[col] = pd.to_numeric(current[col], errors='coerce')
        for col in categorical_cols:
            reference[col] = reference[col].astype(str)
            current[col] = current[col].astype(str)

        # Check for null values
        if reference[numeric_cols].isnull().any().any():
            raise AirflowException("Null values found in numeric columns of reference dataset")
        if current[numeric_cols].isnull().any().any():
            raise AirflowException("Null values found in numeric columns of current dataset")

        # Define column mapping for Evidently
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = numeric_cols
        column_mapping.categorical_features = categorical_cols

        # Run data drift report
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping
        )

        # Save HTML report
        report.save_html(report_html)
        logger.info(f"Data drift HTML report saved at {report_html}")

        # Extract and save drift summary
        drift_results = report.as_dict()
        drift_summary = {
            'drift_detected': drift_results['metrics'][0]['result']['dataset_drift'],
            'number_of_drifted_columns': drift_results['metrics'][0]['result']['number_of_drifted_columns'],
            'share_of_drifted_columns': drift_results['metrics'][0]['result']['share_of_drifted_columns']
        }
        pd.DataFrame([drift_summary]).to_csv(report_csv, index=False)
        logger.info(f"Data drift summary saved at {report_csv}")

        # Check drift against threshold and send email if exceeded
        if drift_summary['share_of_drifted_columns'] > DRIFT_THRESHOLD:
            email_subject = "Data Drift Alert: Bank Dataset"
            email_body = (
                f"Data drift detected!\n"
                f"Share of drifted columns: {drift_summary['share_of_drifted_columns']:.2f}\n"
                f"Number of drifted columns: {drift_summary['number_of_drifted_columns']}\n"
                f"Threshold: {DRIFT_THRESHOLD}\n"
                f"See report at: {report_html}\n"
                f"Summary saved at: {report_csv}"
            )
            try:
                send_email(
                    to=["bishnuupadhyaya590@gmail.com"],
                    subject=email_subject,
                    html_content=email_body
                )
                logger.info("Email notification sent for data drift.")
            except Exception as email_error:
                raise AirflowException(f"Email sending error: {str(email_error)}")
        elif drift_summary['drift_detected']:
            logger.warning(f"Data drift detected but within threshold: "
                          f"{drift_summary['number_of_drifted_columns']} columns drifted")
        else:
            logger.info("No data drift detected")

        return True

    except Exception as e:
        logger.error(f"Data drift monitoring failed: {str(e)}")
        raise AirflowException(f"Data drift monitoring error: {str(e)}")
