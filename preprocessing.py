import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import category_encoders as ce
from imblearn.over_sampling import SMOTE
import logging
import pickle
import redis
import pyarrow.parquet as pq
import pyarrow as pa
import os
from airflow.exceptions import AirflowException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(db_url, table_name, train_arrow_file, train_output_table, output_csv,
        test_arrow_file, test_output_table, test_output_csv,
        redis_host='localhost', redis_port=6379, redis_db=0):
    """
    Preprocess the bank dataset from MySQL with Redis caching and save the processed train and test data.
    Clears all Redis data before processing and storing.
    """

    logger.info("Starting preprocessing on bank dataset")
    try:
        # Initialize Redis connection
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)

        # Step 1: Clear all data from Redis
        logger.info("Clearing all data from Redis")
        redis_client.flushdb()  # Clears all keys in the current Redis database
        logger.info("Redis cache cleared successfully")

        # Step 2: Load raw data from MySQL (no cache check since we cleared Redis)
        engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")
        df_bank_tmp = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
        
        # Cache raw data in Redis
        raw_key = f"bank_data:{table_name}"
        redis_client.set(raw_key, pickle.dumps(df_bank_tmp))
        logger.info(f"Cached raw data in Redis with key: {raw_key}")

        df = df_bank_tmp.copy()
        logger.info(f"Loaded {len(df)} rows from table: {table_name}")

        # Clean column names
        df.columns = [str(col).replace('.', '_') for col in df.columns]

        # Drop unnecessary columns
        cols_to_remove = ['duration', 'month', 'day_of_week', 'poutcome', 'pdays']
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])

        # Remove unknowns and NaNs
        df = df[~(df.astype(str) == "unknown").any(axis=1)]
        df = df.dropna().reset_index(drop=True)

        # Map target
        df['y'] = df['y'].map({'yes': 1, 'no': 0})

        # Split features and target
        X = df.drop('y', axis=1)
        y = df['y']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Impute numerical columns
        numeric_features = ['age', 'campaign', 'previous', 'emp_var_rate',
                            'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']
        numeric_features = [col for col in numeric_features if col in X_train.columns]
        if numeric_features:
            imputer = SimpleImputer(strategy='mean')
            X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features])
            X_test[numeric_features] = imputer.transform(X_test[numeric_features])

        # Binary encoding
        binary_flag_mappings = [
            ('contact', 'contact_flag'),
            ('default', 'default_flag'),
            ('housing', 'housing_flag'),
            ('loan', 'loan_flag')
        ]
        binary_cols = []
        for col, new_col in binary_flag_mappings:
            if col in X_train.columns:
                X_train[new_col] = (X_train[col] == 'yes' if col != 'contact' else X_train[col] == 'cellular').astype(int)
                X_test[new_col] = (X_test[col] == 'yes' if col != 'contact' else X_test[col] == 'cellular').astype(int)
                binary_cols.append(col)
        X_train = X_train.drop(columns=[col for col in binary_cols if col in X_train.columns])
        X_test = X_test.drop(columns=[col for col in binary_cols if col in X_test.columns])

        # Target encoding for categorical columns
        categorical_cols = ['job', 'marital', 'education']
        categorical_cols = [col for col in categorical_cols if col in X_train.columns]
        if categorical_cols:
            target_encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=1.0)
            X_train[categorical_cols] = target_encoder.fit_transform(X_train[categorical_cols], y_train)
            X_test[categorical_cols] = target_encoder.transform(X_test[categorical_cols])

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
        y_train_resampled = pd.Series(y_train_resampled, name='y')

        # Combine train/test
        processed_train = pd.concat([X_train_resampled, y_train_resampled], axis=1)
        processed_test = pd.concat([X_test, y_test], axis=1)

        # Cache processed train/test in Redis
        train_key, test_key = "bank_data:train", "bank_data:test"
        redis_client.set(train_key, pickle.dumps(processed_train))
        redis_client.set(test_key, pickle.dumps(processed_test))
        logger.info("Cached processed train and test data in Redis")

        # === Save to MySQL, CSV, Parquet ===
        try:
            engine = create_engine(db_url)
            processed_train.to_sql(train_output_table, con=engine, if_exists='replace', index=False)
            processed_train.to_csv(output_csv, index=False)
            pq.write_table(pa.Table.from_pandas(processed_train), train_arrow_file)
            logger.info("Saved processed training data to MySQL/CSV/Parquet")

            processed_test.to_sql(test_output_table, con=engine, if_exists='replace', index=False)
            processed_test.to_csv(test_output_csv, index=False)
            pq.write_table(pa.Table.from_pandas(processed_test), test_arrow_file)
            logger.info("Saved processed test data to MySQL/CSV/Parquet")
        except Exception as e:
            logger.error(f"Save error: {str(e)}")
            raise AirflowException(f"Save error: {str(e)}")

        logger.info("Data preprocessing completed successfully")
        return True

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise AirflowException(f"Preprocessing error: {str(e)}")