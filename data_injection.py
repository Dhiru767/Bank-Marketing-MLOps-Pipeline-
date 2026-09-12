import pandas as pd
from sqlalchemy import create_engine
import logging

def run():
    """
    Inject data from bank-additional-full.csv into the MySQL bank_data table.
    """
    # Database connection
    db_url = "mysql+pymysql://bishnu:bishnu%40pass123@localhost:3306/bank"
    csv_path = "/home/dhiraj-bohara/projects/MLops/data/bank-additional-full.csv"
    table_name = "bank_data"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:

        engine = create_engine(db_url)
        logger.info("Database connection established.")


        df = pd.read_csv(csv_path, sep=';')
        logger.info(f"Loaded CSV file with {len(df)} rows.")

        df = df.rename(columns={
            'emp.var.rate': 'emp_var_rate',
            'cons.price.idx': 'cons_price_idx',
            'cons.conf.idx': 'cons_conf_idx',
            'nr.employed': 'nr_employed'
        })

        expected_columns = [
            'age', 'job', 'marital', 'education', 'default', 'housing', 
            'loan', 'contact', 'month', 'day_of_week', 'duration', 
            'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 
            'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'y'
        ]
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in CSV: {missing_columns}")
            raise ValueError(f"Missing columns in CSV: {missing_columns}")

        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        logger.info(f"Successfully injected {len(df)} rows into {table_name} table.")

    except Exception as e:
        logger.error(f"Error during data injection: {str(e)}")
        raise

    return True