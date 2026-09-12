import sqlalchemy as sa
from sqlalchemy import create_engine

def create_bigtable():
    """
    Create the bank_data table in the MySQL database if it doesn't exist.
    The schema is based on the typical bank-additional-full dataset structure.
    """
    
    db_url = "mysql+pymysql://bishnu:bishnu%40pass123@localhost:3306/bank"
    engine = create_engine(db_url)
    metadata = sa.MetaData()
    
    bank_data = sa.Table(
        'bank_data',
        metadata,
        sa.Column('age', sa.Integer, nullable=False),
        sa.Column('job', sa.String(50)),
        sa.Column('marital', sa.String(20)),
        sa.Column('education', sa.String(50)),
        sa.Column('default', sa.String(20)),
        sa.Column('housing', sa.String(20)),
        sa.Column('loan', sa.String(20)),
        sa.Column('contact', sa.String(20)),
        sa.Column('month', sa.String(10)),
        sa.Column('day_of_week', sa.String(10)),
        sa.Column('duration', sa.Integer),
        sa.Column('campaign', sa.Integer),
        sa.Column('pdays', sa.Integer),
        sa.Column('previous', sa.Integer),
        sa.Column('poutcome', sa.String(20)),
        sa.Column('emp_var_rate', sa.Float),
        sa.Column('cons_price_idx', sa.Float),
        sa.Column('cons_conf_idx', sa.Float),
        sa.Column('euribor3m', sa.Float),
        sa.Column('nr_employed', sa.Float),
        sa.Column('y', sa.String(5))  
    )
    
    metadata.create_all(engine)
    
    with engine.connect() as conn:
        result = conn.execute(sa.text("SHOW TABLES LIKE 'bank_data'")).fetchall()
        if result:
            print("Table 'bank_data' created successfully or already exists.")
        else:
            print("Failed to create table 'bank_data'.")