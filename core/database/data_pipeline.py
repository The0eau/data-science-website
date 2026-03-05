import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# ---------------------------------------------------------
# 1. EXTRACT (Extraction)
# Simulation: Creating "dirty" raw data from a source
# ---------------------------------------------------------
print("--- Step 1: Extracting raw data ---")

raw_data = {
    'transaction_id': [101, 102, 103, 104, 105],
    'product': ['  iphone 15  ', 'Macbook Pro', 'ipad air', 'apple watch', None],
    'price_usd': [999.0, 2400.0, 599.0, -50.0, 399.0], # Note: Negative price and None value
    'date': ['2026-01-01', '2026-01-02', '2026-01-02', '2026-01-03', '2026-01-04']
}

df = pd.DataFrame(raw_data)
print(f"Raw data collected:\n{df}\n")

# ---------------------------------------------------------
# 2. TRANSFORM (Transformation)
# Cleaning, filtering, and enriching the data
# ---------------------------------------------------------
print("--- Step 2: Transforming data ---")

# A. Data Cleaning: Remove rows with missing products or invalid prices (price <= 0)
df = df.dropna(subset=['product', 'price_usd'])
df = df[df['price_usd'] > 0]

# B. Normalization: Trim whitespace and convert product names to UPPERCASE
df['product'] = df['product'].str.strip().str.upper()

# C. Calculation: Convert USD to EUR (Exchange rate: 0.92)
df['price_eur'] = df['price_usd'] * 0.92

# D. Enrichment: Add a metadata column (Processing Timestamp)
df['processed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"Cleaned and transformed data:\n{df}\n")

# ---------------------------------------------------------
# 3. LOAD (Loading)
# Sending the data to the Data Warehouse (SQLite for this example)
# ---------------------------------------------------------
print("--- Step 3: Loading into Data Warehouse ---")

# Create SQLAlchemy connection (Change to PostgreSQL/MySQL easily here)
engine = create_engine('sqlite:///company_warehouse.db')

# Load the data (Replaces the table if it already exists)
df.to_sql('final_sales_report', engine, if_exists='replace', index=False)

print("Pipeline executed successfully! Data is now in 'company_warehouse.db'.")

# ---------------------------------------------------------
# VERIFICATION (Querying the result)
# ---------------------------------------------------------
query_result = pd.read_sql('SELECT * FROM final_sales_report', engine)
print(f"\nFinal Table in Database:\n{query_result}")