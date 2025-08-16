# preprocess_data.py
import pandas as pd
import numpy as np
import os

print("Starting data pre-processing...")

# Define file paths
DATA_FOLDER = 'data'
ATTRIBUTES_FILE = os.path.join(DATA_FOLDER, 'attributes.csv')
OUTPUT_FILE = os.path.join(DATA_FOLDER, 'preprocessed_data.parquet')

try:
    print(f"Reading {ATTRIBUTES_FILE}...")
    df_attributes = pd.read_csv(ATTRIBUTES_FILE)
    
    # Clean up column names by removing any leading/trailing whitespace.
    df_attributes.columns = df_attributes.columns.str.strip()
    print("Column headers cleaned.")
    
    print("Reshaping and pivoting data...")
    id_vars = ['Code', 'Event Name', 'Remedial Service Type', 'Industry NAICS']
    
    value_vars = [c for c in df_attributes.columns if any(str(y) in c for y in range(2025, 2031))]
    df_long = pd.melt(df_attributes, id_vars=id_vars, value_vars=value_vars, var_name='Metric_Year', value_name='Value')
    
    # Force the 'Value' column to be numeric, automatically fixing errors.
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
    print("Data values converted to numeric type, fixing errors.")

    df_long[['Metric', 'Year']] = df_long['Metric_Year'].str.rsplit('_', n=1, expand=True)
    df_long['Year'] = pd.to_numeric(df_long['Year'])
    
    # Drop rows where critical data might be missing after coercion
    df_long.dropna(subset=['Value'], inplace=True)

    df_cyber = df_long.pivot_table(
        index=['Year', 'Industry NAICS', 'Code', 'Event Name', 'Remedial Service Type'],
        columns='Metric',
        values='Value'
    ).reset_index()

    print("Calculating simulation parameters...")
    df_cyber['Cost_mu'] = np.log(df_cyber['Cost']) - (0.3**2 / 2)
    df_cyber['Cost_sigma'] = 0.3
    
    print(f"Saving pre-processed data to {OUTPUT_FILE}...")
    df_cyber.to_parquet(OUTPUT_FILE)
    
    print("\n✅ Success! Pre-processing complete.")
    print(f"Your API will now use the fast-loading file: {OUTPUT_FILE}")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")