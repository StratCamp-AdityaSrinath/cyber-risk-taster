# preprocess_data.py (Optimized for size)
import pandas as pd
import numpy as np
import os

print("Starting data pre-processing...")
DATA_FOLDER = 'data'
ATTRIBUTES_FILE = os.path.join(DATA_FOLDER, 'attributes.csv')
OUTPUT_FILE = os.path.join(DATA_FOLDER, 'preprocessed_data.parquet')

try:
    print(f"Reading {ATTRIBUTES_FILE}...")
    df_attributes = pd.read_csv(ATTRIBUTES_FILE)
    df_attributes.columns = df_attributes.columns.str.strip()
    print("Column headers cleaned.")
    
    print("Reshaping and pivoting data...")
    id_vars = ['Code', 'Remedial Service Type', 'Industry NAICS']
    value_vars = [c for c in df_attributes.columns if any(str(y) in c for y in range(2025, 2031))]
    df_long = pd.melt(df_attributes, id_vars=id_vars, value_vars=value_vars, var_name='Metric_Year', value_name='Value')
    
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
    print("Data values converted to numeric type, fixing errors.")
    df_long[['Metric', 'Year']] = df_long['Metric_Year'].str.rsplit('_', n=1, expand=True)
    df_long['Year'] = pd.to_numeric(df_long['Year'])
    df_long.dropna(subset=['Value'], inplace=True)

    df_cyber = df_long.pivot_table(index=['Year', 'Industry NAICS', 'Code', 'Remedial Service Type'], columns='Metric', values='Value').reset_index()

    print("Calculating simulation parameters...")
    df_cyber['Cost_mu'] = np.log(df_cyber['Cost']) - (0.3**2 / 2)
    df_cyber['Cost_sigma'] = 0.3
    
    final_columns = ['Year', 'Industry NAICS', 'Code', 'Remedial Service Type', 'Event_Freq', 'Cost_mu', 'Cost_sigma']
    df_final = df_cyber[final_columns]
    
    print(f"Saving optimized data to {OUTPUT_FILE}...")
    df_final.to_parquet(OUTPUT_FILE, compression='gzip')
    
    print("\n✅ Success! Optimized pre-processing complete.")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")