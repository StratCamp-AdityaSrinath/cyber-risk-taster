# api/index.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os

# --- 1. SET UP THE FASTAPI APP ---
app = FastAPI()

# --- 2. DEFINE DATA MODELS FOR INPUT/OUTPUT ---
class SimulationInput(BaseModel):
    year: int = 2026
    industry_naics: int
    employee_size: str
    deductible: int
    selected_events: list[str]

class SimulationOutput(BaseModel):
    pure_premium_mean: float
    var_95: float
    var_99: float
    error: str | None = None

# --- 3. LOAD AND PRE-PROCESS DATA ONCE ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
try:
    df_attributes = pd.read_csv(os.path.join(DATA_PATH, 'attributes.csv'))
    
    id_vars = ['Code', 'Event Name', 'Remedial Service Type', 'Industry NAICS']
    value_vars = [c for c in df_attributes.columns if '2026' in c]
    df_long = pd.melt(df_attributes, id_vars=id_vars, value_vars=value_vars, var_name='Metric_Year', value_name='Value')
    df_long[['Metric', 'Year']] = df_long['Metric_Year'].str.rsplit('_', n=1, expand=True)
    
    df_cyber = df_long.pivot_table(
        index=['Industry NAICS', 'Code', 'Event Name', 'Remedial Service Type'],
        columns='Metric',
        values='Value'
    ).reset_index()

    df_cyber['Cost_mu'] = np.log(df_cyber['Cost']) - (0.3**2 / 2)
    df_cyber['Cost_sigma'] = 0.3
    
except FileNotFoundError:
    df_cyber = None
    print("ERROR: Data files not found.")


# --- 4. CONFIGURATION ---
N_ITERATIONS = 1000
EMPLOYEE_SIZE_SCALING_FACTORS = {
    "<5": 0.1, "5-9": 0.2, "10-14": 0.35, "15-19": 0.5, "20-24": 0.65,
    "25-29": 0.8, "30-34": 1.0, "35-39": 1.2, "40-49": 1.5, "50-74": 1.9, 
    "75-99": 2.4, "100-149": 3.0, "150-199": 3.7, "200-299": 4.5, 
    "300-399": 5.4, "400-499": 6.4, "500-749": 7.9, "750-999": 9.9, 
    "1,000-1,499": 12.0, "1,500-1,999": 14.0, "2,000-2,499": 16.0, 
    "2,500-4,999": 18.0, "5,000+": 21.0
}
CATASTROPHIC_FREQUENCY = 0.15
CATASTROPHIC_SEVERITY_SHAPE = 1.6
CATASTROPHIC_SEVERITY_SCALE = 0.75

def sample_catastrophic_load():
    if np.random.binomial(1, CATASTROPHIC_FREQUENCY):
        return (np.random.pareto(a=CATASTROPHIC_SEVERITY_SHAPE) + 1) * CATASTROPHIC_SEVERITY_SCALE
    else:
        return 0.05

# --- 5. THE MAIN API ENDPOINT ---
@app.post("/api/simulate", response_model=SimulationOutput)
def run_simulation(inputs: SimulationInput):
    if df_cyber is None:
        return SimulationOutput(pure_premium_mean=0, var_95=0, var_99=0, error="Server data not loaded.")

    mask = (
        (df_cyber['Industry NAICS'] == inputs.industry_naics) &
        (df_cyber['Code'].isin(inputs.selected_events))
    )
    sim_data = df_cyber[mask]
    
    if sim_data.empty:
        return SimulationOutput(pure_premium_mean=0, var_95=0, var_99=0, error="No valid events for this industry.")

    size_scaling_factor = EMPLOYEE_SIZE_SCALING_FACTORS.get(inputs.employee_size, 1.0)
    simulated_loss_per_firm = np.zeros(N_ITERATIONS)

    for _, service_params in sim_data.iterrows():
        scaled_freq_mean = service_params['Event_Freq'] * size_scaling_factor
        n_events = np.random.poisson(scaled_freq_mean, size=N_ITERATIONS)
        sampled_uptake_prob = np.random.beta(2, 2, size=N_ITERATIONS)
        sampled_cost = np.random.lognormal(
            service_params['Cost_mu'], 
            service_params['Cost_sigma'], 
            size=N_ITERATIONS
        )
        service_loss = n_events * sampled_uptake_prob * sampled_cost
        simulated_loss_per_firm += service_loss

    catastrophic_loads = np.array([sample_catastrophic_load() for _ in range(N_ITERATIONS)])
    loaded_simulated_loss = simulated_loss_per_firm * (1 + catastrophic_loads)
    simulated_premium = np.maximum(0, loaded_simulated_loss - inputs.deductible)

    return SimulationOutput(
        pure_premium_mean=simulated_premium.mean(),
        var_95=np.percentile(simulated_premium, 95),
        var_99=np.percentile(simulated_premium, 99)
    )