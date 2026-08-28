import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
n_records = 2500

# Simulate clinical data points typical of low-resource/crisis settings
data = {
    'patient_id': [f"PAT_{i:04d}" for i in range(1, n_records + 1)],
    'age': np.random.randint(14, 46, size=n_records),
    'first_pregnancy': np.random.choice([0, 1], size=n_records, p=[0.45, 0.55]),
    'twin_pregnancy': np.random.choice([0, 1], size=n_records, p=[0.96, 0.04]),
    'history_of_hypertension': np.random.choice([0, 1], size=n_records, p=[0.90, 0.10]),
    'systolic_bp_week12': np.random.randint(90, 150, size=n_records),
    'bmi': np.random.uniform(17.5, 38.0, size=n_records).round(1),
    'proteinuria_trace_strip': np.random.choice([0, 1], size=n_records, p=[0.80, 0.20]),
    'crisis_displacement_flag': np.random.choice([0, 1], size=n_records, p=[0.65, 0.35])
}

df = pd.DataFrame(data)

# Algorithmic risk assignment to determine preeclampsia onset
risk_score = (
    (df['history_of_hypertension'] * 4.5) +
    (df['systolic_bp_week12'] > 130).astype(int) * 3.5 +
    (df['proteinuria_trace_strip'] * 3.0) +
    ((df['age'] > 35) | (df['age'] < 18)).astype(int) * 2.0 +
    (df['bmi'] > 30.0).astype(int) * 1.5 +
    (df['twin_pregnancy'] * 2.5) +
    (df['crisis_displacement_flag'] * 1.0)
)

# Cross threshold to flag clinical onset
df['preeclampsia_onset'] = (risk_score > 6.0).astype(int)

# Export directly to a local CSV file
df.to_csv('preeclampsia_development_dataset.csv', index=False)
print("Success! 'preeclampsia_development_dataset.csv' has been created in this folder.")
