import numpy as np
import pandas as pd

np.random.seed(101)
n_records = 3000

# Simulate raw baseline characteristics
age = np.random.randint(14, 46, size=n_records)
first_preg = np.random.choice([0, 1], size=n_records, p=[0.45, 0.55])
twins = np.random.choice([0, 1], size=n_records, p=[0.96, 0.04])
history_hyper = np.random.choice([0, 1], size=n_records, p=[0.91, 0.09])
bmi = np.random.uniform(17.5, 42.0, size=n_records).round(1)
displacement = np.random.choice([0, 1], size=n_records, p=[0.70, 0.30])

# Simulate blood pressure tracking with baseline noise
systolic_base = np.random.normal(115, 12, size=n_records)
# Add physiological variance based on risk characteristics
systolic_bp = (systolic_base + (history_hyper * 18) + ((bmi > 30) * 8) + (displacement * 4)).astype(int)

# Trace strips (proteinuria) - add random measurement error or noise typical of field settings
protein_base = np.random.choice([0, 1], size=n_records, p=[0.84, 0.16])
protein_strip = np.where((systolic_bp > 135), np.random.choice([0, 1], p=[0.3, 0.7]), protein_base)

# Complex, non-linear biological risk mapping (hidden equations)
log_odds = (
    -4.5 + 
    (history_hyper * 2.1) + 
    ((systolic_bp > 140) * 1.8) + 
    (protein_strip * 1.5) + 
    (twins * 1.2) + 
    (((age > 38) | (age < 18)).astype(int) * 0.9) +
    ((bmi > 32) * 0.7) +
    (displacement * 0.5)
)
# Convert log odds to probability using sigmoid function
probability = 1 / (1 + np.exp(-log_odds))

# Introduce environmental chaos/noise: 5% of cases completely break clinical patterns
noise_mask = np.random.rand(n_records) < 0.05
onset = np.where(noise_mask, np.random.choice([0, 1], p=[0.5, 0.5], size=n_records), (probability > 0.4).astype(int))

df_chaos = pd.DataFrame({
    'patient_id': [f"PAT_{i:04d}" for i in range(1, n_records + 1)],
    'age': age,
    'first_pregnancy': first_preg,
    'twin_pregnancy': twins,
    'history_of_hypertension': history_hyper,
    'systolic_bp_week12': systolic_bp,
    'bmi': bmi,
    'proteinuria_trace_strip': protein_strip,
    'crisis_displacement_flag': displacement,
    'preeclampsia_onset': onset
})

df_chaos.to_csv('preeclampsia_chaos_dataset.csv', index=False)
print("Chaos dataset compiled. Ready for clinical calibration models.")
