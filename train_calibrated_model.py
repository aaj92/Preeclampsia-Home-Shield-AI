import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Ingest chaotic database
df = pd.read_csv('preeclampsia_chaos_dataset.csv')
X = df.drop(columns=['patient_id', 'preeclampsia_onset'])
y = df['preeclampsia_onset']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

# 2. Train a baseline machine learning engine
model = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=101)
model.fit(X_train, y_train)

# 3. Pull raw probability scores rather than default predictions
probabilities = model.predict_proba(X_test)[:, 1]

# 4. Shift Decision Threshold for Clinical Safety
# Default threshold is 0.5. Lowering it to 0.3 means we flag warnings aggressively to catch hidden risks.
custom_threshold = 0.3
calibrated_predictions = (probabilities >= custom_threshold).astype(int)

# 5. Review True Performance
cm = confusion_matrix(y_test, calibrated_predictions)
print("=== CALIBRATED RECALL PERFORMANCE RAPORT ===\n")
print(classification_report(y_test, calibrated_predictions, target_names=['Stable Track', 'High Risk Warning']))

print("=== CONFUSION MATRIX (CLINICALLY OPTIMIZED) ===")
print(f"True Negatives (Correctly Cleared): {cm[0][0]}")
print(f"False Positives (Safely Flagged for extra observation): {cm[0][1]}")
print(f"False Negatives (CRITICAL OVERLOOKS - MUST BE MINIMIZED): {cm[1][0]}")
print(f"True Positives (Complications Pre-detected): {cm[1][1]}")
print(f"\nDiscriminative Power Area Under ROC Curve: {roc_auc_score(y_test, probabilities):.4f}")
