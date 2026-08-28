import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Read the CSV file directly from the local folder
try:
    df = pd.read_csv('preeclampsia_development_dataset.csv')
    print("Successfully loaded clinical database files.\n")
except FileNotFoundError:
    print("Error: 'preeclampsia_development_dataset.csv' not found. Run generate_data.py first.")
    exit()

# 2. Separate administrative data from clinical features
# We drop patient_id (useless for math) and preeclampsia_onset (the target variable)
X = df.drop(columns=['patient_id', 'preeclampsia_onset'])
y = df['preeclampsia_onset']

# 3. Split into Training Data (80%) and Test Data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Initialize and Train the Model
# 'balanced' helps handle situations where there are far more healthy patients than sick ones
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. Generate Diagnostics
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions)

# 6. Print Clinical Performance Evaluations
print("=== CLINICAL MODEL PERFORMANCE ===\n")
print(classification_report(y_test, predictions, target_names=['Stable Track', 'High Risk Warning']))

print("=== CONFUSION MATRIX BREAKDOWN ===")
print(f"True Negatives (Correctly identified as safe): {cm[0][0]}")
print(f"False Positives (Healthy but flagged as high risk): {cm[0][1]}")
print(f"False Negatives (CRITICAL ERROR: High risk missed!): {cm[1][0]}")
print(f"True Positives (Correctly flagged as high risk): {cm[1][1]}")
