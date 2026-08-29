# 🩺 Preeclampsia Early-Detection AI: Crisis-Context Home-Shield Triage Engine

An open-source, production-grade clinical stratification application leveraging machine learning to predict preeclampsia risk profiles within low-resource environments, conflict-displaced populations, and home settings.

## 🌟 Core Innovations

* **Self-Healing Clinical Pipeline:** Integrates an iterative missing-data imputation array (`IterativeImputer`) to calculate missing clinical parameters safely on the fly, eliminating system crashes caused by missing or broken checkup log entry variables.
* **Calibrated Recall Optimization:** Tuned specifically for public health triage frameworks, optimizing decision margins to minimize fatal false-negative errors while maintaining robust statistical classification power.
* **100% Offline Database Architecture:** Leverages a localized SQLite data-logging system, allowing field workers and families to run analytical assessments and track patient profiles securely entirely without internet connectivity.
* **Dual-Tier Emergency Evacuation Protocol:** Features clear, non-technical instructions for family response routines alongside professional intervention protocols for field medics.

## 🏗️ Technical Architecture Blueprint
The system utilizes a multi-layered, offline-first pipeline to process clinical input safely and generate highly critical triage protocols:

```text
[ Mother's Input Form ] 
         │
         ▼
[ Step 1: Symptom Scan Override ] ─── (Active Checkboxes Detected) ───► [ AUTOMATIC EMERGENCY ALERT ]
         │                                                                             ▲
         │ (All Symptoms Clear)                                                        │
         ▼                                                                             │
[ Step 2: Self-Healing Imputer ] ───► (Fills Blank Inputs / NaN Data)                  │ (Risk >= 28% OR
         │                                                                             │  Vitals Critical)
         ▼                                                                             │
[ Step 3: Calibrated Random Forest Engine ] ──► (Calculates True Probability Score) ───┘
         │
         ▼ (Risk < 28%)
[ AUTOMATIC STABLE TRACK STATUS ]
         │
         ▼
[ Step 4: Local SQL Vault ] ───► (Commits Timestamped Row Data Directly to Disk Offline)
```

### Data Pipeline Logic
1. **Symptom Scan:** Prioritizes biological indicators (severe headache, blurred vision, abdominal pain, sudden edema) as an immediate fail-safe. If any symptom is flagged, the AI logic triggers an emergency warning regardless of vital metrics.
2. **Missing-Data Resolution:** If numerical metrics are missing, the pipeline maps incoming parameters to an `IterativeImputer` baseline to estimate systolic tracking thresholds without a code crash.
3. **Calibrated Machine Learning Engine:** Ingests the unified feature matrix to isolate probability trends using an aggressive 28%-30% evaluation ceiling optimized specifically to protect human life by preventing false-negative omissions.
4. **Relational Offline Data Ledger:** Logs entry characteristics, risk calculation flags, and medical response directives into an offline SQLite cache data architecture.
   

