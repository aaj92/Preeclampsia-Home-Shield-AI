import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# Database Configuration
DB_FILE = "preeclampsia_patient_registry.db"

# --------------------------------------------------------
# CORE ANALYTICS ENGINE INFRASTRUCTURE
# --------------------------------------------------------
@st.cache_resource
def load_clinical_model():
    df = pd.read_csv('preeclampsia_chaos_dataset.csv')
    X = df.drop(columns=['patient_id', 'preeclampsia_onset'])
    y = df['preeclampsia_onset']
    engine = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=101)
    engine.fit(X, y)
    return engine

try:
    clinical_engine = load_clinical_model()
except FileNotFoundError:
    st.error("Core database file missing.")
    st.stop()

# --------------------------------------------------------
# INTERACTIVE UI FRAMEWORK
# --------------------------------------------------------
st.set_page_config(page_title="Home Maternal Triage Shield", layout="wide")

st.title("🏡 Maternal Home-Shield Triage System")
st.markdown("### Daily Offline Risk Assessment & Emergency Action Protocol")
st.markdown("---")

# Setup Sidebar Context for Personal Baseline
st.sidebar.header("👤 Mother's Baseline Profile")
age = st.sidebar.slider("Age", 14, 50, 25)
first_preg = st.sidebar.selectbox("First Pregnancy?", ["No", "Yes"])
twins = st.sidebar.selectbox("Carrying Twins?", ["No", "Yes"])
chronic_hyper = st.sidebar.selectbox("History of Chronic High Blood Pressure?", ["No", "Yes"])
displaced = st.sidebar.selectbox("Living in a Displacement Camp/Crisis Zone?", ["No", "Yes"])

map_binary = lambda text: 1 if text == "Yes" else 0

# Main App Navigation Tabs
tab_screen, tab_about = st.tabs(["🌤️ Daily Morning & Urgent Screening", "📖 System Instruction Manual"])

with tab_screen:
    st.subheader("🛑 Step 1: Check for Feeling Unwell (Symptom Scan)")
    st.write("Does the mother currently feel unwell or have any of these specific symptoms right now?")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s_headache = st.checkbox("❌ Severe, throbbing headache that won't go away")
        s_vision = st.checkbox("❌ Blurry vision, flashing lights, or dark spots in front of the eyes")
    with col_s2:
        s_pain = st.checkbox("❌ Sharp pain right below the ribs or upper stomach area")
        s_swelling = st.checkbox("❌ Sudden, massive swelling in the face, eyes, or hands")

    st.markdown("---")
    st.subheader("📈 Step 2: Input Morning Vital Signs")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        systolic = st.number_input("Systolic Blood Pressure (Top Reading from Cuff mmHg)", min_value=70, max_value=220, value=115)
        diastolic = st.number_input("Diastolic Blood Pressure (Bottom Reading from Cuff mmHg)", min_value=40, max_value=140, value=75)
    with col_v2:
        st.markdown("**Urine Paper Strip Color Match:**")
        urine_score = st.radio(
            "Match the dipped morning paper strip color to the kit card options:",
            options=[
                "0: Yellow / Light Yellow (Negative / Normal)",
                "1: Light Green (Trace Protein detected)",
                "2: Medium Green (High Protein detected)",
                "3: Deep Dark Green (Severe Risk Level)"
            ]
        )

    # FIXED: Added [0] slice selector to safely clean string data before parsing float operations
    protein_numeric = float(urine_score.split(":")[0].strip())
    symptom_active = s_headache or s_vision or s_pain or s_swelling
    
    st.markdown("---")
    
    if st.button("🔍 RUN EMERGENCY RISK CALCULATION", use_container_width=True):
        
        # Structure payload vector matching the random forest format requirements
        input_vector = pd.DataFrame([{
            'age': age,
            'first_pregnancy': map_binary(first_preg),
            'twin_pregnancy': map_binary(twins),
            'history_of_hypertension': map_binary(chronic_hyper),
            'systolic_bp_week12': float(systolic), 
            'bmi': 24.5, 
            'proteinuria_trace_strip': 1 if protein_numeric >= 1 else 0,
            'crisis_displacement_flag': map_binary(displaced)
        }])
        
        raw_prob = float(clinical_engine.predict_proba(input_vector)[0, 1])

        
        # Override calculation safety ceiling if severe home red flags are explicitly triggered
        critical_vitals = (systolic >= 140) or (diastolic >= 90) or (protein_numeric >= 2)
        is_high_risk = (raw_prob >= 0.28) or symptom_active or critical_vitals
        
        st.subheader("📊 Ultimate Risk Assessment Outcome")
        
        if is_high_risk:
            st.error("🚨 CRITICAL ALERT: HIGH RISK OF DEVELOPING PREECLAMPASIA / ECLAMPSIA DETECTED")
            
            # Action Plan Framework Split Box
            act_col1, act_col2 = st.columns(2)
            with act_col1:
                st.markdown("""
                ### 🚨 IMMEDIATE ACTION PLAN (For the Family)
                1. **GET TO A HOSPITAL IMMEDIATELY:** Do not wait. Leave the house right away. Preeclampsia is a fast-acting emergency.
                2. **DO NOT LAY FLAT ON YOUR BACK:** If resting while waiting for transportation, lay on your **left side**. This improves blood flow to the baby and kidneys.
                3. **STAY CALM & REDUCE LIGHTS:** High blood pressure mixed with stress can trigger seizures. Keep the mother in a quiet, dark area while moving.
                """)
            with act_col2:
                # FIXED: Swapped LaTeX syntax symbol formatting to fix text parser compilation exceptions
                st.markdown("""
                ### 🏥 CLINICAL INTERVENTION PROTOCOL (For Field Medics)
                * **Antihypertensive Administration:** Prepare safe emergency blood pressure medications (e.g., oral Labetalol or Nifedipine) if systolic is >= 160 or diastolic is >= 110.
                * **Seizure Prophylaxis:** Administer an immediate loading dose of **Magnesium Sulfate (MgSO4)** intravenously/intramuscularly to prevent eclamptic seizures.
                * **Delivery Planning:** Assess gestational age. If the pregnancy is past 37 weeks, prepare for urgent delivery to save both lives.
                """)
        else:
            st.success("✅ STABLE TRACK: SCREENING COMPLETED SUCCESSFULLY")
            st.markdown("""
            **Next Actions:**
            * Everything looks normal this morning.
            * **Repeat this test tomorrow morning** at the exact same time.
            * **CRITICAL RULE:** If the mother feels unwell later today (headache, vision changes, or pain), do not wait for tomorrow. **Run this screening test again immediately.**
            """)

with tab_about:
    st.subheader("📋 Low-Resource Home Triage Manual")
    st.markdown("""
    ### How to Use the At-Home Kit in Displacement Settings
    
    1. **The Morning Routine:** Every morning, before eating or walking around, the mother should sit quietly for 5 minutes, then take her blood pressure. 
    2. **The Dipstick Method:** Collect a tiny amount of urine in a clean cup. Dip the paper strip for 2 seconds. Shake off excess fluid. Wait 60 seconds, then match the color to the cardboard reference card.
    3. **The Unwell Trigger:** Preeclampsia does not care about schedules. If a mother says *'I feel strange'* or complains of a headache, her family must run this app immediately.
    """)
