import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------
# CORE SYSTEM OPERATIONS & CONFIGURATIONS
# --------------------------------------------------------
DB_FILE = "preeclampsia_patient_registry.db"

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
    st.error("Core engine training dataset files missing.")
    st.stop()

map_binary = lambda text: 1 if text == "Yes" else 0

# --------------------------------------------------------
# PREMIUM USER INTERFACE CONFIGURATION
# --------------------------------------------------------
st.set_page_config(page_title="Maternal Home-Shield AI", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS Styling for a High-End Clinical Visual Presentation
st.markdown("""
    <style>
        @import url('https://googleapis.com');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main { background-color: #f8fafc; }
        div[data-testid="stSidebarUserContent"] { background-color: #0f172a; color: #ffffff; }
        .stButton>button { 
            background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%); 
            color: white; border: none; font-weight: 600; 
            padding: 0.6rem 2rem; border-radius: 8px; transition: all 0.3s;
        }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(13,148,136,0.3); }
        .metric-card { 
            background-color: white; border-radius: 12px; padding: 1.5rem; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; 
        }
        .protocol-box { border-radius: 12px; padding: 1.5rem; margin-top: 1rem; color: #1e293b; }
        .family-box { background-color: #fef2f2; border-left: 5px solid #ef4444; }
        .medic-box { background-color: #f0fdfa; border-left: 5px solid #0d9488; }
    </style>
""", unsafe_content_with_markup=True)

# App Core Visual Header Banner Area
header_col1, header_col2 = st.columns([0.15, 0.85])
with header_col1:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.subheader("🛡️")
with header_col2:
    st.markdown("<h1 style='color: #0f172a; margin-bottom: 0;'>MATERNAL HOME-SHIELD</h1>", unsafe_content_with_markup=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-top: 0;'>Predictive Early-Detection & Triage Ecosystem</p>", unsafe_content_with_markup=True)

st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 2rem;'>", unsafe_content_with_markup=True)

# Sidebar Patient Characteristics Configurations
st.sidebar.markdown("<h3 style='color: #f8fafc;'>👤 Baseline Profile</h3>", unsafe_content_with_markup=True)
age = st.sidebar.slider("Maternal Age", 14, 50, 25)
first_preg = st.sidebar.selectbox("First-Time Pregnancy?", ["No", "Yes"])
twins = st.sidebar.selectbox("Carrying Twins/Multiples?", ["No", "Yes"])
chronic_hyper = st.sidebar.selectbox("History of Hypertension?", ["No", "Yes"])
displaced = st.sidebar.selectbox("Crisis Displacement Status?", ["No", "Yes"])

# Dual Screening Tab Interfaces
tab_screen, tab_manual = st.tabs(["🌤️ Active Diagnostic Triage Screen", "📖 Field Deployment Protocol Manual"])

with tab_screen:
    # Diagnostic Screen Section Layout Arrays
    input_panel_col, feedback_panel_col = st.columns([1.1, 0.9])
    
    with input_panel_col:
        st.markdown("<h4 style='color: #0f766e;'>🛑 Step 1: Systematic Symptom Checklist</h4>", unsafe_content_with_markup=True)
        st.caption("Check all active discomfort vectors expressed or experienced by the mother right now:")
        
        sym_col1, sym_col2 = st.columns(2)
        with sym_col1:
            s_headache = st.checkbox("Severe, throbbing persistent headache")
            s_vision = st.checkbox("Blurry vision / flashing dark spots")
        with sym_col2:
            s_pain = st.checkbox("Sharp upper right abdominal/rib pain")
            s_swelling = st.checkbox("Sudden swelling of face, eyes, or hands")
            
        st.markdown("<br><h4 style='color: #0f766e;'>📈 Step 2: Clinical Vitals & Urine Strips</h4>", unsafe_content_with_markup=True)
        
        vits_col1, vits_col2 = st.columns(2)
        with vits_col1:
            systolic = st.number_input("Systolic Pressure (Top Cuff Number - mmHg)", min_value=70, max_value=220, value=115)
            diastolic = st.number_input("Diastolic Pressure (Bottom Cuff Number - mmHg)", min_value=40, max_value=140, value=75)
        with vits_col2:
            urine_score = st.radio(
                "Dipstick Card Color Matching Scale:",
                options=[
                    "0: Normal (Yellow / Clear)",
                    "1: Trace Protein (Light Green)",
                    "2: High Protein (Medium Green)",
                    "3: Severe Protein (Dark Obsidian Green)"
                ]
            )
            
        protein_numeric = float(urine_score.split(":"))
        symptom_active = s_headache or s_vision or s_pain or s_swelling
        
        st.markdown("<br>", unsafe_content_with_markup=True)
        execute_analysis = st.button("🚀 EXECUTE CLINICAL ASSESSMENT RUN", use_container_width=True)

    with feedback_panel_col:
        st.markdown("<h4 style='color: #1e293b;'>📊 Diagnostic Assessment Matrix</h4>", unsafe_content_with_markup=True)
        
        if execute_analysis:
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
            
            raw_prob = float(clinical_engine.predict_proba(input_vector))
            critical_vitals = (systolic >= 140) or (diastolic >= 90) or (protein_numeric >= 2)
            is_high_risk = (raw_prob >= 0.28) or symptom_active or critical_vitals
            
            # Premium Custom Cards Layout Rendering
            st.markdown(f"""
                <div class="metric-card">
                    <p style="color: #64748b; font-weight: 600; margin: 0; text-transform: uppercase; font-size: 0.85rem;">Calculated Statistical Onset Probability</p>
                    <h2 style="color: {'#ef4444' if is_high_risk else '#0d9488'}; margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: 700;">{raw_prob:.1%}</h2>
                </div>
            """, unsafe_content_with_markup=True)
            
            if is_high_risk:
                st.markdown("<div class='protocol-box family-box'><h4>🚨 IMMEDIATE HOUSEHOLD EMERGENCY ACTIONS</h4>"
                            "1. <b>EVACUATE TO CLINIC IMMEDIATELY:</b> Do not wait for appointments. Leave now.<br>"
                            "2. <b>REST ON LEFT SIDE ONLY:</b> Avoid flat back postures to sustain uterine and renal vascular flows.<br>"
                            "3. <b>MINIMIZE VISUAL STIMULI:</b> Keep rooms dim and quiet. Sensory stress drops seizure thresholds under hypertensive states.</div>", unsafe_content_with_markup=True)
                
                st.markdown("<div class='protocol-box medic-box'><h4>🏥 EMERGENCY MEDIC RESPONSE SEQUENCES</h4>"
                            "• <b>Vascular Control:</b> Administer oral Labetalol or fast-acting Nifedipine immediately if Systolic $\ge$ 160 mmHg.<br>"
                            "• <b>Seizure Blockades:</b> Deploy full <b>Magnesium Sulfate (MgSO4)</b> loading protocols.<br>"
                            "• <b>Obstetric Clearance:</b> Assess fetal maturity parameters. Prepare logistics for stabilization or active triage transport.</div>", unsafe_content_with_markup=True)
            else:
                st.markdown("<div class='protocol-box medic-box' style='border-left-color: #0d9488;'><h4>✅ SYSTEM RISK CLASSIFICATION: STABLE TRACK</h4>"
                            "Patient is tracking within safe algorithmic norms. Secure regular checkup intervals.<br>"
                            "<b>CRITICAL INSTRUCTION:</b> If maternal headaches, vision spots, or acute right side stomach discomfort manifest later today, rerunning this triage assessment module immediately is mandatory.</div>", unsafe_content_with_markup=True)
        else:
            st.info("Awaiting input initialization metrics panel. Complete and execute Step 1 & 2 to populate diagnostic triage response logs.")

with tab_manual:
    st.subheader("📋 Low-Resource Deployment Operational Directives")
    st.markdown("""
    ### System Workflow Synchronization Overview
    1. **The Baselines Matrix:** Use the structural sidebar panels to input background genetic, demographic, and geographical contexts before reviewing ongoing parameters.
    2. **Urine Protein Testing Cards:** Dip standard validation testing strip layers inside early morning urine samples. Align color gradients closely with reference panels, logging results as numeric values `0` through `3`.
    3. **The Unwell Override Protocol:** Preeclampsia operates on variable timelines. Any single warning sign checkbox trigger enforces a clinical high-risk output flag automatically to protect human life.
    """)
