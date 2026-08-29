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

map_binary = lambda text: 1 if text in ["Yes", "Oui"] else 0

# --------------------------------------------------------
# GLOBAL TRANSLATION DICTIONARY MAPS
# --------------------------------------------------------
LANG_DICT = {
    "English": {
        "title": "MATERNAL HOME-SHIELD",
        "subtitle": "Predictive Early-Detection & Triage Ecosystem",
        "baseline_header": "👤 Baseline Profile",
        "age_label": "Maternal Age",
        "first_preg_label": "First-Time Pregnancy?",
        "twins_label": "Carrying Twins/Multiples?",
        "hyper_label": "History of Hypertension?",
        "displaced_label": "Crisis Displacement Status?",
        "tab_screen": "🌤️ Active Diagnostic Triage Screen",
        "tab_manual": "📖 Field Deployment Protocol Manual",
        "step1_header": "🛑 Step 1: Systematic Symptom Checklist",
        "step1_caption": "Check all active discomfort vectors expressed or experienced by the mother right now:",
        "sym_headache": "Severe, throbbing persistent headache",
        "sym_vision": "Blurry vision / flashing dark spots",
        "sym_pain": "Sharp upper right abdominal/rib pain",
        "sym_swelling": "Sudden swelling of face, eyes, or hands",
        "step2_header": "📈 Step 2: Clinical Vitals & Urine Strips",
        "sys_label": "Systolic Pressure (Top Cuff Number - mmHg)",
        "dia_label": "Diastolic Pressure (Bottom Cuff Number - mmHg)",
        "urine_label": "Dipstick Card Color Matching Scale:",
        "urine_opts": [
            "0: Normal (Yellow / Clear)",
            "1: Trace Protein (Light Green)",
            "2: High Protein (Medium Green)",
            "3: Severe Protein (Dark Obsidian Green)"
        ],
        "btn_run": "🚀 EXECUTE CLINICAL ASSESSMENT RUN",
        "matrix_header": "📊 Diagnostic Assessment Matrix",
        "prob_label": "Calculated Statistical Onset Probability",
        "awaiting_inputs": "Awaiting input initialization metrics panel. Complete and execute Step 1 & 2 to populate diagnostic triage response logs.",
        "stable_header": "✅ SYSTEM RISK CLASSIFICATION: STABLE TRACK",
        "stable_txt": "Patient is tracking within safe algorithmic norms. Secure regular checkup intervals.<br><b>CRITICAL INSTRUCTION:</b> If maternal headaches, vision spots, or acute right side stomach discomfort manifest later today, rerunning this triage assessment module immediately is mandatory.",
        "family_header": "🚨 IMMEDIATE HOUSEHOLD EMERGENCY ACTIONS",
        "family_txt": "1. <b>EVACUATE TO CLINIC IMMEDIATELY:</b> Do not wait for appointments. Leave now.<br>2. <b>REST ON LEFT SIDE ONLY:</b> Avoid flat back postures to sustain uterine and renal vascular flows.<br>3. <b>MINIMIZE VISUAL STIMULI:</b> Keep rooms dim and quiet. Sensory stress drops seizure thresholds under hypertensive states.",
        "medic_header": "🏥 CRITICAL ECLAMPSIA MANAGEMENT INTEGRATION PROTOCOLS",
        "medic_txt": """
        <ul>
            <li><b>Anticonvulsant Administration (Magnesium Sulfate):</b>
                <ul>
                    <li><i>Pritchard Regimen (IM + IV):</i> Loading dose of 4g IV slowly + 10g IM (5g inside each buttock). Maintain with 5g IM every 4 hours.</li>
                    <li><i>Zuspan Regimen (IV):</i> Loading dose of 4g IV slowly. Maintain with a continuous 1 g/hour IV infusion.</li>
                </ul>
            </li>
            <li><b>Mandatory Hourly Clinical Safety Monitoring:</b>
                <ul>
                    <li>Evaluate respiratory rate (RR) and patellar deep-tendon reflexes every hour.</li>
                    <li>Monitor fluid balance hourly. Trigger emergency alert guidelines if urine output drops below 25–30 mL/hour.</li>
                </ul>
            </li>
            <li><b>Magnesium Toxicity Reversal Protocol:</b>
                <ul>
                    <li>If patellar reflexes vanish, RR drops below 12 breaths/minute, or severe oliguria occurs: <b>Stop MgSO4 immediately.</b></li>
                    <li>Administer 10 mL of 10% <b>Calcium Gluconate (1g)</b> intravenously slowly. Provide respiratory support and secure senior medical call assistance.</li>
                </ul>
            </li>
        </ul>
        """
    },
    "Français": {
        "title": "BOUCLIER MATERNEL DOMESTIQUE",
        "subtitle": "Écosystème Prédictif de Détection Précoce et de Triage",
        "baseline_header": "👤 Profil de Base de la Mère",
        "age_label": "Âge Maternel",
        "first_preg_label": "Première Grossesse ?",
        "twins_label": "Grossesse Gémellaire / Multiple ?",
        "hyper_label": "Antécédents d'Hypertension ?",
        "displaced_label": "Statut de Déplacement de Crise ?",
        "tab_screen": "🌤️ Écran de Triage Diagnostique Actif",
        "tab_manual": "📖 Manuel des Protocoles de Déploiement",
        "step1_header": "🛑 Étape 1 : Liste des Symptômes Systématiques",
        "step1_caption": "Cochez tous les facteurs d'inconfort exprimés ou ressentis par la mère en ce moment :",
        "sym_headache": "Maux de tête graves, lancinants et persistants",
        "sym_vision": "Vision floue / taches sombres clignotantes",
        "sym_pain": "Douleur abdominale supérieure droite / côtes aiguës",
        "sym_swelling": "Gonflement soudain du visage, des yeux ou des mains",
        "step2_header": "📈 Étape 2 : Signes Vitaux Cliniques et Bandelettes Urinaires",
        "sys_label": "Pression Systolique (Chiffre Supérieur du Brassard - mmHg)",
        "dia_label": "Pression Diastolique (Chiffre Inférieur du Brassard - mmHg)",
        "urine_label": "Échelle de Correspondance des Couleurs de la Bandelette :",
        "urine_opts": [
            "0 : Normal (Jaune / Clair)",
            "1 : Traces de Protéines (Vert Clair)",
            "2 : Protéines Élevées (Vert Moyen)",
            "3 : Protéines Sévères (Vert Obsidienne Foncé)"
        ],
        "btn_run": "🚀 EXÉCUTER L'ÉVALUATION CLINIQUE",
        "matrix_header": "📊 Matrice d'Évaluation Diagnostique",
        "prob_label": "Probabilité Statistique de Début Calculée",
        "awaiting_inputs": "En attente des paramètres d'initialisation. Remplissez les étapes 1 et 2 pour afficher le plan de triage.",
        "stable_header": "✅ CLASSIFICATION DU RISQUE SYSTEME : SUIVI STABLE",
        "stable_txt": "La patiente suit des normes algorithmiques sûres. Planifiez des examens réguliers.<br><b>INSTRUCTION CRITIQUE :</b> Si des maux de tête maternels, des troubles visuels ou une douleur aiguë à l'estomac droit apparaissent plus tard aujourd'hui, réexécutez ce module immédiatement.",
        "family_header": "🚨 ACTIONS D'URGENCE IMMÉDIATES POUR LA FAMILLE",
        "family_txt": "1. <b>ÉVACUER IMMÉDIATEMENT À LA CLINIQUE :</b> N'attendez pas de rendez-vous. Partez maintenant.<br>2. <b>REPOS SUR LE CÔTÉ GAUCHE UNIQUEMENT :</b> Évitez de vous allonger sur le dos pour maintenir les flux sanguins rénaux.<br>3. <b>MINIMISER LES STIMULI VISUELS :</b> Gardez les pièces sombres et calmes pour prévenir les crises.",
        "medic_header": "🏥 PROTOCOLES CRITIQUES DE GESTION DE L'ÉCLAMPSIE",
        "medic_txt": """
        <ul>
            <li><b>Administration d'Anticonvulsivants (Sulfate de Magnésium) :</b>
                <ul>
                    <li><i>Protocole de Pritchard (IM + IV) :</i> Dose de charge de 4g IV lentement + 10g IM (5g dans chaque fesse). Maintenir avec 5g IM toutes les 4 heures.</li>
                    <li><i>Protocole de Zuspan (IV) :</i> Dose de charge de 4g IV lentement. Maintenir avec une perfusion IV continue de 1 g/heure.</li>
                </ul>
            </li>
            <li><b>Surveillance Horaire Obligatoire de la Sécurité Clinique :</b>
                <ul>
                    <li>Évaluez la fréquence respiratoire (FR) et les réflexes rotuliens toutes les heures.</li>
                    <li>Surveillez l'équilibre hydrique heure par heure. Alerte si la production d'urine tombe en dessous de 25–30 mL/heure.</li>
                </ul>
            </li>
            <li><b>Protocole d'Inversion de la Toxicité du Magnésium :</b>
                <ul>
                    <li>Si les réflexes disparaissent ou la FR tombe en dessous de 12 cycles/min : <b>Arrêtez le MgSO4 immédiatement.</b></li>
                    <li>Administrer 10 mL de <b>Gluconate de Calcium à 10% (1g)</b> par voie intraveineuse lente. Appelez une assistance médicale.</li>
                </ul>
            </li>
        </ul>
        """
    }
}

# --------------------------------------------------------
# PREMIUM USER INTERFACE CONFIGURATION
# --------------------------------------------------------
st.set_page_config(page_title="Maternal Home-Shield AI", layout="wide", initial_sidebar_state="expanded")

# Inject Custom High-End CSS Styling
st.markdown("""
    <style>
        @import url('https://googleapis.com');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main { background-color: #f8fafc; }
    'step1_caption': {'en': 'Check all active discomfort vectors expressed or experienced by the mother right now:',
                      'fr': "Cochez tous les facteurs d'inconfort exprimés ou ressentis par la mère en ce moment :",
                      'pcm': 'Tick all di wah body pain or wah di mama dey feel now:'},
    'headache': {'en': 'Severe, throbbing persistent headache', 'fr': 'Maux de tête sévères et persistants', 'pcm': 'Bad head pain weh no stop'},
    'vision': {'en': 'Blurry vision / flashing dark spots', 'fr': 'Vision floue / taches sombres clignotantes', 'pcm': 'Vision blurry or dark spot dey flash'},
    'pain': {'en': 'Sharp upper right abdominal/rib pain', 'fr': 'Douleur abdominale/aux côtes supérieures droites aiguë', 'pcm': 'Sharp pain for right side of belle/rib'},
    'swelling': {'en': 'Sudden swelling of face, eyes, or hands', 'fr': 'Gonflement soudain du visage, des yeux ou des mains', 'pcm': 'Face, eye or hand swell quick'},
    'step2': {'en': '📈 Step 2: Clinical Vitals & Urine Strips', 'fr': '📈 Étape 2 : Signes vitaux cliniques et bandelettes urinaires', 'pcm': '📈 Step 2: Vitals and Urine Strip'},
    'systolic': {'en': 'Systolic Pressure (Top Cuff Number - mmHg)', 'fr': 'Pression systolique (mmHg)', 'pcm': 'Systolic Pressure'},
    'diastolic': {'en': 'Diastolic Pressure (Bottom Cuff Number - mmHg)', 'fr': 'Pression diastolique (mmHg)', 'pcm': 'Diastolic Pressure'},
    'urine_score': {'en': 'Dipstick Card Color Matching Scale:', 'fr': 'Échelle de correspondance des couleurs de la bandelette :', 'pcm': 'Dipstick color scale'},
    'execute': {'en': '🚀 EXECUTE CLINICAL ASSESSMENT RUN', 'fr': "🚀 EXÉCUTER L'ÉVALUATION CLINIQUE", 'pcm': '🚀 RUN ASSESSMENT'},
    'calculated': {'en': 'Calculated Statistical Onset Probability', 'fr': "Probabilité calculée d'apparition", 'pcm': 'Calculated Onset Probability'},
    'immediate_actions_title': {'en': '🚨 IMMEDIATE HOUSEHOLD EMERGENCY ACTIONS', 'fr': "🚨 ACTIONS D'URGENCE IMMÉDIATES", 'pcm': '🚨 EMERGENCY ACTIONS'},
    'stable_track': {'en': '✅ SYSTEM RISK CLASSIFICATION: STABLE TRACK', 'fr': '✅ CLASSIFICATION DU RISQUE : SUIVI STABLE', 'pcm': '✅ SYSTEM SAY: STABLE'},
    'awaiting': {'en': 'Awaiting input initialization metrics panel. Complete and execute Step 1 & 2 to populate diagnostic triage response logs.',
                 'fr': "En attente de l'initialisation des métriques. Complétez et exécutez l'étape 1 et 2.",
                 'pcm': 'Waiting for input. Do Step 1 & 2 then run.'}
}

def t(key, lang='en'):
    return TRANSLATIONS.get(key, {}).get(lang, TRANSLATIONS.get(key, {}).get('en', key))

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
        .medical-box { background-color: #f0fdfa; border-left: 5px solid #0d9488; }
    </style>
""", unsafe_allow_html=True)

# App Core Visual Header Banner Area
header_col1, header_col2 = st.columns([0.15, 0.85])
with header_col1:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.subheader("🛡️")

# Language selector placed prominently at top-right area
with header_col2:
    # default language is English
    lang_key = st.selectbox("Language / Langue / Pidgin", options=list(LANG_OPTIONS.keys()), format_func=lambda k: LANG_OPTIONS[k], index=0)
    st.markdown(f"<h1 style='color: #0f172a; margin-bottom: 0;'>{t('title', lang_key)}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #64748b; font-size: 1.1rem; margin-top: 0;'>{t('subtitle', lang_key)}</p>", unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)

# Sidebar Patient Characteristics Configurations
st.sidebar.markdown(f"<h3 style='color: #f8fafc;'>{t('baseline_profile', lang_key)}</h3>", unsafe_allow_html=True)
age = st.sidebar.slider(t('age', lang_key), 14, 50, 25)
first_preg = st.sidebar.selectbox(t('first_preg', lang_key), ["No", "Yes"]) 
twins = st.sidebar.selectbox(t('twins', lang_key), ["No", "Yes"]) 
chronic_hyper = st.sidebar.selectbox(t('history_hyper', lang_key), ["No", "Yes"]) 
displaced = st.sidebar.selectbox(t('displaced', lang_key), ["No", "Yes"]) 

# --------------------------------------------------------
# Advanced Biomarker Optional Slots (don't change model features)
# These affect a safe, rule-based adjustment to the model probability
# --------------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("Advanced Biomarkers (Optional)"):
    st.write("Use if local clinic can provide low-cost lab markers. These are optional and will be used to adjust risk after the model prediction.")
    platelet_count = st.number_input("Platelet count (x10^9/L)", min_value=10, max_value=1000, value=250)
    serum_creatinine = st.number_input("Serum creatinine (mg/dL)", min_value=0.2, max_value=10.0, value=0.7, format="%.2f")
    include_biomarkers = st.checkbox("Include advanced biomarkers in risk adjustment?", value=False)

# Dual Screening Tab Interfaces
tab_screen, tab_manual = st.tabs(["🌤️ Active Diagnostic Triage Screen", "📖 Field Deployment Protocol Manual"]) 

with tab_screen:
    # Diagnostic Screen Section Layout Arrays
    input_panel_col, feedback_panel_col = st.columns([1.1, 0.9])
    
    with input_panel_col:
        st.markdown(f"<h4 style='color: #0f766e;'>{t('step1', lang_key)}</h4>", unsafe_allow_html=True)
        st.caption(t('step1_caption', lang_key))
        
        sym_col1, sym_col2 = st.columns(2)
        with sym_col1:
            s_headache = st.checkbox(t('headache', lang_key))
            s_vision = st.checkbox(t('vision', lang_key))
        with sym_col2:
            s_pain = st.checkbox(t('pain', lang_key))
            s_swelling = st.checkbox(t('swelling', lang_key))
            
        st.markdown("<br><h4 style='color: #0f766e;'>📈 Step 2: Clinical Vitals & Urine Strips</h4>", unsafe_allow_html=True)
        
        vits_col1, vits_col2 = st.columns(2)
        with vits_col1:
            systolic = st.number_input(t('systolic', lang_key), min_value=70, max_value=220, value=115)
            diastolic = st.number_input(t('diastolic', lang_key), min_value=40, max_value=140, value=75)
        with vits_col2:
            urine_score = st.radio(
                t('urine_score', lang_key),
                options=[
                    "0: Normal (Yellow / Clear)",
                    "1: Trace Protein (Light Green)",
                    "2: High Protein (Medium Green)",
                    "3: Severe Protein (Dark Obsidian Green)"
                ]
            )
            
        # Use helper to parse the numeric prefix of the radio label safely
        protein_numeric = parse_urine_score(urine_score)

        symptom_active = s_headache or s_vision or s_pain or s_swelling
        
        st.markdown("<br>", unsafe_allow_html=True)
        execute_analysis = st.button(t('execute', lang_key), use_container_width=True)

    with feedback_panel_col:
        st.markdown(f"<h4 style='color: #1e293b;'>📊 {t('calculated', lang_key)}</h4>", unsafe_allow_html=True)
        
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
            
            # Validate the input vector columns against the trained model's expected features
            valid, msg = validate_input_vector_columns(input_vector, clinical_engine)
            if not valid:
                st.error(f"Feature mismatch: {msg}")
            else:
                # Ensure we extract the positive-class probability safely
                prob_array = clinical_engine.predict_proba(input_vector)
                raw_prob = extract_positive_probability(prob_array)

                # Apply a safe, rule-based adjustment using optional biomarkers and MAP
                def compute_map(s, d):
                    return (s + 2*d) / 3.0

                def adjust_probability(base_prob, include_bio, platelet, creatinine, s, d):
                    adj = 0.0
                    map_val = compute_map(s, d)
                    # MAP (mean arterial pressure) thresholds
                    if map_val >= 120:
                        adj += 0.15
                    elif map_val >= 105:
                        adj += 0.07
                    # Platelets: thrombocytopenia increases risk
                    if include_bio:
                        if platelet < 100:
                            adj += 0.12
                        elif platelet < 150:
                            adj += 0.04
                        # Creatinine: renal dysfunction
                        if creatinine >= 1.1:
                            adj += 0.10
                        elif creatinine >= 0.9:
                            adj += 0.03
                    # Ensure adjustments are bounded
                    new_prob = min(1.0, max(0.0, base_prob + adj))
                    return new_prob, map_val

                adjusted_prob, map_value = adjust_probability(raw_prob, include_biomarkers, platelet_count, serum_creatinine, systolic, diastolic)

                critical_vitals = (systolic >= 140) or (diastolic >= 90) or (protein_numeric >= 2)
                is_high_risk = (adjusted_prob >= 0.28) or symptom_active or critical_vitals
                
                # Premium Custom Cards Layout Rendering
                st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: #64748b; font-weight: 600; margin: 0; text-transform: uppercase; font-size: 0.85rem;">{t('calculated', lang_key)}</p>
                        <h2 style="color: {'#ef4444' if is_high_risk else '#0d9488'}; margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: 700;">{adjusted_prob:.1%}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show MAP and biomarker summary
                st.markdown(f"<div style='margin-top:0.75rem; color:#475569;'>Mean arterial pressure (MAP): {map_value:.1f} mmHg</div>", unsafe_allow_html=True)
                if include_biomarkers:
                    st.markdown(f"<div style='margin-top:0.25rem; color:#475569;'>Platelets: {platelet_count} x10^9/L — Creatinine: {serum_creatinine:.2f} mg/dL</div>", unsafe_allow_html=True)

                if is_high_risk:
                    st.markdown(f"<div class='protocol-box family-box'><h4>{t('immediate_actions_title', lang_key)}</h4>"
                                "1. <b>EVACUATE TO CLINIC IMMEDIATELY:</b> Do not wait for appointments. Leave now.<br>"
                                "2. <b>REST ON LEFT SIDE ONLY:</b> Avoid flat back postures to sustain uterine and renal vascular flows.<br>"
                                "3. <b>MINIMIZE VISUAL STIMULI:</b> Keep rooms dim and quiet. Sensory stress drops seizure thresholds under hypertensive states.</div>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class='protocol-box medical-box'>
                      <h4>🏥 EMERGENCY MEDICAL RESPONSE (Key points)</h4>
                      <ul>
                        <li><b>Anticonvulsant — Magnesium sulfate</b>: Pritchard (IM+IV) — loading 4 g IV slow + 10 g IM (5 g each buttock); then 5 g IM every 4 h; or Zuspan (IV) — loading 4 g I[...]
                        <li><b>Monitoring:</b> Respiratory rate and deep‑tendon reflexes hourly; urine output hourly (alert if &lt;25–30 mL/h); continuous BP and fetal monitoring as available.</li[...]
                        <li><b>Toxicity — immediate actions:</b> If absent reflexes, RR &lt; 12/min, or oliguria → stop MgSO4, give calcium gluconate 10 mL of 10% (1 g) IV slowly, provide respirat[...]
                        <li><b>Duration:</b> Continue for 24 hours after last seizure or after delivery (whichever is later).</li>
                        <li><b>Practical:</b> Confirm local MgSO4 concentration before converting g → mL (example: 50% = 500 mg/mL → 4 g = 8 mL); keep calcium gluconate at bedside.</li>
                      </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='protocol-box medical-box' style='border-left-color: #0d9488;'><h4>{t('stable_track', lang_key)}</h4>"
                                "Patient is tracking within safe algorithmic norms. Secure regular checkup intervals.<br>"
                                "<b>CRITICAL INSTRUCTION:</b> If maternal headaches, vision spots, or acute right side stomach discomfort manifest later today, rerunning this triage assessment module [...]</div>", unsafe_allow_html=True)
        else:
            st.info(t('awaiting', lang_key))

with tab_manual:
    st.subheader("📋 Low-Resource Deployment Operational Directives")
    st.markdown("""
    ### System Workflow Synchronization Overview
    1. **The Baselines Matrix:** Use the structural sidebar panels to input background genetic, demographic, and geographical contexts before reviewing ongoing parameters.
    2. **Urine Protein Testing Cards:** Dip standard validation testing strip layers inside early morning urine samples. Align color gradients closely with reference panels, logging results as numeric[...]
    3. **The Unwell Override Protocol:** Preeclampsia operates on variable timelines. Any single warning sign checkbox trigger enforces a clinical high-risk output flag automatically to protect human [...]
    """, unsafe_allow_html=True)
