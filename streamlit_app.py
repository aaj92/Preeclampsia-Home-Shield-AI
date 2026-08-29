import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ========================================
# TRANSLATION DICTIONARIES
# ========================================
TRANSLATIONS = {
    "en": {
        # Page & Header
        "page_title": "Maternal Health Home Triage",
        "app_title": "Maternal Health Screening System",
        "app_subtitle": "Preeclampsia Risk Assessment & Clinical Guidance",
        
        # Sidebar
        "mother_profile": "Patient Information",
        "age": "Age (years)",
        "first_pregnancy": "First Pregnancy?",
        "twins": "Multiple Pregnancy (Twins/Triplets)?",
        "chronic_hypertension": "History of High Blood Pressure?",
        "displaced": "Living in Crisis/Displacement Setting?",
        "yes": "Yes",
        "no": "No",
        
        # Main Screening
        "screening_tab": "Daily Screening",
        "manual_tab": "User Guide",
        "step_1": "Step 1: Symptom Assessment",
        "step_1_desc": "Select any symptoms present today:",
        "symptom_headache": "Severe headache",
        "symptom_vision": "Vision changes (blurred, flashing lights, dark spots)",
        "symptom_pain": "Upper abdominal or right upper quadrant pain",
        "symptom_swelling": "Facial or hand swelling",
        
        # Vital Signs
        "step_2": "Step 2: Vital Signs",
        "systolic": "Systolic Blood Pressure (mmHg)",
        "diastolic": "Diastolic Blood Pressure (mmHg)",
        "urine_label": "Urine Protein Detection",
        "urine_instructions": "Match strip color to reference card:",
        "urine_0": "Negative (No protein)",
        "urine_1": "Trace (1+)",
        "urine_2": "Moderate (2+)",
        "urine_3": "High (3+)",
        
        # Button & Results
        "run_assessment": "Calculate Risk Assessment",
        "risk_outcome": "Risk Assessment Result",
        "high_risk_alert": "HIGH RISK: Preeclampsia/Eclampsia Suspected",
        "high_risk_desc": "Immediate medical evaluation required",
        "risk_percentage": "Preeclampsia Risk Score",
        "action_plan": "Recommended Actions",
        "family_actions": "Family Should:",
        "action_1": "Proceed to hospital without delay",
        "action_2": "Rest on left side to improve blood flow",
        "action_3": "Keep in quiet environment; minimize stress",
        
        "medical_protocol": "Medical Team Protocol:",
        "medical_1": "Monitor blood pressure continuously",
        "medical_2": "Prepare magnesium sulfate (seizure prophylaxis)",
        "medical_3": "Prepare for possible urgent delivery after 37 weeks",
        
        "low_risk": "LOW RISK: Screening Completed",
        "low_risk_desc": "No acute risk indicators detected",
        "next_steps": "Recommended Next Steps:",
        "next_1": "Continue daily morning screening",
        "next_2": "Seek immediate care if symptoms develop",
        "next_3": "Schedule antenatal follow-up as planned",
        
        # Manual
        "manual_title": "User Guide & Instructions",
        "section_1": "How to Use This System",
        "section_1_desc": """
        This system provides daily screening for preeclampsia risk in resource-limited settings.
        Use each morning and whenever symptoms develop.
        """,
        "section_2": "Vital Signs Collection",
        "section_2_desc": """
        **Blood Pressure:** Use an automated cuff if available. Sit for 5 minutes before measuring.
        **Urine Test:** Use provided dipstick kit. Dip for 2 seconds, wait 60 seconds, compare color.
        """,
        "section_3": "When to Seek Emergency Care",
        "section_3_desc": """
        • Sudden severe headache
        • Vision changes or seeing flashing lights
        • Severe upper abdominal pain
        • Facial or hand swelling
        • Blood pressure reading >140/90 mmHg
        • Protein detected in urine
        """,
        "data_note": "Note: This app does not store personal data. Results are calculated locally.",
        
        # Errors
        "error_missing_model": "System initialization error. Clinical model unavailable.",
    },
    "fr": {
        # Page & Header
        "page_title": "Dépistage Maternel à Domicile",
        "app_title": "Système de Dépistage de la Santé Maternelle",
        "app_subtitle": "Évaluation du Risque de Prééclampsie et Guidance Clinique",
        
        # Sidebar
        "mother_profile": "Informations de la Patiente",
        "age": "Âge (années)",
        "first_pregnancy": "Première grossesse?",
        "twins": "Grossesse Multiple (Jumeaux/Triplés)?",
        "chronic_hypertension": "Antécédents d'Hypertension?",
        "displaced": "Situation de Crise/Déplacement?",
        "yes": "Oui",
        "no": "Non",
        
        # Main Screening
        "screening_tab": "Dépistage Quotidien",
        "manual_tab": "Guide d'Utilisation",
        "step_1": "Étape 1: Évaluation des Symptômes",
        "step_1_desc": "Sélectionnez les symptômes présents aujourd'hui:",
        "symptom_headache": "Mal de tête sévère",
        "symptom_vision": "Troubles visuels (flou, lumières clignotantes, points noirs)",
        "symptom_pain": "Douleur abdominale ou en haut à droite",
        "symptom_swelling": "Gonflement du visage ou des mains",
        
        # Vital Signs
        "step_2": "Étape 2: Signes Vitaux",
        "systolic": "Pression Artérielle Systolique (mmHg)",
        "diastolic": "Pression Artérielle Diastolique (mmHg)",
        "urine_label": "Détection de Protéines Urinaires",
        "urine_instructions": "Comparez la couleur de la bandelette à la référence:",
        "urine_0": "Négatif (Pas de protéine)",
        "urine_1": "Trace (1+)",
        "urine_2": "Modéré (2+)",
        "urine_3": "Élevé (3+)",
        
        # Button & Results
        "run_assessment": "Calculer l'Évaluation du Risque",
        "risk_outcome": "Résultat de l'Évaluation",
        "high_risk_alert": "RISQUE ÉLEVÉ: Prééclampsie/Éclampsie Suspectée",
        "high_risk_desc": "Évaluation médicale immédiate requise",
        "risk_percentage": "Score de Risque de Prééclampsie",
        "action_plan": "Actions Recommandées",
        "family_actions": "La Famille Doit:",
        "action_1": "Se rendre à l'hôpital sans délai",
        "action_2": "Reposer sur le côté gauche pour améliorer la circulation",
        "action_3": "Maintenir un environnement calme; minimiser le stress",
        
        "medical_protocol": "Protocole pour l'Équipe Médicale:",
        "medical_1": "Surveiller la pression artérielle en continu",
        "medical_2": "Préparer le sulfate de magnésium (prophylaxie des crises)",
        "medical_3": "Préparer un accouchement d'urgence possible après 37 semaines",
        
        "low_risk": "RISQUE FAIBLE: Dépistage Terminé",
        "low_risk_desc": "Aucun indicateur de risque aigu détecté",
        "next_steps": "Étapes Recommandées:",
        "next_1": "Poursuivre le dépistage matinal quotidien",
        "next_2": "Chercher des soins immédiats si les symptômes se développent",
        "next_3": "Planifier le suivi prénatal comme prévu",
        
        # Manual
        "manual_title": "Guide d'Utilisation et Instructions",
        "section_1": "Comment Utiliser Ce Système",
        "section_1_desc": """
        Ce système fournit un dépistage quotidien du risque de prééclampsie dans les environnements aux ressources limitées.
        À utiliser chaque matin et en cas de symptômes.
        """,
        "section_2": "Collecte des Signes Vitaux",
        "section_2_desc": """
        **Pression Artérielle:** Utilisez un tensiomètre automatique si disponible. Asseyez-vous 5 minutes avant de mesurer.
        **Test Urinaire:** Utilisez le kit de bandelette fourni. Trempez pendant 2 secondes, attendez 60 secondes, comparez la couleur.
        """,
        "section_3": "Quand Chercher des Soins d'Urgence",
        "section_3_desc": """
        • Mal de tête soudain et sévère
        • Troubles visuels ou lumières clignotantes
        • Douleur abdominale sévère
        • Gonflement du visage ou des mains
        • Lecture de pression artérielle >140/90 mmHg
        • Protéines détectées dans l'urine
        """,
        "data_note": "Remarque: Cette application ne stocke pas les données personnelles. Les résultats sont calculés localement.",
        
        # Errors
        "error_missing_model": "Erreur d'initialisation du système. Modèle clinique non disponible.",
    }
}

# ========================================
# DATABASE & MODEL CONFIGURATION
# ========================================
DB_FILE = "preeclampsia_patient_registry.db"

@st.cache_resource
def load_clinical_model():
    df = pd.read_csv('preeclampsia_chaos_dataset.csv')
    X = df.drop(columns=['patient_id', 'preeclampsia_onset'])
    y = df['preeclampsia_onset']
    engine = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=101)
    engine.fit(X, y)
    return engine

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Maternal Health Screening",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for professional styling
st.markdown("""
    <style>
        :root {
            --primary-color: #1f77b4;
            --success-color: #2ca02c;
            --danger-color: #d62728;
            --warning-color: #ff7f0e;
        }
        .main-header {
            text-align: center;
            padding: 2rem 0 1rem 0;
            border-bottom: 3px solid #1f77b4;
            margin-bottom: 2rem;
        }
        .metric-box {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            border-left: 5px solid #1f77b4;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .risk-high {
            background: linear-gradient(135deg, #ffe6e6 0%, #ffcccc 100%);
            border-left: 5px solid #d62728;
        }
        .risk-low {
            background: linear-gradient(135deg, #e6ffe6 0%, #ccffcc 100%);
            border-left: 5px solid #2ca02c;
        }
        .risk-percentage-display {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 1rem 0;
            padding: 1rem;
            border-radius: 8px;
        }
        .risk-percentage-high {
            color: #d62728;
            background-color: rgba(214, 39, 40, 0.1);
        }
        .risk-percentage-low {
            color: #2ca02c;
            background-color: rgba(44, 160, 44, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ========================================
# INITIALIZE SESSION & TRANSLATIONS
# ========================================
if 'language' not in st.session_state:
    st.session_state.language = 'en'

def t(key):
    """Get translation for key"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

# ========================================
# SIDEBAR - LANGUAGE & PATIENT INFO
# ========================================
with st.sidebar:
    st.markdown("---")
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button("🇬🇧 English", key="lang_en", use_container_width=True):
            st.session_state.language = "en"
            st.rerun()
    with lang_col2:
        if st.button("🇫🇷 Français", key="lang_fr", use_container_width=True):
            st.session_state.language = "fr"
            st.rerun()
    st.markdown("---")
    
    st.header(t("mother_profile"))
    age = st.slider(t("age"), 14, 50, 25)
    first_preg = st.selectbox(t("first_pregnancy"), [t("no"), t("yes")])
    twins = st.selectbox(t("twins"), [t("no"), t("yes")])
    chronic_hyper = st.selectbox(t("chronic_hypertension"), [t("no"), t("yes")])
    displaced = st.selectbox(t("displaced"), [t("no"), t("yes")])

map_binary = lambda text: 1 if text == t("yes") else 0

# ========================================
# MAIN HEADER
# ========================================
st.markdown(f"<div class='main-header'><h1>{t('app_title')}</h1><p>{t('app_subtitle')}</p></div>", unsafe_allow_html=True)

# ========================================
# LOAD MODEL
# ========================================
try:
    clinical_engine = load_clinical_model()
except FileNotFoundError:
    st.error(t("error_missing_model"))
    st.stop()

# ========================================
# MAIN CONTENT TABS
# ========================================
tab_screening, tab_manual = st.tabs([t("screening_tab"), t("manual_tab")])

with tab_screening:
    # Step 1: Symptoms
    st.subheader(t("step_1"))
    st.write(t("step_1_desc"))
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s_headache = st.checkbox(t("symptom_headache"))
        s_vision = st.checkbox(t("symptom_vision"))
    with col_s2:
        s_pain = st.checkbox(t("symptom_pain"))
        s_swelling = st.checkbox(t("symptom_swelling"))
    
    st.markdown("---")
    
    # Step 2: Vital Signs
    st.subheader(t("step_2"))
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        systolic = st.number_input(t("systolic"), min_value=70, max_value=220, value=115)
        diastolic = st.number_input(t("diastolic"), min_value=40, max_value=140, value=75)
    
    with col_v2:
        st.markdown(f"**{t('urine_label')}**")
        st.markdown(f"*{t('urine_instructions')}*")
        urine_score = st.radio(
            "Select protein level:",
            options=[
                f"0: {t('urine_0')}",
                f"1: {t('urine_1')}",
                f"2: {t('urine_2')}",
                f"3: {t('urine_3')}"
            ],
            label_visibility="collapsed"
        )
    
    protein_numeric = float(urine_score.split(":")[0].strip())
    symptom_active = s_headache or s_vision or s_pain or s_swelling
    
    st.markdown("---")
    
    # Run Assessment Button
    if st.button(t("run_assessment"), use_container_width=True, type="primary"):
        # Build prediction vector
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
        risk_percentage = raw_prob * 100
        
        # Risk determination
        critical_vitals = (systolic >= 140) or (diastolic >= 90) or (protein_numeric >= 2)
        is_high_risk = (raw_prob >= 0.28) or symptom_active or critical_vitals
        
        st.markdown("---")
        st.subheader(t("risk_outcome"))
        
        if is_high_risk:
            # High Risk Display
            st.markdown(f"""
                <div class='metric-box risk-high'>
                    <h2 style='color: #d62728; text-align: center;'>{t('high_risk_alert')}</h2>
                    <p style='text-align: center; font-size: 1.1em;'>{t('high_risk_desc')}</p>
                    <div class='risk-percentage-display risk-percentage-high'>{risk_percentage:.1f}%</div>
                    <p style='text-align: center; font-size: 0.95em;'>{t('risk_percentage')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                with st.container(border=True):
                    st.markdown(f"### {t('family_actions')}")
                    st.markdown(f"1. **{t('action_1')}**")
                    st.markdown(f"2. **{t('action_2')}**")
                    st.markdown(f"3. **{t('action_3')}**")
            
            with risk_col2:
                with st.container(border=True):
                    st.markdown(f"### {t('medical_protocol')}")
                    st.markdown(f"• {t('medical_1')}")
                    st.markdown(f"• {t('medical_2')}")
                    st.markdown(f"• {t('medical_3')}")
        
        else:
            # Low Risk Display
            st.markdown(f"""
                <div class='metric-box risk-low'>
                    <h2 style='color: #2ca02c; text-align: center;'>{t('low_risk')}</h2>
                    <p style='text-align: center; font-size: 1.1em;'>{t('low_risk_desc')}</p>
                    <div class='risk-percentage-display risk-percentage-low'>{risk_percentage:.1f}%</div>
                    <p style='text-align: center; font-size: 0.95em;'>{t('risk_percentage')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.container(border=True):
                st.markdown(f"### {t('next_steps')}")
                st.markdown(f"• {t('next_1')}")
                st.markdown(f"• {t('next_2')}")
                st.markdown(f"• {t('next_3')}")

with tab_manual:
    st.subheader(t("manual_title"))
    
    with st.expander(t("section_1"), expanded=True):
        st.markdown(t("section_1_desc"))
    
    with st.expander(t("section_2")):
        st.markdown(t("section_2_desc"))
    
    with st.expander(t("section_3")):
        st.markdown(t("section_3_desc"))
    
    st.markdown("---")
    st.info(t("data_note"))
