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
        "urine_instructions": "Dip the strip as directed, wait the specified time, then compare it with the color chart supplied with your test kit:",
        "urine_0": "No color change",
        "urine_1": "Slight color change",
        "urine_2": "Moderate color change",
        "urine_3": "Strong color change",
        
        # Button & Results
        "run_assessment": "Calculate Risk Assessment",
        "risk_outcome": "Risk Assessment Result",
        "high_risk_alert": "HIGH RISK: Preeclampsia/Eclampsia Suspected",
        "high_risk_desc": "Immediate medical evaluation required",
        "risk_percentage": "Preeclampsia Risk Score",
        
        # Family Actions (Left Column)
        "family_response": "IMMEDIATE FAMILY RESPONSE",
        "action_1": "Seek hospital care immediately - do not delay",
        "action_2": "Position mother on LEFT SIDE to improve blood flow",
        "action_3": "Keep environment quiet and dark - reduce stress",
        "action_4": "Inform hospital staff of high preeclampsia risk",
        "action_5": "Bring this screening result to hospital",
        
        # Medical Response (Right Column)
        "medical_response": "EMERGENCY MEDICAL RESPONSE",
        "medical_response_subtitle": "(Key Points)",
        
        "anticonvulsant_title": "Anticonvulsant — Magnesium sulfate:",
        "pritchard_regimen": "Pritchard (IM+IV) — loading 4 g IV slow + 10 g IM (5 g each buttock); then 5 g IM every 4 h; or",
        "zuspan_regimen": "Zuspan (IV) — loading 4 g IV slow; then 1 g/hr IV infusion.",
        
        "monitoring_title": "Monitoring:",
        "monitoring_text": "Respiratory rate and deep-tendon reflexes hourly; urine output hourly (alert if <25-30 mL/h); continuous BP and fetal monitoring as available.",
        
        "toxicity_title": "Toxicity — immediate actions:",
        "toxicity_text": "If absent reflexes, RR < 12/min, or oliguria → stop MgSO₄, give calcium gluconate 10 mL of 10% (1 g) IV slowly, provide respiratory support, call senior help.",
        
        "duration_title": "Duration:",
        "duration_text": "Continue for 24 hours after last seizure or after delivery (whichever is later).",
        
        "practical_title": "Practical:",
        "practical_text": "Confirm local MgSO₄ concentration before converting g → mL (example: 50% = 500 mg/mL → 4 g = 8 mL); keep calcium gluconate at bedside.",
        
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
        "urine_home_title": "🧪 HOME URINE CHECK",
        "urine_home_note": "Select the color change that most closely matches your test strip.",
        "home_action_label": "FOR FAMILY / CAREGIVER",
        "medical_action_label": "FOR HEALTHCARE PROFESSIONALS",
        "why_high_risk": "Why did the system trigger this alert?",
        "screening_disclaimer": "SCREENING RESULT — NOT A DIAGNOSIS",
        "screening_disclaimer_text": "This tool does not replace professional medical assessment. If severe symptoms are present, seek emergency care immediately regardless of the calculated score.",
        "screening_completed": "Screening completed",
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
        "urine_instructions": "Utilisez la bandelette selon les instructions, attendez le temps indiqué, puis comparez-la au nuancier fourni avec votre kit:",
        "urine_0": "Aucun changement de couleur",
        "urine_1": "Slight color change",
        "urine_2": "Changement de couleur modéré",
        "urine_3": "Changement de couleur important",
        
        # Button & Results
        "run_assessment": "Calculer l'Évaluation du Risque",
        "risk_outcome": "Résultat de l'Évaluation",
        "high_risk_alert": "RISQUE ÉLEVÉ: Prééclampsie/Éclampsie Suspectée",
        "high_risk_desc": "Évaluation médicale immédiate requise",
        "risk_percentage": "Score de Risque de Prééclampsie",
        
        # Family Actions (Left Column)
        "family_response": "RÉPONSE FAMILIALE IMMÉDIATE",
        "action_1": "Chercher les soins hospitaliers immédiatement - ne pas attendre",
        "action_2": "Positionner la mère sur le CÔTÉ GAUCHE pour améliorer la circulation",
        "action_3": "Garder l'environnement calme et sombre - réduire le stress",
        "action_4": "Informer le personnel hospitalier du risque élevé de prééclampsie",
        "action_5": "Apporter ce résultat de dépistage à l'hôpital",
        
        # Medical Response (Right Column)
        "medical_response": "RÉPONSE MÉDICALE D'URGENCE",
        "medical_response_subtitle": "(Points Clés)",
        
        "anticonvulsant_title": "Anticonvulsivant — Sulfate de magnésium:",
        "pritchard_regimen": "Pritchard (IM+IV) — charge 4 g IV lent + 10 g IM (5 g chaque fesse); puis 5 g IM toutes les 4 h; ou",
        "zuspan_regimen": "Zuspan (IV) — charge 4 g IV lent; puis 1 g/h perfusion IV.",
        
        "monitoring_title": "Surveillance:",
        "monitoring_text": "Fréquence respiratoire et réflexes rotuliens toutes les heures; débit urinaire horaire (alerte si <25-30 mL/h); surveillance continue PA et fœtale si disponible.",
        
        "toxicity_title": "Toxicité — actions immédiates:",
        "toxicity_text": "Si réflexes absents, FR < 12/min, ou oligurie → arrêter MgSO₄, donner gluconate de calcium 10 mL de 10% (1 g) IV lent, soutien respiratoire, appeler clinicien senior.",
        
        "duration_title": "Durée:",
        "duration_text": "Continuer pendant 24 heures après la dernière crise ou après l'accouchement (le plus tard).",
        
        "practical_title": "Pratique:",
        "practical_text": "Confirmer la concentration locale de MgSO₄ avant conversion g → mL (exemple: 50% = 500 mg/mL → 4 g = 8 mL); garder gluconate de calcium à côté du lit.",
        
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
        "urine_home_title": "🧪 TEST URINAIRE À DOMICILE",
        "urine_home_note": "Sélectionnez le changement de couleur qui correspond le mieux à votre bandelette.",
        "home_action_label": "POUR LA FAMILLE / L’AIDANT",
        "medical_action_label": "POUR LES PROFESSIONNELS DE SANTÉ",
        "why_high_risk": "Pourquoi cette alerte a-t-elle été déclenchée ?",
        "screening_disclaimer": "RÉSULTAT DE DÉPISTAGE — PAS UN DIAGNOSTIC",
        "screening_disclaimer_text": "Cet outil ne remplace pas une évaluation médicale professionnelle. En présence de symptômes sévères, recherchez immédiatement des soins d’urgence, quel que soit le score calculé.",
        "screening_completed": "Dépistage terminé",
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
        .response-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin: 2rem 0;
        }
        .response-card {
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .family-card {
            background-color: #e8f4ff;
            border-top: 6px solid #0066cc;
        }
        .medical-card {
            background-color: #ffe8e8;
            border-top: 6px solid #d62728;
        }
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 3px solid;
        }
        .family-card .card-header {
            border-bottom-color: #0066cc;
        }
        .medical-card .card-header {
            border-bottom-color: #d62728;
        }
        .card-header-icon {
            font-size: 2em;
            margin-right: 1rem;
        }
        .card-header-title {
            font-size: 1.3em;
            font-weight: bold;
            margin: 0;
        }
        .family-card .card-header-title {
            color: #0066cc;
        }
        .medical-card .card-header-title {
            color: #d62728;
        }
        .card-subtitle {
            font-size: 0.9em;
            color: #666;
            margin-top: 0.5rem;
            font-style: italic;
        }
        .action-item {
            display: flex;
            align-items: flex-start;
            margin: 0.8rem 0;
            font-size: 0.95em;
            line-height: 1.5;
            color: #333;
        }
        .action-icon {
            font-size: 1.2em;
            margin-right: 0.8rem;
            flex-shrink: 0;
        }
        .medical-section {
            margin: 1.2rem 0;
            padding: 1rem;
            background-color: rgba(255,255,255,0.8);
            border-left: 4px solid #d62728;
            border-radius: 6px;
        }
        .medical-section-title {
            font-weight: bold;
            color: #d62728;
            margin-bottom: 0.6rem;
            font-size: 0.95em;
        }
        .medical-section-text {
            color: #333;
            font-size: 0.9em;
            line-height: 1.6;
        }
        @media (max-width: 1024px) {
            .response-grid {
                grid-template-columns: 1fr;
            }
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
        st.markdown(f"### {t('urine_home_title')}")
        st.markdown(f"*{t('urine_instructions')}*")
        st.caption(t("urine_home_note"))
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
            
            st.markdown("---")
            
            # Side-by-Side Response Cards
            # Render the complete nested HTML in ONE component. This prevents
            # Streamlit's Markdown renderer from exposing the HTML as text.
            response_html = f"""
            <style>
                .response-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 24px;
                    margin: 24px 0;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                }}
                .response-card {{
                    border-radius: 20px;
                    padding: 26px;
                    box-shadow: 0 10px 30px rgba(15,23,42,.10);
                    border: 1px solid rgba(15,23,42,.08);
                    box-sizing: border-box;
                    background: #fff;
                }}
                .family-card {{
                    background: linear-gradient(145deg,#f0f8ff 0%,#fff 100%);
                    border-top: 5px solid #1976d2;
                }}
                .medical-card {{
                    background: linear-gradient(145deg,#fff4f4 0%,#fff 100%);
                    border-top: 5px solid #d32f2f;
                }}
                .card-header {{
                    display: flex;
                    align-items: center;
                    gap: 14px;
                    padding-bottom: 18px;
                    margin-bottom: 18px;
                    border-bottom: 1px solid rgba(15,23,42,.10);
                }}
                .card-header-icon {{
                    width: 48px;
                    height: 48px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 14px;
                    font-size: 24px;
                    background: rgba(255,255,255,.85);
                    box-shadow: 0 4px 12px rgba(15,23,42,.08);
                    flex-shrink: 0;
                }}
                .audience-label {{
                    font-size: .68rem;
                    font-weight: 800;
                    letter-spacing: .08em;
                    margin-bottom: 4px;
                    text-transform: uppercase;
                }}
                .family-audience {{ color: #1976d2; }}
                .medical-audience {{ color: #d32f2f; }}
                .card-header-title {{
                    font-size: 1.12rem;
                    line-height: 1.3;
                    font-weight: 750;
                    margin: 0;
                }}
                .card-subtitle {{
                    margin-top: 4px;
                    color: #64748b;
                    font-size: .84rem;
                }}
                .action-item {{
                    display: flex;
                    align-items: flex-start;
                    gap: 12px;
                    padding: 12px 0;
                    color: #263238;
                    font-size: .94rem;
                    line-height: 1.55;
                }}
                .action-icon {{
                    width: 25px;
                    height: 25px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    background: #dbeafe;
                    color: #1565c0;
                    font-weight: 800;
                    flex-shrink: 0;
                }}
                .medical-section {{
                    margin: 12px 0;
                    padding: 15px 16px;
                    background: rgba(255,255,255,.88);
                    border: 1px solid rgba(211,47,47,.12);
                    border-left: 4px solid #d32f2f;
                    border-radius: 12px;
                }}
                .medical-section-title {{
                    color: #b71c1c;
                    font-weight: 750;
                    font-size: .90rem;
                    margin-bottom: 7px;
                }}
                .medical-section-text {{
                    color: #374151;
                    font-size: .86rem;
                    line-height: 1.62;
                }}
                @media (max-width: 900px) {{
                    .response-grid {{ grid-template-columns: 1fr; }}
                }}
            </style>

            <div class="response-grid">
                <section class="response-card family-card">
                    <div class="card-header">
                        <div class="card-header-icon">👨‍👩‍👧</div>
                        <div>
                            <div class="audience-label family-audience">{t('home_action_label')}</div>
                            <div class="card-header-title">{t('family_response')}</div>
                        </div>
                    </div>
                    <div class="action-item"><div class="action-icon">✓</div><div>{t('action_1')}</div></div>
                    <div class="action-item"><div class="action-icon">✓</div><div>{t('action_2')}</div></div>
                    <div class="action-item"><div class="action-icon">✓</div><div>{t('action_3')}</div></div>
                    <div class="action-item"><div class="action-icon">✓</div><div>{t('action_4')}</div></div>
                    <div class="action-item"><div class="action-icon">✓</div><div>{t('action_5')}</div></div>
                </section>

                <section class="response-card medical-card">
                    <div class="card-header">
                        <div class="card-header-icon">🏥</div>
                        <div>
                            <div class="audience-label medical-audience">{t('medical_action_label')}</div>
                            <div class="card-header-title">{t('medical_response')}</div>
                            <div class="card-subtitle">{t('medical_response_subtitle')}</div>
                        </div>
                    </div>
                    <div class="medical-section">
                        <div class="medical-section-title">{t('anticonvulsant_title')}</div>
                        <div class="medical-section-text">• {t('pritchard_regimen')}<br>{t('zuspan_regimen')}</div>
                    </div>
                    <div class="medical-section">
                        <div class="medical-section-title">{t('monitoring_title')}</div>
                        <div class="medical-section-text">{t('monitoring_text')}</div>
                    </div>
                    <div class="medical-section">
                        <div class="medical-section-title">{t('toxicity_title')}</div>
                        <div class="medical-section-text">{t('toxicity_text')}</div>
                    </div>
                    <div class="medical-section">
                        <div class="medical-section-title">{t('duration_title')}</div>
                        <div class="medical-section-text">{t('duration_text')}</div>
                    </div>
                    <div class="medical-section">
                        <div class="medical-section-title">{t('practical_title')}</div>
                        <div class="medical-section-text">{t('practical_text')}</div>
                    </div>
                </section>
            </div>
            """

            st.html(response_html)

            # Show the concrete reasons for the emergency flag.
            triggers = []
            if systolic >= 140 or diastolic >= 90:
                triggers.append(f"🔴 Blood pressure: {systolic}/{diastolic} mmHg")
            if protein_numeric >= 2:
                triggers.append("🔴 Strong urine-strip color change")
            elif protein_numeric >= 1:
                triggers.append("🟠 Urine-strip color change detected")
            if s_headache:
                triggers.append("🔴 Severe headache reported")
            if s_vision:
                triggers.append("🔴 Vision changes reported")
            if s_pain:
                triggers.append("🔴 Upper abdominal/right-upper-quadrant pain reported")
            if s_swelling:
                triggers.append("🟠 Facial or hand swelling reported")
            if raw_prob >= 0.28:
                triggers.append(f"🟠 Model-estimated risk: {risk_percentage:.1f}%")

            with st.container(border=True):
                st.markdown(f"### {t('why_high_risk')}")
                for trigger in triggers:
                    st.markdown(f"- {trigger}")
                st.caption(
                    f"⏱ {t('screening_completed')}: "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )

            st.warning(
                f"**{t('screening_disclaimer')}**\n\n"
                f"{t('screening_disclaimer_text')}"
            )

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
                st.markdown(f"✓ {t('next_1')}")
                st.markdown(f"✓ {t('next_2')}")
                st.markdown(f"✓ {t('next_3')}")

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
