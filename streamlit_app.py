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
        
        # Family Actions (Left Column)
        "family_response": "👨‍👩‍👧 IMMEDIATE FAMILY RESPONSE",
        "family_actions_title": "Actions for Family/Caregiver",
        "action_1": "Seek hospital care immediately - do not delay",
        "action_2": "Position mother on LEFT SIDE to improve blood flow",
        "action_3": "Keep environment quiet and dark - reduce stress",
        "action_4": "Inform hospital staff of high preeclampsia risk",
        "action_5": "Bring this screening result to hospital",
        
        # Medical Response (Right Column) - WHO Protocol
        "medical_response": "🏥 EMERGENCY MEDICAL RESPONSE",
        "medical_response_subtitle": "(WHO Clinical Management Protocol)",
        
        "anti_seizure": "Anti-Seizure Prophylaxis",
        "anti_seizure_drug": "Magnesium Sulfate (MgSO₄) - First Line",
        "loading_dose": "Loading Dose:",
        "loading_text": "4-6 g IV over 20-30 minutes",
        "maintenance_dose": "Maintenance Dose:",
        "maintenance_text": "1-2 g/hour IV continuous OR 5 g IM every 4 hours",
        
        "vitals_monitoring": "Continuous Monitoring",
        "monitor_1": "Blood Pressure: every 15 min initially, then hourly",
        "monitor_2": "Respiratory rate: every hour (alert if <12/min)",
        "monitor_3": "Patellar reflexes: every hour (assess for hyperreflexia)",
        "monitor_4": "Urine output: hourly (alert if <30 mL/h)",
        "monitor_5": "Fetal heart rate: continuous if available",
        
        "urgent_action": "Urgent Actions",
        "urgent_1": "Prepare for DELIVERY - definitive treatment",
        "urgent_2": "If ≥37 weeks gestation: proceed to delivery",
        "urgent_3": "If <37 weeks: assess delivery vs expectant management",
        "urgent_4": "Have ICU/HDU bed available",
        
        "toxicity_alert": "⚠️ MgSO₄ Toxicity Signs",
        "toxicity_1": "STOP MgSO₄ if: Absent reflexes OR RR <12/min OR oliguria (<30 mL/h)",
        "toxicity_action": "Immediate Action: Give calcium gluconate 10 mL of 10% (1 g) IV slowly",
        "toxicity_support": "Support: Airway management, O₂, call senior clinician immediately",
        
        "post_eclampsia": "Post-Seizure/Delivery Care",
        "post_1": "Continue MgSO₄ for 24 hours after last seizure or delivery (whichever later)",
        "post_2": "Monitor for pulmonary edema, renal dysfunction, cerebral hemorrhage",
        "post_3": "Consider ICU admission if severe preeclampsia/eclampsia",
        
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
        
        # Family Actions (Left Column)
        "family_response": "👨‍👩‍👧 RÉPONSE FAMILIALE IMMÉDIATE",
        "family_actions_title": "Actions pour la Famille/Soignant",
        "action_1": "Chercher les soins hospitaliers immédiatement - ne pas attendre",
        "action_2": "Positionner la mère sur le CÔTÉ GAUCHE pour améliorer la circulation",
        "action_3": "Garder l'environnement calme et sombre - réduire le stress",
        "action_4": "Informer le personnel hospitalier du risque élevé de prééclampsie",
        "action_5": "Apporter ce résultat de dépistage à l'hôpital",
        
        # Medical Response (Right Column) - WHO Protocol
        "medical_response": "🏥 RÉPONSE MÉDICALE D'URGENCE",
        "medical_response_subtitle": "(Protocole de Gestion Clinique OMS)",
        
        "anti_seizure": "Prophylaxie Anti-Convulsante",
        "anti_seizure_drug": "Sulfate de Magnésium (MgSO₄) - Première Intention",
        "loading_dose": "Dose de Charge:",
        "loading_text": "4-6 g IV sur 20-30 minutes",
        "maintenance_dose": "Dose d'Entretien:",
        "maintenance_text": "1-2 g/heure IV continu OU 5 g IM toutes les 4 heures",
        
        "vitals_monitoring": "Surveillance Continue",
        "monitor_1": "Pression Artérielle: toutes les 15 min initialement, puis toutes les heures",
        "monitor_2": "Fréquence respiratoire: toutes les heures (alerte si <12/min)",
        "monitor_3": "Réflexes patellaires: toutes les heures (évaluer l'hyperréflexie)",
        "monitor_4": "Débit urinaire: horaire (alerte si <30 mL/h)",
        "monitor_5": "Fréquence cardiaque fœtale: continue si disponible",
        
        "urgent_action": "Actions Urgentes",
        "urgent_1": "Préparer l'ACCOUCHEMENT - traitement définitif",
        "urgent_2": "Si ≥37 semaines: procéder à l'accouchement",
        "urgent_3": "Si <37 semaines: évaluer l'accouchement vs gestion expectante",
        "urgent_4": "Avoir un lit USI/HDU disponible",
        
        "toxicity_alert": "⚠️ Signes de Toxicité du MgSO₄",
        "toxicity_1": "ARRÊTER MgSO₄ si: Réflexes absents OU FR <12/min OU oligurie (<30 mL/h)",
        "toxicity_action": "Action Immédiate: Donner gluconate de calcium 10 mL de 10% (1 g) IV lentement",
        "toxicity_support": "Soutien: Gestion des voies aériennes, O₂, appeler le clinicien senior immédiatement",
        
        "post_eclampsia": "Soins Post-Crise/Accouchement",
        "post_1": "Continuer MgSO₄ pendant 24 heures après la dernière crise ou l'accouchement (le plus tard)",
        "post_2": "Surveiller l'œdème pulmonaire, la dysfonction rénale, l'hémorragie cérébrale",
        "post_3": "Envisager l'admission à l'USI si prééclampsie/éclampsie sévère",
        
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
        .response-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-top: 4px solid;
            height: 100%;
        }
        .family-card {
            border-top-color: #0066cc;
        }
        .medical-card {
            border-top-color: #d62728;
        }
        .card-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid;
        }
        .family-card .card-title {
            color: #0066cc;
            border-bottom-color: #0066cc;
        }
        .medical-card .card-title {
            color: #d62728;
            border-bottom-color: #d62728;
        }
        .card-section {
            margin: 1rem 0;
            padding: 0.8rem;
            background-color: rgba(255,255,255,0.8);
            border-left: 3px solid;
            border-radius: 4px;
        }
        .family-card .card-section {
            border-left-color: #0066cc;
        }
        .medical-card .card-section {
            border-left-color: #d62728;
        }
        .section-title {
            font-weight: bold;
            font-size: 0.95em;
            margin-bottom: 0.5rem;
            color: #333;
        }
        .section-text {
            font-size: 0.9em;
            line-height: 1.5;
            color: #555;
        }
        .toxicity-alert {
            background-color: rgba(214, 39, 40, 0.15);
            border-left-color: #d62728 !important;
            color: #d62728;
            font-weight: bold;
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
            
            st.markdown("---")
            
            # Side-by-Side Response Cards
            left_col, right_col = st.columns(2)
            
            with left_col:
                st.markdown(f"""
                    <div class='response-card family-card'>
                        <div class='card-title'>{t('family_response')}</div>
                        
                        <div class='card-section'>
                            <div class='section-text'>
                                ✓ {t('action_1')}<br>
                                ✓ {t('action_2')}<br>
                                ✓ {t('action_3')}<br>
                                ✓ {t('action_4')}<br>
                                ✓ {t('action_5')}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with right_col:
                st.markdown(f"""
                    <div class='response-card medical-card'>
                        <div class='card-title'>{t('medical_response')}</div>
                        <div style='font-size: 0.9em; color: #666; margin-bottom: 1rem;'>{t('medical_response_subtitle')}</div>
                        
                        <div class='card-section'>
                            <div class='section-title'>{t('anti_seizure')}</div>
                            <div class='section-text'>{t('anti_seizure_drug')}</div>
                            <div style='margin-top: 0.5rem;'>
                                <strong>{t('loading_dose')}</strong> {t('loading_text')}<br>
                                <strong>{t('maintenance_dose')}</strong> {t('maintenance_text')}
                            </div>
                        </div>
                        
                        <div class='card-section'>
                            <div class='section-title'>{t('vitals_monitoring')}</div>
                            <div class='section-text'>
                                • {t('monitor_1')}<br>
                                • {t('monitor_2')}<br>
                                • {t('monitor_3')}<br>
                                • {t('monitor_4')}<br>
                                • {t('monitor_5')}
                            </div>
                        </div>
                        
                        <div class='card-section'>
                            <div class='section-title'>{t('urgent_action')}</div>
                            <div class='section-text'>
                                • {t('urgent_1')}<br>
                                • {t('urgent_2')}<br>
                                • {t('urgent_3')}<br>
                                • {t('urgent_4')}
                            </div>
                        </div>
                        
                        <div class='card-section toxicity-alert'>
                            <div class='section-title'>{t('toxicity_alert')}</div>
                            <div class='section-text'>
                                • {t('toxicity_1')}<br>
                                • {t('toxicity_action')}<br>
                                • {t('toxicity_support')}
                            </div>
                        </div>
                        
                        <div class='card-section'>
                            <div class='section-title'>{t('post_eclampsia')}</div>
                            <div class='section-text'>
                                • {t('post_1')}<br>
                                • {t('post_2')}<br>
                                • {t('post_3')}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
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
