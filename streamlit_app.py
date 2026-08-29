import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------
# CORE SYSTEM OPERATIONS & CONFIGURATIONS
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
                    <li>If patellar reflexes vanish, RR drops below 12 breaths/minute, or severe oliguria occurs: Stop MgSO4 immediately.</li>
                    <li>Administer 10 mL of 10% Calcium Gluconate (1g) intravenously slowly. Provide respiratory support and secure senior medical call assistance.</li>
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
                    <li>Si les réflexes disparaissent ou la FR tombe en dessous de 12 cycles/min : Arrêtez le MgSO4 immédiatement.</li>
                    <li>Administrer 10 mL de Gluconate de Calcium à 10% (1g) par voie intraveineuse lente. Appelez une assistance médicale.</li>
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

# Raw clean style injection variable to stop compiler exceptions
CSS_DATA = """
<style>
    .main { background-color: #f8fafc; }
    .metric-card { background-color: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
    .protocol-box { border-radius: 12px; padding: 1.5rem; margin-top: 1rem; color: #1e293b; }
    .family-box { background-color: #fef2f2; border-left: 5px solid #ef4444; }
