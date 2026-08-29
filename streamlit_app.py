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
# GLOBAL DICTIONARY DATA SYSTEM (PREVENTS PARSING ERRORS)
# --------------------------------------------------------
LANG_DICT = {
    "English": {
        "title": "🏡 Maternal Home-Shield Triage System",
        "subtitle": "### Daily Offline Risk Assessment & Emergency Action Protocol",
        "sidebar_title": "👤 Mother's Baseline Profile",
        "age": "Age",
        "first_preg": "First Pregnancy?",
        "twins": "Carrying Twins?",
        "chronic_hyper": "History of Chronic High Blood Pressure?",
        "displaced": "Living in a Displacement Camp/Crisis Zone?",
        "opts": ["No", "Yes"],
        "tab_screen": "🌤️ Daily Morning & Urgent Screening",
        "tab_about": "📖 System Instruction Manual",
        "step1_title": "🛑 Step 1: Check for Feeling Unwell (Symptom Scan)",
        "step1_txt": "Does the mother currently feel unwell or have any of these specific symptoms right now?",
        "s1": "❌ Severe, throbbing headache that won't go away",
        "s2": "❌ Blurry vision, flashing lights, or dark spots in front of the eyes",
        "s3": "❌ Sharp pain right below the ribs or upper stomach area",
        "s4": "❌ Sudden, massive swelling in the face, eyes, or hands",
        "step2_title": "📈 Step 2: Input Morning Vital Signs",
        "sys_label": "Systolic Blood Pressure (Top Reading from Cuff mmHg)",
        "dia_label": "Diastolic Blood Pressure (Bottom Reading from Cuff mmHg)",
        "urine_title": "**Urine Paper Strip Color Match:**",
        "urine_label": "Match the dipped morning paper strip color to the kit card options:",
        "u0": "0: Yellow / Light Yellow (Negative / Normal)",
        "u1": "1: Light Green (Trace Protein detected)",
        "u2": "2: Medium Green (High Protein detected)",
        "u3": "3: Deep Dark Green (Severe Risk Level)",
        "btn": "🔍 RUN EMERGENCY RISK CALCULATION",
        "out_title": "📊 Ultimate Risk Assessment Outcome",
        "err_msg": "🚨 CRITICAL ALERT: HIGH RISK OF DEVELOPING PREECLAMPASIA / ECLAMPSIA DETECTED",
        "act_title": "### 🚨 IMMEDIATE ACTION PLAN (For the Family)",
        "act_txt": (
            "1. **GET TO A HOSPITAL IMMEDIATELY:** Do not wait. Leave the house right away. Preeclampsia is a fast-acting emergency.\n"
            "2. **DO NOT LAY FLAT ON YOUR BACK:** If resting while waiting for transportation, lay on your **left side**. This improves blood flow to the baby and kidneys.\n"
            "3. **STAY CALM & REDUCE LIGHTS:** High blood pressure mixed with stress can trigger seizures. Keep the mother in a quiet, dark area while moving."
        ),
        "cli_title": "### 🏥 CLINICAL INTERVENTION PROTOCOL (For Field Medics)",
        "cli_txt": (
            "* **Antihypertensive Administration:** Prepare safe emergency blood pressure medications (e.g., oral Labetalol or Nifedipine) if systolic is >= 160 or diastolic is >= 110.\n"
            "* **Seizure Prophylaxis:** Administer an immediate loading dose of **Magnesium Sulfate (MgSO4)** intravenously/intramuscularly to prevent eclamptic seizures.\n"
            "* **Delivery Planning:** Assess gestational age. If the pregnancy is past 37 weeks, prepare for urgent delivery to save both lives."
        ),
        "succ_msg": "✅ STABLE TRACK: SCREENING COMPLETED SUCCESSFULLY",
        "succ_txt": (
            "**Next Actions:**\n"
            "* Everything looks normal this morning.\n"
            "* **Repeat this test tomorrow morning** at the exact same time.\n"
            "* **CRITICAL RULE:** If the mother feels unwell later today (headache, vision changes, or pain), do not wait for tomorrow. **Run this screening test again immediately.**"
        ),
        "man_title": "📋 Low-Resource Home Triage Manual",
        "man_txt": (
            "### How to Use the At-Home Kit in Displacement Settings\n\n"
            "1. **The Morning Routine:** Every morning, before eating or walking around, the mother should sit quietly for 5 minutes, then take her blood pressure.\n"
            "2. **The Dipstick Method:** Collect a tiny amount of urine in a clean cup. Dip the paper strip for 2 seconds. Shake off excess fluid. Wait 60 seconds, then match the color to the cardboard reference strip.\n"
            "3. **The Unwell Trigger:** Preeclampsia does not care about schedules. If a mother says *'I feel strange'* or complains of a headache, her family must run this app immediately."
        )
    },
    "Français": {
        "title": "🏡 Système de Triage Maternelle à Domicile",
        "subtitle": "### Évaluation Quotidienne du Risque Hors-Ligne et Protocole d'Urgence",
        "sidebar_title": "👤 Profil de Base de la Mère",
        "age": "Âge",
        "first_preg": "Première Grossesse ?",
        "twins": "Grossesse Gémellaire ?",
        "chronic_hyper": "Antécédents d'Hypertension Artérielle Chronique ?",
        "displaced": "Vit dans un Camp de Réfugiés / Zone de Crise ?",
        "opts": ["Non", "Oui"],
        "tab_screen": "🌤️ Évaluation Quotidienne et Urgente",
        "tab_about": "📖 Manuel d'Instructions du Système",
        "step1_title": "🛑 Étape 1 : Vérification des Symptômes (Malaise)",
        "step1_txt": "La mère se sent-elle actuellement mal ou présente-t-elle l'un de ces symptômes en ce moment ?",
        "s1": "❌ Maux de tête graves et lancinants qui ne disparaissent pas",
        "s2": "❌ Vision floue, lumières clignotantes ou taches sombres devant les yeux",
        "s3": "❌ Douleur aiguë juste sous les côtes ou dans la zone supérieure de l'estomac",
        "s4": "❌ Gonflement soudain et massif du visage, des yeux ou des mains",
        "step2_title": "📈 Étape 2 : Saisir les Signes Vitaux du Matin",
        "sys_label": "Pression Systolique (Chiffre Supérieur du Brassard - mmHg)",
        "dia_label": "Pression Diastolique (Chiffre Inférieur du Brassard - mmHg)",
        "urine_title": "**Correspondance des Couleurs de la Bandelette :**",
        "urine_label": "Faites correspondre la couleur de la bandelette urinaire du matin aux options de la carte :",
        "u0": "0 : Jaune / Jaune Clair (Négatif / Normal)",
        "u1": "1 : Vert Clair (Traces de protéines détectées)",
        "u2": "2 : Vert Moyen (Taux de protéines élevé détecté)",
        "u3": "3 : Vert Foncé (Niveau de risque grave)",
        "btn": "🔍 LANCER LE CALCUL DU RISQUE D'URGENCE",
        "out_title": "📊 Résultat de l'Évaluation Finale du Risque",
        "err_msg": "🚨 ALERTE CRITIQUE : RISQUE ÉLEVÉ DE PRÉÉCLAMPSIE / ÉCLAMPSIE DÉTECTÉ",
        "act_title": "### 🚨 PLAN D'ACTION IMMÉDIAT (Pour la Famille)",
        "act_txt": (
            "1. **RENDEZ-VOUS IMMÉDIATEMENT À L'HÔPITAL :** N'attendez pas. Partez tout de suite. La prééclampsie est une urgence à évolution rapide.\n"
            "2. **NE VOUS ALLONGEZ PAS SUR LE DOS :** Si vous vous reposez en attendant le transport, allongez-vous sur le **côté gauche**. Cela améliore le flux sanguin vers le bébé et les reins.\n"
            "3. **RESTEZ CALME ET RÉDUISEZ LA LUMIÈRE :** Une tension artérielle élevée combinée au stress peut déclencher des crises. Gardez la mère dans un endroit calme et sombre pendant le déplacement."
        ),
        "cli_title": "### 🏥 PROTOCOLE D'INTERVENTION CLINIQUE (Pour les Médicaux sur le Terrain)",
        "cli_txt": (
            "* **Administration d'Antihypertenseurs :** Préparer des médicaments sûrs pour la tension artérielle d'urgence (ex. Labétalol oral ou Nifédipine) si la systolique est >= 160 ou la diastolique est >= 110.\n"
            "* **Prophylaxie des Crises :** Administrer immédiatement une dose de charge de **Sulfate de Magnésium (MgSO4)** par voie intraveineuse/intramusculaire pour prévenir les crises d'éclampsie.\n"
            "* **Planification de l'Accouchement :** Évaluer l'âge gestationnel. Si la grossesse dépasse 37 semaines, préparer un accouchement urgent pour sauver les deux vies."
        ),
        "succ_msg": "✅ SUIVI STABLE : ÉVALUATION TERMINÉE AVEC SUCCÈS",
        "succ_txt": (
            "**Actions Suivantes :**\n"
            "* Tout semble normal ce matin.\n"
            "* **Répétez ce test demain matin** exactement à la même heure.\n"
            "* **RÈGLE CRITIQUE :** Si la mère se sent mal plus tard aujourd'hui (maux de tête, changements de vision ou douleur), n'attendez pas demain. **Refaites ce test de dépistage immédiatement.**"
        ),
        "man_title": "📋 Manuel de Triage à Domicile (Faibles Ressources)",
        "man_txt": (
            "### Comment Utiliser le Kit à Domicile en Contexte de Crise\n\n"
            "1. **La Routine du Matin :** Chaque matin, avant de manger ou de marcher, la mère doit s'asseoir calmement pendant 5 minutes, puis prendre sa tension artérielle.\n"
            "2. **La Méthode de la Bandelette :** Recueillir une petite quantité d'urine dans un gobelet propre. Tremper la bandelette de papier pendant 2 secondes. Secouer l'excès de liquide. Attendre 60 secondes, puis faire correspondre la couleur à la bandelette de référence en carton.\n"
