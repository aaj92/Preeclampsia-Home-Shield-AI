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
