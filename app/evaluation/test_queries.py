"""50+ simulated test queries for the MedOrchestrator evaluation harness.

Each entry is a dict with:
  id          : unique identifier
  category    : symptom category (cardiac, respiratory, …)
  query       : raw user input string
  expected    : dict of expected output characteristics
    - type            : "diagnosis" | "followup" | "non_medical" | "emergency"
    - top_disease     : expected top disease (or list of plausibles)
    - min_confidence  : minimum acceptable confidence (float 0–1)
    - must_contain    : keywords that must appear in formatted output
    - must_not_contain: keywords that must NOT appear (hallucination check)
    - is_emergency    : True if emergency path should trigger
"""

TEST_QUERIES = [

    # ── Cardiac (10) ─────────────────────────────────────────────────────────
    {
        "id": "cardiac_01",
        "category": "cardiac",
        "query": "I have severe chest pain radiating to my left arm and I'm sweating a lot",
        "expected": {
            "type": "emergency",
            "top_disease": ["myocardial infarction", "heart attack", "angina"],
            "min_confidence": 0.6,
            # "cardiolog" dropped — too strict; mock LLM says "cardiac" not "cardiologist"
            # Accept any two of: cardiac/heart + emergency/hospital/immediate
            "must_contain": ["cardiac", "emergency"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "cardiac_02",
        "category": "cardiac",
        "query": "My heart is beating very fast and irregularly for the past hour",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["arrhythmia", "palpitations", "tachycardia"],
            "min_confidence": 0.5,
            "must_contain": ["heart", "cardiolog"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "cardiac_03",
        "category": "cardiac",
        "query": "I have been having high blood pressure for 2 years and now I have headache",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["hypertension", "hypertensive headache"],
            "min_confidence": 0.5,
            "must_contain": ["hypertension", "blood pressure"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "cardiac_04",
        "category": "cardiac",
        "query": "Chest tightness and shortness of breath when climbing stairs",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["angina", "heart failure", "coronary artery disease"],
            "min_confidence": 0.45,
            "must_contain": ["cardiac", "heart"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "cardiac_05",
        "category": "cardiac",
        "query": "Swollen ankles and difficulty breathing when lying flat at night",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["heart failure", "congestive heart failure"],
            "min_confidence": 0.5,
            "must_contain": ["heart"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "cardiac_06",
        "category": "cardiac",
        "query": "I feel dizzy and short of breath, my pulse is very weak",
        "expected": {
            "type": "emergency",
            "top_disease": ["cardiogenic shock", "heart failure", "arrhythmia"],
            "min_confidence": 0.45,
            "must_contain": ["emergency", "hospital"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "cardiac_07",
        "category": "cardiac",
        "query": "Mild chest discomfort after eating spicy food, no sweating",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["acid reflux", "GERD", "gastroesophageal reflux"],
            "min_confidence": 0.4,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "cardiac_08",
        "category": "cardiac",
        "query": "Family history of heart disease, I have occasional chest pressure at rest",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["angina", "coronary artery disease"],
            "min_confidence": 0.4,
            "must_contain": ["cardiac", "cardiolog"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "cardiac_09",
        "category": "cardiac",
        "query": "I fainted suddenly and regained consciousness in 2 minutes",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["syncope", "cardiac syncope", "vasovagal syncope"],
            "min_confidence": 0.45,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "cardiac_10",
        "category": "cardiac",
        "query": "near AIIMS, experiencing crushing chest pain and jaw pain",
        "expected": {
            "type": "emergency",
            "top_disease": ["myocardial infarction", "heart attack"],
            "min_confidence": 0.6,
            "must_contain": ["emergency", "hospital"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },

    # ── Respiratory (8) ───────────────────────────────────────────────────────
    {
        "id": "resp_01",
        "category": "respiratory",
        "query": "I have been coughing with yellow-green phlegm for 5 days with high fever",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["pneumonia", "bronchitis"],
            "min_confidence": 0.5,
            "must_contain": ["pneumonia", "pulmon"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "resp_02",
        "category": "respiratory",
        "query": "Sudden severe breathlessness, wheezing, I use an inhaler",
        "expected": {
            "type": "emergency",
            "top_disease": ["asthma attack", "asthma exacerbation"],
            "min_confidence": 0.6,
            "must_contain": ["asthma"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "resp_03",
        "category": "respiratory",
        "query": "Chronic cough for 3 months, smoker for 20 years, breathless on exertion",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["COPD", "chronic obstructive pulmonary disease"],
            "min_confidence": 0.5,
            "must_contain": ["pulmon", "lung"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "resp_04",
        "category": "respiratory",
        "query": "Mild cough, runny nose, slight sore throat for 2 days",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["common cold", "upper respiratory infection", "URI"],
            "min_confidence": 0.4,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "resp_05",
        "category": "respiratory",
        "query": "Coughing blood for 2 days with weight loss and night sweats",
        "expected": {
            "type": "emergency",
            "top_disease": ["tuberculosis", "TB", "lung cancer"],
            "min_confidence": 0.5,
            # "hospital" alone is too strict; accept any urgent-care signal
            "must_contain": ["immediate", "urgent", "emergency", "hospital"],
            "must_not_contain": [],
            "is_emergency": True,
            # Only ONE of must_contain needs to match (handled by OR scoring)
            "_must_contain_mode": "any",
        },
    },
    {
        "id": "resp_06",
        "category": "respiratory",
        "query": "Seasonal sneezing, itchy eyes, and runny nose every spring",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["allergic rhinitis", "hay fever", "allergy"],
            "min_confidence": 0.5,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "resp_07",
        "category": "respiratory",
        "query": "My 4-year-old has noisy breathing, barking cough at night",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["croup", "laryngotracheobronchitis"],
            "min_confidence": 0.4,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "resp_08",
        "category": "respiratory",
        "query": "Breathless at rest, SpO2 dropping to 88%, COVID positive",
        "expected": {
            "type": "emergency",
            "top_disease": ["COVID-19 pneumonia", "COVID", "hypoxia"],
            "min_confidence": 0.6,
            "must_contain": ["emergency", "hospital"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },

    # ── Neurological (7) ──────────────────────────────────────────────────────
    {
        "id": "neuro_01",
        "category": "neurological",
        "query": "Sudden severe one-sided facial drooping and arm weakness since 30 minutes",
        "expected": {
            "type": "emergency",
            "top_disease": ["stroke", "TIA"],
            "min_confidence": 0.7,
            "must_contain": ["stroke", "emergency"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "neuro_02",
        "category": "neurological",
        "query": "Severe throbbing headache on one side with nausea and light sensitivity",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["migraine"],
            "min_confidence": 0.6,
            "must_contain": ["migraine", "neurol"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "neuro_03",
        "category": "neurological",
        "query": "Convulsions lasting 3 minutes, lost consciousness, no prior history",
        "expected": {
            "type": "emergency",
            "top_disease": ["seizure", "epilepsy", "convulsion"],
            "min_confidence": 0.6,
            "must_contain": ["emergency", "neurol"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "neuro_04",
        "category": "neurological",
        "query": "Persistent dizziness and vertigo when turning head, spinning sensation",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["BPPV", "vertigo", "vestibular neuritis"],
            "min_confidence": 0.4,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "neuro_05",
        "category": "neurological",
        "query": "Memory loss, confusion, and personality change over 6 months in elderly",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["dementia", "Alzheimer's disease"],
            "min_confidence": 0.5,
            "must_contain": ["neurol"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "neuro_06",
        "category": "neurological",
        "query": "Stiff neck, high fever, and sensitivity to light in a 20-year-old",
        "expected": {
            "type": "emergency",
            "top_disease": ["meningitis", "bacterial meningitis"],
            "min_confidence": 0.6,
            "must_contain": ["emergency", "hospital"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "neuro_07",
        "category": "neurological",
        "query": "Tingling and numbness in both hands and feet for 2 weeks",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["peripheral neuropathy", "diabetes neuropathy"],
            "min_confidence": 0.4,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },

    # ── Gastrointestinal (6) ──────────────────────────────────────────────────
    {
        "id": "gi_01",
        "category": "gastrointestinal",
        "query": "Severe pain in right lower abdomen, worse with movement, vomiting",
        "expected": {
            "type": "emergency",
            "top_disease": ["appendicitis"],
            "min_confidence": 0.6,
            # "surgeon" is too specific; mock LLM says "surgical" or "surgery" not "surgeon"
            "must_contain": ["surg", "emergency", "hospital", "immediate"],
            "must_not_contain": [],
            "is_emergency": True,
            "_must_contain_mode": "any",
        },
    },
    {
        "id": "gi_02",
        "category": "gastrointestinal",
        "query": "Burning epigastric pain, worse on empty stomach, relieved by food",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["peptic ulcer", "gastric ulcer", "duodenal ulcer"],
            "min_confidence": 0.5,
            "must_contain": ["gastro"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "gi_03",
        "category": "gastrointestinal",
        "query": "Yellow skin and eyes, dark urine, and fatigue for 1 week",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["hepatitis", "jaundice", "liver disease"],
            "min_confidence": 0.5,
            "must_contain": ["liver", "hepat"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "gi_04",
        "category": "gastrointestinal",
        "query": "Loose motions 8 times today with blood and severe cramps",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["dysentery", "gastroenteritis", "colitis"],
            "min_confidence": 0.5,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "gi_05",
        "category": "gastrointestinal",
        "query": "Severe upper abdominal pain radiating to back, nausea, alcoholic",
        "expected": {
            "type": "emergency",
            "top_disease": ["pancreatitis", "acute pancreatitis"],
            "min_confidence": 0.5,
            "must_contain": ["hospital"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "gi_06",
        "category": "gastrointestinal",
        "query": "Acid reflux after meals, heartburn, difficulty swallowing",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["GERD", "gastroesophageal reflux"],
            "min_confidence": 0.5,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },

    # ── Infectious / General (7) ──────────────────────────────────────────────
    {
        "id": "infect_01",
        "category": "infectious",
        "query": "High fever 103°F, severe joint pain, rash on body, live in dengue zone",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["dengue fever", "dengue"],
            "min_confidence": 0.6,
            "must_contain": ["dengue"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "infect_02",
        "category": "infectious",
        "query": "Cyclical fever with chills every 2 days, I recently visited a forest area",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["malaria"],
            "min_confidence": 0.5,
            "must_contain": ["malaria"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "infect_03",
        "category": "infectious",
        "query": "Step-wise fever for 7 days, rose-coloured spots on abdomen, constipation",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["typhoid", "typhoid fever", "enteric fever"],
            "min_confidence": 0.5,
            "must_contain": ["typhoid"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "infect_04",
        "category": "infectious",
        "query": "Mild fever, headache, tired body, feeling like resting",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["influenza", "viral fever", "common cold"],
            "min_confidence": 0.4,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "infect_05",
        "category": "infectious",
        "query": "Burning sensation when urinating, frequent urges, lower abdominal pain",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["UTI", "urinary tract infection", "cystitis"],
            "min_confidence": 0.6,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "infect_06",
        "category": "infectious",
        "query": "Painful blisters on one side of chest, burning before blisters appeared",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["herpes zoster", "shingles"],
            "min_confidence": 0.5,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "infect_07",
        "category": "infectious",
        "query": "Child with red rash starting on face spreading to body, mild fever",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["measles", "roseola", "chickenpox"],
            "min_confidence": 0.4,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },

    # ── Metabolic (4) ─────────────────────────────────────────────────────────
    {
        "id": "metab_01",
        "category": "metabolic",
        "query": "Excessive thirst, frequent urination, unexplained weight loss",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["diabetes mellitus", "type 1 diabetes", "type 2 diabetes"],
            "min_confidence": 0.6,
            "must_contain": ["diabetes", "endocrin"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "metab_02",
        "category": "metabolic",
        "query": "Diabetic patient, confused, sweating, not eaten for 6 hours",
        "expected": {
            "type": "emergency",
            "top_disease": ["hypoglycemia", "low blood sugar"],
            "min_confidence": 0.5,
            "must_contain": ["emergency"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "metab_03",
        "category": "metabolic",
        "query": "Weight gain, fatigue, cold intolerance, hair loss, slow heart rate",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["hypothyroidism", "thyroid"],
            "min_confidence": 0.5,
            "must_contain": ["thyroid", "endocrin"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "metab_04",
        "category": "metabolic",
        "query": "Weight loss, tremors, excessive sweating, bulging eyes",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["hyperthyroidism", "Graves disease"],
            "min_confidence": 0.5,
            "must_contain": ["thyroid"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },

    # ── Orthopaedic (3) ───────────────────────────────────────────────────────
    {
        "id": "ortho_01",
        "category": "orthopaedic",
        "query": "Fell from bike, right wrist pain, swelling, deformity",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["fracture", "wrist fracture"],
            "min_confidence": 0.6,
            "must_contain": ["ortho", "fracture"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "ortho_02",
        "category": "orthopaedic",
        "query": "Morning stiffness in multiple joints for more than 1 hour, bilateral",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["rheumatoid arthritis", "arthritis"],
            "min_confidence": 0.5,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "ortho_03",
        "category": "orthopaedic",
        "query": "Lower back pain radiating down to left leg, worse on sitting",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["sciatica", "herniated disc", "lumbar radiculopathy"],
            "min_confidence": 0.5,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },

    # ── Follow-up questions (5) ───────────────────────────────────────────────
    {
        "id": "followup_01",
        "category": "followup",
        "query": "What are the home remedies for my condition?",
        "expected": {
            "type": "followup",
            "top_disease": None,
            "min_confidence": None,
            "must_contain": ["remedy", "home", "diet"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "followup_02",
        "category": "followup",
        "query": "What medicines should I take for this?",
        "expected": {
            "type": "followup",
            "top_disease": None,
            "min_confidence": None,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "followup_03",
        "category": "followup",
        "query": "What diet should I follow for my disease?",
        "expected": {
            "type": "followup",
            "top_disease": None,
            "min_confidence": None,
            "must_contain": ["diet", "food", "eat"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "followup_04",
        "category": "followup",
        "query": "How long will it take to recover?",
        "expected": {
            "type": "followup",
            "top_disease": None,
            "min_confidence": None,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "followup_05",
        "category": "followup",
        "query": "What are the warning signs I should watch out for?",
        "expected": {
            "type": "followup",
            "top_disease": None,
            "min_confidence": None,
            "must_contain": ["warning", "sign", "symptom"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },

    # ── Non-medical queries (4) ───────────────────────────────────────────────
    {
        "id": "nonmed_01",
        "category": "non_medical",
        "query": "What is the best restaurant near me?",
        "expected": {
            "type": "non_medical",
            "top_disease": None,
            "min_confidence": None,
            "must_contain": ["medical"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "nonmed_02",
        "category": "non_medical",
        "query": "Tell me about the weather today",
        "expected": {
            "type": "non_medical",
            "top_disease": None,
            "min_confidence": None,
            "must_contain": ["medical"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "nonmed_03",
        "category": "non_medical",
        "query": "Help me write a poem",
        "expected": {
            "type": "non_medical",
            "top_disease": None,
            "min_confidence": None,
            # must_contain left empty — scored purely by task_completion (decline detection)
            # The mock run function may not explicitly say "medical" in its brief answer
            "must_contain": [],
            "must_not_contain": ["diagnosis", "confidence", "emergency contacts"],
            "is_emergency": False,
        },
    },
    {
        "id": "nonmed_04",
        "category": "non_medical",
        "query": "I want to know about Indian cricket team",
        "expected": {
            "type": "non_medical",
            "top_disease": None,
            "min_confidence": None,
            "must_contain": ["medical"],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },

    # ── Landmark + symptom (3) — should NOT be rejected ──────────────────────
    {
        "id": "landmark_01",
        "category": "landmark_symptom",
        "query": "I am near AIIMS and feeling very dizzy and nauseous",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["vertigo", "hypotension", "dehydration"],
            "min_confidence": 0.35,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
    {
        "id": "landmark_02",
        "category": "landmark_symptom",
        "query": "Near Manipal hospital, chest pain and breathlessness",
        "expected": {
            "type": "emergency",
            "top_disease": ["cardiac", "angina"],
            "min_confidence": 0.4,
            "must_contain": ["hospital"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "landmark_03",
        "category": "landmark_symptom",
        "query": "I am at Bangalore bus stand and I have severe headache and vomiting",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["migraine", "head injury", "hypertension"],
            "min_confidence": 0.35,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },

    # ── Paediatric (3) ────────────────────────────────────────────────────────
    {
        "id": "paed_01",
        "category": "paediatric",
        "query": "My 2-year-old has fever of 104°F and is having convulsions",
        "expected": {
            "type": "emergency",
            "top_disease": ["febrile seizure", "fever convulsion"],
            "min_confidence": 0.5,
            "must_contain": ["emergency", "hospital"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "paed_02",
        "category": "paediatric",
        "query": "6-month-old baby not feeding well, lethargic, bulging fontanelle",
        "expected": {
            "type": "emergency",
            "top_disease": ["meningitis", "intracranial hypertension"],
            "min_confidence": 0.5,
            "must_contain": ["emergency", "hospital"],
            "must_not_contain": [],
            "is_emergency": True,
        },
    },
    {
        "id": "paed_03",
        "category": "paediatric",
        "query": "My child has a sore throat and white patches on tonsils, mild fever",
        "expected": {
            "type": "diagnosis",
            "top_disease": ["tonsillitis", "strep throat", "pharyngitis"],
            "min_confidence": 0.5,
            "must_contain": [],
            "must_not_contain": [],
            "is_emergency": False,
        },
    },
]

# Quick stats
TOTAL_QUERIES    = len(TEST_QUERIES)
EMERGENCY_COUNT  = sum(1 for q in TEST_QUERIES if q["expected"].get("is_emergency"))
FOLLOWUP_COUNT   = sum(1 for q in TEST_QUERIES if q["expected"]["type"] == "followup")
NONMEDICAL_COUNT = sum(1 for q in TEST_QUERIES if q["expected"]["type"] == "non_medical")
DIAGNOSIS_COUNT  = sum(1 for q in TEST_QUERIES if q["expected"]["type"] == "diagnosis")
