def apply_medical_rules(patient_data):
    risks = []

    symptoms = [s.lower() for s in patient_data.symptoms]

    # 🚨 Emergency rules
    if "chest pain" in symptoms:
        risks.append("Possible cardiac emergency")

    if "shortness of breath" in symptoms:
        risks.append("Respiratory distress")

    # 🧪 Dengue risk rule
    if "fever" in symptoms and "body pain" in symptoms:
        risks.append("Possible dengue - monitor platelets")

    # ⚠️ Severe condition
    if patient_data.severity == "severe":
        risks.append("Severe condition - immediate medical attention needed")

    return risks