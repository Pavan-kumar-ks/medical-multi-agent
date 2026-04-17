"""Conflict Detector — compares 4 panel opinions and marks structured disagreements.

Detects:
- diagnosis_mismatch   : panelists named different top diseases
- confidence_mismatch  : same disease, confidence gap > 0.3
- missing_evidence     : a diagnosis has no evidence refs
- urgency_mismatch     : panelists disagree on urgency level
"""
from typing import Dict, Any, List


_URGENCY_RANK = {"routine": 0, "urgent": 1, "emergency": 2}


def _get_top_disease(opinion: Dict[str, Any]) -> str:
    diagnoses = opinion.get("diagnoses") or []
    if diagnoses:
        return (diagnoses[0].get("disease") or "").strip().lower()
    return ""


def _get_disease_confidence(opinion: Dict[str, Any], disease_lower: str) -> float:
    for d in (opinion.get("diagnoses") or []):
        if (d.get("disease") or "").strip().lower() == disease_lower:
            return float(d.get("confidence", 0))
    return 0.0


def conflict_detector(opinions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare 4 panel opinions and return a structured conflict report.

    Args:
        opinions: list of 4 PanelOpinion dicts (one per role)

    Returns:
        {
            "conflicts": [...],
            "conflict_count": int,
            "emergency_flagged": bool,
            "all_urgencies": [...],
            "all_top_diseases": [...],
            "consensus_diseases": [...],  # diseases named by ≥2 panelists
        }
    """
    conflicts: List[Dict[str, Any]] = []

    # ── Collect per-opinion summaries ──
    top_diseases   = [_get_top_disease(op) for op in opinions]
    urgencies      = [op.get("urgency", "routine") for op in opinions]
    emergency_flag = any(
        op.get("emergency_override") or op.get("urgency") == "emergency"
        for op in opinions
    )

    # ── 1. Diagnosis mismatch ──
    unique_tops = set(d for d in top_diseases if d)
    if len(unique_tops) > 1:
        conflicts.append({
            "type": "diagnosis_mismatch",
            "description": (
                f"Panelists disagree on the primary diagnosis. "
                f"Top picks: {', '.join(unique_tops)}"
            ),
            "involved_roles": [op.get("role", "unknown") for op in opinions],
            "severity": "high" if len(unique_tops) >= 3 else "medium",
            "details": {role_op.get("role"): top_diseases[i] for i, role_op in enumerate(opinions)},
        })

    # ── 2. Confidence mismatch (same disease, wide spread) ──
    all_disease_names: List[str] = []
    for op in opinions:
        for d in (op.get("diagnoses") or []):
            name = (d.get("disease") or "").strip().lower()
            if name:
                all_disease_names.append(name)

    seen_pairs: set = set()
    for disease in set(all_disease_names):
        confidences = []
        roles_with = []
        for op in opinions:
            conf = _get_disease_confidence(op, disease)
            if conf > 0:
                confidences.append(conf)
                roles_with.append(op.get("role", "unknown"))
        if len(confidences) >= 2:
            spread = max(confidences) - min(confidences)
            key = disease
            if spread > 0.3 and key not in seen_pairs:
                seen_pairs.add(key)
                conflicts.append({
                    "type": "confidence_mismatch",
                    "description": (
                        f"Confidence spread of {spread:.0%} for '{disease}' "
                        f"(range {min(confidences):.0%}–{max(confidences):.0%})"
                    ),
                    "involved_roles": roles_with,
                    "severity": "high" if spread > 0.5 else "medium",
                    "details": {"disease": disease, "spread": round(spread, 2)},
                })

    # ── 3. Missing evidence ──
    for op in opinions:
        for d in (op.get("diagnoses") or []):
            name = (d.get("disease") or "").strip()
            refs = d.get("evidence_refs") or []
            if name and not refs:
                conflicts.append({
                    "type": "missing_evidence",
                    "description": (
                        f"'{name}' proposed by {op.get('role')} has no evidence references."
                    ),
                    "involved_roles": [op.get("role", "unknown")],
                    "severity": "low",
                    "details": {"disease": name},
                })

    # ── 4. Urgency mismatch ──
    urgency_set = set(urgencies)
    if len(urgency_set) > 1:
        max_urgency = max(urgencies, key=lambda u: _URGENCY_RANK.get(u, 0))
        min_urgency = min(urgencies, key=lambda u: _URGENCY_RANK.get(u, 0))
        conflicts.append({
            "type": "urgency_mismatch",
            "description": (
                f"Urgency disagreement: range from '{min_urgency}' to '{max_urgency}'. "
                f"Safety policy: apply highest urgency."
            ),
            "involved_roles": [op.get("role", "unknown") for op in opinions],
            "severity": (
                "high"
                if _URGENCY_RANK.get(max_urgency, 0) - _URGENCY_RANK.get(min_urgency, 0) >= 2
                else "medium"
            ),
            "details": {op.get("role"): op.get("urgency") for op in opinions},
        })

    # ── Consensus diseases (≥ 2 panelists agree) ──
    from collections import Counter
    disease_counter = Counter(all_disease_names)
    consensus_diseases = [d for d, count in disease_counter.most_common() if count >= 2]

    return {
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "emergency_flagged": emergency_flag,
        "all_urgencies": urgencies,
        "all_top_diseases": top_diseases,
        "consensus_diseases": consensus_diseases,
    }
