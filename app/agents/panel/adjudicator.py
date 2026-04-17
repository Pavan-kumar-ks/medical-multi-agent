"""Adjudicator — applies deterministic scoring policy to resolve panel conflicts.

Scoring formula:
  final_score = 0.5 * evidence_strength
              + 0.3 * panel_agreement
              + 0.2 * clinical_fit

Priority order (overrides score):
  1. Safety override  — life-threatening possibilities always surfaced
  2. Evidence quality — evidence-backed beats unsupported
  3. Consensus score  — formula above
"""
import json
from collections import defaultdict
from typing import Dict, Any, List

from app.config import llm_call


_URGENCY_RANK = {"routine": 0, "urgent": 1, "emergency": 2}


# ── Deterministic scoring ─────────────────────────────────────────────────────

def _score_disease(
    disease_lower: str,
    opinions: List[Dict[str, Any]],
    conflict_report: Dict[str, Any],
) -> float:
    """Compute final_score for a disease across all panelist opinions."""
    n = len(opinions)
    if n == 0:
        return 0.0

    confidences, evidence_counts = [], []
    for op in opinions:
        for d in (op.get("diagnoses") or []):
            if (d.get("disease") or "").strip().lower() == disease_lower:
                confidences.append(float(d.get("confidence", 0)))
                evidence_counts.append(len(d.get("evidence_refs") or []))

    if not confidences:
        return 0.0

    # panel_agreement: fraction of panelists who mentioned this disease
    panel_agreement = len(confidences) / n

    # evidence_strength: fraction of mentioning panelists who cited evidence
    evidence_strength = sum(1 for e in evidence_counts if e > 0) / len(evidence_counts)

    # clinical_fit: average confidence across mentioning panelists
    clinical_fit = sum(confidences) / len(confidences)

    return 0.5 * evidence_strength + 0.3 * panel_agreement + 0.2 * clinical_fit


def _build_candidate_list(opinions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate all mentioned diseases with their best metadata."""
    disease_map: Dict[str, Dict] = {}
    for op in opinions:
        for d in (op.get("diagnoses") or []):
            name = (d.get("disease") or "").strip()
            key = name.lower()
            if not key:
                continue
            if key not in disease_map:
                disease_map[key] = {
                    "disease": name,
                    "all_reasons": [],
                    "all_refs": [],
                    "all_confidences": [],
                    "mentioning_roles": [],
                }
            disease_map[key]["all_reasons"].append(d.get("reason", ""))
            disease_map[key]["all_refs"].extend(d.get("evidence_refs") or [])
            disease_map[key]["all_confidences"].append(float(d.get("confidence", 0)))
            disease_map[key]["mentioning_roles"].append(op.get("role", "unknown"))
    return list(disease_map.values())


def _resolve_with_llm(
    opinions: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
    scored_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Ask LLM to generate human-readable adjudication summary."""
    top3 = scored_candidates[:3]

    prompt = f"""You are the ADJUDICATOR for a medical expert panel.

Four specialists (Primary Diagnostician, Skeptical Reviewer, Evidence Auditor, Safety Triage Lead) have independently reviewed a patient and disagreed.

Panel Opinions Summary:
{json.dumps([{
    "role": op.get("role"),
    "top_diagnosis": (op.get("diagnoses") or [{}])[0].get("disease", "N/A"),
    "urgency": op.get("urgency"),
    "red_flags": op.get("red_flags", [])[:3],
    "notes": op.get("notes", "")
} for op in opinions], indent=2)}

Conflicts Detected:
{json.dumps(conflicts, indent=2)}

Deterministically Scored Top Candidates (by formula: 0.5*evidence + 0.3*agreement + 0.2*fit):
{json.dumps(top3, indent=2)}

Your task — return ONLY valid JSON:
{{
  "conflict_reason": "1-2 sentences: why the panel disagreed",
  "why_final_won": "1-2 sentences: why the top-scored diagnosis is the best choice",
  "resolving_test": "single most important test that would quickly confirm or overturn the final diagnosis",
  "alternate_considered": ["list of other diseases seriously considered"],
  "uncertainty_flag": false,
  "adjudicator_notes": "brief note on panel quality or remaining uncertainty"
}}

Set uncertainty_flag = true if conflicts remain unresolved and a specialist referral is needed.
"""

    response = llm_call(prompt)
    try:
        data = json.loads(response)
        return data
    except Exception:
        start, end = response.find("{"), response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except Exception:
                pass
    return {
        "conflict_reason": "Panel disagreed on primary diagnosis and urgency.",
        "why_final_won": "Selected by highest combined evidence + agreement score.",
        "resolving_test": "Clinical examination and basic blood panel.",
        "alternate_considered": [],
        "uncertainty_flag": True,
        "adjudicator_notes": "Adjudicator LLM parse failed; deterministic result used.",
    }


# ── Main adjudicator ─────────────────────────────────────────────────────────

def adjudicator(
    opinions: List[Dict[str, Any]],
    conflict_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve panel conflicts using deterministic scoring + LLM summary.

    Returns a PanelDecision dict containing:
    - final_diagnoses   : ranked list with final_score
    - panel_summary     : what the panel agreed/disagreed on
    - emergency_triggered
    - uncertainty_flag
    - conflict_reason, why_final_won, resolving_test (for user display)
    """

    # ── Safety override first ──────────────────────────────────────────────
    emergency_triggered = conflict_report.get("emergency_flagged", False)

    # Collect "cannot_miss" diagnoses from safety triage lead
    cannot_miss: List[str] = []
    safety_ops = [op for op in opinions if op.get("role") == "safety_triage_lead"]
    for sop in safety_ops:
        cannot_miss.extend(sop.get("cannot_miss_diagnoses") or [])

    # ── Score all candidate diseases ───────────────────────────────────────
    candidates = _build_candidate_list(opinions)
    for c in candidates:
        c["final_score"] = round(
            _score_disease(c["disease"].lower(), opinions, conflict_report), 3
        )
        c["panel_agreement_count"] = len(c["mentioning_roles"])

    # Sort: cannot-miss diseases float to top, then by score
    cannot_miss_lower = [d.lower() for d in cannot_miss]

    def _sort_key(c):
        is_cannot_miss = any(cm in c["disease"].lower() for cm in cannot_miss_lower)
        return (0 if is_cannot_miss else 1, -c["final_score"])

    candidates.sort(key=_sort_key)

    # Build final_diagnoses list (de-duped, top 3)
    final_diagnoses = []
    for c in candidates[:3]:
        avg_conf = (
            sum(c["all_confidences"]) / len(c["all_confidences"])
            if c["all_confidences"] else 0.0
        )
        final_diagnoses.append({
            "disease":        c["disease"],
            "confidence":     round(avg_conf, 2),
            "final_score":    c["final_score"],
            "reason":         c["all_reasons"][0] if c["all_reasons"] else "",
            "evidence_refs":  list(set(c["all_refs"]))[:5],
            "panel_agreement": c["panel_agreement_count"],
        })

    # ── Resolve urgency by safety policy (take highest) ───────────────────
    all_urgencies = conflict_report.get("all_urgencies", ["routine"])
    resolved_urgency = max(all_urgencies, key=lambda u: _URGENCY_RANK.get(u, 0))

    # ── LLM adjudication summary ───────────────────────────────────────────
    llm_adj = _resolve_with_llm(opinions, conflict_report.get("conflicts", []), candidates[:3])

    # ── Panel summary string ───────────────────────────────────────────────
    n_conflicts   = conflict_report.get("conflict_count", 0)
    consensus_list = conflict_report.get("consensus_diseases", [])

    if n_conflicts == 0:
        panel_summary = f"Full consensus: all panelists agreed on {final_diagnoses[0]['disease'] if final_diagnoses else 'the diagnosis'}."
    else:
        panel_summary = (
            f"{n_conflicts} conflict(s) detected. "
            f"Consensus on: {', '.join(consensus_list) if consensus_list else 'none'}. "
            f"Resolved by deterministic scoring (evidence × agreement × fit)."
        )

    return {
        "final_diagnoses":     final_diagnoses,
        "emergency_triggered": emergency_triggered,
        "resolved_urgency":    resolved_urgency,
        "cannot_miss":         cannot_miss,
        "panel_summary":       panel_summary,
        "conflict_reason":     llm_adj.get("conflict_reason", ""),
        "why_final_won":       llm_adj.get("why_final_won", ""),
        "resolving_test":      llm_adj.get("resolving_test", ""),
        "alternate_considered": llm_adj.get("alternate_considered", []),
        "uncertainty_flag":    llm_adj.get("uncertainty_flag", False),
        "adjudicator_notes":   llm_adj.get("adjudicator_notes", ""),
        "conflict_count":      n_conflicts,
        "consensus_diseases":  consensus_list,
    }
