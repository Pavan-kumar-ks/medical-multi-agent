from typing import Dict, Any, List
import logging

from pydantic import ValidationError
import os

from app.schemas.diagnosis import DiagnosisOutput
from app.schemas.verification import VerificationReport, ItemReport

logger = logging.getLogger("verifier")


def verifier_agent(state: Dict[str, Any], min_confidence: float | None = None) -> VerificationReport:
    """
    Validating verifier agent that returns a structured `VerificationReport`.

    - Uses Pydantic `DiagnosisOutput` to parse and validate the diagnosis payload.
    - Produces per-item reports and collects human-readable issues.
    - Returns `ok=False` if any critical problems are found.
    """
    issues: List[str] = []
    per_items: List[ItemReport] = []

    # allow env override for min_confidence
    if min_confidence is None:
        try:
            min_confidence = float(os.getenv("MED_AGENT_VERIFIER_MIN_CONF", "0.2"))
        except Exception:
            min_confidence = 0.2

    diag_payload = state.get("diagnosis")
    if diag_payload is None:
        issues.append("diagnosis key missing from state")
        return VerificationReport(ok=False, issues=issues, per_item=[])

    # Use pydantic to enforce schema shape for diagnosis
    try:
        diag = DiagnosisOutput(**diag_payload)
    except ValidationError as e:
        logger.debug("Diagnosis parsing failed: %s", e)
        issues.append(f"diagnosis parse error: {e}")
        return VerificationReport(ok=False, issues=issues, per_item=[])

    overall_ok = True

    for idx, item in enumerate(diag.diagnoses):
        valid = True
        reason = "ok"
        note = None

        # Confidence checks
        conf = item.confidence
        if conf < 0 or conf > 1:
            valid = False
            reason = "confidence out of range"
        elif conf < min_confidence:
            valid = False
            reason = f"low confidence ({conf})"

        # evidence_refs checks
        if not isinstance(item.evidence_refs, list):
            valid = False
            reason = "evidence_refs not a list"

        if valid and len(item.evidence_refs) == 0:
            note = "no evidence_refs provided"
            issues.append(f"diagnosis[{idx}] has no evidence_refs")
            overall_ok = False

        if not valid:
            issues.append(f"diagnosis[{idx}] invalid: {reason}")
            overall_ok = False

        per_items.append(ItemReport(index=idx, disease=item.disease, valid=valid, reason=reason, note=note))

    return VerificationReport(ok=overall_ok, issues=issues, per_item=per_items)
