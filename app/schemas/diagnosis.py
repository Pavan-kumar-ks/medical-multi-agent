from pydantic import BaseModel
from typing import List


class DiagnosisItem(BaseModel):
    disease: str
    reason: str
    confidence: float
    evidence_refs: List[str] = []


class DiagnosisOutput(BaseModel):
    diagnoses: List[DiagnosisItem]