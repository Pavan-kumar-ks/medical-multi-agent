from pydantic import BaseModel
from typing import List


class DiagnosisItem(BaseModel):
    disease: str
    reason: str
    confidence: float


class DiagnosisOutput(BaseModel):
    diagnoses: List[DiagnosisItem]