from pydantic import BaseModel, Field
from typing import List, Optional


class PatientData(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    duration_days: Optional[int] = None
    severity: Optional[str] = None  # mild / moderate / severe
    age: Optional[int] = None
    gender: Optional[str] = None