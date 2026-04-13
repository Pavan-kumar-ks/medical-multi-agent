from pydantic import BaseModel
from typing import List, Optional


class ItemReport(BaseModel):
    index: int
    disease: str
    valid: bool
    reason: str
    note: Optional[str] = None


class VerificationReport(BaseModel):
    ok: bool
    issues: List[str] = []
    per_item: List[ItemReport] = []
