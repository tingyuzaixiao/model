from typing import List, Any, Dict, Optional

from pydantic import BaseModel


class RerankResponse(BaseModel):
    success: bool
    ranked_documents: List[str]
    scores: List[float]
    processing_time: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
    system_info: Dict[str, Any]
    service_uptime: float


class ErrorResponse(BaseModel):
    success: bool = False
    error_message: str
    error_code: str