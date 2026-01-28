from typing import Any, Dict, List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
    system_info: Dict[str, Any]
    service_uptime: float

class EmbeddingData(BaseModel):
    dense_vec: List[float]
    lexical_weights: Dict[int, float]

class EmbeddingResponse(BaseModel):
    code: int
    msg: str
    data: EmbeddingData

class ErrorResponse(BaseModel):
    error_message: str
    error_code: str