from typing import Any, Dict

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
    system_info: Dict[str, Any]
    service_uptime: float

class EmbeddingResponse(BaseModel):
    code: int
    dense_vec: list
    lexical_weights: dict

class ErrorResponse(BaseModel):
    error_message: str
    error_code: str