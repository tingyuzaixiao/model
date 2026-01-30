from pydantic import BaseModel, Field


class EmbeddingsRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="查询文本")