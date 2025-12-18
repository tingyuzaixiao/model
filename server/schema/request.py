from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="查询文本")
    documents: List[str] = Field(..., min_length=1, max_length=100, description="待排序文档列表")
    top_k: Optional[int] = Field(10, ge=1, le=100, description="返回top K个结果")
    batch_size: Optional[int] = Field(None, ge=1, le=128, description="批处理大小")