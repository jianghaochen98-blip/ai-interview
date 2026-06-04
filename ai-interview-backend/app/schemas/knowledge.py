from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class KnowledgeItemCreate(BaseModel):
    title: str = ""
    content: str
    category: str = "general"
    difficulty: str = "medium"
    source: str = ""


class KnowledgeItemUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    source: Optional[str] = None


class KnowledgeItemOut(BaseModel):
    id: int
    title: str
    content: str
    category: str
    difficulty: str
    source: str
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class KnowledgeItemListOut(BaseModel):
    total: int
    items: List[KnowledgeItemOut]


class KnowledgeSearchIn(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    category: Optional[str] = None
    difficulty: Optional[str] = None
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class KnowledgeSearchOut(BaseModel):
    query: str
    results: List[KnowledgeSearchResult]


class KnowledgeSearchResult(BaseModel):
    knowledge_id: int
    content: str
    category: str
    difficulty: str
    score: float


class KnowledgeBatchCreate(BaseModel):
    items: List[KnowledgeItemCreate]
