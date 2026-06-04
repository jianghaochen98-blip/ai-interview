from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeItem:
    """知识条目"""

    content: str
    category: str = "general"
    difficulty: str = "medium"
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    knowledge_id: Optional[int] = None


@dataclass
class SearchResult:
    """检索结果"""

    knowledge_id: int
    content: str
    category: str
    difficulty: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchRequest:
    """检索请求"""

    query: str
    top_k: int = 5
    category: Optional[str] = None
    difficulty: Optional[str] = None
    min_score: float = 0.0
