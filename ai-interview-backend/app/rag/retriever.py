from __future__ import annotations

import logging
from typing import List, Optional

from app.rag.embedding_service import embedding_service
from app.rag.milvus_client import milvus_store
from app.rag.schemas import SearchResult

logger = logging.getLogger(__name__)


class Retriever:
    """知识库检索服务"""

    def __init__(self):
        self._store = milvus_store
        self._embedding = embedding_service

    async def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """向量检索"""
        query_embedding = await self._embedding.embed_query(query)

        hits = self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            category=category,
            difficulty=difficulty,
            min_score=min_score,
        )

        results = [
            SearchResult(
                knowledge_id=h["knowledge_id"],
                content=h["content"],
                category=h["category"],
                difficulty=h["difficulty"],
                score=h["score"],
            )
            for h in hits
        ]
        logger.info(
            f"检索完成: query='{query[:50]}...' top_k={top_k} "
            f"category={category} difficulty={difficulty} got={len(results)}"
        )
        return results

    async def search_formatted(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> str:
        """检索并格式化为 Agent 可用的文本"""
        results = await self.search(
            query=query,
            top_k=top_k,
            category=category,
            difficulty=difficulty,
        )

        if not results:
            return "未找到相关知识条目"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] ({r.category}/{r.difficulty}, 相似度: {r.score:.3f})\n"
                f"{r.content}"
            )
        return "\n\n---\n\n".join(lines)


retriever = Retriever()
